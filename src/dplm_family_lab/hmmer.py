"""Pfam curated GA scoring through HMMER or the optional pyhmmer backend."""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile

from .common import sha256_file, write_fasta


def hmm_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    return (gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw).decode("utf-8")


def read_hmm_metadata(path: str | Path) -> dict:
    values = {}
    for line in hmm_text(path).splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] in {"NAME", "ACC", "LENG", "GA"}:
            values[fields[0]] = fields[1:]
        if line.startswith("HMM "):
            break
    if not all(key in values for key in ("ACC", "LENG", "GA")):
        raise ValueError("The HMM must contain ACC, LENG and curated GA thresholds")
    if hmm_text(path).count("\n//") != 1:
        raise ValueError("Supply exactly one family HMM")
    ga = [float(x.rstrip(";")) for x in values["GA"]]
    if len(ga) != 2:
        raise ValueError("The HMM GA line must specify sequence and domain thresholds")
    return {"accession": values["ACC"][0], "family": values["ACC"][0].split(".")[0],
            "name": " ".join(values.get("NAME", [])), "length": int(values["LENG"][0]),
            "ga_sequence": ga[0], "ga_domain": ga[1], "sha256": sha256_file(path)}


def parse_domtblout(text: str, metadata: dict, min_coverage: float = 0.8) -> dict[str, dict]:
    """Parse hmmsearch (not hmmscan) --domtblout with inclusive model coordinates."""
    rows = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split(maxsplit=22)
        if len(cols) < 22:
            raise ValueError(f"Malformed HMMER domain row: {line[:100]}")
        if cols[4] != metadata["accession"] or int(cols[5]) != metadata["length"]:
            raise ValueError("Domain results refer to a different HMM")
        row = rows.setdefault(cols[0], {"sequence_score": float(cols[7]), "domains": []})
        start, end = int(cols[15]), int(cols[16])
        if not (1 <= start <= end <= metadata["length"]):
            raise ValueError("Invalid HMM domain coordinates")
        row["domains"].append({"score": float(cols[13]), "hmm_from": start,
                               "hmm_to": end, "coverage": (end - start + 1) / metadata["length"]})
    return {name: _summarize(row, metadata, min_coverage) for name, row in rows.items()}


def _summarize(row: dict, metadata: dict, min_coverage: float) -> dict:
    seq_pass = row["sequence_score"] >= metadata["ga_sequence"]
    passing = [d for d in row["domains"] if seq_pass and d["score"] >= metadata["ga_domain"]]
    best = max(passing, key=lambda d: (d["coverage"], d["score"]), default=None)
    return {"sequence_score": row["sequence_score"], "ga_pass": bool(passing),
            "hmm_coverage": best["coverage"] if best else 0.0,
            "domain_score": best["score"] if best else None,
            "family_pass": any(d["coverage"] + 1e-12 >= min_coverage for d in passing),
            "domains": row["domains"]}


def _empty() -> dict:
    return {"sequence_score": None, "ga_pass": False, "hmm_coverage": 0.0,
            "domain_score": None, "family_pass": False, "domains": []}


def score_sequences(records, hmm_path: str | Path, binary: str = "hmmsearch",
                    backend: str = "auto", min_coverage: float = 0.8,
                    work_dir: str | Path | None = None, cpu: int = 1) -> dict[str, dict]:
    """All sequences receive a row, including misses. No E-value replacement for GA."""
    if not 0 < min_coverage <= 1:
        raise ValueError("HMM coverage must be in (0,1]")
    records = list(records)
    if len({r.id for r in records}) != len(records):
        raise ValueError("HMM input IDs must be unique")
    metadata = read_hmm_metadata(hmm_path)
    if not records:
        return {}
    if backend not in {"auto", "external", "pyhmmer"}:
        raise ValueError("HMM backend must be auto, external or pyhmmer")
    resolved = shutil.which(binary)
    if backend == "auto":
        backend = "external" if resolved else "pyhmmer"
    if backend == "pyhmmer":
        if importlib.util.find_spec("pyhmmer") is None:
            raise RuntimeError("Install HMMER (hmmsearch) on Linux/WSL, or install optional pyhmmer; no HMM scores were fabricated.")
        results = _score_pyhmmer(records, hmm_path, metadata, min_coverage, cpu)
    else:
        if not resolved:
            raise RuntimeError(f"HMMER executable not found: {binary}. Install HMMER or select pyhmmer.")
        if work_dir is not None:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hmmer_", dir=work_dir) as tmp:
            tmp = Path(tmp)
            fasta = tmp / "sequences.fasta"
            model = tmp / "family.hmm"
            domtbl = tmp / "domains.tsv"
            write_fasta(records, fasta)
            model.write_text(hmm_text(hmm_path), encoding="utf-8")
            run = subprocess.run([resolved, "--cut_ga", "--noali", "--cpu", str(cpu),
                                  "--domtblout", str(domtbl), str(model), str(fasta)],
                                 capture_output=True, text=True, check=False)
            if run.returncode:
                raise RuntimeError(f"hmmsearch failed ({run.returncode}): {run.stderr[-3000:]}")
            results = parse_domtblout(domtbl.read_text(), metadata, min_coverage)
    unknown = set(results) - {r.id for r in records}
    if unknown:
        raise RuntimeError(f"HMMER returned unknown sequence IDs: {sorted(unknown)[:3]}")
    return {r.id: results.get(r.id, _empty()) for r in records}


def _score_pyhmmer(records, path, metadata, coverage, cpu):
    import pyhmmer

    alphabet = pyhmmer.easel.Alphabet.amino()
    # HMMFile transparently handles gzip; this route does not require a subprocess.
    with pyhmmer.plan7.HMMFile(str(path)) as handle:
        hmm = handle.read()
    sequences = pyhmmer.easel.DigitalSequenceBlock(alphabet)
    for record in records:
        sequences.append(pyhmmer.easel.TextSequence(name=record.id.encode("ascii"),
                                                    sequence=record.sequence).digitize(alphabet))
    output = {}
    for hits in pyhmmer.hmmer.hmmsearch(hmm, sequences, cpus=cpu, bit_cutoffs="gathering"):
        for hit in hits:
            row = {"sequence_score": float(hit.score), "domains": []}
            for domain in hit.domains:
                if not domain.reported:
                    continue
                start, end = domain.alignment.hmm_from, domain.alignment.hmm_to
                row["domains"].append({"score": float(domain.score), "hmm_from": start,
                                       "hmm_to": end, "coverage": (end - start + 1) / metadata["length"]})
            # pyhmmer 0.12 returns str; 0.11 and older return bytes.
            name = hit.name.decode("ascii") if isinstance(hit.name, bytes) else hit.name
            output[name] = _summarize(row, metadata, coverage)
    return output
