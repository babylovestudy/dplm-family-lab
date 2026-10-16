"""Download, clean, isolate and freeze the public family adaptation dataset."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import time

import requests

from .common import SequenceRecord, read_fasta, sha256_file, write_fasta, write_json
from .hmmer import read_hmm_metadata, score_sequences
from .similarity import METRIC, connected_clusters, cross_split_audit, sequence_hash

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checked(path, expected):
    if not Path(path).is_file() or sha256_file(path) != expected:
        raise ValueError(f"Frozen or cached file checksum mismatch: {path}")


def fetch(config: dict) -> dict:
    """Fetch only. Individual immutable response pages allow interrupted restart."""
    cfg = config["data"]
    raw = Path(cfg.get("raw_dir", "data/raw"))
    raw.mkdir(parents=True, exist_ok=True)
    provenance_path = raw / "provenance.json"
    fasta_path = raw / "uniprot.fasta"
    if provenance_path.exists():
        provenance = _load(provenance_path)
        if provenance.get("query") != cfg["query"]:
            raise ValueError("Cached UniProt query differs; choose a new raw_dir")
        _checked(fasta_path, provenance["fasta_sha256"])
    else:
        pages_dir = raw / "pages"
        pages_dir.mkdir(exist_ok=True)
        params = {"query": cfg["query"], "format": "tsv", "size": 500,
                  "fields": "accession,length,organism_id,reviewed,sequence"}
        next_url = requests.Request("GET", "https://rest.uniprot.org/uniprotkb/search", params=params).prepare().url
        pages, rows, visited = [], [], set()
        with requests.Session() as session:
            session.headers["User-Agent"] = "dplm-family-lab/0.1 public-research-data"
            while next_url:
                if next_url in visited:
                    raise RuntimeError("UniProt pagination returned a repeated page")
                visited.add(next_url)
                key = hashlib.sha256(next_url.encode()).hexdigest()
                page_path, meta_path = pages_dir / f"{key}.tsv", pages_dir / f"{key}.json"
                if meta_path.exists():
                    meta = _load(meta_path)
                    _checked(page_path, meta["sha256"])
                    if meta["url"] != next_url:
                        raise ValueError("Pagination cache URL mismatch")
                else:
                    response = session.get(next_url, timeout=(15, 120))
                    response.raise_for_status()
                    page_path.write_bytes(response.content)
                    meta = {"url": next_url, "retrieved_at": _now(), "sha256": sha256_file(page_path),
                            "uniprot_release": response.headers.get("X-UniProt-Release"),
                            "uniprot_release_date": response.headers.get("X-UniProt-Release-Date"),
                            "http_date": response.headers.get("Date"),
                            "total_results": response.headers.get("X-Total-Results"),
                            "next_url": response.links.get("next", {}).get("url")}
                    write_json(meta_path, meta)
                table = list(csv.DictReader(io.StringIO(page_path.read_text(encoding="utf-8-sig")), delimiter="\t"))
                for row in table:
                    if not row.get("Entry") or not row.get("Sequence"):
                        raise ValueError("UniProt response lacks Entry or Sequence fields")
                    if int(row["Length"]) != len(row["Sequence"]):
                        raise ValueError("UniProt sequence length disagrees with the response")
                rows.extend(table)
                pages.append(meta)
                next_url = meta["next_url"]
        releases = {p["uniprot_release"] for p in pages if p["uniprot_release"]}
        if len(releases) > 1:
            raise RuntimeError("UniProt changed release between cached pages; use a fresh raw_dir")
        if len({r["Entry"] for r in rows}) != len(rows):
            raise RuntimeError("Duplicate accession across UniProt pages; refetch a stable release")
        reported = {int(p["total_results"]) for p in pages if p["total_results"] is not None}
        if reported and (len(reported) != 1 or len(rows) != next(iter(reported))):
            raise RuntimeError("UniProt pagination count differs from X-Total-Results")
        if not rows:
            raise ValueError("UniProt query returned no sequences")
        records = [SequenceRecord(r["Entry"], r["Sequence"],
                                  f"taxid={r.get('Organism (ID)', '')} reviewed={r.get('Reviewed', '')}") for r in rows]
        write_fasta(sorted(records, key=lambda r: r.id), fasta_path)
        provenance = {"schema_version": 1, "source": "UniProtKB REST", "query": cfg["query"],
                      "completed_at": _now(), "records": len(records), "pages": pages,
                      "uniprot_release": next(iter(releases), None),
                      "uniprot_release_date": pages[0]["uniprot_release_date"],
                      "fasta_sha256": sha256_file(fasta_path)}
        write_json(provenance_path, provenance)
    hmm_cfg = config["hmm"]
    hmm_path = Path(hmm_cfg["path"])
    hmm_provenance = Path(str(hmm_path) + ".provenance.json")
    if hmm_path.exists():
        if not hmm_provenance.exists():
            raise ValueError(f"Existing HMM has no download provenance: {hmm_path}; use prepare(download=False) for an explicitly local file")
        _checked(hmm_path, _load(hmm_provenance)["sha256"])
    else:
        response = requests.get(hmm_cfg["url"], timeout=(15, 120))
        response.raise_for_status()
        raw_bytes = response.content
        plain = gzip.decompress(raw_bytes) if raw_bytes.startswith(b"\x1f\x8b") else raw_bytes
        hmm_path.parent.mkdir(parents=True, exist_ok=True)
        hmm_path.write_bytes(plain)
        metadata = read_hmm_metadata(hmm_path)
        if metadata["family"] != cfg["family"]:
            raise ValueError("Downloaded HMM family does not match the configured family")
        write_json(hmm_provenance, {"url": hmm_cfg["url"], "retrieved_at": _now(),
                                   "http_date": response.headers.get("Date"),
                                   "last_modified": response.headers.get("Last-Modified"),
                                   "download_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                                   "sha256": sha256_file(hmm_path), "model": metadata})
    return {"fasta_path": str(fasta_path), "hmm_path": str(hmm_path),
            "provenance_path": str(provenance_path), "records": provenance["records"],
            "uniprot_release": provenance.get("uniprot_release")}


def clean_records(records, min_length: int, max_length: int):
    counts = {"input": 0, "invalid_alphabet": 0, "outside_length": 0, "exact_duplicates": 0}
    retained, aliases = {}, {}
    seen_ids = set()
    for record in sorted(records, key=lambda r: r.id):
        counts["input"] += 1
        if record.id in seen_ids:
            raise ValueError(f"Duplicate FASTA ID: {record.id}")
        seen_ids.add(record.id)
        sequence = record.sequence.upper()
        if not sequence or not set(sequence) <= STANDARD_AA:
            counts["invalid_alphabet"] += 1
            continue
        if not min_length <= len(sequence) <= max_length:
            counts["outside_length"] += 1
            continue
        if sequence in retained:
            counts["exact_duplicates"] += 1
            aliases.setdefault(retained[sequence].id, []).append(record.id)
        else:
            retained[sequence] = SequenceRecord(record.id, sequence, record.description)
    result = sorted(retained.values(), key=lambda r: r.id)
    counts["retained"] = len(result)
    return result, counts, aliases


def assign_splits(clusters: dict[str, str], ratios=(0.8, 0.1, 0.1), seed=42):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Three positive split fractions summing to one are required")
    groups = {}
    for record_id, cluster in sorted(clusters.items()):
        groups.setdefault(cluster, []).append(record_id)
    if len(groups) < 3:
        raise ValueError("At least three independent homology components are required; never split a component to fill partitions")
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: -len(item[1]))
    names = ("train", "val", "test")
    targets = {name: ratios[i] * len(clusters) for i, name in enumerate(names)}
    sizes = dict.fromkeys(names, 0)
    assignments = {}
    for index, (_, members) in enumerate(items):
        missing = [name for name in names if sizes[name] == 0]
        choices = missing if len(items) - index == len(missing) else names
        # Minimize final count error, reserving whole components for every split.
        chosen = min(choices, key=lambda name: sum((sizes[n] + (len(members) if n == name else 0) - targets[n]) ** 2 for n in names))
        sizes[chosen] += len(members)
        assignments.update({member: chosen for member in members})
    return assignments, sizes


def _preparation_signature(config, raw_path, hmm_path):
    cfg = config["data"]
    return {"raw_sha256": sha256_file(raw_path), "hmm_sha256": sha256_file(hmm_path),
            "family": cfg["family"], "query": cfg["query"], "min_length": cfg["min_length"],
            "max_length": cfg["max_length"], "min_hmm_coverage": cfg["min_hmm_coverage"],
            "identity": cfg["identity"], "coverage": cfg["coverage"],
            "split": cfg["split"], "seed": cfg["seed"], "metric": METRIC}


def _write_training_reference(output, manifest):
    metadata = manifest["report"]["hmm"]
    write_json(output / "training_ref.json", {"schema_version": 1,
               "family": manifest["signature"]["family"],
               "hmm_sha256": metadata["sha256"], "hmm_metadata": metadata,
               "min_hmm_coverage": manifest["signature"]["min_hmm_coverage"],
               "train_fasta_sha256": manifest["files"]["train.fasta"],
               "split_manifest_sha256": sha256_file(output / "split_manifest.json"),
               "source": manifest["source"], "hmm_source": manifest["hmm_source"]})


def prepare(config: dict, download: bool = True) -> dict:
    cfg = config["data"]
    if download:
        fetch(config)
    raw = Path(cfg.get("raw_fasta", Path(cfg["raw_dir"]) / "uniprot.fasta"))
    hmm_path = Path(config["hmm"]["path"])
    output = Path(cfg["processed_dir"])
    output.mkdir(parents=True, exist_ok=True)
    signature = _preparation_signature(config, raw, hmm_path)
    manifest_path = output / "split_manifest.json"
    if manifest_path.exists():
        manifest = _load(manifest_path)
        if manifest["signature"] != signature:
            raise ValueError("Frozen split inputs or configuration changed; use a new processed_dir")
        for name, digest in manifest["files"].items():
            _checked(output / name, digest)
        if not (output / "training_ref.json").exists():
            # Recover a process interrupted immediately after atomic manifest
            # creation. All frozen input/output hashes were checked above.
            _write_training_reference(output, manifest)
        reference = _load(output / "training_ref.json")
        if reference["split_manifest_sha256"] != sha256_file(manifest_path):
            raise ValueError("Frozen training reference differs from split manifest")
        return manifest["report"]
    provenance_path = Path(cfg["raw_dir"]) / "provenance.json"
    if provenance_path.exists():
        source = _load(provenance_path)
        _checked(raw, source["fasta_sha256"])
        if source.get("query") != cfg["query"]:
            raise ValueError("Offline source provenance query differs from the configuration")
    else:
        source = {"source": "local_offline", "path": str(raw), "sha256": sha256_file(raw),
                  "note": "Local input; no UniProt release provenance was supplied"}
    metadata = read_hmm_metadata(hmm_path)
    if metadata["family"] != cfg["family"]:
        raise ValueError("Configured HMM and target family differ")
    candidates, cleaning, aliases = clean_records(read_fasta(raw), cfg["min_length"], cfg["max_length"])
    hmm_cache = output / "hmm_scores.json"
    score_key = _digest({"sequences": [(r.id, sequence_hash(r.sequence)) for r in candidates],
                         "hmm_sha256": metadata["sha256"], "coverage": cfg["min_hmm_coverage"]})
    if hmm_cache.exists() and _load(hmm_cache).get("key") == score_key:
        scores = _load(hmm_cache)["scores"]
    else:
        scores = score_sequences(candidates, hmm_path, binary=config["hmm"].get("binary", "hmmsearch"),
                                 backend=config["hmm"].get("backend", "auto"),
                                 min_coverage=cfg["min_hmm_coverage"], work_dir=output / ".tmp")
        write_json(hmm_cache, {"key": score_key, "hmm": metadata, "scores": scores})
    retained = [r for r in candidates if scores[r.id]["family_pass"]]
    if len(retained) < 3:
        raise ValueError("Fewer than three sequences passed GA and single-domain coverage; cannot create independent splits")
    cache_path = output / "pairwise.sqlite"
    started = time.perf_counter()
    clusters, stats = connected_clusters(retained, cfg["identity"], cfg["coverage"], cache_path)
    stats["elapsed_seconds"] = time.perf_counter() - started
    splits, sizes = assign_splits(clusters, cfg["split"], cfg["seed"])
    checked = cross_split_audit(retained, splits, cfg["identity"], cfg["coverage"], cache_path)
    if not checked["passed"]:
        raise RuntimeError("Cross-split homology audit failed; no dataset was frozen")
    for name in ("train", "val", "test"):
        write_fasta([r for r in retained if splits[r.id] == name], output / f"{name}.fasta")
    warnings = []
    for i, name in enumerate(("train", "val", "test")):
        if abs(sizes[name] / len(retained) - cfg["split"][i]) > 0.1:
            warnings.append(f"{name} differs from target fraction by >10 percentage points because components remain intact")
        if sizes[name] < 30:
            warnings.append(f"{name} contains fewer than 30 sequences; interpret estimates cautiously")
    report = {"cleaning": cleaning, "hmm_pass": len(retained), "hmm_removed": len(candidates) - len(retained),
              "clusters": stats, "split_sizes": sizes, "cross_split_audit": checked,
              "warnings": warnings, "hmm": metadata, "processed_dir": str(output)}
    files = {f"{name}.fasta": sha256_file(output / f"{name}.fasta") for name in ("train", "val", "test")}
    files["hmm_scores.json"] = sha256_file(hmm_cache)
    manifest = {"schema_version": 1, "created_at": _now(), "signature": signature,
                "sequences": [{"id": r.id, "sha256": sequence_hash(r.sequence), "length": len(r.sequence),
                               "cluster": clusters[r.id], "split": splits[r.id]} for r in retained],
                "duplicate_aliases": aliases, "files": files, "report": report, "source": source,
                "hmm_source": _load(str(hmm_path) + ".provenance.json") if Path(str(hmm_path) + ".provenance.json").exists()
                              else {"source": "local_offline", "sha256": metadata["sha256"]}}
    write_json(manifest_path, manifest)
    _write_training_reference(output, manifest)
    write_json(output / "audit.json", checked)
    return report


def audit(config: dict) -> dict:
    cfg = config["data"]
    output = Path(cfg["processed_dir"])
    manifest = _load(output / "split_manifest.json")
    reference = _load(output / "training_ref.json")
    _checked(output / "split_manifest.json", reference["split_manifest_sha256"])
    _checked(config["hmm"]["path"], reference["hmm_sha256"])
    for name, digest in manifest["files"].items():
        _checked(output / name, digest)
    records, assignments = [], {}
    expected = {r["id"]: r for r in manifest["sequences"]}
    for name in ("train", "val", "test"):
        path = output / f"{name}.fasta"
        _checked(path, manifest["files"][path.name])
        for record in read_fasta(path):
            if record.id in assignments or record.id not in expected:
                raise ValueError("Repeated or unknown ID in frozen dataset")
            item = expected[record.id]
            if item["split"] != name or item["sha256"] != sequence_hash(record.sequence):
                raise ValueError("Frozen sequence or partition changed")
            assignments[record.id] = name
            records.append(record)
    if set(assignments) != set(expected):
        raise ValueError("Frozen manifest sequence is missing")
    signature = manifest["signature"]
    if cfg["identity"] != signature["identity"] or cfg["coverage"] != signature["coverage"]:
        raise ValueError("Audit thresholds differ from the frozen dataset")
    result = cross_split_audit(records, assignments, signature["identity"], signature["coverage"], output / "pairwise.sqlite")
    write_json(output / "audit.json", result)
    return result
