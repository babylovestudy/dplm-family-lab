"""Generation evaluation anchored to the immutable training data and Pfam HMM."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .common import read_fasta, sha256_file, write_json
from .data import STANDARD_AA
from .hmmer import read_hmm_metadata, score_sequences
from .similarity import METRIC, PairwiseCache


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _interval(values, draws, rng):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"value": None, "ci95": None, "n": 0}
    point = float(values.mean())
    if not draws or len(values) < 2:
        return {"value": point, "ci95": None, "n": len(values)}
    replicates = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(draws)]
    low, high = np.quantile(replicates, [0.025, 0.975])
    return {"value": point, "ci95": [float(low), float(high)], "n": len(values)}


def _diversity_interval(matrix, draws, rng):
    n = len(matrix)
    if n < 2:
        return {"value": None, "ci95": None, "n": n, "pairs": 0}
    upper = np.triu_indices(n, 1)
    point = float(matrix[upper].mean())
    ci = None
    if draws:
        replicates = []
        for _ in range(draws):
            sampled = rng.integers(0, n, n)
            replicates.append(float(matrix[sampled[upper[0]], sampled[upper[1]]].mean()))
        ci = [float(x) for x in np.quantile(replicates, [0.025, 0.975])]
    return {"value": point, "ci95": ci, "n": n, "pairs": len(upper[0])}


def evaluate(config: dict, fasta_path: str | Path, output_dir: str | Path | None = None) -> dict:
    cfg = config.get("evaluation", {})
    train_path = Path(cfg.get("train_fasta") or config.get("train", {}).get("train_fasta")
                      or Path(config["data"]["processed_dir"]) / "train.fasta")
    reference_path = Path(cfg.get("training_reference", train_path.parent / "training_ref.json"))
    if not reference_path.is_file():
        raise ValueError("Evaluation requires training_ref.json from the frozen prepare workflow")
    reference = _load(reference_path)
    hmm_path = Path(cfg.get("hmm_path", config["hmm"]["path"]))
    if sha256_file(hmm_path) != reference["hmm_sha256"]:
        raise ValueError("Evaluation HMM differs from the frozen training HMM")
    if sha256_file(train_path) != reference["train_fasta_sha256"]:
        raise ValueError("Evaluation training FASTA differs from the frozen reference")
    manifest_path = reference_path.parent / "split_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != reference["split_manifest_sha256"]:
        raise ValueError("Frozen split manifest is missing or changed")
    metadata = read_hmm_metadata(hmm_path)
    if metadata["family"] != reference["family"]:
        raise ValueError("Evaluation family and training reference differ")
    min_coverage = cfg.get("min_hmm_coverage", reference["min_hmm_coverage"])
    if min_coverage != reference["min_hmm_coverage"]:
        raise ValueError("Evaluation coverage threshold differs from the frozen reference")
    training = read_fasta(train_path)
    if not training:
        raise ValueError("Training reference FASTA is empty")
    generated = read_fasta(fasta_path)
    valid = [r for r in generated if r.sequence and set(r.sequence) <= STANDARD_AA]
    output = Path(output_dir or cfg.get("output_dir", "outputs/evaluation"))
    output.mkdir(parents=True, exist_ok=True)
    scores = score_sequences(valid, hmm_path, binary=config["hmm"].get("binary", "hmmsearch"),
                             backend=config["hmm"].get("backend", "auto"), min_coverage=min_coverage,
                             work_dir=output / ".tmp")
    multiplicity = Counter(r.sequence for r in valid)
    train_set = {r.sequence for r in training}
    nearest_by_sequence = {}
    diversity = np.zeros((len(valid), len(valid)), dtype=np.float64)
    pair_rows = []
    with PairwiseCache(output / "pairwise.sqlite") as cache:
        for record in valid:
            if record.sequence in nearest_by_sequence:
                continue
            best_identity, best_id = -1.0, None
            for train in training:
                # Identity can never exceed shorter/longer length, even with gaps.
                upper = min(len(record.sequence), len(train.sequence)) / max(len(record.sequence), len(train.sequence))
                if upper + 1e-12 < best_identity:
                    continue
                identity = cache.compare(record.sequence, train.sequence).identity
                if identity > best_identity or (identity == best_identity and (best_id is None or train.id < best_id)):
                    best_identity, best_id = identity, train.id
            nearest_by_sequence[record.sequence] = (best_identity, best_id)
        for i, a in enumerate(valid):
            for j in range(i + 1, len(valid)):
                b = valid[j]
                result = cache.compare(a.sequence, b.sequence)
                distance = 1 - result.identity
                diversity[i, j] = diversity[j, i] = distance
                pair_rows.append({"a": a.id, "b": b.id, "identity": result.identity,
                                  "distance": distance, "coverage_a": result.coverage_a,
                                  "coverage_b": result.coverage_b})
    seen = set()
    rows = []
    min_length, max_length = config["data"]["min_length"], config["data"]["max_length"]
    for record in generated:
        is_valid = bool(record.sequence) and set(record.sequence) <= STANDARD_AA
        score = scores.get(record.id, {})
        nearest, nearest_id = nearest_by_sequence.get(record.sequence, (None, None))
        rows.append({"id": record.id, "sequence": record.sequence, "length": len(record.sequence),
                     "valid_amino_acids": is_valid,
                     "within_training_length": min_length <= len(record.sequence) <= max_length,
                     "ga_pass": score.get("ga_pass", False), "family_pass": score.get("family_pass", False),
                     "hmm_coverage": score.get("hmm_coverage", 0.0),
                     "sequence_score": score.get("sequence_score"), "domain_score": score.get("domain_score"),
                     "duplicate_of_earlier": is_valid and record.sequence in seen,
                     "sequence_multiplicity": multiplicity.get(record.sequence, 0),
                     "exact_training_match": is_valid and record.sequence in train_set,
                     "nearest_training_identity": nearest, "nearest_training_id": nearest_id})
        if is_valid:
            seen.add(record.sequence)
    valid_rows = [r for r in rows if r["valid_amino_acids"]]
    draws = int(cfg.get("bootstrap_samples", 1000))
    if draws < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    metrics = {}
    for name, field in (("ga_pass_rate", "ga_pass"), ("family_pass_rate", "family_pass"),
                        ("mean_hmm_coverage", "hmm_coverage"),
                        ("exact_training_match_rate", "exact_training_match"),
                        ("mean_nearest_training_identity", "nearest_training_identity"),
                        ("within_training_length_rate", "within_training_length")):
        metrics[name] = _interval([r[field] for r in valid_rows], draws, rng)
    metrics["mean_pairwise_distance"] = _diversity_interval(diversity, draws, rng)
    n = len(valid)
    counts = {"emitted_fasta_records": len(generated), "valid_amino_acid_sequences": n,
              "invalid_amino_acid_sequences": len(generated) - n,
              "unique_valid_sequences": len(multiplicity), "excess_duplicate_records": n - len(multiplicity),
              "ga_pass": sum(r["ga_pass"] for r in valid_rows),
              "family_pass": sum(r["family_pass"] for r in valid_rows)}
    metrics["excess_duplicate_rate"] = {"value": (n - len(multiplicity)) / n if n else None,
                                         "ci95": None, "n": n}
    result = {"schema_version": 1, "input_fasta": str(fasta_path), "input_sha256": sha256_file(fasta_path),
              "counts": counts, "metrics": metrics, "hmm": metadata,
              "denominator": "All emitted nonempty sequences containing only the 20 standard amino acids; duplicates and out-of-range lengths remain in denominator",
              "bootstrap": {"samples": draws, "seed": seed, "unit": "generated sequence",
                            "interval": "percentile 95%; diversity resamples sequences, not independent pairs",
                            "note": "Descriptive sampling uncertainty; does not establish biochemical activity. Duplicate excess is reported without a bootstrap interval."},
              "similarity_metric": METRIC, "training_reference": reference,
              "limitations": ["Pfam GA and domain coverage measure family sequence compatibility, not catalytic activity or expression.",
                              "The split isolates fine-tuning data; absence from the foundation model pretraining set is not established."],
              "artifacts": {"metrics": str(output / "metrics.json"),
                            "per_sequence": str(output / "per_sequence.csv"),
                            "pairwise_diversity": str(output / "pairwise_diversity.csv")}}
    fields = ["id", "sequence", "length", "valid_amino_acids", "within_training_length", "ga_pass", "family_pass",
              "hmm_coverage", "sequence_score", "domain_score", "duplicate_of_earlier", "sequence_multiplicity",
              "exact_training_match", "nearest_training_identity", "nearest_training_id"]
    _csv(output / "per_sequence.csv", rows, fields)
    _csv(output / "pairwise_diversity.csv", pair_rows, ["a", "b", "identity", "distance", "coverage_a", "coverage_b"])
    write_json(output / "metrics.json", result)
    return result
