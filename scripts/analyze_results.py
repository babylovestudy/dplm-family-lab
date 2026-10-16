"""Analyze completed recorded generations; never run a model or rescore an HMM.

Example: python scripts/analyze_results.py --mode pilot --input-root results/pilot
         python scripts/analyze_results.py --mode formal --input-root results/formal

Paired percentile intervals resample the same sequence indices in both arms.
This implementation requires every requested index to have a valid, correctly
sized sequence. Invalid/missing outputs are rejected, never silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import platform

import numpy as np

from dplm_family_lab.common import read_fasta, sha256_file, write_json
from dplm_family_lab.reporting import _official_source_chain, _verified_export_identity


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
METRICS = ("ga_pass_rate", "family_pass_rate", "excess_duplicate_rate",
           "mean_nearest_training_identity", "mean_pairwise_distance", "exact_training_match_rate")
LIMITATIONS = [
    "Analysis checks recorded metadata and hashes; it does not authenticate model weights or rerun biological scoring.",
    "Paired intervals resample generated sequence indices, not independent training replicates; homologous or dependent sequences limit their interpretation.",
    "The three generation seeds are repeated samples from each trained model; nine adapted groups are not nine independent training runs.",
    "Across-generation-seed means and ranges are descriptive. Only three formal training seeds are available; no p-values are computed.",
    "Zero observed family hits can produce a degenerate empirical bootstrap interval [0,0]; this does not establish zero population hit probability.",
    "HMM GA/coverage measures family compatibility, not folding, catalytic activity, expression or stability.",
    "The same PF00194 HMM defines data completeness screening and generation-family evaluation; this is an objective-aligned profile measure, not independent functional validation.",
    "Nearest training identity and exact matches concern the adaptation set, not all protein databases; pretraining overlap remains possible.",
    "In this experiment validation/test sequences are singleton components under the 80/80 graph threshold; that does not establish remote-homology or functional generalization.",
    "Fixed-noise weighted test denoising CE is not exact sequence likelihood or perplexity.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, probability=False):
    require(not isinstance(value, bool) and isinstance(value, (float, int))
            and math.isfinite(value), f"Missing/nonfinite value: {label}")
    require(not probability or 0 <= value <= 1, f"Outside [0,1]: {label}")
    return float(value)


def boolean(value, label):
    require(value in ("True", "False"), f"Invalid boolean in {label}: {value!r}")
    return value == "True"


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"Recorded metric/count disagrees with sequence rows: {label}")


def paired_interval(baseline, adapted, *, samples, seed, training_seed, generation_seed):
    """Resample one common index vector per draw, keeping both arms paired."""
    baseline, adapted = np.asarray(baseline, dtype=np.float64), np.asarray(adapted, dtype=np.float64)
    require(baseline.shape == adapted.shape, "Paired outcome shapes differ")
    delta = adapted - baseline
    require(delta.ndim == 1 and delta.size > 0, "Empty or invalid paired outcomes")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [seed, training_seed, generation_seed])))
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975], method="linear")
    return {"value": float(delta.mean()), "ci95": [float(low), float(high)]}


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}

    def record(self, path):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), f"Input is outside evidence root: {path.name}")
        require(path.is_file(), f"Missing input: {path.relative_to(self.root)}")
        key = path.relative_to(self.root).as_posix()
        self.sources[key] = {"path": key, "sha256": sha256_file(path)}
        return path

    def json(self, path):
        path = self.record(path)
        with path.open(encoding="utf-8-sig") as handle:
            value = json.load(handle)
        require(isinstance(value, dict), f"Expected JSON object: {path.name}")
        return value

    def reference(self, value):
        """Resolve portable bundle paths and original remote /.../outputs paths."""
        value = str(value or "").replace("\\", "/")
        if value.startswith("outputs/"):
            return self.root / value
        if "/outputs/" in value:
            return self.root / "outputs" / value.split("/outputs/", 1)[1]
        return self.root / value


def read_generation(evidence, directory, *, seed, count, expected_reference, expected_weights_sha=None):
    manifest = evidence.json(directory / "manifest.json")
    schedule_path = directory / "length_schedule.json"
    schedule = evidence.json(schedule_path)
    shared_path = evidence.reference(manifest.get("length_schedule_path"))
    shared = evidence.json(shared_path)
    require(schedule == shared, f"Local/shared length schedule mismatch: {directory.name}")
    schedule_sha = sha256_file(schedule_path)
    require(schedule_sha == manifest.get("length_schedule_sha256") == sha256_file(shared_path),
            f"Length schedule SHA mismatch: {directory}")
    require(manifest.get("status") == "complete" and manifest.get("seed") == seed
            and manifest.get("sequences") == count, f"Incomplete/wrong generation budget: {directory}")
    require(manifest.get("requested_sequences", count) == count,
            f"Wrong requested sequence count: {directory}")
    require(schedule.get("seed") == seed and schedule.get("total_sequences") == count
            and len(schedule.get("lengths", [])) == count, f"Wrong length schedule budget: {directory}")
    require(manifest.get("output_alphabet") == AMINO_ACIDS
            and manifest.get("sampling_strategy") == "gumbel_argmax"
            and manifest.get("max_iter") == 100, f"Unexpected sampling/alphabet policy: {directory}")
    batch_size = manifest.get("batch_size")
    require(isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0,
            f"Missing/invalid generation batch size: {directory}")
    batches = manifest.get("batches", [])
    offsets = list(range(0, count, batch_size))
    require(isinstance(batches, list) and len(batches) == len(offsets), f"Missing batch seed schedule: {directory}")
    for batch, offset in zip(batches, offsets):
        require(isinstance(batch, dict) and batch.get("offset") == offset
                and batch.get("count") == min(batch_size, count - offset)
                and isinstance(batch.get("seed"), int) and not isinstance(batch["seed"], bool)
                and 0 <= batch["seed"] < 2**32, f"Invalid batch seed schedule: {directory}")
    require(_official_source_chain(manifest.get("provenance", {}))
            and manifest.get("provenance", {}).get("parameter_count", 0) >= 100_000_000,
            f"Missing pinned original-model provenance: {directory}")
    if expected_weights_sha is not None:
        require(manifest.get("provenance", {}).get("loaded_weights_sha256") == expected_weights_sha,
                f"Adapted loaded weights differ from the independently sealed training export: {directory}")
    fasta = evidence.record(directory / "sequences.fasta")
    require(sha256_file(fasta) == manifest.get("fasta_sha256"), f"FASTA SHA mismatch: {directory}")
    records = read_fasta(fasta)
    metrics_path = directory / "evaluation/metrics.json"
    metrics = evidence.json(metrics_path)
    require(metrics.get("input_sha256") == sha256_file(fasta)
            and evidence.reference(metrics.get("input_fasta")).resolve() == fasta,
            f"Evaluation input mismatch: {directory}")
    reference = {key: metrics.get("training_reference", {}).get(key) for key in (
        "train_fasta_sha256", "hmm_sha256", "split_manifest_sha256")}
    require(all(isinstance(v, str) and len(v) == 64 for v in reference.values()),
            f"Missing frozen training/HMM reference: {directory}")
    reference["min_hmm_coverage"] = finite(metrics.get("training_reference", {}).get("min_hmm_coverage"),
                                            f"HMM coverage threshold: {directory}", probability=True)
    reference["similarity_metric"] = metrics.get("similarity_metric")
    require(isinstance(reference["similarity_metric"], str)
            and reference["similarity_metric"].startswith("global-v1:"),
            f"Missing/unsupported similarity definition: {directory}")
    require(expected_reference is None or reference == expected_reference,
            f"Different training/HMM reference between arms: {directory}")
    require(schedule.get("source_sha256") == reference["train_fasta_sha256"],
            f"Length schedule was not drawn from the frozen training set: {directory}")
    rows_path = evidence.record(directory / "evaluation/per_sequence.csv")
    with rows_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(records) == count, f"Missing emitted/requested index: {directory}")
    ids = [row.get("id") for row in rows]
    require(len(set(ids)) == count and all(ids), f"Duplicate/missing sequence IDs: {directory}")
    require(ids == [record.id for record in records], f"FASTA/CSV sequence ID order mismatch: {directory}")
    family, ga, exact, nearest, duplicate = [], [], [], [], []
    seen = set()
    for index, (row, record, length) in enumerate(zip(rows, records, schedule["lengths"])):
        label = f"{directory.name}/{record.id}"
        require(isinstance(length, int) and not isinstance(length, bool) and length > 0,
                f"Invalid planned length: {label}")
        require(row.get("sequence") == record.sequence and int(row["length"]) == len(record.sequence) == length,
                f"CSV/FASTA/planned length mismatch at index {index}: {label}")
        valid = bool(record.sequence) and set(record.sequence) <= set(AMINO_ACIDS)
        require(boolean(row["valid_amino_acids"], label) == valid, f"Validity flag mismatch: {label}")
        require(valid and boolean(row["within_training_length"], label),
                f"Paired analysis requires all requested indices valid and in range; invalid index: {label}")
        family.append(boolean(row["family_pass"], label))
        ga.append(boolean(row["ga_pass"], label))
        coverage = finite(float(row["hmm_coverage"]), label, probability=True)
        require(family[-1] == (ga[-1] and coverage + 1e-12 >= reference["min_hmm_coverage"]),
                f"Family/GA/coverage flags disagree: {label}")
        exact.append(boolean(row["exact_training_match"], label))
        nearest.append(finite(float(row["nearest_training_identity"]), label, probability=True))
        earlier = record.sequence in seen
        require(boolean(row["duplicate_of_earlier"], label) == earlier, f"Duplicate flag mismatch: {label}")
        duplicate.append(earlier)
        seen.add(record.sequence)
    counts = metrics.get("counts", {})
    for key, actual in (("emitted_fasta_records", count), ("valid_amino_acid_sequences", count),
                        ("invalid_amino_acid_sequences", 0), ("unique_valid_sequences", len(seen)),
                        ("excess_duplicate_records", sum(duplicate)),
                        ("family_pass", sum(family)), ("ga_pass", sum(ga))):
        require(counts.get(key) == actual, f"Counts disagree with sequence rows: {directory}/{key}")
    summary = {key: finite(metrics.get("metrics", {}).get(key, {}).get("value"), key, probability=True)
               for key in METRICS}
    for key, actual in (("family_pass_rate", np.mean(family)), ("ga_pass_rate", np.mean(ga)),
                        ("exact_training_match_rate", np.mean(exact)),
                        ("excess_duplicate_rate", np.mean(duplicate)),
                        ("mean_nearest_training_identity", np.mean(nearest))):
        close(summary[key], float(actual), f"{directory}/{key}")
    summary.update(requested=count, emitted=count, valid=count, invalid=0, missing=0,
                   valid_rate=1.0, overall_family_yield=sum(family) / count,
                   family_hits=sum(family), ga_hits=sum(ga),
                   length_schedule_sha256=schedule_sha)
    return {"manifest": manifest, "metrics": metrics, "reference": reference,
            "ids": ids, "lengths": schedule["lengths"], "family": family, "summary": summary}


def analyze(mode, input_root, output_dir, *, bootstrap_samples=1000, bootstrap_seed=20261016):
    require(mode in {"pilot", "formal"}, "Mode must be pilot or formal")
    require(bootstrap_samples == 1000, "The fixed analysis protocol uses 1000 bootstrap draws")
    require(isinstance(bootstrap_seed, int) and 0 <= bootstrap_seed < 2**32,
            "Bootstrap seed must be an unsigned 32-bit integer")
    evidence = Evidence(input_root)
    output = Path(output_dir).resolve()
    require(not output.is_relative_to(evidence.root / "outputs")
            and not output.is_relative_to(evidence.root / "runs"), "Analysis output cannot overwrite original evidence")
    directory = evidence.root / "outputs" / mode
    journal = evidence.json(directory / "experiment_status.json")
    require(journal.get("mode") == mode and journal.get("status") == "complete",
            f"Cannot analyze an incomplete {mode} experiment")
    entries = journal.get("completed", [])
    require(isinstance(entries, list), "Invalid completed-stage journal")
    completed = {entry.get("stage"): entry.get("result") for entry in entries if isinstance(entry, dict)}
    require(len(completed) == len(entries), "Duplicate/invalid journal stages")
    training_seeds = [42] if mode == "pilot" else [42, 43, 44]
    generation_seeds = [101] if mode == "pilot" else [101, 102, 103]
    count, steps = (32, 100) if mode == "pilot" else (256, 1000)
    arms = ["baseline", *(f"train_seed_{seed}" for seed in training_seeds)]
    export_identities = {}
    for seed in training_seeds:
        train = completed.get(f"train_{seed}", {})
        require(train.get("status") == "complete" and train.get("step") == steps,
                f"Missing completed {mode} training seed {seed}")
        suffix = f"pilot_seed_{seed}" if mode == "pilot" else f"seed_{seed}"
        run = evidence.root / "runs/pf00194" / suffix
        identity, identity_sources = _verified_export_identity(run)
        for source in identity_sources:
            evidence.record(source)
        require(evidence.json(run / "training_report.json") == train,
                f"Training journal differs from the sealed source report: {seed}")
        export_identities[f"train_seed_{seed}"] = identity
    groups, reference, denoising = {}, None, []
    test_reference = None
    for arm in arms:
        for seed in generation_seeds:
            group = read_generation(evidence, directory / arm / f"seed_{seed}", seed=seed,
                                    count=count, expected_reference=reference,
                                    expected_weights_sha=export_identities.get(arm, {}).get("weights_file_sha256"))
            reference = group["reference"]
            require(completed.get(f"generate_{arm}_{seed}") == group["manifest"]
                    and completed.get(f"evaluate_{arm}_{seed}") == group["metrics"],
                    f"Missing/mismatched journal result: {arm}/{seed}")
            groups[(arm, seed)] = group
        name = "denoising_baseline.json" if arm == "baseline" else "denoising_adapted.json"
        test = evidence.json(directory / arm / "test_denoising" / name)
        require(completed.get(f"test_denoising_{arm}") == test and test.get("split") == "test",
                f"Missing/mismatched test denoising stage: {arm}")
        require(_official_source_chain(test.get("provenance", {})), f"Unverified test model source: {arm}")
        if arm != "baseline":
            require(test.get("provenance", {}).get("loaded_weights_sha256")
                    == export_identities[arm]["weights_file_sha256"],
                    f"Adapted test weights differ from the independently sealed training export: {arm}")
        identity = (test.get("source_sha256"), test.get("noise_seed"), test.get("sequences"), test.get("masked_residues"))
        require(isinstance(identity[0], str) and len(identity[0]) == 64
                and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in identity[1:]),
                f"Incomplete fixed-noise test metadata: {arm}")
        require(test_reference is None or identity == test_reference, f"Different test data/noise between arms: {arm}")
        test_reference = identity
        require(all(groups[(arm, seed)]["manifest"].get("model") == test.get("model")
                    for seed in generation_seeds), f"Generation/test model mismatch: {arm}")
        denoising.append({"arm": arm, "training_seed": None if arm == "baseline" else int(arm[-2:]),
                          "weighted_masked_test_denoising_ce": finite(test.get("weighted_masked_denoising_loss"), arm),
                          "test_sha256": identity[0], "noise_seed": identity[1],
                          "test_sequences": identity[2], "masked_residues": identity[3]})
    comparisons = []
    for training_seed in training_seeds:
        arm = f"train_seed_{training_seed}"
        for generation_seed in generation_seeds:
            baseline, adapted = groups[("baseline", generation_seed)], groups[(arm, generation_seed)]
            require(baseline["ids"] == adapted["ids"] and baseline["lengths"] == adapted["lengths"]
                    and baseline["summary"]["length_schedule_sha256"] == adapted["summary"]["length_schedule_sha256"],
                    f"Unpaired sequence IDs or length schedule: train {training_seed}, generation {generation_seed}")
            for key in ("seed", "max_iter", "batch_size", "batches", "sampling_strategy", "output_alphabet"):
                require(baseline["manifest"].get(key) == adapted["manifest"].get(key), f"Unpaired sampling policy: {key}")
            interval = paired_interval(baseline["family"], adapted["family"], samples=bootstrap_samples,
                                       seed=bootstrap_seed, training_seed=training_seed, generation_seed=generation_seed)
            row = {"mode": mode, "training_seed": training_seed, "generation_seed": generation_seed,
                   "requested_per_arm": count, "paired_indices": count,
                   "family_rate_difference": interval["value"], "family_difference_ci95_low": interval["ci95"][0],
                   "family_difference_ci95_high": interval["ci95"][1],
                   "overall_family_yield_difference": interval["value"],
                   "overall_yield_difference_ci95_low": interval["ci95"][0],
                   "overall_yield_difference_ci95_high": interval["ci95"][1],
                   "baseline_fixed_noise_test_ce": denoising[0]["weighted_masked_test_denoising_ce"],
                   "adapted_fixed_noise_test_ce": next(item["weighted_masked_test_denoising_ce"]
                                                      for item in denoising if item["training_seed"] == training_seed)}
            for prefix, group in (("baseline", baseline), ("adapted", adapted)):
                row.update({f"{prefix}_{key}": value for key, value in group["summary"].items()})
            comparisons.append(row)
    across = []
    for seed in training_seeds:
        rows = [row for row in comparisons if row["training_seed"] == seed]
        values = [row["family_rate_difference"] for row in rows]
        descriptive = {}
        for key in ("family_rate_difference", "overall_family_yield_difference",
                    *(f"{arm}_{key}" for arm in ("baseline", "adapted")
                      for key in (*METRICS, "overall_family_yield", "valid_rate"))):
            observed = [row[key] for row in rows]
            descriptive[key] = {"mean": float(np.mean(observed)), "range": [min(observed), max(observed)]}
        across.append({"training_seed": seed, "generation_seeds": generation_seeds,
                       "family_rate_difference_mean": float(np.mean(values)),
                       "family_rate_difference_range": [min(values), max(values)],
                       "descriptive_metrics": descriptive,
                       "note": "Descriptive average/range across generation seeds; no independent-training CI."})
    result = {"schema_version": 1, "analysis_status": "complete", "mode": mode,
              "training_seeds": training_seeds, "generation_seeds": generation_seeds,
              "generation_groups": len(groups), "paired_comparisons": comparisons,
              "across_generation_seeds": across, "test_denoising": denoising,
              "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed, "rng": "NumPy PCG64",
                            "seed_components": ["bootstrap_seed", "training_seed", "generation_seed"],
                            "unit": "paired requested sequence index", "interval": "95% percentile; linear quantiles"},
              "denominator_policy": "Complete all-valid pairs required; conditional family rates equal family_pass/requested. Missing/invalid indices cause refusal, never deletion.",
              "limitations": LIMITATIONS,
              "reproducibility": {"script_sha256": sha256_file(__file__), "python": platform.python_version(),
                                  "numpy": np.__version__},
              "sources": [evidence.sources[key] for key in sorted(evidence.sources)]}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis.json", result)
    (output / "generation_comparison.csv").write_text(buffer.getvalue(), encoding="utf-8", newline="")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "formal"], required=True)
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-seed", type=int, default=20261016)
    args = parser.parse_args()
    try:
        result = analyze(args.mode, args.input_root or f"results/{args.mode}",
                         args.output_dir or f"results/analysis/{args.mode}", bootstrap_seed=args.bootstrap_seed)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Analysis refused: {error}\n")
    print(json.dumps({"mode": args.mode, "status": result["analysis_status"],
                      "paired_comparisons": len(result["paired_comparisons"]),
                      "test_denoising_groups": len(result["test_denoising"])}))


if __name__ == "__main__":
    main()
