"""Supplementary diagnostics from recorded formal CSV/FASTA/metrics only.

python scripts/diagnose_generations.py --input-root results/formal

The default path does not load a model, rescore an HMM, or align new sequences.
Optional --train-fasta independently checks exact-copy flags and identities to
the named training references. This is not a new global nearest-neighbor search.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dplm_family_lab.common import read_fasta, sha256_file  # noqa: E402
from dplm_family_lab.similarity import METRIC  # noqa: E402

ARMS = ["baseline", "train_seed_42", "train_seed_43", "train_seed_44"]
GEN_SEEDS = [101, 102, 103]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def boolean(value):
    require(value in ("True", "False"), f"Invalid serialized bool: {value!r}")
    return value == "True"


def optional_float(value):
    return None if value == "" else float(value)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{label}: {actual} != {expected}")


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def distribution(rows):
    values = np.array([r["nearest_training_identity"] for r in rows], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "quantiles": None,
                "at_least_80_count": 0, "at_least_80_fraction": None,
                "at_least_90_count": 0, "at_least_90_fraction": None}
    levels = [0, .25, .5, .75, .9, .95, .99, 1]
    quantiles = {str(q): float(v) for q, v in zip(levels, np.quantile(values, levels))}
    return {"n": len(values), "mean": float(values.mean()), "quantiles": quantiles,
            "at_least_80_count": int(np.sum(values + 1e-12 >= .8)),
            "at_least_80_fraction": float(np.mean(values + 1e-12 >= .8)),
            "at_least_90_count": int(np.sum(values + 1e-12 >= .9)),
            "at_least_90_fraction": float(np.mean(values + 1e-12 >= .9))}


def inspect_group(inputs, arm, seed, training=None, aligner=None):
    folder = inputs / arm / f"seed_{seed}"
    metrics_path = folder / "evaluation/metrics.json"
    if not metrics_path.exists():
        return None
    metric = json.loads(metrics_path.read_text(encoding="utf-8"))
    fasta_path = folder / "sequences.fasta"
    csv_path = folder / "evaluation/per_sequence.csv"
    require(sha256_file(fasta_path) == metric["input_sha256"], f"Changed input: {arm}/{seed}")
    require(metric["similarity_metric"] == METRIC, "Unexpected alignment metric")
    require(metric["hmm"]["accession"] == "PF00194.27", "Unexpected HMM version")
    require(metric["hmm"]["sha256"] == "75ace6724895a6b7fe31c050b7ab9d73968feae442f40fda487ef4aecda711a3", "Unexpected HMM bytes")
    require(metric["hmm"]["ga_sequence"] == 27 and metric["hmm"]["ga_domain"] == 27,
            "Changed curated GA cutoff")
    reference = metric["training_reference"]
    require(reference["train_fasta_sha256"] == "ba3ef41022933ea83105f3ad09d095ed726c1c20e69da9be4d744c0b765738a7", "Changed training reference")
    require(reference["min_hmm_coverage"] == .8, "Changed family coverage cutoff")
    require(reference["split_manifest_sha256"] == "d0611fd93b331a12dab9407597cba246adad2b10115f86e1f26b314c32dbd345", "Changed split manifest")
    fasta = read_fasta(fasta_path)
    records = {record.id: record.sequence for record in fasta}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(records) == len(rows) == 256, f"Emitted budget is not 256: {arm}/{seed}")
    require(len({row['id'] for row in rows}) == len(rows), "Repeated CSV IDs")
    require(set(records) == {row["id"] for row in rows}, "CSV and FASTA IDs differ")
    train_set = set(training.values()) if training else None
    multiplicity = Counter(record.sequence for record in fasta if record.sequence and set(record.sequence) <= AA)
    seen = set()
    for row in rows:
        sequence = row["sequence"]
        require(sequence == records[row["id"]], "CSV and FASTA residues differ")
        for key in ("valid_amino_acids", "within_training_length", "ga_pass", "family_pass",
                    "duplicate_of_earlier", "exact_training_match"):
            row[key] = boolean(row[key])
        for key in ("length", "sequence_multiplicity"):
            row[key] = int(row[key])
        for key in ("hmm_coverage", "sequence_score", "domain_score", "nearest_training_identity"):
            row[key] = optional_float(row[key])
        valid = bool(sequence) and set(sequence) <= AA
        require(row["valid_amino_acids"] == valid, "Invalid validity flag")
        require(valid, f"Formal diagnostics requires 256 valid records per group: {arm}/{seed}/{row['id']}")
        require(row["length"] == len(sequence), "Length mismatch")
        require(row["within_training_length"] == (180 <= len(sequence) <= 256), "Length flag mismatch")
        require(row["sequence_multiplicity"] == multiplicity.get(sequence, 0), "Multiplicity mismatch")
        require(row["duplicate_of_earlier"] == (valid and sequence in seen), "Duplicate flag mismatch")
        if train_set is not None:
            require(row["exact_training_match"] == (valid and sequence in train_set), "Exact-copy flag mismatch")
        sequence_score, domain_score = row["sequence_score"], row["domain_score"]
        inferred_ga = (sequence_score is not None and domain_score is not None
                       and sequence_score >= 27 and domain_score >= 27)
        require(row["ga_pass"] == inferred_ga, "GA flags disagree with selected-domain scores")
        require(row["family_pass"] == (row["ga_pass"] and row["hmm_coverage"] + 1e-12 >= .8),
                "Family flag disagrees with GA+single-domain coverage")
        require(0 <= row["hmm_coverage"] <= 1, "Invalid HMM coverage")
        close(row["hmm_coverage"] * 255, round(row["hmm_coverage"] * 255), "HMM model-coordinate count")
        if valid:
            nearest_id = row["nearest_training_id"]
            require(nearest_id and 0 <= row["nearest_training_identity"] <= 1,
                    "Invalid reported nearest ID or identity")
            if training is not None:
                from dplm_family_lab.similarity import global_similarity
                require(nearest_id in training, "Nearest training ID is absent from the reference")
                result = global_similarity(sequence, training[nearest_id], aligner)
                close(row["nearest_training_identity"], result.identity, "Reported nearest-pair identity")
            seen.add(sequence)
        row.update(arm=arm, generation_seed=seed, sequence_sha256=sequence_hash(sequence))
    valid_rows = [row for row in rows if row["valid_amino_acids"]]
    n = len(valid_rows)
    expected_counts = {
        "emitted_fasta_records": len(rows), "valid_amino_acid_sequences": n,
        "invalid_amino_acid_sequences": len(rows)-n, "unique_valid_sequences": len(multiplicity),
        "excess_duplicate_records": n-len(multiplicity),
        "ga_pass": sum(row["ga_pass"] for row in valid_rows),
        "family_pass": sum(row["family_pass"] for row in valid_rows),
    }
    require(metric["counts"] == expected_counts, f"Counts mismatch: {arm}/{seed}")
    for name, field in (("ga_pass_rate", "ga_pass"), ("family_pass_rate", "family_pass"),
                        ("mean_hmm_coverage", "hmm_coverage"),
                        ("exact_training_match_rate", "exact_training_match"),
                        ("mean_nearest_training_identity", "nearest_training_identity"),
                        ("within_training_length_rate", "within_training_length")):
        require(metric["metrics"][name]["n"] == n, "Metric denominator mismatch")
        close(metric["metrics"][name]["value"], float(np.mean([row[field] for row in valid_rows])), name)
    close(metric["metrics"]["excess_duplicate_rate"]["value"], (n-len(multiplicity))/n, "Duplicate excess rate")
    require(metric["metrics"]["excess_duplicate_rate"]["n"] == n, "Duplicate denominator mismatch")
    require(metric["metrics"]["mean_pairwise_distance"]["n"] == n, "Diversity n mismatch")
    require(metric["metrics"]["mean_pairwise_distance"]["pairs"] == n*(n-1)//2, "Diversity pair budget mismatch")
    require(0 <= metric["metrics"]["mean_pairwise_distance"]["value"] <= 1, "Invalid diversity range")
    summary = {"arm": arm, "generation_seed": seed, **expected_counts,
               "all_256_are_valid": n == 256, "family_pass_rate": expected_counts["family_pass"]/n,
               "ga_pass_but_coverage_fail": sum(row["ga_pass"] and not row["family_pass"] for row in valid_rows),
               "ga_miss": sum(not row["ga_pass"] for row in valid_rows),
               "exact_training_match_records": sum(row["exact_training_match"] for row in valid_rows),
               "mean_nearest_training_identity": metric["metrics"]["mean_nearest_training_identity"]["value"],
               "mean_pairwise_distance": metric["metrics"]["mean_pairwise_distance"]["value"],
               "nearest_at_least_80_records": sum(row["nearest_training_identity"] + 1e-12 >= .8 for row in valid_rows),
               "nearest_at_least_90_records": sum(row["nearest_training_identity"] + 1e-12 >= .9 for row in valid_rows)}
    sources = {f"outputs/formal/{arm}/seed_{seed}/{name}": sha256_file(folder/name)
               for name in ("sequences.fasta", "evaluation/metrics.json", "evaluation/per_sequence.csv")}
    return summary, valid_rows, sources


def trace(row, selection):
    fields = ["arm", "generation_seed", "id", "sequence_sha256", "length", "ga_pass", "family_pass",
              "hmm_coverage", "sequence_score", "domain_score", "nearest_training_identity",
              "nearest_training_id", "exact_training_match", "sequence_multiplicity"]
    return {"selection_rule": selection, **{key: row[key] for key in fields},
            "source_csv": f"outputs/formal/{row['arm']}/seed_{row['generation_seed']}/evaluation/per_sequence.csv"}


def summarize_arm(rows, completed_seeds):
    seed_presence = defaultdict(set)
    for row in rows:
        seed_presence[row["sequence"]].add(row["generation_seed"])
    per_seed_unique = sum(len({r["sequence"] for r in rows if r["generation_seed"] == seed})
                          for seed in completed_seeds)
    passing = [row for row in rows if row["family_pass"]]
    count = len(rows)
    return {"completed_generation_seeds": sorted(completed_seeds), "complete": len(completed_seeds) == 3,
            "valid_records": count, "family_pass": len(passing), "family_pass_rate": len(passing)/count,
            "nearest_all_valid": distribution(rows), "nearest_family_pass_only": distribution(passing),
            "exact_training_match_records": sum(row["exact_training_match"] for row in rows),
            "exact_training_match_unique_sequences": len({r["sequence"] for r in rows if r["exact_training_match"]}),
            "unique_sequences_across_generation_seeds": len(seed_presence),
            "excess_duplicate_records_across_all_generation_seeds": count-len(seed_presence),
            "within_generation_seed_duplicate_excess_total": count-per_seed_unique,
            "cross_generation_seed_unique_membership_excess": per_seed_unique-len(seed_presence),
            "distinct_sequences_present_in_multiple_generation_seeds": sum(len(seeds)>1 for seeds in seed_presence.values()),
            "records_whose_sequence_occurs_in_multiple_generation_seeds": sum(len(seed_presence[row["sequence"]])>1 for row in rows),
            "ga_miss": sum(not row["ga_pass"] for row in rows),
            "ga_pass_but_coverage_fail": sum(row["ga_pass"] and not row["family_pass"] for row in rows)}


def pooled_adapted(rows):
    """Exact union accounting; (model, generation seed) is an occurrence group."""
    selected = [row for row in rows if row["arm"] != "baseline"]
    if not selected:
        return {"valid_records": 0, "completed_groups": 0, "complete": False}
    memberships = defaultdict(set)
    model_presence = defaultdict(set)
    for row in selected:
        memberships[row["sequence"]].add((row["arm"], row["generation_seed"]))
        model_presence[row["sequence"]].add(row["arm"])
    groups = {(row["arm"], row["generation_seed"]) for row in selected}
    per_group_unique = sum(len(value) for value in memberships.values())
    return {"valid_records": len(selected), "completed_groups": len(groups), "complete": len(groups) == 9,
            "unique_sequences": len(memberships), "excess_duplicate_records": len(selected)-len(memberships),
            "within_group_duplicate_excess": len(selected)-per_group_unique,
            "across_group_unique_membership_excess": per_group_unique-len(memberships),
            "distinct_sequences_in_multiple_groups": sum(len(value)>1 for value in memberships.values()),
            "records_whose_sequence_occurs_in_multiple_groups": sum(len(memberships[row["sequence"]])>1 for row in selected),
            "distinct_sequences_in_multiple_training_seeds": sum(len(value)>1 for value in model_presence.values()),
            "records_whose_sequence_occurs_in_multiple_training_seeds": sum(len(model_presence[row["sequence"]])>1 for row in selected),
            "nearest_all_valid": distribution(selected),
            "exact_training_match_records": sum(row["exact_training_match"] for row in selected),
            "note": "Supplementary exact equality diagnostics. Models share the same three sampling seeds; groups are not independent training replicates."}


def composition(sequence):
    """Descriptive casebook features, not primary evaluation outcomes."""
    runs = [(len(list(group)), residue) for residue, group in groupby(sequence)]
    longest, residue = max(runs, key=lambda item: (item[0], item[1]))
    return {"amino_acid_composition": dict(sorted(Counter(sequence).items())),
            "longest_homopolymer_residue": residue, "longest_homopolymer_length": longest,
            "longest_homopolymer_fraction": longest/len(sequence)}


def duplicate_cases(rows, scope):
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row["sequence"]].append(row)
    repeated = []
    for sequence, members in sorted(by_sequence.items()):
        groups = {(row["arm"], row["generation_seed"]) for row in members}
        if len(groups) <= 1:
            continue
        repeated.append({"scope": scope, "sequence_sha256": sequence_hash(sequence), "length": members[0]["length"],
                         "record_count": len(members), "generation_seed_count": len({row["generation_seed"] for row in members}),
                         "model_arm_count": len({row["arm"] for row in members}), "group_count": len(groups),
                         "members": ";".join(f"{r['arm']}/seed_{r['generation_seed']}/{r['id']}" for r in sorted(members, key=lambda row: (row["arm"], row["generation_seed"], row["id"]))),
                         "family_pass": members[0]["family_pass"], "exact_training_match": members[0]["exact_training_match"],
                         **composition(members[0]["sequence"])})
    return repeated


def validate_repeated_consistency(rows):
    """The same sequence and frozen references must have the same annotation."""
    first = {}
    exact_fields = ("length", "ga_pass", "family_pass", "exact_training_match", "nearest_training_id")
    float_fields = ("hmm_coverage", "nearest_training_identity")
    for row in rows:
        previous = first.setdefault(row["sequence"], row)
        for field in exact_fields:
            require(row[field] == previous[field], f"Conflicting {field} for identical generated sequence")
        for field in float_fields:
            close(row[field], previous[field], f"Conflicting {field} for identical generated sequence")


def select_examples(rows):
    examples = []
    def tie(row):
        return row["generation_seed"], row["id"]
    for arm in ARMS:
        candidates = [row for row in rows if row["arm"] == arm]
        if not candidates:
            continue
        highest = sorted(candidates, key=lambda row: (-row["nearest_training_identity"], *tie(row)))[0]
        examples.append((highest, "highest_nearest_identity; ties generation_seed then id"))
        partial = [row for row in candidates if row["ga_pass"] and not row["family_pass"]]
        if partial:
            chosen = sorted(partial, key=lambda row: (-row["hmm_coverage"], *tie(row)))[0]
            examples.append((chosen, "coverage_failure_closest_to_0.8; ties generation_seed then id"))
        misses = [row for row in candidates if not row["ga_pass"]]
        if misses:
            median = float(np.median([row["nearest_training_identity"] for row in misses]))
            chosen = sorted(misses, key=lambda row: (abs(row["nearest_training_identity"]-median), *tie(row)))[0]
            examples.append((chosen, "GA_miss_nearest_to_median_identity; ties generation_seed then id"))
        passing = [row for row in candidates if row["family_pass"]]
        if passing:
            median = float(np.median([row["nearest_training_identity"] for row in passing]))
            chosen = sorted(passing, key=lambda row: (abs(row["nearest_training_identity"]-median), *tie(row)))[0]
            examples.append((chosen, "ordinary_family_pass_nearest_to_median_identity; ties generation_seed then id"))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", "--inputs", dest="input_root", type=Path, default=Path("results/formal"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/formal_diagnostics"))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--train-fasta", type=Path,
                        help="optional frozen reference for exact-copy and named-pair identity checks; no global nearest search")
    args = parser.parse_args()
    inputs = args.input_root
    if (inputs / "outputs/formal").is_dir():
        inputs = inputs / "outputs/formal"
    training = aligner = None
    if args.train_fasta:
        require(sha256_file(args.train_fasta) == "ba3ef41022933ea83105f3ad09d095ed726c1c20e69da9be4d744c0b765738a7", "Optional training FASTA differs")
        from dplm_family_lab.similarity import _aligner
        training = {record.id: record.sequence for record in read_fasta(args.train_fasta)}
        aligner = _aligner()
    groups, rows, sources, missing = [], [], {}, []
    for arm in ARMS:
        for seed in GEN_SEEDS:
            inspected = inspect_group(inputs, arm, seed, training, aligner)
            if inspected is None:
                missing.append(f"{arm}/seed_{seed}")
                continue
            summary, valid_rows, group_sources = inspected
            groups.append(summary)
            rows.extend(valid_rows)
            sources.update(group_sources)
    require(groups, "No completed result files are available")
    require(args.allow_incomplete or not missing, f"Formal evaluation incomplete: {missing}")
    validate_repeated_consistency(rows)
    selected = select_examples(rows)
    examples = [trace(row, rule) for row, rule in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = {arm: summarize_arm([row for row in rows if row["arm"] == arm],
                              {group["generation_seed"] for group in groups if group["arm"] == arm})
            for arm in ARMS if any(row["arm"] == arm for row in rows)}
    repeated = [case for arm in arms for case in duplicate_cases([row for row in rows if row["arm"] == arm], arm)]
    adapted_repeated = duplicate_cases([row for row in rows if row["arm"] != "baseline"], "all_adapted")
    result = {"schema_version": 1, "kind": "supplementary_read_only_formal_diagnostics",
              "complete": not missing, "completed_groups": len(groups), "expected_groups": 12,
              "missing_groups": missing, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "checks": {"emitted_256_records_each": True,
                         "all_groups_have_256_valid_denominator": all(group["all_256_are_valid"] for group in groups),
                         "CSV_and_FASTA_residues_and_hashes_match": True,
                         "recorded_frozen_train_and_HMM_reference_hashes_match": True,
                         "GA_and_single_domain_coverage_flags_match_reported_scores": True,
                         "per_sequence_aggregates_match_metrics": True,
                         "identical_sequences_have_consistent_recorded_annotations": True,
                         "duplicate_flags_independently_recounted": True,
                         "exact_training_copy_flags_independently_recounted": training is not None,
                         "reported_nearest_pair_identity_recomputed_for_every_valid_record": training is not None,
                         "global_nearest_neighbor_search_reexecuted": False},
              "groups": groups, "arms": arms, "all_adapted_pooled": pooled_adapted(rows), "examples": examples,
              "cross_generation_seed_repeated_sequences": repeated,
              "all_adapted_pooled_repeated_sequences": adapted_repeated,
              "source_files_sha256": sources, "analysis_script_sha256": sha256_file(Path(__file__)),
              "methods": {"identity": METRIC, "quantiles": "numpy.quantile method=linear, all valid records including duplicates",
                          "cross_seed_duplicates": "Per arm and separately across all nine adapted groups: exact residue equality. Membership excess = sum(unique sequences in each model/generation-seed group) - unique sequences in their union; distinguished from within-group excess.",
                          "casebook_composition": "Counts of each residue and longest consecutive identical-residue run divided by full sequence length; supplementary description of observed repeated sequences only, not a primary endpoint or a functional diagnosis.",
                          "example_selection": "Deterministic descriptive cases: highest nearest identity; nearest coverage failure to the unchanged threshold; nearest-to-median GA miss; nearest-to-median passing sequence. Ties by generation seed then sequence ID.",
                          "scope": "Supplementary diagnostics only; does not change primary metrics, thresholds, sequence selection or generation budgets."},
              "planning_status": "Post hoc supplementary diagnostics defined during result inspection, not preregistered primary endpoints.",
              "limitations": ["Three generation seeds share an arm's trained model and are sampling repetitions, not independent training replicates.",
                              "High sequence identity is not exact copying; exact copies are counted separately.",
                              "Copy flags and nearest identity concern the adaptation training set, not all databases or foundation-model pretraining data.",
                              "A Pfam GA and coverage hit supports family compatibility, not catalytic activity, expression or folding.",
                              "GA/coverage reconciliation uses recorded scores and flags; this script does not rerun HMMER or authenticate the scoring process.",
                              "Nearest argmax is taken from completed evaluation. Optional --train-fasta verifies exact-copy flags and identities to the named references, not whether those references are globally nearest; no new global nearest search is performed.",
                              "No new p-values, hyperparameter tuning or GPU computations were performed."]}
    (args.output_dir/"diagnostics.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    csv_write(args.output_dir/"group_checks.csv", groups)
    csv_write(args.output_dir/"examples.csv", examples)
    csv_write(args.output_dir/"sequence_diagnostics.csv", [trace(row, "all_valid_records") for row in rows])
    flattened_arms = []
    for arm, arm_result in arms.items():
        flat = {"arm": arm, **{key: value for key, value in arm_result.items()
                             if key not in ("nearest_all_valid", "nearest_family_pass_only")}}
        flat["completed_generation_seeds"] = ";".join(str(seed) for seed in arm_result["completed_generation_seeds"])
        for subset in ("nearest_all_valid", "nearest_family_pass_only"):
            for key, value in arm_result[subset].items():
                if key == "quantiles":
                    for quantile, estimate in (value or {}).items():
                        flat[f"{subset}_quantile_{quantile}"] = estimate
                else:
                    flat[f"{subset}_{key}"] = value
        flattened_arms.append(flat)
    csv_write(args.output_dir/"arm_diagnostics.csv", flattened_arms)
    duplicate_fields = ["scope", "sequence_sha256", "length", "record_count", "generation_seed_count", "model_arm_count", "group_count", "members", "family_pass", "exact_training_match", "amino_acid_composition", "longest_homopolymer_residue", "longest_homopolymer_length", "longest_homopolymer_fraction"]
    csv_write(args.output_dir/"cross_generation_seed_duplicates.csv", repeated, duplicate_fields)
    csv_write(args.output_dir/"all_adapted_pooled_duplicates.csv", adapted_repeated, duplicate_fields)
    print(json.dumps({"complete": not missing, "completed_groups": len(groups), "family_pass_by_arm": {arm: f"{value['family_pass']}/{value['valid_records']}" for arm, value in arms.items()}}, indent=2))


if __name__ == "__main__":
    main()
