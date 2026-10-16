"""Small fixtures exercise supplementary union and casebook accounting."""

import importlib.util
import csv
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "diagnose_generations", Path(__file__).resolve().parents[1]/"scripts/diagnose_generations.py")
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


def row(arm, seed, identifier, sequence, identity=.5):
    return {"arm": arm, "generation_seed": seed, "id": identifier, "sequence": sequence,
            "sequence_sha256": DIAG.sequence_hash(sequence), "length": len(sequence),
            "family_pass": True, "ga_pass": True, "exact_training_match": False,
            "nearest_training_identity": identity, "hmm_coverage": .9,
            "sequence_score": 50, "domain_score": 50, "nearest_training_id": "reference",
            "sequence_multiplicity": 1}


def test_union_distinguishes_within_group_from_across_group_duplicates():
    rows = [row("train_seed_42", 101, "a", "AAA"),
            row("train_seed_42", 101, "b", "AAA"),
            row("train_seed_42", 102, "a", "AAA"),
            row("train_seed_42", 103, "a", "CCC")]
    summary = DIAG.summarize_arm(rows, {101, 102, 103})
    assert summary["complete"]
    assert summary["unique_sequences_across_generation_seeds"] == 2
    assert summary["excess_duplicate_records_across_all_generation_seeds"] == 2
    assert summary["within_generation_seed_duplicate_excess_total"] == 1
    assert summary["cross_generation_seed_unique_membership_excess"] == 1
    assert summary["distinct_sequences_present_in_multiple_generation_seeds"] == 1
    assert summary["records_whose_sequence_occurs_in_multiple_generation_seeds"] == 3


def test_all_adapted_union_excludes_baseline_and_tracks_model_overlap():
    rows = [row("baseline", 101, "a", "WWW"),
            row("train_seed_42", 101, "a", "AAA"),
            row("train_seed_43", 101, "a", "AAA"),
            row("train_seed_44", 102, "a", "CCC")]
    pooled = DIAG.pooled_adapted(rows)
    assert not pooled["complete"]
    assert pooled["valid_records"] == 3
    assert pooled["unique_sequences"] == 2
    assert pooled["excess_duplicate_records"] == 1
    assert pooled["distinct_sequences_in_multiple_training_seeds"] == 1
    cases = DIAG.duplicate_cases(rows[1:], "all_adapted")
    assert len(cases) == 1
    assert cases[0]["model_arm_count"] == 2
    assert cases[0]["generation_seed_count"] == 1


def test_quantile_thresholds_and_casebook_composition_are_descriptive():
    rows = [row("train_seed_42", 101, "a", "ACCCAA", .8),
            row("train_seed_42", 102, "a", "ACCCAD", .9)]
    stats = DIAG.distribution(rows)
    assert stats["quantiles"]["0.5"] == pytest.approx(.85)
    assert stats["at_least_80_count"] == 2
    assert stats["at_least_90_count"] == 1
    composition = DIAG.composition("ACCCAA")
    assert composition["amino_acid_composition"] == {"A": 3, "C": 3}
    assert composition["longest_homopolymer_residue"] == "C"
    assert composition["longest_homopolymer_length"] == 3
    assert composition["longest_homopolymer_fraction"] == .5


def test_casebook_selection_is_deterministic_without_function_claims():
    rows = [row("train_seed_42", 103, "a", "AAA", .9),
            row("train_seed_42", 101, "a", "CCC", .5),
            row("train_seed_42", 102, "a", "DDD", .7)]
    examples = DIAG.select_examples(rows)
    assert examples[0][0]["generation_seed"] == 103
    assert examples[1][0]["generation_seed"] == 102
    assert "median" in examples[1][1]
    assert DIAG.select_examples(list(reversed(rows))) == examples


def test_identical_sequences_cannot_have_conflicting_recorded_annotations():
    rows = [row("train_seed_42", 101, "a", "AAA"),
            row("train_seed_43", 101, "a", "AAA")]
    DIAG.validate_repeated_consistency(rows)
    rows[1]["nearest_training_identity"] = .6
    with pytest.raises(ValueError, match="Conflicting nearest_training_identity"):
        DIAG.validate_repeated_consistency(rows)


@pytest.fixture
def recorded_group(tmp_path):
    folder = tmp_path/"outputs/formal/baseline/seed_101"
    (folder/"evaluation").mkdir(parents=True)
    fasta = folder/"sequences.fasta"
    fasta.write_text("".join(f">seq_{index:04d}\n{'A'*180}\n" for index in range(256)))
    rows = [{"id": f"seq_{index:04d}", "sequence": "A"*180, "length": 180,
             "valid_amino_acids": True, "within_training_length": True,
             "ga_pass": False, "family_pass": False, "hmm_coverage": 0,
             "sequence_score": None, "domain_score": None,
             "duplicate_of_earlier": index > 0, "sequence_multiplicity": 256,
             "exact_training_match": False, "nearest_training_identity": .1,
             "nearest_training_id": "reference"} for index in range(256)]
    DIAG.csv_write(folder/"evaluation/per_sequence.csv", rows)
    metrics = {name: {"value": value, "n": 256} for name, value in {
        "ga_pass_rate": 0, "family_pass_rate": 0, "mean_hmm_coverage": 0,
        "exact_training_match_rate": 0, "mean_nearest_training_identity": .1,
        "within_training_length_rate": 1, "excess_duplicate_rate": 255/256}.items()}
    metrics["mean_pairwise_distance"] = {"value": 0, "n": 256, "pairs": 32640}
    value = {"input_sha256": DIAG.sha256_file(fasta), "similarity_metric": DIAG.METRIC,
             "hmm": {"accession": "PF00194.27", "sha256": "75ace6724895a6b7fe31c050b7ab9d73968feae442f40fda487ef4aecda711a3", "ga_sequence": 27, "ga_domain": 27},
             "training_reference": {"train_fasta_sha256": "ba3ef41022933ea83105f3ad09d095ed726c1c20e69da9be4d744c0b765738a7", "min_hmm_coverage": .8, "split_manifest_sha256": "d0611fd93b331a12dab9407597cba246adad2b10115f86e1f26b314c32dbd345"},
             "counts": {"emitted_fasta_records": 256, "valid_amino_acid_sequences": 256,
                        "invalid_amino_acid_sequences": 0, "unique_valid_sequences": 1,
                        "excess_duplicate_records": 255, "ga_pass": 0, "family_pass": 0},
             "metrics": metrics}
    (folder/"evaluation/metrics.json").write_text(json.dumps(value))
    return tmp_path, folder


def test_recorded_group_checks_hash_flags_and_optional_copy_reference(recorded_group):
    root, folder = recorded_group
    summary, records, sources = DIAG.inspect_group(root/"outputs/formal", "baseline", 101)
    assert summary["excess_duplicate_records"] == 255
    assert len(records) == 256 and len(sources) == 3
    with pytest.raises(ValueError, match="Exact-copy flag mismatch"):
        DIAG.inspect_group(root/"outputs/formal", "baseline", 101, {"reference": "A"*180})
    with (folder/"sequences.fasta").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Changed input"):
        DIAG.inspect_group(root/"outputs/formal", "baseline", 101)


def test_invalid_record_cannot_be_reported_as_256_valid(recorded_group):
    root, folder = recorded_group
    fasta = folder/"sequences.fasta"
    fasta.write_text(fasta.read_text().replace("A"*180, "B"*180, 1))
    metrics_path = folder/"evaluation/metrics.json"
    value = json.loads(metrics_path.read_text())
    value["input_sha256"] = DIAG.sha256_file(fasta)
    metrics_path.write_text(json.dumps(value))
    csv_path = folder/"evaluation/per_sequence.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["sequence"] = "B"*180
    rows[0]["valid_amino_acids"] = "False"
    DIAG.csv_write(csv_path, rows)
    with pytest.raises(ValueError, match="requires 256 valid records"):
        DIAG.inspect_group(root/"outputs/formal", "baseline", 101)


def test_incomplete_public_bundle_requires_explicit_flag(recorded_group, monkeypatch):
    root, _ = recorded_group
    destination = root/"analysis"
    arguments = ["diagnose_generations.py", "--input-root", str(root), "--output-dir", str(destination)]
    monkeypatch.setattr(DIAG.sys, "argv", arguments)
    with pytest.raises(ValueError, match="Formal evaluation incomplete"):
        DIAG.main()
    assert not destination.exists()
    monkeypatch.setattr(DIAG.sys, "argv", arguments+["--allow-incomplete"])
    DIAG.main()
    result = json.loads((destination/"diagnostics.json").read_text())
    assert result["complete"] is False and result["completed_groups"] == 1
    assert len(result["missing_groups"]) == 11
    assert result["checks"]["global_nearest_neighbor_search_reexecuted"] is False
    assert not result["checks"]["reported_nearest_pair_identity_recomputed_for_every_valid_record"]
