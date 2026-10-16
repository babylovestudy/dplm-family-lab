"""Small synthetic metadata fixtures; no trained-model measurements are made."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dplm_family_lab.backend import DPLM_150M_REVISION, DPLM_150M_SHA256, UPSTREAM_COMMIT
from dplm_family_lab.common import SequenceRecord, sha256_file, write_fasta, write_json


spec = importlib.util.spec_from_file_location("analyze_results", Path(__file__).resolve().parents[1]
                                            / "scripts/analyze_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def synthetic_pilot(root):
    """32 rows only satisfy metadata checks, not actual HMM/model evaluation."""
    directory = root / "outputs/pilot"
    reference = {"train_fasta_sha256": "0" * 64, "hmm_sha256": "1" * 64,
                 "split_manifest_sha256": "2" * 64, "min_hmm_coverage": 0.8}
    provenance = {"source": "airkingbd/dplm_150m", "revision": DPLM_150M_REVISION,
                  "checkpoint_sha256": DPLM_150M_SHA256, "upstream_commit": UPSTREAM_COMMIT,
                  "parameter_count": 150_000_000, "tiny_fixture": False,
                  "synthetic_metadata_fixture": True}
    schedule = {"seed": 101, "total_sequences": 32, "lengths": [4] * 32,
                "source_sha256": reference["train_fasta_sha256"]}
    shared = root / "outputs/length_schedules/seed_101_n32.json"
    write_json(shared, schedule)
    run = root / "runs/pf00194/pilot_seed_42"
    roundtrip = {"passed": True, "max_logit_error_cpu_fp32": 0.0, "weights_sha256": "a" * 64}
    train = {"status": "complete", "step": 100, "export_roundtrip": roundtrip}
    write_json(run / "training_report.json", train)
    write_json(run / "export_best/roundtrip.json", roundtrip)
    adapted_provenance = {**provenance, "artifact_role": "adapted_export", "loaded_weights_sha256": "b" * 64}
    write_json(run / "export_best/family_lab_provenance.json", adapted_provenance)
    write_json(run / "export_best/export_identity.json", {
        "schema_version": 1, "weights_file": "pytorch_model.bin", "weights_file_sha256": "b" * 64,
        "tensor_sha256": roundtrip["weights_sha256"], "training_report_sha256": sha256_file(run / "training_report.json"),
        "roundtrip_json_sha256": sha256_file(run / "export_best/roundtrip.json"),
        "provenance_json_sha256": sha256_file(run / "export_best/family_lab_provenance.json"),
        "synthetic_metadata_fixture": True,
    })
    entries = [{"stage": "train_42", "result": train}]
    for arm in ("baseline", "train_seed_42"):
        path = directory / arm / "seed_101"
        model = "model_cache/dplm_150m" if arm == "baseline" else "runs/pf00194/pilot_seed_42/export_best"
        records = [SequenceRecord(f"seq_{index:04d}", f"AC{analysis.AMINO_ACIDS[index // 20]}{analysis.AMINO_ACIDS[index % 20]}")
                   for index in range(32)]
        write_fasta(records, path / "sequences.fasta")
        write_json(path / "length_schedule.json", schedule)
        manifest = {"status": "complete", "seed": 101, "sequences": 32, "model": model,
                    "length_schedule_path": "outputs/length_schedules/seed_101_n32.json",
                    "length_schedule_sha256": sha256_file(shared), "fasta_sha256": sha256_file(path / "sequences.fasta"),
                    "output_alphabet": analysis.AMINO_ACIDS, "sampling_strategy": "gumbel_argmax",
                    "max_iter": 100, "batch_size": 1,
                    "provenance": provenance if arm == "baseline" else adapted_provenance,
                    "batches": [{"offset": index, "count": 1, "seed": index} for index in range(32)]}
        write_json(path / "manifest.json", manifest)
        metrics = {"input_fasta": f"outputs/pilot/{arm}/seed_101/sequences.fasta",
                   "input_sha256": manifest["fasta_sha256"], "training_reference": reference,
                   "similarity_metric": "global-v1:synthetic-metadata-test-only",
                   "counts": {"emitted_fasta_records": 32, "valid_amino_acid_sequences": 32,
                              "invalid_amino_acid_sequences": 0, "unique_valid_sequences": 32,
                              "excess_duplicate_records": 0, "family_pass": 0, "ga_pass": 0},
                   "metrics": {key: {"value": 0.5 if key == "mean_pairwise_distance" else
                                     0.25 if key == "mean_nearest_training_identity" else 0.0}
                               for key in analysis.METRICS}}
        write_json(path / "evaluation/metrics.json", metrics)
        rows = [{"id": record.id, "sequence": record.sequence, "length": 4,
                 "valid_amino_acids": True, "within_training_length": True,
                 "family_pass": False, "ga_pass": False, "exact_training_match": False,
                 "hmm_coverage": 0.0, "nearest_training_identity": 0.25,
                 "duplicate_of_earlier": False} for record in records]
        with (path / "evaluation/per_sequence.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        denoising = {"split": "test", "source_sha256": "3" * 64, "noise_seed": 9174,
                     "sequences": 10, "masked_residues": 20, "weighted_masked_denoising_loss": 1.0,
                     "provenance": provenance if arm == "baseline" else adapted_provenance, "model": model}
        name = "denoising_baseline.json" if arm == "baseline" else "denoising_adapted.json"
        write_json(directory / arm / "test_denoising" / name, denoising)
        entries.extend([{"stage": f"generate_{arm}_101", "result": manifest},
                        {"stage": f"evaluate_{arm}_101", "result": metrics},
                        {"stage": f"test_denoising_{arm}", "result": denoising}])
    write_json(directory / "experiment_status.json", {"mode": "pilot", "status": "complete", "completed": entries})


def test_complete_synthetic_pilot_is_reproducible_and_explicitly_pilot(tmp_path):
    synthetic_pilot(tmp_path / "input")
    first = analysis.analyze("pilot", tmp_path / "input", tmp_path / "a")
    second = analysis.analyze("pilot", tmp_path / "input", tmp_path / "b")
    assert first == second
    assert (tmp_path / "a/analysis.json").read_bytes() == (tmp_path / "b/analysis.json").read_bytes()
    assert first["mode"] == "pilot" and first["generation_groups"] == 2
    assert len(first["test_denoising"]) == 2
    row = first["paired_comparisons"][0]
    assert row["training_seed"] == 42 and row["generation_seed"] == 101
    assert row["family_rate_difference"] == 0 and row["family_difference_ci95_low"] == 0
    assert "zero population" in " ".join(first["limitations"])


def test_paired_bootstrap_preserves_index_correlation():
    outcomes = [0, 1] * 16
    interval = analysis.paired_interval(outcomes, outcomes, samples=1000, seed=42,
                                        training_seed=42, generation_seed=101)
    assert interval == {"value": 0.0, "ci95": [0.0, 0.0]}
    positive = analysis.paired_interval(np.zeros(32), np.ones(32), samples=1000, seed=42,
                                        training_seed=42, generation_seed=101)
    assert positive == {"value": 1.0, "ci95": [1.0, 1.0]}


@pytest.mark.parametrize("mismatch", ["missing", "ids", "length_sha", "invalid", "journal", "seed"])
def test_incomplete_or_unpaired_metadata_refuses_analysis(tmp_path, mismatch):
    root = tmp_path / "input"
    synthetic_pilot(root)
    arm = root / "outputs/pilot/train_seed_42/seed_101"
    if mismatch == "missing":
        (arm / "evaluation/per_sequence.csv").unlink()
    elif mismatch in {"ids", "invalid"}:
        path = arm / "evaluation/per_sequence.csv"
        text = path.read_text()
        path.write_text(text.replace("seq_0000", "wrong_id", 1) if mismatch == "ids"
                        else text.replace("True,True", "False,True", 1))
    elif mismatch == "length_sha":
        path = arm / "length_schedule.json"
        value = json.loads(path.read_text())
        value["lengths"][0] = 5
        write_json(path, value)
    elif mismatch == "seed":
        path = arm / "manifest.json"
        value = json.loads(path.read_text())
        value["seed"] = 102
        write_json(path, value)
    else:
        path = root / "outputs/pilot/experiment_status.json"
        value = json.loads(path.read_text())
        value["status"] = "running"
        write_json(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        analysis.analyze("pilot", root, tmp_path / "analysis")
    assert not (tmp_path / "analysis/analysis.json").exists()


@pytest.mark.parametrize("kind", ["generation", "denoising", "identity"])
def test_wrong_or_unsealed_adapted_weights_refuse_analysis(tmp_path, kind):
    root = tmp_path / "input"
    synthetic_pilot(root)
    if kind == "identity":
        (root / "runs/pf00194/pilot_seed_42/export_best/export_identity.json").unlink()
    else:
        if kind == "generation":
            path = root / "outputs/pilot/train_seed_42/seed_101/manifest.json"
            stage = "generate_train_seed_42_101"
        else:
            path = root / "outputs/pilot/train_seed_42/test_denoising/denoising_adapted.json"
            stage = "test_denoising_train_seed_42"
        value = json.loads(path.read_text())
        value["provenance"]["loaded_weights_sha256"] = "c" * 64
        write_json(path, value)
        journal_path = root / "outputs/pilot/experiment_status.json"
        journal = json.loads(journal_path.read_text())
        next(entry for entry in journal["completed"] if entry["stage"] == stage)["result"] = value
        write_json(journal_path, journal)
    with pytest.raises((ValueError, FileNotFoundError)):
        analysis.analyze("pilot", root, tmp_path / "analysis")
    assert not (tmp_path / "analysis/analysis.json").exists()
