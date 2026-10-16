"""Synthetic metadata fixtures only; these are not measured model results."""

import csv
import json
from pathlib import Path

from dplm_family_lab.common import sha256_file, write_json
from dplm_family_lab.reporting import build_report


REVISION = "synthetic-fixture-revision-not-a-real-measurement"


def config():
    return {"model": {"source": "airkingbd/dplm_150m", "revision": REVISION},
            "train": {"max_steps": 1000}}


def provenance(tiny=False):
    return {"source": "airkingbd/dplm_150m", "revision": REVISION,
            "upstream_commit": "synthetic-source-commit", "tiny_fixture": tiny,
            "parameter_count": 1000 if tiny else 150_000_000}


def training(root, name, maximum, step, *, tiny=False):
    directory = root / "runs" / name
    write_json(directory / "config.json", {"train": {"max_steps": maximum, "seed": 42}})
    path = directory / "training_report.json"
    write_json(path, {"status": "complete" if step == maximum else "interrupted_at_requested_step",
                      "step": step, "output_dir": str(directory), "provenance": provenance(tiny),
                      "proof": {"backbone_updated": True}, "best_validation_loss": 1.5,
                      "elapsed_seconds_this_session": 12.0, "peak_gpu_allocated_bytes": 1024 ** 3,
                      "data_sha256": {"train_fasta": "synthetic-data-hash"}})
    return path


def generation(root, arm, model):
    sample = root / "outputs" / arm
    sample.mkdir(parents=True, exist_ok=True)
    fasta = sample / "sequences.fasta"
    fasta.write_text(">synthetic\nACDE\n", encoding="utf-8")
    write_json(sample / "length_schedule.json", {"total_sequences": 256, "lengths": [4] * 256})
    write_json(sample / "manifest.json", {"status": "complete", "model": model,
               "provenance": provenance(), "seed": 101, "elapsed_seconds": 9.0})
    metrics = root / "outputs" / "evaluation" / arm / "metrics.json"
    write_json(metrics, {"input_fasta": str(fasta),
               "counts": {"emitted_fasta_records": 256, "valid_amino_acid_sequences": 250,
                          "family_pass": 125},
               "metrics": {"family_pass_rate": {"value": 0.5, "ci95": [0.4, 0.6]},
                           "ga_pass_rate": {"value": 0.6},
                           "excess_duplicate_rate": {"value": 0.1},
                           "mean_pairwise_distance": {"value": 0.8},
                           "mean_nearest_training_identity": {"value": 0.4}},
               "denominator": "synthetic fixture: valid emitted sequences"})
    return metrics


def test_no_results_is_pending_and_report_output_is_excluded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "outputs" / "report"
    write_json(output / "metrics.json", {"metrics": {"family_pass_rate": {"value": 1.0}}})
    report = build_report(config(), output)
    assert report["status"] == "pending" and report["formal_status"] == "pending"
    assert report["rows"] == []
    assert "pending" in (output / "report.md").read_text(encoding="utf-8")
    assert list(csv.DictReader((output / "summary.csv").open(encoding="utf-8"))) == []


def test_pilot_tiny_smoke_incomplete_and_formal_are_distinct(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    training(tmp_path, "pilot_seed_42", 100, 100)
    training(tmp_path, "smoke/seed_42", 20, 20)
    training(tmp_path, "tiny/seed_42", 1000, 1000, tiny=True)
    training(tmp_path, "incomplete/seed_42", 1000, 200)
    formal_path = training(tmp_path, "formal/seed_42", 1000, 1000)
    result = build_report(config())
    assert {row["stage"] for row in result["rows"]} == {"pilot", "smoke", "tiny", "incomplete", "formal"}
    assert result["formal_training_runs"] == 1
    formal = next(r for r in result["rows"] if r["stage"] == "formal")
    assert formal["source_sha256"] == sha256_file(formal_path)
    assert formal["model_revision"] == REVISION
    assert formal["elapsed_seconds"] == 12.0
    assert formal["peak_gpu_allocated_bytes"] == 1024 ** 3
    before = sha256_file(formal_path)
    build_report(config())
    assert sha256_file(formal_path) == before


def test_short_unmarked_runs_and_missing_identity_cannot_be_formal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    training(tmp_path, "seed_42", 100, 100)
    path = training(tmp_path, "seed_43", 1000, 1000)
    payload = json.loads(path.read_text())
    payload["provenance"].pop("revision")
    write_json(path, payload)
    result = build_report(config())
    assert result["formal_status"] == "pending"
    assert {r["stage"] for r in result["rows"]} == {"pilot", "unverified"}


def test_generation_metrics_linked_to_pilot_and_formal_with_real_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    training(tmp_path, "pilot_seed_42", 100, 100)
    training(tmp_path, "pf00194/seed_42", 1000, 1000)
    pilot_metrics = generation(tmp_path, "pilot_eval", "runs/pilot_seed_42/export_best")
    generation(tmp_path, "formal_eval", "/remote/work/dplm-family-lab/runs/pf00194/seed_42/export_best")
    generation(tmp_path, "baseline", "airkingbd/dplm_150m")
    result = build_report(config())
    generated = [r for r in result["rows"] if r["source_type"] == "generation"]
    assert {r["stage"] for r in generated} == {"pilot", "formal", "baseline"}
    row = next(r for r in generated if r["stage"] == "pilot")
    assert row["family_pass_rate"] == 0.5
    assert row["overall_family_yield"] == 125 / 256
    assert row["mean_nearest_training_identity"] == 0.4
    assert row["source_sha256"] == sha256_file(pilot_metrics)
    assert row["model_revision"] == REVISION
    text = Path(result["artifacts"]["report"]).read_text(encoding="utf-8")
    assert "pretraining overlap" in text and "not exact sequence likelihood" in text
    assert REVISION in text


def test_denoising_and_malformed_sources_do_not_fabricate_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "outputs" / "test_eval"
    write_json(directory / "test_baseline.json", {"split": "test", "model": "airkingbd/dplm_150m",
               "weighted_masked_denoising_loss": 2.0, "source_sha256": "synthetic-test-hash",
               "provenance": provenance()})
    (directory / "metrics.json").write_text("{incomplete", encoding="utf-8")
    result = build_report(config())
    assert len(result["rows"]) == 1 and len(result["warnings"]) == 1
    assert result["rows"][0]["denoising_loss"] == 2.0
    assert result["formal_status"] == "pending"
    assert result["rows"][0]["family_pass_rate"] is None


def test_formal_model_with_short_generation_is_still_a_pilot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    training(tmp_path, "pf00194/seed_42", 1000, 1000)
    generation(tmp_path, "short_generation", "runs/pf00194/seed_42/export_best")
    schedule = tmp_path / "outputs" / "short_generation" / "length_schedule.json"
    write_json(schedule, {"total_sequences": 32, "lengths": [4] * 32})
    result = build_report(config())
    generated = next(row for row in result["rows"] if row["source_type"] == "generation")
    assert generated["stage"] == "pilot"
    assert generated["requested_sequences"] == 32
    assert "below" in generated["notes"]
