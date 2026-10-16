"""Synthetic metadata fixtures only; these are not measured model results."""

import csv
import json
from pathlib import Path

import pytest

from dplm_family_lab.backend import DPLM_150M_REVISION, DPLM_150M_SHA256, UPSTREAM_COMMIT
from dplm_family_lab.common import sha256_file, write_json
from dplm_family_lab.reporting import _real_identity, build_report


# Fixed identifiers exercise the metadata gate. All generated fixture results
# below are synthetic metadata, never authenticated model measurements.
REVISION = DPLM_150M_REVISION
ADAPTED_FILE_SHA = "b" * 64  # Synthetic file identity, not measured weights.


def config():
    return {"model": {"source": "airkingbd/dplm_150m", "revision": REVISION,
                      "upstream_commit": UPSTREAM_COMMIT},
            "train": {"max_steps": 1000}}


def provenance(tiny=False):
    return {"source": "airkingbd/dplm_150m", "revision": REVISION,
            "upstream_commit": UPSTREAM_COMMIT, "checkpoint_sha256": DPLM_150M_SHA256,
            "synthetic_metadata_fixture": True, "tiny_fixture": tiny,
            "parameter_count": 1000 if tiny else 150_000_000}


def seal_metadata_fixture(training_path):
    """Construct metadata only; never claim a CPU tensor seal was executed."""
    payload = json.loads(training_path.read_text())
    export = training_path.parent / "export_best"
    roundtrip = payload["export_roundtrip"]
    exported_provenance = {**payload["provenance"], "artifact_role": "adapted_export"}
    write_json(export / "roundtrip.json", roundtrip)
    write_json(export / "family_lab_provenance.json", exported_provenance)
    write_json(export / "export_identity.json", {
        "schema_version": 1, "weights_file": "pytorch_model.bin", "weights_file_sha256": ADAPTED_FILE_SHA,
        "tensor_sha256": roundtrip["weights_sha256"], "training_report_sha256": sha256_file(training_path),
        "roundtrip_json_sha256": sha256_file(export / "roundtrip.json"),
        "provenance_json_sha256": sha256_file(export / "family_lab_provenance.json"),
        "synthetic_metadata_fixture": True,
    })


def adapted_provenance():
    return {**provenance(), "artifact_role": "adapted_export", "loaded_weights_sha256": ADAPTED_FILE_SHA}


def training(root, name, maximum, step, *, tiny=False):
    directory = root / "runs" / name
    write_json(directory / "config.json", {"model": config()["model"],
                                          "train": {"max_steps": maximum, "seed": 42}})
    path = directory / "training_report.json"
    write_json(path, {"status": "complete" if step == maximum else "interrupted_at_requested_step",
                      "step": step, "output_dir": str(directory), "provenance": provenance(tiny),
                      "proof": {"backbone_updated": True}, "best_validation_loss": 1.5,
                      "export_roundtrip": {"passed": True, "max_logit_error_cpu_fp32": 0.0,
                                           "weights_sha256": "a" * 64},
                      "elapsed_seconds_this_session": 12.0, "peak_gpu_allocated_bytes": 1024 ** 3,
                      "data_sha256": {"train_fasta": "synthetic-data-hash"}})
    seal_metadata_fixture(path)
    return path


def generation(root, arm, model):
    sample = root / "outputs" / arm
    sample.mkdir(parents=True, exist_ok=True)
    fasta = sample / "sequences.fasta"
    fasta.write_text(">synthetic\nACDE\n", encoding="utf-8")
    write_json(sample / "length_schedule.json", {"total_sequences": 256, "lengths": [4] * 256})
    write_json(sample / "manifest.json", {"status": "complete", "model": model,
               "provenance": provenance() if model == "airkingbd/dplm_150m" else adapted_provenance(),
               "seed": 101, "elapsed_seconds": 9.0})
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


def complete_metadata_fixture(root):
    """Construct a complete *synthetic* journal, not real DPLM measurements.

    These tiny text assets only test metadata/hash consistency. No model,
    sequence generation, HMM scoring or biological experiment is performed.
    """
    cfg = config()
    cfg["train"].update({"output_dir": "runs/pf00194/seed_42", "validation_seed": 9173})
    cfg["data"] = {"processed_dir": "data/processed"}
    cfg["hmm"] = {"path": "data/raw/synthetic.hmm"}
    cfg["generate"] = {"total_sequences": 256, "max_iter": 100, "batch_size": 1}
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    for split in ("train", "val", "test"):
        path = processed / f"{split}.fasta"
        path.write_text(f">synthetic_{split}\nACDE\n", encoding="utf-8")
        cfg["train"][f"{split}_fasta"] = str(path.relative_to(root))
    hmm = root / cfg["hmm"]["path"]
    hmm.parent.mkdir(parents=True)
    hmm.write_text("Synthetic metadata fixture, not a scoreable HMM.\n", encoding="utf-8")
    split = {"files": {f"{name}.fasta": sha256_file(processed / f"{name}.fasta")
                       for name in ("train", "val", "test")}}
    write_json(processed / "split_manifest.json", split)
    reference = {"train_fasta_sha256": sha256_file(processed / "train.fasta"),
                 "hmm_sha256": sha256_file(hmm),
                 "split_manifest_sha256": sha256_file(processed / "split_manifest.json")}
    write_json(processed / "training_ref.json", reference)
    completed = [{"stage": "data_audit", "result": {"passed": True, "synthetic_fixture": True}}]
    for seed in (42, 43, 44):
        path = training(root, f"pf00194/seed_{seed}", 1000, 1000)
        saved = json.loads((path.parent / "config.json").read_text())
        saved["train"]["seed"] = seed
        write_json(path.parent / "config.json", saved)
        payload = json.loads(path.read_text())
        payload["data_sha256"] = {f"{name}_fasta": sha256_file(processed / f"{name}.fasta")
                                  for name in ("train", "val")}
        payload["export_roundtrip"] = {"passed": True, "max_logit_error_cpu_fp32": 0.0,
                                       "weights_sha256": f"{seed:064x}"}
        write_json(path, payload)
        seal_metadata_fixture(path)
        completed.append({"stage": f"train_{seed}", "result": payload})

    arms = ("baseline", "train_seed_42", "train_seed_43", "train_seed_44")
    for arm in arms:
        model = "airkingbd/dplm_150m" if arm == "baseline" else f"runs/pf00194/seed_{arm[-2:]}/export_best"
        model_provenance = provenance() if arm == "baseline" else adapted_provenance()
        denoising = {"split": "test", "model": model, "provenance": model_provenance,
                     "weighted_masked_denoising_loss": 2.0, "masked_residues": 2,
                     "source_sha256": sha256_file(processed / "test.fasta"), "noise_seed": 9174}
        filename = "denoising_baseline.json" if arm == "baseline" else "denoising_adapted.json"
        write_json(root / "outputs/formal" / arm / "test_denoising" / filename, denoising)
        completed.append({"stage": f"test_denoising_{arm}", "result": denoising})
        for seed in (101, 102, 103):
            directory = root / "outputs/formal" / arm / f"seed_{seed}"
            directory.mkdir(parents=True)
            fasta = directory / "sequences.fasta"
            fasta.write_text("".join(f">synthetic_{i}\nACDE\n" for i in range(256)), encoding="utf-8")
            schedule = {"seed": seed, "total_sequences": 256, "lengths": [4] * 256,
                        "source_sha256": reference["train_fasta_sha256"]}
            shared = root / "outputs/shared" / f"seed_{seed}.json"
            write_json(shared, schedule)
            write_json(directory / "length_schedule.json", schedule)
            manifest = {"status": "complete", "model": model, "provenance": model_provenance,
                        "seed": seed, "sequences": 256, "batch_size": 1, "max_iter": 100,
                        "sampling_strategy": "gumbel_argmax", "fasta_sha256": sha256_file(fasta),
                        "length_schedule_path": str(shared), "length_schedule_sha256": sha256_file(shared)}
            write_json(directory / "manifest.json", manifest)
            metrics = {"input_fasta": str(fasta), "input_sha256": sha256_file(fasta),
                       "counts": {"emitted_fasta_records": 256, "valid_amino_acid_sequences": 256,
                                  "family_pass": 128}, "training_reference": reference,
                       "metrics": {name: {"value": 0.5} for name in (
                           "ga_pass_rate", "family_pass_rate", "excess_duplicate_rate",
                           "exact_training_match_rate", "mean_pairwise_distance", "mean_nearest_training_identity")},
                       "denominator": "Synthetic metadata fixture, not real measurements."}
            write_json(directory / "evaluation/metrics.json", metrics)
            completed.extend([{"stage": f"generate_{arm}_{seed}", "result": manifest},
                              {"stage": f"evaluate_{arm}_{seed}", "result": metrics}])
    write_json(root / "outputs/formal/experiment_status.json",
               {"mode": "formal", "status": "complete", "completed": completed})
    return cfg


def test_forged_complete_journal_without_metrics_is_still_partial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    (tmp_path / "outputs/formal/baseline/seed_101/evaluation/metrics.json").unlink()
    result = build_report(cfg)
    assert result["formal_status"] == "partial"
    assert "evaluate_baseline_101" in result["missing_expected_stages"]
    assert "required parsed result row" in " ".join(result["formal_validation_issues"]["evaluate_baseline_101"])


def test_complete_synthetic_metadata_protocol_can_be_marked_complete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    result = build_report(cfg)
    assert result["formal_status"] == "complete", result["formal_validation_issues"]
    assert result["formal_training_runs"] == 3
    assert result["missing_expected_stages"] == []
    assert len(result["verified_expected_stages"]) == 34
    assert len([r for r in result["rows"] if r["source_type"] == "generation"]) == 12
    assert len([r for r in result["rows"] if r["source_type"] == "denoising"]) == 4


def test_complete_string_cannot_hide_changed_source_hash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    fasta = tmp_path / "outputs/formal/train_seed_43/seed_102/sequences.fasta"
    fasta.write_text(fasta.read_text() + ">tampered\nACDE\n", encoding="utf-8")
    result = build_report(cfg)
    assert result["formal_status"] == "partial"
    assert "generate_train_seed_43_102" in result["missing_expected_stages"]
    assert "source hash differs" in " ".join(result["formal_validation_issues"]["generate_train_seed_43_102"])


def test_running_journal_with_all_results_is_not_declared_complete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    path = tmp_path / "outputs/formal/experiment_status.json"
    journal = json.loads(path.read_text())
    journal["status"] = "running"
    write_json(path, journal)
    result = build_report(cfg)
    assert result["formal_status"] == "partial"
    assert result["missing_expected_stages"] == ["experiment_status"]


def test_invalid_completed_stage_collection_stays_partial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json(tmp_path / "outputs/formal/experiment_status.json",
               {"mode": "formal", "status": "complete", "completed": None})
    result = build_report(config())
    assert result["formal_status"] == "partial"
    assert "experiment_status" in result["missing_expected_stages"]
    assert result["formal_training_runs"] == 0


def test_pinned_baseline_can_use_cache_path_and_best_need_not_equal_last(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    journal_path = tmp_path / "outputs/formal/experiment_status.json"
    journal = json.loads(journal_path.read_text())
    manifest_path = tmp_path / "outputs/formal/baseline/seed_101/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"] = f"/offline/cache/models--airkingbd--dplm_150m/snapshots/{REVISION}"
    write_json(manifest_path, manifest)
    training_path = tmp_path / "runs/pf00194/seed_42/training_report.json"
    payload = json.loads(training_path.read_text())
    payload["proof"]["current_model_sha256"] = "synthetic-last-weights"
    payload["export_roundtrip"]["weights_sha256"] = "a" * 64
    write_json(training_path, payload)
    seal_metadata_fixture(training_path)
    for entry in journal["completed"]:
        if entry["stage"] == "generate_baseline_101":
            entry["result"] = manifest
        if entry["stage"] == "train_42":
            entry["result"] = payload
    write_json(journal_path, journal)
    result = build_report(cfg)
    assert result["formal_status"] == "complete", result["formal_validation_issues"]


def test_complete_synthetic_protocol_accepts_verified_original_cache_source(tmp_path, monkeypatch):
    """All scores here remain synthetic; this exercises source-chain equality."""
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    cfg["model"]["source"] = "model_cache/dplm_150m"
    for path in tmp_path.glob("runs/**/config.json"):
        saved = json.loads(path.read_text())
        saved["model"] = dict(cfg["model"])
        write_json(path, saved)
    for path in tmp_path.glob("outputs/**/*.json"):
        value = json.loads(path.read_text())
        candidates = [value, *(entry["result"] for entry in value.get("completed", []))]
        for payload in candidates:
            if payload.get("model") == "airkingbd/dplm_150m":
                payload["model"] = "model_cache/dplm_150m"
            elif "train_seed_" in str(path) or "export_best" in str(payload.get("model", "")):
                if "provenance" in payload:
                    payload["provenance"].update(artifact_role="adapted_export",
                                                  loaded_weights_sha256=ADAPTED_FILE_SHA)
        write_json(path, value)
    result = build_report(cfg)
    assert result["formal_status"] == "complete", result["formal_validation_issues"]
    assert result["formal_training_runs"] == 3
    assert result["missing_expected_stages"] == []
    public_cfg = {**cfg, "model": {**cfg["model"], "source": "airkingbd/dplm_150m"}}
    mismatched = build_report(public_cfg, "artifacts/public-config-mismatch")
    assert mismatched["formal_status"] == "partial"
    assert "saved model source/revision" in " ".join(mismatched["formal_validation_issues"]["train_42"])


@pytest.mark.parametrize("field,bad_value", [
    ("checkpoint_sha256", "0" * 64), ("revision", "unknown-revision"),
    ("upstream_commit", "unknown-commit"), ("source", "unknown/model"),
    ("tiny_fixture", True), ("loaded_weights_sha256", "0" * 64),
    ("artifact_role", "unknown-role"),
])
def test_official_and_cache_identity_both_require_original_pinned_chain(field, bad_value):
    for source in ("airkingbd/dplm_150m", "model_cache/dplm_150m"):
        cfg = config()
        cfg["model"]["source"] = source
        payload = provenance()
        assert _real_identity(payload, cfg)
        payload[field] = bad_value
        assert not _real_identity(payload, cfg)


def test_unknown_cache_path_needs_matching_official_source_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config()
    cfg["model"]["source"] = "unknown/local/model"
    assert not _real_identity(provenance(), cfg)
    sidecar = tmp_path / cfg["model"]["source"] / "family_lab_provenance.json"
    write_json(sidecar, provenance())
    assert _real_identity(provenance(), cfg)
    invalid = provenance()
    invalid["checkpoint_sha256"] = "0" * 64
    write_json(sidecar, invalid)
    assert not _real_identity(provenance(), cfg)


def test_unknown_baseline_model_path_does_not_complete_formal_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    path = tmp_path / "outputs/formal/baseline/seed_101/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model"] = "/unknown/unverified-model"
    write_json(path, manifest)
    journal_path = tmp_path / "outputs/formal/experiment_status.json"
    journal = json.loads(journal_path.read_text())
    next(entry for entry in journal["completed"]
         if entry["stage"] == "generate_baseline_101")["result"] = manifest
    write_json(journal_path, journal)
    result = build_report(cfg)
    assert result["formal_status"] == "partial"
    assert "generate_baseline_101" in result["missing_expected_stages"]


def test_changed_best_export_roundtrip_cannot_complete_formal_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    path = tmp_path / "runs/pf00194/seed_42/export_best/roundtrip.json"
    exported = json.loads(path.read_text())
    exported["weights_sha256"] = "f" * 64
    write_json(path, exported)
    result = build_report(cfg)
    assert result["formal_status"] == "partial"
    assert "train_42" in result["missing_expected_stages"]


def test_published_real_pilot_metadata_keeps_pilot_and_verified_cached_identity(tmp_path, monkeypatch):
    """Read actual recorded metadata when published, without running a model."""
    source = Path(__file__).resolve().parents[1] / "results/pilot/runs/pf00194/pilot_seed_42"
    if not (source / "training_report.json").is_file():
        pytest.skip("The separately collected real-pilot evidence bundle is not present")
    cfg = json.loads((source / "config.json").read_text())
    payload = json.loads((source / "training_report.json").read_text())
    assert cfg["model"]["source"] == "model_cache/dplm_150m"
    assert _real_identity(payload["provenance"], cfg)
    destination = tmp_path / "runs/pf00194/pilot_seed_42"
    write_json(destination / "config.json", cfg)
    write_json(destination / "training_report.json", payload)
    monkeypatch.chdir(tmp_path)
    result = build_report(cfg)
    assert result["rows"][0]["stage"] == "pilot"
    assert result["formal_training_runs"] == 0
    assert result["formal_status"] == "pending"


@pytest.mark.parametrize("kind", ["generation", "denoising"])
def test_same_adapted_path_with_wrong_loaded_weights_is_rejected(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    if kind == "generation":
        path = tmp_path / "outputs/formal/train_seed_42/seed_101/manifest.json"
        stage = "generate_train_seed_42_101"
    else:
        path = tmp_path / "outputs/formal/train_seed_42/test_denoising/denoising_adapted.json"
        stage = "test_denoising_train_seed_42"
    payload = json.loads(path.read_text())
    # The model path and original pretrained provenance are unchanged.
    payload["provenance"]["loaded_weights_sha256"] = "c" * 64
    write_json(path, payload)
    journal_path = tmp_path / "outputs/formal/experiment_status.json"
    journal = json.loads(journal_path.read_text())
    next(entry for entry in journal["completed"] if entry["stage"] == stage)["result"] = payload
    write_json(journal_path, journal)
    result = build_report(cfg)
    assert result["formal_status"] == "partial"
    assert stage in result["missing_expected_stages"]
    assert any(row["stage"] == "unverified" and "sealed training export" in row["notes"]
               for row in result["rows"])


@pytest.mark.parametrize("field", ["tensor_sha256", "roundtrip_json_sha256", "training_report_sha256",
                                   "provenance_json_sha256", "missing_identity"])
def test_missing_or_changed_export_identity_cannot_complete_formal_protocol(tmp_path, monkeypatch, field):
    monkeypatch.chdir(tmp_path)
    cfg = complete_metadata_fixture(tmp_path)
    path = tmp_path / "runs/pf00194/seed_42/export_best/export_identity.json"
    if field == "missing_identity":
        path.unlink()
    else:
        identity = json.loads(path.read_text())
        identity[field] = "f" * 64
        write_json(path, identity)
    result = build_report(cfg)
    assert result["formal_status"] == "partial"
    assert "train_42" in result["missing_expected_stages"]
    assert "generate_train_seed_42_101" in result["missing_expected_stages"]
