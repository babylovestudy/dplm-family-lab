import copy

import pytest

from dplm_family_lab.common import SequenceRecord, sha256_file, write_fasta, write_json
from dplm_family_lab.experiment import _generation_run, _training_run
from dplm_family_lab.sampling import length_schedule


def test_orchestrator_refuses_reuse_after_training_input_changes(tmp_path):
    fasta = tmp_path / "train.fasta"
    write_fasta([SequenceRecord("a", "ACDE")], fasta)
    output = tmp_path / "run"
    config = {"train": {"output_dir": str(output), "train_fasta": str(fasta)}}
    write_json(output / "config.json", config)
    write_json(output / "training_report.json", {
        "status": "complete", "data_sha256": {"train_fasta": sha256_file(fasta)}
    })
    write_json(output / "export_best" / "roundtrip.json", {"passed": True})
    assert _training_run(config)["status"] == "complete"
    write_fasta([SequenceRecord("a", "ACDF")], fasta)
    with pytest.raises(ValueError, match="Training input changed"):
        _training_run(config)


def test_orchestrator_refuses_changed_config(tmp_path):
    output = tmp_path / "run"
    config = {"train": {"output_dir": str(output), "seed": 42}}
    write_json(output / "config.json", config)
    changed = copy.deepcopy(config)
    changed["train"]["seed"] = 43
    with pytest.raises(ValueError, match="configuration changed"):
        _training_run(changed)


def test_orchestrator_generation_reuse_checks_fasta_checksum(tmp_path):
    source = tmp_path / "train.fasta"
    write_fasta([SequenceRecord("a", "ACDE")], source)
    output = tmp_path / "generated"
    config = {
        "model": {"source": "test-source"}, "data": {"max_length": 256},
        "generate": {"lengths_source": str(source), "seed": 101, "total_sequences": 1,
                     "length_schedule_path": str(tmp_path / "lengths.json"),
                     "batch_size": 1, "max_iter": 100},
    }
    _, schedule_path = length_schedule(config)
    write_fasta([SequenceRecord("seq_0000", "ACDE")], output / "sequences.fasta")
    manifest = {
        "status": "complete", "model": "test-source", "seed": 101,
        "batch_size": 1, "max_iter": 100, "sequences": 1,
        "length_schedule_sha256": sha256_file(schedule_path),
        "fasta_sha256": sha256_file(output / "sequences.fasta"),
    }
    write_json(output / "manifest.json", manifest)
    assert _generation_run(config, output) == manifest
    write_fasta([SequenceRecord("seq_0000", "ACDF")], output / "sequences.fasta")
    with pytest.raises(ValueError, match="differs from this protocol"):
        _generation_run(config, output)


def test_generation_reuse_refuses_weight_replacement_at_same_path(tmp_path):
    source = tmp_path / "train.fasta"
    write_fasta([SequenceRecord("a", "ACDE")], source)
    model = tmp_path / "export_best"
    model.mkdir()
    weights = model / "pytorch_model.bin"
    weights.write_bytes(b"test-only original weight bytes")
    output = tmp_path / "generated"
    config = {
        "model": {"source": str(model)}, "data": {"max_length": 256},
        "generate": {"lengths_source": str(source), "seed": 101, "total_sequences": 1,
                     "length_schedule_path": str(tmp_path / "lengths.json"),
                     "batch_size": 1, "max_iter": 100},
    }
    _, schedule_path = length_schedule(config)
    write_fasta([SequenceRecord("seq_0000", "ACDE")], output / "sequences.fasta")
    manifest = {
        "status": "complete", "model": str(model), "seed": 101,
        "batch_size": 1, "max_iter": 100, "sequences": 1,
        "length_schedule_sha256": sha256_file(schedule_path),
        "fasta_sha256": sha256_file(output / "sequences.fasta"),
        "provenance": {"artifact_role": "adapted_export", "loaded_weights_sha256": sha256_file(weights)},
    }
    write_json(output / "manifest.json", manifest)
    assert _generation_run(config, output) == manifest
    weights.write_bytes(b"test-only replacement weight bytes")
    with pytest.raises(ValueError, match="weights changed at the same path"):
        _generation_run(config, output)
