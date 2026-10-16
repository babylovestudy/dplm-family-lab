"""Offline integration tests exercise the vendored ESM/DPLM network itself."""
# ruff: noqa: E402

from copy import deepcopy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from dplm_family_lab.backend import (
    DPLM_150M_REVISION,
    create_tiny_fixture,
    load_backend,
    safe_denoising_loss,
    seal_export_identity,
    tensor_digest,
)
from dplm_family_lab.common import (
    SequenceRecord,
    seed_everything,
    sha256_file,
    write_fasta,
    write_json,
)
from dplm_family_lab.training import evaluate_denoising, train


@pytest.fixture
def tiny_config(tmp_path):
    torch.set_num_threads(1)
    seed_everything(2)
    source = create_tiny_fixture(tmp_path / "tiny_weights")
    for split in ("train", "val", "test"):
        write_fasta(
            [
                SequenceRecord(f"{split}_1", "ACDEFGHIKLMNPQ"),
                SequenceRecord(f"{split}_2", "RSTVWYACDEFGHI"),
            ],
            tmp_path / f"{split}.fasta",
        )
    return {
        "model": {"source": str(source), "allow_tiny_fixture": True},
        "data": {"max_length": 256},
        "train": {
            "output_dir": str(tmp_path / "run"),
            "seed": 7,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.001,
            "max_steps": 2,
            "eval_every": 1,
            "checkpoint_every": 1,
            "gradient_checkpointing": True,
            "mixed_precision": "fp32",
            "warmup_steps": 0,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "validation_seed": 9173,
            "train_fasta": str(tmp_path / "train.fasta"),
            "val_fasta": str(tmp_path / "val.fasta"),
            "test_fasta": str(tmp_path / "test.fasta"),
            "device": "cpu",
        },
        "evaluation": {"output_dir": str(tmp_path / "eval")},
    }


def test_zero_mask_is_finite_and_differentiable():
    scores = torch.randn(1, 3, 33, requires_grad=True)
    loss, _, count = safe_denoising_loss(
        scores,
        torch.zeros(1, 3, dtype=torch.long),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.ones(1, 1),
    )
    assert count == 0 and loss.item() == 0
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_training_updates_backbone_exports_and_evaluates(tiny_config):
    report = train(tiny_config)
    assert report["proof"]["backbone_updated"]
    assert report["proof"]["gradient_checkpointing_enabled"]
    assert report["export_roundtrip"]["passed"]
    assert report["provenance"]["tiny_fixture"]
    for path in ("best.pt", "last.pt", "validation_noise.pt", "export_best/roundtrip.json"):
        assert (Path(report["output_dir"]) / path).exists()
    run = Path(report["output_dir"])
    identity = json.loads((run / "export_best/export_identity.json").read_text())
    assert identity["training_report_sha256"] == sha256_file(run / "training_report.json")
    assert identity["roundtrip_json_sha256"] == sha256_file(run / "export_best/roundtrip.json")
    assert identity["tensor_sha256"] == report["export_roundtrip"]["weights_sha256"]
    assert identity["weights_file_sha256"] == sha256_file(run / "export_best/pytorch_model.bin")
    base = evaluate_denoising(tiny_config)
    adapted = evaluate_denoising(tiny_config, report["best_checkpoint"])
    assert base["source_sha256"] == adapted["source_sha256"]
    assert base["noise_seed"] == adapted["noise_seed"]
    with pytest.raises(FileExistsError):
        train(tiny_config)


def test_seal_existing_export_preserves_original_bytes_and_binds_final_report(tiny_config):
    report = train(tiny_config)
    run = Path(report["output_dir"])
    exported = run / "export_best"
    (exported / "export_identity.json").unlink()
    originals = [
        run / "training_report.json",
        exported / "roundtrip.json",
        exported / "family_lab_provenance.json",
        exported / "pytorch_model.bin",
    ]
    before = {str(path): sha256_file(path) for path in originals}
    identity = seal_export_identity(exported, run / "training_report.json")
    assert identity["training_report_sha256"] == before[str(run / "training_report.json")]
    assert {str(path): sha256_file(path) for path in originals} == before
    weights = exported / "pytorch_model.bin"
    state = torch.load(weights, weights_only=False)
    next(value for value in state.values() if value.is_floating_point()).add_(1)
    torch.save(state, weights)
    with pytest.raises(ValueError, match="tensors do not match"):
        seal_export_identity(exported, run / "training_report.json")


@pytest.mark.parametrize("legacy_signature", [False, True])
def test_optimizer_boundary_resume_matches_uninterrupted(tiny_config, legacy_signature):
    full = train(tiny_config)
    interrupted = deepcopy(tiny_config)
    interrupted["train"]["output_dir"] += "_resumed"
    interrupted["train"]["stop_after_steps"] = 1
    first = train(interrupted)
    assert first["step"] == 1
    if legacy_signature:
        checkpoint = torch.load(first["last_checkpoint"], weights_only=False)
        checkpoint["signature"]["train"].pop("eval_every")
        torch.save(checkpoint, first["last_checkpoint"])
    interrupted["train"].pop("stop_after_steps")
    resumed = train(interrupted, resume=first["last_checkpoint"])
    assert full["proof"]["current_model_sha256"] == resumed["proof"]["current_model_sha256"]
    state_a = torch.load(full["last_checkpoint"], weights_only=False)
    state_b = torch.load(resumed["last_checkpoint"], weights_only=False)
    assert state_a["sampler"] == state_b["sampler"]
    assert torch.equal(state_a["rng"]["torch"], state_b["rng"]["torch"])


@pytest.mark.parametrize("legacy_signature", [False, True])
def test_resume_rejects_changed_validation_cadence(tiny_config, legacy_signature):
    tiny_config["train"]["stop_after_steps"] = 1
    partial = train(tiny_config)
    if legacy_signature:
        checkpoint = torch.load(partial["last_checkpoint"], weights_only=False)
        checkpoint["signature"]["train"].pop("eval_every")
        torch.save(checkpoint, partial["last_checkpoint"])
    tiny_config["train"].pop("stop_after_steps")
    tiny_config["train"]["eval_every"] = 2
    with pytest.raises(ValueError, match="signature differs"):
        train(tiny_config, resume=partial["last_checkpoint"])


def test_unplanned_stop_does_not_change_best_selection_schedule(tiny_config):
    tiny_config["train"]["eval_every"] = 2
    reference = train(tiny_config)
    interrupted = deepcopy(tiny_config)
    interrupted["train"]["output_dir"] += "_early_stop"
    interrupted["train"]["stop_after_steps"] = 1
    partial = train(interrupted)
    assert partial["best_checkpoint"] is None
    assert partial["best_validation_loss"] is None
    interrupted["train"].pop("stop_after_steps")
    resumed = train(interrupted, resume=partial["last_checkpoint"])
    assert reference["proof"]["current_model_sha256"] == resumed["proof"]["current_model_sha256"]
    assert reference["best_validation_loss"] == resumed["best_validation_loss"]


def test_tiny_fixture_cannot_masquerade_as_real_model(tiny_config):
    fixture = load_backend(tiny_config)
    assert fixture.provenance["tiny_fixture"] is True
    assert "checkpoint_sha256" not in fixture.provenance
    assert fixture.provenance["source"] != "airkingbd/dplm_150m"
    tiny_config["model"]["allow_tiny_fixture"] = False
    with pytest.raises(ValueError, match="not DPLM-150M"):
        load_backend(tiny_config)


def test_original_source_claim_requires_official_weight_hash(tiny_config):
    write_json(
        Path(tiny_config["model"]["source"]) / "family_lab_provenance.json",
        {"source": "airkingbd/dplm_150m", "revision": DPLM_150M_REVISION},
    )
    with pytest.raises(ValueError, match="official revision/size/SHA"):
        load_backend(tiny_config)


@pytest.mark.parametrize("keep_bin", [True, False])
def test_loader_uses_the_same_weight_file_it_hashes(tiny_config, keep_bin):
    from safetensors.torch import save_file

    original = load_backend(tiny_config)
    original_digest = tensor_digest(original.model.net.state_dict())
    safe_state = {
        key: value.detach().cpu().clone() for key, value in original.model.net.state_dict().items()
    }
    probe = next(key for key in safe_state if "query.weight" in key)
    safe_state[probe].add_(0.25)
    safe_digest = tensor_digest(safe_state)
    assert safe_digest != original_digest
    source = Path(tiny_config["model"]["source"])
    safe_path = source / "model.safetensors"
    save_file(safe_state, safe_path, metadata={"format": "pt"})
    if not keep_bin:
        (source / "pytorch_model.bin").unlink()
    restored = load_backend(tiny_config)
    selected = source / "pytorch_model.bin" if keep_bin else safe_path
    assert restored.provenance["loaded_weights_sha256"] == sha256_file(selected)
    assert tensor_digest(restored.model.net.state_dict()) == (
        original_digest if keep_bin else safe_digest
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for mixed-precision smoke test"
)
def test_cuda_training_smoke(tiny_config):
    tiny_config["train"]["device"] = "cuda"
    tiny_config["train"]["mixed_precision"] = "auto"
    report = train(tiny_config)
    assert report["peak_gpu_allocated_bytes"] > 0
    assert report["export_roundtrip"]["passed"]
