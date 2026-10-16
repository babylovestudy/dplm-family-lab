"""Offline integration tests exercise the vendored ESM/DPLM network itself."""
# ruff: noqa: E402

from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from dplm_family_lab.backend import (
    DPLM_150M_REVISION,
    create_tiny_fixture,
    load_backend,
    safe_denoising_loss,
)
from dplm_family_lab.common import SequenceRecord, seed_everything, write_fasta, write_json
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
    base = evaluate_denoising(tiny_config)
    adapted = evaluate_denoising(tiny_config, report["best_checkpoint"])
    assert base["source_sha256"] == adapted["source_sha256"]
    assert base["noise_seed"] == adapted["noise_seed"]
    with pytest.raises(FileExistsError):
        train(tiny_config)


def test_optimizer_boundary_resume_matches_uninterrupted(tiny_config):
    full = train(tiny_config)
    interrupted = deepcopy(tiny_config)
    interrupted["train"]["output_dir"] += "_resumed"
    interrupted["train"]["stop_after_steps"] = 1
    first = train(interrupted)
    assert first["step"] == 1
    interrupted["train"].pop("stop_after_steps")
    resumed = train(interrupted, resume=first["last_checkpoint"])
    assert full["proof"]["current_model_sha256"] == resumed["proof"]["current_model_sha256"]
    state_a = torch.load(full["last_checkpoint"], weights_only=False)
    state_b = torch.load(resumed["last_checkpoint"], weights_only=False)
    assert state_a["sampler"] == state_b["sampler"]
    assert torch.equal(state_a["rng"]["torch"], state_b["rng"]["torch"])


def test_tiny_fixture_cannot_masquerade_as_real_model(tiny_config):
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for mixed-precision smoke test"
)
def test_cuda_training_smoke(tiny_config):
    tiny_config["train"]["device"] = "cuda"
    tiny_config["train"]["mixed_precision"] = "auto"
    report = train(tiny_config)
    assert report["peak_gpu_allocated_bytes"] > 0
    assert report["export_roundtrip"]["passed"]
