"""Lazy, registry-free access to the pinned official DPLM-1 implementation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json

UPSTREAM_COMMIT = "8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d"
DPLM_150M_REVISION = "49b7125a5d28c6418fcc2f3c4fe799352ac1488b"
DPLM_150M_SHA256 = "ea4eaa99536b60ed76f945f71a1a5e604f08447ec3def5104a93ca6001a59961"
DPLM_150M_SIZE = 595359662
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ESM_VOCAB = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "<null_1>",
    "<mask>",
]


def require_torch():
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Training/generation needs the optional train dependencies: pip install -e '.[train]'."
        ) from exc
    return torch


@dataclass
class Backend:
    model: Any
    tokenizer: Any
    device: Any
    precision: str
    provenance: dict

    def autocast(self):
        torch = require_torch()
        if self.device.type != "cuda" or self.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def encode(self, sequences):
        batch = self.tokenizer(
            sequences, padding=True, add_special_tokens=True, return_tensors="pt"
        )
        return batch["input_ids"].to(self.device)


def _device_precision(config):
    torch = require_torch()
    requested = config.get("train", {}).get("device", "auto")
    device = torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    precision = config.get("train", {}).get("mixed_precision", "auto")
    if precision == "auto":
        precision = (
            "bf16"
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else "fp16"
            if device.type == "cuda"
            else "fp32"
        )
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("mixed_precision must be auto, bf16, fp16, or fp32")
    if device.type == "cpu":
        precision = "fp32"
    return device, precision


def load_backend(config: dict, model_path=None, *, training=False) -> Backend:
    """Load actual DPLM weights, never silently substitute ESM2 or random weights."""
    torch = require_torch()
    from transformers import EsmConfig, EsmTokenizer
    from ._vendor.esm import EsmForDPLM
    from ._vendor.model import DiffusionProteinLanguageModel

    model_cfg = config.get("model", {})
    device, precision = _device_precision(config)
    source = str(model_path or model_cfg.get("source", "airkingbd/dplm_150m"))
    checkpoint = None
    if Path(source).is_file():
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "dplm-family-training-v1":
            raise ValueError("Unrecognized training checkpoint format")
        tokenizer_dir = Path(checkpoint["tokenizer_dir"])
        if not tokenizer_dir.exists():
            tokenizer_dir = Path(source).resolve().parent / "tokenizer"
        if not tokenizer_dir.exists():
            raise FileNotFoundError("Training checkpoint needs its adjacent tokenizer directory")
        net_config = EsmConfig.from_dict(checkpoint["net_config"])
        net_config._name_or_path = str(tokenizer_dir.resolve())
        net = EsmForDPLM(net_config)
        provenance = dict(checkpoint["provenance"])
        from .common import sha256_file

        provenance["training_checkpoint_sha256"] = sha256_file(source)
        tokenizer = EsmTokenizer.from_pretrained(tokenizer_dir)
    else:
        source_path = Path(source)
        if source_path.exists():
            local_path = source_path.resolve()
            revision = None
        else:
            from huggingface_hub import model_info, snapshot_download

            revision = model_info(source, revision=model_cfg.get("revision")).sha
            local_path = Path(
                snapshot_download(
                    source,
                    revision=revision,
                    allow_patterns=[
                        "config.json",
                        "*token*.json",
                        "vocab.txt",
                        "pytorch_model.bin",
                        "model.safetensors",
                    ],
                )
            )
        from .common import sha256_file

        provenance_path = local_path / "family_lab_provenance.json"
        provenance = (
            json.loads(provenance_path.read_text())
            if provenance_path.exists()
            else {"source": source, "revision": revision, "upstream_commit": UPSTREAM_COMMIT}
        )
        weights = local_path / "pytorch_model.bin"
        if not weights.exists():
            weights = local_path / "model.safetensors"
        actual_hash = sha256_file(weights)
        if (
            provenance.get("source") == "airkingbd/dplm_150m"
            and provenance.get("artifact_role", "pretrained_source") == "pretrained_source"
        ):
            if (
                provenance.get("revision") != DPLM_150M_REVISION
                or actual_hash != DPLM_150M_SHA256
                or weights.stat().st_size != DPLM_150M_SIZE
            ):
                raise ValueError(
                    "Original DPLM-150M source weights failed the pinned official revision/size/SHA-256 check"
                )
        provenance["loaded_weights_sha256"] = actual_hash
        provenance["loaded_config_sha256"] = sha256_file(local_path / "config.json")
        net, loading = EsmForDPLM.from_pretrained(local_path, output_loading_info=True)
        if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
            raise RuntimeError(f"DPLM checkpoint did not load exactly: {loading}")
        if getattr(net.config, "dplm_type", None) != "dplm_esm":
            raise ValueError(
                "Checkpoint must declare dplm_type=dplm_esm; plain ESM2 is not accepted"
            )
        tokenizer = EsmTokenizer.from_pretrained(local_path)
        provenance["tiny_fixture"] = bool(getattr(net.config, "family_lab_tiny_fixture", False))
        if provenance["tiny_fixture"] and not model_cfg.get("allow_tiny_fixture", False):
            raise ValueError(
                "Tiny test weights require model.allow_tiny_fixture=true; they are not DPLM-150M"
            )
    if bool(getattr(net.config, "family_lab_tiny_fixture", False)) and not model_cfg.get(
        "allow_tiny_fixture", False
    ):
        raise ValueError(
            "Tiny test weights require model.allow_tiny_fixture=true; they are not DPLM-150M"
        )
    model = DiffusionProteinLanguageModel(
        cfg={
            "num_diffusion_timesteps": 500,
            "rdm_couple": False,
            "gradient_ckpt": bool(
                training and config.get("train", {}).get("gradient_checkpointing", True)
            ),
        },
        net=net,
    )
    allowed = torch.zeros(net.config.vocab_size, dtype=torch.bool)
    allowed[[tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS]] = True
    model.register_buffer("canonical_token_mask", allowed, persistent=False)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.train(training)
    for parameter in model.parameters():
        parameter.requires_grad_(training)
    provenance["parameter_count"] = sum(p.numel() for p in model.parameters())
    provenance["device"] = str(device)
    provenance["precision"] = precision
    return Backend(model, tokenizer, device, precision, provenance)


def safe_denoising_loss(logits, target, mask, weights):
    """Official zero-smoothing weighted masked CE, with a zero-mask-safe reduction."""
    torch = require_torch()
    ce = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape_as(target)
    count = mask.sum()
    numerator = (ce * mask * weights).sum()
    return numerator / count.clamp_min(1), numerator.detach(), int(count)


def rng_state():
    import random
    import numpy as np

    torch = require_torch()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    import random
    import numpy as np

    torch = require_torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_backend(backend: Backend, path, probe_sequences):
    """Export with a verified exact CPU FP32 inference round trip."""
    torch = require_torch()
    from .common import write_json

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    backend.model.net.save_pretrained(path, safe_serialization=False)
    backend.tokenizer.save_pretrained(path)
    write_json(
        path / "family_lab_provenance.json",
        {**backend.provenance, "artifact_role": "adapted_export"},
    )
    cfg = {
        "model": {
            "source": str(path),
            "allow_tiny_fixture": backend.provenance.get("tiny_fixture", False),
        },
        "train": {"device": "cpu", "mixed_precision": "fp32"},
    }
    restored = load_backend(cfg)
    original_hash = tensor_digest(backend.model.state_dict())
    restored_hash = tensor_digest(restored.model.state_dict())
    if original_hash != restored_hash:
        raise RuntimeError("Export round trip changed model tensors")
    # Compare saved/reloaded networks on the same CPU backend, avoiding CUDA/CPU tolerances.
    saved_mode = backend.model.training
    original_device = backend.device
    backend.model.to("cpu").eval()
    ids = restored.encode(probe_sequences)
    with torch.no_grad():
        expected = backend.model(ids)
        actual = restored.model(ids)
    max_error = float((expected - actual).abs().max())
    backend.model.to(original_device).train(saved_mode)
    if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"Export round-trip logits differ: {max_error}")
    result = {
        "weights_sha256": original_hash,
        "max_logit_error_cpu_fp32": max_error,
        "passed": True,
    }
    write_json(path / "roundtrip.json", result)
    return result


def create_tiny_fixture(path):
    """Offline test fixture using the exact vendored network, never a mock model."""
    require_torch()
    from transformers import EsmConfig, EsmTokenizer
    from ._vendor.esm import EsmForDPLM

    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(ESM_VOCAB) + "\n", encoding="utf-8")
    tokenizer = EsmTokenizer(vocab_file=str(path / "vocab.txt"))
    tokenizer.save_pretrained(path)
    cfg = EsmConfig(
        vocab_size=33,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        pad_token_id=1,
        mask_token_id=32,
        position_embedding_type="rotary",
        max_position_embeddings=1026,
        token_dropout=True,
        hidden_dropout_prob=0.1,
    )
    cfg.dplm_type = "dplm_esm"
    cfg.family_lab_tiny_fixture = True
    cfg._name_or_path = str(path)
    net = EsmForDPLM(cfg)
    net.save_pretrained(path, safe_serialization=False)
    return path
