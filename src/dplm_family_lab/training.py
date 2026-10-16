"""Single-GPU family adaptation with exact optimizer-boundary resume."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

from .backend import (
    AMINO_ACIDS,
    export_backend,
    load_backend,
    require_torch,
    restore_rng,
    rng_state,
    safe_denoising_loss,
    tensor_digest,
)


def _atomic_save(value, path):
    torch = require_torch()
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _records(path, config):
    from .common import read_fasta

    records = read_fasta(path)
    if not records:
        raise ValueError(f"Empty sequence split: {path}")
    maximum = config.get("data", {}).get("max_length", 256)
    for record in records:
        if not record.sequence or set(record.sequence) - set(AMINO_ACIDS):
            raise ValueError(f"Noncanonical/empty sequence: {record.id}")
        if len(record.sequence) > maximum:
            raise ValueError(
                f"Sequence {record.id} exceeds {maximum}; filter explicitly, do not silently crop"
            )
    return records


def _noise_bank(backend, records, seed):
    torch = require_torch()
    before = rng_state()
    from .common import seed_everything

    seed_everything(seed)
    bank = []
    try:
        for record in records:
            target = backend.encode([record.sequence])
            timestep = torch.randint(1, 501, (1,), device=backend.device)
            noisy = backend.model.q_sample(
                target, timestep, backend.model.get_non_special_symbol_mask(target)
            )
            bank.append(
                {
                    "id": record.id,
                    "input_ids": noisy["x_t"].cpu(),
                    "targets": target.cpu(),
                    "mask": noisy["mask_mask"].cpu(),
                    "weight": ((501 - timestep).float() / 500).cpu()[:, None],
                }
            )
    finally:
        restore_rng(before)
    return bank


def _evaluate_bank(backend, bank):
    torch = require_torch()
    before = rng_state()
    mode = backend.model.training
    backend.model.eval()
    numerator, count, zero = 0.0, 0, 0
    try:
        with torch.no_grad():
            for entry in bank:
                ids = entry["input_ids"].to(backend.device)
                with backend.autocast():
                    logits = backend.model(ids)
                    _, total, masked = safe_denoising_loss(
                        logits,
                        entry["targets"].to(backend.device),
                        entry["mask"].to(backend.device),
                        entry["weight"].to(backend.device),
                    )
                numerator += float(total)
                count += masked
                zero += masked == 0
    finally:
        backend.model.train(mode)
        restore_rng(before)
    if count == 0:
        raise ValueError(
            "Validation noise bank contains no masked residues; use more sequences or another validation seed"
        )
    return {
        "weighted_masked_denoising_loss": numerator / count,
        "masked_residues": count,
        "sequences": len(bank),
        "zero_mask_sequences": zero,
        "metric_note": "Weighted masked denoising CE, not exact sequence likelihood/perplexity.",
    }


def _lr_factor(step, warmup, maximum):
    if warmup and step < warmup:
        return (step + 1) / warmup
    return max(0.5, 1.0 - 0.5 * max(0, step - warmup) / max(1, maximum - warmup))


def _signature(config, hashes):
    train_cfg = config["train"]
    selected = [
        "seed",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_steps",
        "gradient_checkpointing",
        "mixed_precision",
        "warmup_steps",
        "weight_decay",
        "grad_clip",
        "validation_seed",
    ]
    return {
        "model": config["model"],
        "train": {key: train_cfg.get(key) for key in selected},
        "data_sha256": hashes,
    }


def _next_batch(records, batch_size, sampler, seed):
    indices = []
    while len(indices) < batch_size:
        order = list(range(len(records)))
        random.Random(seed + sampler["epoch"]).shuffle(order)
        remaining = batch_size - len(indices)
        selected = order[sampler["position"] : sampler["position"] + remaining]
        indices.extend(selected)
        sampler["position"] += len(selected)
        if sampler["position"] == len(order):
            sampler["epoch"] += 1
            sampler["position"] = 0
    return [records[index].sequence for index in indices]


def train(config: dict, resume=None) -> dict:
    from .common import environment_info, seed_everything, sha256_file, write_json

    torch = require_torch()
    tc = config["train"]
    output = Path(tc["output_dir"]).resolve()
    maximum = int(tc["max_steps"])
    accumulation = int(tc.get("gradient_accumulation_steps", 16))
    batch_size = int(tc.get("micro_batch_size", 1))
    if (
        min(
            maximum,
            accumulation,
            batch_size,
            int(tc.get("eval_every", 100)),
            int(tc.get("checkpoint_every", 100)),
        )
        < 1
    ):
        raise ValueError("Step, batch and interval settings must be positive")
    resume_path = Path(resume).resolve() if resume else None
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty run: {output}; use --resume or a new output_dir"
        )
    output.mkdir(parents=True, exist_ok=True)
    train_records = _records(tc["train_fasta"], config)
    validation_records = _records(tc["val_fasta"], config)
    hashes = {name: sha256_file(tc[name]) for name in ["train_fasta", "val_fasta"]}
    signature = _signature(config, hashes)
    seed_everything(int(tc.get("seed", 42)))
    backend = load_backend(config, resume_path, training=True)
    optimizer = torch.optim.AdamW(
        backend.model.parameters(),
        lr=float(tc["learning_rate"]),
        weight_decay=float(tc.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_factor(step, int(tc.get("warmup_steps", 50)), maximum)
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=backend.device.type == "cuda" and backend.precision == "fp16"
    )
    step, sampler, best_loss = 0, {"epoch": 0, "position": 0}, math.inf
    probe_names = [
        name
        for name, _ in backend.model.named_parameters()
        if "encoder.layer" in name and "query.weight" in name
    ]
    probe_names = list(dict.fromkeys([probe_names[0], probe_names[-1]]))
    initial_probe = {
        name: dict(backend.model.named_parameters())[name].detach().cpu().clone()
        for name in probe_names
    }
    initial_hash = tensor_digest(backend.model.state_dict())
    gradient_proof = {name: 0.0 for name in probe_names}
    noise_path = output / "validation_noise.pt"
    if resume_path:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Resume model/data/training signature differs from checkpoint")
        if state["output_dir"] != str(output):
            raise ValueError(
                "Resume must use the original output_dir, preserving its noise bank and provenance"
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, sampler, best_loss = state["step"], state["sampler"], state["best_loss"]
        initial_probe, initial_hash = state["initial_probe"], state["initial_hash"]
        gradient_proof = state["gradient_proof"]
        if not noise_path.exists() or sha256_file(noise_path) != state["validation_noise_sha256"]:
            raise ValueError("Validation noise bank is missing or changed")
        restore_rng(state["rng"])
        noise = torch.load(noise_path, map_location="cpu", weights_only=False)
    else:
        noise = _noise_bank(backend, validation_records, int(tc.get("validation_seed", 9173)))
        _atomic_save(noise, noise_path)
        backend.tokenizer.save_pretrained(output / "tokenizer")
        write_json(output / "config.json", config)
        write_json(output / "environment.json", environment_info())
        write_json(output / "baseline_validation.json", _evaluate_bank(backend, noise))
    noise_hash = sha256_file(noise_path)
    start_time = time.perf_counter()
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    log_path = output / "training.jsonl"
    # A killed process can leave log rows newer than the last durable checkpoint.
    if resume_path and log_path.exists():
        rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= step))

    def checkpoint(path):
        _atomic_save(
            {
                "format": "dplm-family-training-v1",
                "model_state": backend.model.state_dict(),
                "net_config": backend.model.net.config.to_dict(),
                "tokenizer_dir": str(output / "tokenizer"),
                "provenance": backend.provenance,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "sampler": dict(sampler),
                "best_loss": best_loss,
                "rng": rng_state(),
                "signature": signature,
                "output_dir": str(output),
                "initial_probe": initial_probe,
                "initial_hash": initial_hash,
                "gradient_proof": gradient_proof,
                "validation_noise_sha256": noise_hash,
            },
            path,
        )

    stop = min(maximum, int(tc.get("stop_after_steps", maximum)))
    while step < stop:
        backend.model.train()
        optimizer.zero_grad(set_to_none=True)
        total_num, total_masked = 0.0, 0
        for _ in range(accumulation):
            sequences = _next_batch(train_records, batch_size, sampler, int(tc.get("seed", 42)))
            targets = backend.encode(sequences)
            with backend.autocast():
                logits, target, mask, weights = backend.model.compute_loss(
                    {"targets": targets}, weighting="linear"
                )
                loss, numerator, masked = safe_denoising_loss(logits, target, mask, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at optimizer step {step}")
            scaler.scale(loss / accumulation).backward()
            total_num += float(numerator)
            total_masked += masked
        scaler.unscale_(optimizer)
        for name in probe_names:
            grad = dict(backend.model.named_parameters())[name].grad
            if grad is not None:
                gradient_proof[name] = max(gradient_proof[name], float(grad.float().norm()))
        norm = torch.nn.utils.clip_grad_norm_(
            backend.model.parameters(), float(tc.get("grad_clip", 1.0))
        )
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm; checkpoint not advanced")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        record = {
            "step": step,
            "weighted_masked_denoising_loss": total_num / max(1, total_masked),
            "masked_residues": total_masked,
            "gradient_norm": float(norm),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "sampler": dict(sampler),
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        if step % int(tc.get("eval_every", 100)) == 0 or step == stop:
            evaluation = _evaluate_bank(backend, noise)
            record["validation"] = evaluation
            value = evaluation["weighted_masked_denoising_loss"]
            if value < best_loss:
                best_loss = value
                checkpoint(output / "best.pt")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if step % int(tc.get("checkpoint_every", 100)) == 0 or step == stop:
            checkpoint(output / "last.pt")
    elapsed = time.perf_counter() - start_time
    current = dict(backend.model.named_parameters())
    deltas = {
        name: float((current[name].detach().cpu() - initial_probe[name]).abs().max())
        for name in probe_names
    }
    proof = {
        "initial_model_sha256": initial_hash,
        "current_model_sha256": tensor_digest(backend.model.state_dict()),
        "probe_max_weight_changes": deltas,
        "probe_max_gradient_norms": gradient_proof,
        "backbone_updated": all(value > 0 for value in deltas.values())
        and all(value > 0 for value in gradient_proof.values()),
        "trainable_parameters": sum(
            p.numel() for p in backend.model.parameters() if p.requires_grad
        ),
        "gradient_checkpointing_enabled": bool(
            backend.model.net.esm.encoder.gradient_checkpointing
        ),
    }
    if step and not proof["backbone_updated"]:
        raise RuntimeError("Training did not demonstrate backbone gradients and weight changes")
    result = {
        "status": "complete" if step == maximum else "interrupted_at_requested_step",
        "step": step,
        "best_validation_loss": best_loss,
        "output_dir": str(output),
        "best_checkpoint": str(output / "best.pt"),
        "last_checkpoint": str(output / "last.pt"),
        "validation_noise_sha256": noise_hash,
        "data_sha256": hashes,
        "elapsed_seconds_this_session": elapsed,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(backend.device)
        if backend.device.type == "cuda"
        else None,
        "provenance": backend.provenance,
        "proof": proof,
    }
    write_json(
        output / "training_report.json",
        {**result, "status": "export_pending"} if step == maximum else result,
    )
    if step == maximum:
        best_backend = load_backend(config, output / "best.pt")
        result["export_roundtrip"] = export_backend(
            best_backend, output / "export_best", [validation_records[0].sequence]
        )
        write_json(output / "training_report.json", result)
    return result


def evaluate_denoising(config: dict, model_path=None) -> dict:
    from .common import sha256_file, write_json

    require_torch()
    backend = load_backend(config, model_path)
    tc = config["train"]
    records = _records(tc["test_fasta"], config)
    bank = _noise_bank(backend, records, int(tc.get("validation_seed", 9173)) + 1)
    report = _evaluate_bank(backend, bank)
    report.update(
        {
            "split": "test",
            "source_sha256": sha256_file(tc["test_fasta"]),
            "noise_seed": int(tc.get("validation_seed", 9173)) + 1,
            "model": str(model_path or config["model"]["source"]),
            "provenance": backend.provenance,
        }
    )
    destination = Path(config.get("evaluation", {}).get("output_dir", tc["output_dir"]))
    destination.mkdir(parents=True, exist_ok=True)
    name = "denoising_baseline.json" if model_path is None else "denoising_adapted.json"
    if (destination / name).exists():
        raise FileExistsError(f"Refusing to overwrite {destination / name}")
    write_json(destination / name, report)
    return report
