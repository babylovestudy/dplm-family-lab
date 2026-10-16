"""Restartable pilot/formal experiment orchestration, with separate namespaces."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .common import sha256_file, utc_now, write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _training_run(config):
    from .training import train

    output = Path(config["train"]["output_dir"])
    snapshot = output / "config.json"
    report = output / "training_report.json"
    if snapshot.exists() and _read(snapshot) != config:
        raise ValueError(f"Existing training configuration changed: {snapshot}")
    if report.exists():
        previous = _read(report)
        if previous.get("status") == "complete":
            for name, digest in previous["data_sha256"].items():
                if sha256_file(config["train"][name]) != digest:
                    raise ValueError(f"Training input changed for {output}")
            if not (output / "export_best" / "roundtrip.json").exists():
                raise ValueError(f"Completed training lacks verified export: {output}")
            return previous
    last = output / "last.pt"
    return train(config, resume=str(last) if last.exists() else None)


def _generation_run(config, output, model_path=None):
    from .sampling import generate, length_schedule

    output = Path(output)
    manifest_path = output / "manifest.json"
    schedule, schedule_path = length_schedule(config)
    if manifest_path.exists():
        manifest = _read(manifest_path)
        expected_model = str(model_path or config["model"]["source"])
        if manifest.get("status") != "complete":
            raise ValueError(
                f"Generation was interrupted: {output}; retain its failure evidence and select a fresh output namespace"
            )
        if (
            manifest["model"] != expected_model
            or manifest["length_schedule_sha256"] != sha256_file(schedule_path)
            or manifest["max_iter"] != config["generate"]["max_iter"]
            or manifest["batch_size"] != config["generate"]["batch_size"]
            or manifest["seed"] != schedule["seed"]
            or manifest["sequences"] != len(schedule["lengths"])
            or sha256_file(output / "sequences.fasta") != manifest["fasta_sha256"]
        ):
            raise ValueError(f"Existing generation differs from this protocol: {output}")
        return manifest
    return generate(config, model_path=model_path, output_dir=str(output))


def run_experiment(config: dict, pilot: bool = False) -> dict:
    """Pilot: 100 steps + 32 paired sequences. Formal: 3x1000 + 3x256 per arm.

    Each durable training/generation result is checked before being reused.
    The caller can rerun this command after a process exits at a saved training
    boundary. Partial generation outputs remain immutable failure evidence.
    """
    from .data import audit
    from .evaluation import evaluate
    from .training import evaluate_denoising

    mode = "pilot" if pilot else "formal"
    output = Path("outputs") / mode
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "experiment_status.json"
    status = {"mode": mode, "status": "running", "started_at": utc_now(), "completed": []}
    write_json(status_path, status)

    def record(stage, result):
        status["completed"].append({"stage": stage, "result": result, "time": utc_now()})
        write_json(status_path, status)

    try:
        record("data_audit", audit(config))
        training_seeds = [42] if pilot else [42, 43, 44]
        generation_seeds = [101] if pilot else [101, 102, 103]
        models = [("baseline", None)]
        for seed in training_seeds:
            current = copy.deepcopy(config)
            parent = Path(config["train"]["output_dir"]).parent
            suffix = f"pilot_seed_{seed}" if pilot else f"seed_{seed}"
            current["train"]["seed"] = seed
            current["train"]["output_dir"] = str(parent / suffix)
            if pilot:
                current["train"]["max_steps"] = 100
            result = _training_run(current)
            record(f"train_{seed}", result)
            models.append((f"train_seed_{seed}", str(parent / suffix / "export_best")))

        for arm, model_path in models:
            current = copy.deepcopy(config)
            current["evaluation"]["output_dir"] = str(output / arm / "test_denoising")
            denoise_file = Path(current["evaluation"]["output_dir"]) / (
                "denoising_adapted.json" if model_path else "denoising_baseline.json"
            )
            if denoise_file.exists():
                result = _read(denoise_file)
                if (
                    result["source_sha256"] != sha256_file(config["train"]["test_fasta"])
                    or result["model"] != str(model_path or config["model"]["source"])
                    or result["noise_seed"] != config["train"]["validation_seed"] + 1
                ):
                    raise ValueError(f"Existing test evaluation differs from protocol: {denoise_file}")
            else:
                result = evaluate_denoising(current, model_path=model_path)
            record(f"test_denoising_{arm}", result)
            for seed in generation_seeds:
                current = copy.deepcopy(config)
                current["generate"]["seed"] = seed
                current["generate"]["total_sequences"] = 32 if pilot else 256
                generation_dir = output / arm / f"seed_{seed}"
                manifest = _generation_run(current, generation_dir, model_path=model_path)
                record(f"generate_{arm}_{seed}", manifest)
                evaluation_dir = generation_dir / "evaluation"
                metrics = evaluate(
                    current,
                    fasta_path=generation_dir / "sequences.fasta",
                    output_dir=evaluation_dir,
                )
                record(f"evaluate_{arm}_{seed}", metrics)
        status["status"] = "complete"
        status["finished_at"] = utc_now()
        write_json(status_path, status)
    except Exception as exc:
        status.update({"status": "failed", "error": str(exc), "failed_at": utc_now()})
        write_json(status_path, status)
        raise
    return status
