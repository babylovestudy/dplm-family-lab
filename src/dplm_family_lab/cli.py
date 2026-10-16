"""Command-line entry points. Data-only commands do not import PyTorch."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from .common import (
    SequenceRecord,
    environment_info,
    read_fasta,
    sha256_file,
    utc_now,
    write_fasta,
    write_json,
)


def load_config(path: str | Path) -> dict:
    import yaml

    source = Path(path).resolve()
    with source.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    root = source.parent.parent if source.parent.name == "configs" else source.parent
    os.chdir(root)
    return config


def doctor(output: str | None = None) -> dict:
    result = environment_info()
    result["time_utc"] = utc_now()
    result["executables"] = {
        name: shutil.which(name) for name in ("hmmsearch", "mmseqs", "git", "ssh")
    }
    result["python_supported"] = (3, 10) <= sys.version_info[:2] < (3, 13)
    result["gpu"] = {"available": False, "status": "training extra is not installed"}
    if importlib.util.find_spec("torch") is not None:
        import torch

        result["gpu"] = {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "devices": [],
        }
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(index)
                result["gpu"]["devices"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "free_bytes": free,
                        "total_bytes": total,
                        "bf16_supported": torch.cuda.is_bf16_supported(),
                    }
                )
    if output:
        write_json(output, result)
    return result


def demo(output_dir: str) -> dict:
    """An explicitly synthetic, offline FASTA quality-control learning exercise."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records = [
        SequenceRecord("toy_1", "ACDEFGHIKLMNPQRSTVWY", "synthetic teaching example"),
        SequenceRecord("toy_2", "ACDEFGHIKLMNPQRSTVWF"),
        SequenceRecord("toy_duplicate", "ACDEFGHIKLMNPQRSTVWY"),
        SequenceRecord("toy_invalid", "ACDEFGHIKX"),
    ]
    write_fasta(records, output / "input.fasta")
    accepted = []
    seen = set()
    rows = []
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    for record in read_fasta(output / "input.fasta"):
        reason = "accepted"
        if not set(record.sequence) <= alphabet:
            reason = "nonstandard_amino_acid"
        elif record.sequence in seen:
            reason = "duplicate_sequence"
        else:
            accepted.append(record)
            seen.add(record.sequence)
        rows.append({"id": record.id, "length": len(record.sequence), "status": reason})
    write_fasta(accepted, output / "clean.fasta")
    with (output / "qc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "length", "status"])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "kind": "synthetic_learning_demo",
        "input_records": len(records),
        "accepted_records": len(accepted),
        "input_sha256": sha256_file(output / "input.fasta"),
        "hmm_evaluation": "not run",
        "model_training": "not run",
        "biological_claims": "none; these sequences are synthetic",
    }
    write_json(output / "report.json", report)
    return report


def run_seeds(config: dict, pilot: bool = False) -> dict:
    from .training import train

    parent = Path(config["train"]["output_dir"]).parent
    seeds = [42] if pilot else [42, 43, 44]
    runs = []
    for seed in seeds:
        current = copy.deepcopy(config)
        current["train"]["seed"] = seed
        suffix = f"pilot_seed_{seed}" if pilot else f"seed_{seed}"
        current["train"]["output_dir"] = str(parent / suffix)
        if pilot:
            current["train"]["max_steps"] = 100
        runs.append(train(current))
        write_json(parent / ("pilot_runs.json" if pilot else "training_runs.json"), runs)
    return {"runs": runs, "pilot": pilot}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPLM protein family adaptation laboratory")
    parser.add_argument("--debug", action="store_true", help="show full exception trace")
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("doctor", help="inspect local dependencies and GPU")
    diagnostic.add_argument("--output")
    example = commands.add_parser("demo", help="run an offline synthetic FASTA exercise")
    example.add_argument("--output-dir", default="artifacts/demo")
    for name in (
        "fetch", "prepare-data", "audit-data", "train", "generate", "evaluate",
        "denoise-eval", "run-seeds", "run-experiment", "report",
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/pf00194.yaml")
        if name == "prepare-data":
            command.add_argument("--offline", action="store_true")
        elif name == "train":
            command.add_argument("--resume")
            command.add_argument("--max-steps", type=int)
            command.add_argument("--seed", type=int)
        elif name == "generate":
            command.add_argument("--model-path")
            command.add_argument("--output-dir")
            command.add_argument("--seed", type=int)
            command.add_argument("--count", type=int)
            command.add_argument("--lengths-file")
        elif name == "evaluate":
            command.add_argument("--fasta", required=True)
            command.add_argument("--output-dir")
        elif name == "denoise-eval":
            command.add_argument("--model-path")
        elif name in {"run-seeds", "run-experiment"}:
            command.add_argument("--pilot", action="store_true")
        elif name == "report":
            command.add_argument("--output-dir", default="artifacts/report")
    return parser


def _console_result(command: str, result: dict) -> dict:
    if command == "report":
        return {
            "status": result["status"],
            "formal_status": result["formal_status"],
            "recorded_rows": result.get("recorded_rows", 0),
            "missing_stage_count": len(result.get("missing_expected_stages", [])),
            "artifacts": result["artifacts"],
        }
    if command == "run-experiment":
        return {
            "mode": result["mode"], "status": result["status"],
            "completed_stages": len(result["completed"]),
            "journal": f"outputs/{result['mode']}/experiment_status.json",
        }
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args.output)
        elif args.command == "demo":
            result = demo(args.output_dir)
        else:
            config = load_config(args.config)
            if args.command == "fetch":
                from .data import fetch

                result = fetch(config)
            elif args.command == "prepare-data":
                from .data import prepare

                result = prepare(config, download=not args.offline)
            elif args.command == "audit-data":
                from .data import audit

                result = audit(config)
            elif args.command == "train":
                from .training import train

                if args.max_steps is not None:
                    config["train"]["max_steps"] = args.max_steps
                if args.seed is not None:
                    config["train"]["seed"] = args.seed
                    config["train"]["output_dir"] = str(
                        Path(config["train"]["output_dir"]).parent / f"seed_{args.seed}"
                    )
                result = train(config, resume=args.resume)
            elif args.command == "generate":
                from .sampling import generate

                if args.seed is not None:
                    config["generate"]["seed"] = args.seed
                if args.count is not None:
                    config["generate"]["total_sequences"] = args.count
                if args.lengths_file:
                    config["generate"]["lengths_file"] = args.lengths_file
                result = generate(config, model_path=args.model_path, output_dir=args.output_dir)
            elif args.command == "evaluate":
                from .evaluation import evaluate

                result = evaluate(config, fasta_path=args.fasta, output_dir=args.output_dir)
            elif args.command == "denoise-eval":
                from .training import evaluate_denoising

                result = evaluate_denoising(config, model_path=args.model_path)
            elif args.command == "run-seeds":
                result = run_seeds(config, pilot=args.pilot)
            elif args.command == "run-experiment":
                from .experiment import run_experiment

                result = run_experiment(config, pilot=args.pilot)
            else:
                from .reporting import build_report

                result = build_report(config, output_dir=args.output_dir)
        print(json.dumps(_console_result(args.command, result), indent=2, ensure_ascii=False, default=str))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, ImportError) as error:
        if args.debug:
            raise
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
