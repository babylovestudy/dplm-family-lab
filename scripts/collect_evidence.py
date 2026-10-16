"""Collect small, auditable scientific outputs; never publish model/cache files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from dplm_family_lab.common import sha256_file, utc_now, write_json


def collect(mode: str, destination: Path) -> dict:
    output_root = Path("outputs") / mode
    if not (output_root / "experiment_status.json").is_file():
        raise FileNotFoundError(f"Experiment status is missing: {output_root}")
    allowed = {
        "experiment_status.json", "manifest.json", "length_schedule.json", "sequences.fasta",
        "metrics.json", "per_sequence.csv", "pairwise_diversity.csv",
        "denoising_baseline.json", "denoising_adapted.json",
    }
    sources = [p for p in output_root.rglob("*") if p.is_file() and p.name in allowed]
    for manifest in list(sources):
        if manifest.name != "manifest.json":
            continue
        value = json.loads(manifest.read_text())["length_schedule_path"].replace("\\", "/")
        if "/outputs/" in value:
            value = "outputs/" + value.split("/outputs/", 1)[1]
        schedule = Path(value)
        if not schedule.resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError(f"Shared schedule must be within this project: {schedule}")
        if not schedule.is_file():
            raise FileNotFoundError(f"Shared schedule was not transferred: {schedule}")
        sources.append(schedule)
    seeds = [42] if mode == "pilot" else [42, 43, 44]
    for seed in seeds:
        suffix = f"pilot_seed_{seed}" if mode == "pilot" else f"seed_{seed}"
        run = Path("runs/pf00194") / suffix
        for name in (
            "training_report.json", "config.json", "environment.json",
            "baseline_validation.json", "training.jsonl", "export_best/roundtrip.json",
            "export_best/export_identity.json", "export_best/family_lab_provenance.json",
        ):
            if (run / name).is_file():
                sources.append(run / name)
    files = []
    for source in sorted(set(sources)):
        target = destination / source
        digest = sha256_file(source)
        if target.exists() and sha256_file(target) != digest:
            raise FileExistsError(f"Refusing to replace different published evidence: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files.append({"path": source.as_posix(), "sha256": digest, "bytes": source.stat().st_size})
    status = json.loads((output_root / "experiment_status.json").read_text())
    index = {
        "mode": mode, "collected_at": utc_now(), "experiment_status": status["status"],
        "files": files,
        "scope": "Original small outputs and provenance; no checkpoints, SQLite caches or raw training FASTA.",
        "biological_validation": "not performed; family compatibility is a computational metric",
    }
    write_json(destination / "index.json", index)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "formal"], required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    report = collect(args.mode, Path(args.output_dir or f"results/{args.mode}"))
    print(json.dumps({"mode": args.mode, "status": report["experiment_status"], "files": len(report["files"])}))


if __name__ == "__main__":
    main()
