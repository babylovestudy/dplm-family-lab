"""Seeded DPLM generation with a persisted shared empirical length schedule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

from .backend import AMINO_ACIDS, load_backend, require_torch


def _batch_seed(run_seed, batch_index):
    """Domain-separated seeds prevent adjacent runs from reusing the same streams."""
    digest = hashlib.sha256(f"dplm-family-generation-v1:{run_seed}:{batch_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def length_schedule(config: dict):
    from .common import read_fasta, sha256_file, write_json

    gc = config["generate"]
    total, seed = int(gc.get("total_sequences", 256)), int(gc.get("seed", 101))
    if total < 1:
        raise ValueError("total_sequences must be positive")
    records = read_fasta(gc["lengths_source"])
    lengths = [len(record.sequence) for record in records]
    if (
        not lengths
        or min(lengths) < 1
        or max(lengths) > config.get("data", {}).get("max_length", 256)
    ):
        raise ValueError("Length source is empty or outside the configured maximum")
    signature = {
        "seed": seed,
        "total_sequences": total,
        "source_sha256": sha256_file(gc["lengths_source"]),
    }
    path = Path(
        gc.get("lengths_file")
        or gc.get("length_schedule_path", f"outputs/length_schedules/seed_{seed}_n{total}.json")
    )
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if any(result.get(key) != value for key, value in signature.items()):
            raise ValueError(f"Shared length schedule signature differs: {path}")
        if len(result.get("lengths", [])) != total:
            raise ValueError("Shared length schedule has the wrong number of entries")
    else:
        generator = random.Random(seed)
        result = {
            **signature,
            "lengths": [generator.choice(lengths) for _ in range(total)],
            "policy": "sample training lengths with replacement; shared across baseline/adapted arms",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, result)
    return result, path


def _initial_tokens(backend, lengths, segments):
    sequences = []
    for length in lengths:
        sequence = ["<mask>"] * length
        last_end = -1
        for segment in sorted(segments, key=lambda segment: int(segment["start"])):
            start = int(segment["start"])
            motif = segment["sequence"]
            end = start + len(motif)
            if not motif or set(motif) - set(AMINO_ACIDS):
                raise ValueError("Fixed segments must contain canonical amino acids")
            if start <= last_end or start < 0 or end > length:
                raise ValueError("Fixed segments overlap or lie outside a generated sequence")
            sequence[start:end] = list(motif)
            last_end = end - 1
        sequences.append("".join(sequence))
    return backend.encode(sequences)


def generate(config: dict, model_path=None, output_dir=None) -> dict:
    from .common import (
        SequenceRecord,
        environment_info,
        seed_everything,
        sha256_file,
        write_fasta,
        write_json,
    )

    torch = require_torch()
    gc = config["generate"]
    output = Path(output_dir or gc["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing generation: {output}")
    batch_size = int(gc.get("batch_size", 1))
    steps = int(gc.get("max_iter", 100))
    if batch_size < 1 or steps < 1:
        raise ValueError("Generation batch_size and max_iter must be positive")
    if gc.get("sampling_strategy", "gumbel_argmax") != "gumbel_argmax":
        raise ValueError(
            "Version 1 fixes sampling_strategy=gumbel_argmax for controlled comparisons"
        )
    if "temperature" in gc:
        raise ValueError(
            "The official gumbel_argmax sampler ignores temperature; omit this setting"
        )
    schedule, schedule_path = length_schedule(config)
    seed_everything(int(gc.get("seed", 101)))
    backend = load_backend(config, model_path)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "length_schedule.json", schedule)
    manifest = {
        "status": "running",
        "model": str(model_path or config["model"]["source"]),
        "provenance": backend.provenance,
        "seed": schedule["seed"],
        "batch_size": batch_size,
        "max_iter": steps,
        "sampling_strategy": "gumbel_argmax",
        "output_alphabet": AMINO_ACIDS,
        "length_schedule_path": str(schedule_path.resolve()),
        "length_schedule_sha256": sha256_file(schedule_path),
        "environment": environment_info(),
        "batches": [],
    }
    write_json(output / "manifest.json", manifest)
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    started = time.perf_counter()
    generated = []
    segments = gc.get("fixed_segments", [])
    try:
        for offset in range(0, len(schedule["lengths"]), batch_size):
            lengths = schedule["lengths"][offset : offset + batch_size]
            batch_seed = _batch_seed(schedule["seed"], offset // batch_size)
            seed_everything(batch_seed)
            tokens = _initial_tokens(backend, lengths, segments)
            fixed = tokens.ne(backend.model.mask_id)
            with torch.inference_mode(), backend.autocast():
                outputs = backend.model.generate(
                    input_tokens=tokens,
                    max_iter=steps,
                    sampling_strategy="gumbel_argmax",
                    partial_masks=fixed,
                )
            sequences = [
                "".join(sequence.split())
                for sequence in backend.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ]
            for index, (sequence, expected_length) in enumerate(
                zip(sequences, lengths), start=offset
            ):
                if len(sequence) != expected_length:
                    raise RuntimeError(
                        f"Generated length differs at index {index}: {len(sequence)} != {expected_length}"
                    )
                if any(
                    sequence[
                        int(segment["start"]) : int(segment["start"]) + len(segment["sequence"])
                    ]
                    != segment["sequence"]
                    for segment in segments
                ):
                    raise RuntimeError("DPLM generation changed a protected fixed segment")
                generated.append(
                    SequenceRecord(
                        f"seq_{index:04d}",
                        sequence,
                        f"length={expected_length} batch_seed={batch_seed}",
                    )
                )
            manifest["batches"].append(
                {"offset": offset, "count": len(lengths), "seed": batch_seed}
            )
        temporary = output / "sequences.fasta.tmp"
        write_fasta(generated, temporary)
        temporary.replace(output / "sequences.fasta")
        manifest.update(
            {
                "status": "complete",
                "sequences": len(generated),
                "invalid_amino_acid_sequences": sum(
                    bool(set(record.sequence) - set(AMINO_ACIDS)) for record in generated
                ),
                "fasta": str(output / "sequences.fasta"),
                "fasta_sha256": sha256_file(output / "sequences.fasta"),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(backend.device)
                if backend.device.type == "cuda"
                else None,
            }
        )
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        write_json(output / "manifest.json", manifest)
        raise
    write_json(output / "manifest.json", manifest)
    return manifest
