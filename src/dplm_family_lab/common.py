"""Small, dependency-light utilities shared by data and model workflows."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SequenceRecord:
    id: str
    sequence: str
    description: str = ""


def read_fasta(path: str | Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    current_id = None
    description = ""
    sequence: list[str] = []
    seen: set[str] = set()

    def finish():
        if current_id is None:
            return
        value = "".join(sequence).upper()
        if not value:
            raise ValueError(f"Empty sequence for FASTA record {current_id!r}")
        if current_id in seen:
            raise ValueError(f"Duplicate FASTA identifier: {current_id}")
        seen.add(current_id)
        records.append(SequenceRecord(current_id, value, description))

    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                finish()
                header = line[1:].strip().split(maxsplit=1)
                if not header:
                    raise ValueError(f"Empty FASTA header at line {number}")
                current_id = header[0]
                description = header[1] if len(header) == 2 else ""
                sequence = []
            else:
                if current_id is None:
                    raise ValueError(f"Sequence before first FASTA header at line {number}")
                sequence.append("".join(line.split()))
    finish()
    return records


def write_fasta(records: Iterable[SequenceRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            if not record.id or any(c.isspace() for c in record.id):
                raise ValueError("FASTA identifiers must be nonempty and contain no whitespace")
            header = record.id + (f" {record.description}" if record.description else "")
            handle.write(f">{header}\n")
            for start in range(0, len(record.sequence), 80):
                handle.write(record.sequence[start : start + 80] + "\n")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(obj, handle, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_info() -> dict:
    versions = {}
    for package in ("dplm-family-lab", "numpy", "biopython", "torch", "transformers", "pyhmmer"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_root = Path(__file__).resolve().parent
    source_hashes = {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in sorted(source_root.rglob("*.py"))
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "project_source_sha256": source_hashes,
    }
