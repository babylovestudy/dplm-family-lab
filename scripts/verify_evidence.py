"""Verify published scientific files without data downloads or a GPU."""
from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify(base: Path, item: dict) -> None:
    path = (base / item["path"]).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-scope evidence: {item['path']}")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != item["sha256"]:
        raise ValueError(f"Evidence SHA-256 differs: {path.relative_to(ROOT)}")
    if "bytes" in item and len(contents) != item["bytes"]:
        raise ValueError(f"Evidence size differs: {path.relative_to(ROOT)}")


def main() -> None:
    counts = {}
    for mode in ("pilot", "formal", "engineering"):
        directory = ROOT / "results" / mode
        index = directory / "index.json"
        if not index.is_file():
            continue
        payload = json.loads(index.read_text(encoding="utf-8"))
        if (mode != "engineering" and payload.get("mode") != mode) or not payload.get("files"):
            raise ValueError(f"Invalid evidence index: {index}")
        for item in payload["files"]:
            verify(directory, item)
        counts[mode] = len(payload["files"])
    snapshot = ROOT / "results/dataset-snapshot"
    if (snapshot / "manifest.json").exists():
        payload = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        for item in payload["files"]:
            verify(snapshot, item)
        counts["dataset_snapshot"] = len(payload["files"])
    figures = 0
    for manifest in sorted((ROOT / "results/figures").glob("manifest_*.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        verify(ROOT, {"path": "scripts/plot_results.py", "sha256": payload["plot_script_sha256"]})
        for item in payload["outputs"]:
            verify(ROOT, item)
            figures += 1
        for item in payload["sources"]:
            verify(ROOT / "results" / payload["mode"], item)
    counts["figures"] = figures
    analyses = 0
    for manifest in sorted((ROOT / "results/analysis").glob("*/analysis.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        verify(ROOT, {"path": "scripts/analyze_results.py",
                      "sha256": payload["reproducibility"]["script_sha256"]})
        for item in payload["sources"]:
            verify(ROOT / "results" / payload["mode"], item)
        with (manifest.parent / "generation_comparison.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            actual = list(csv.DictReader(handle))
        expected = [{key: "" if value is None else str(value) for key, value in row.items()}
                    for row in payload["paired_comparisons"]]
        if actual != expected:
            raise ValueError(f"Analysis CSV and JSON disagree: {manifest.parent}")
        analyses += 1
    counts["analyses"] = analyses
    if not counts.get("pilot"):
        raise ValueError("Published pilot evidence is missing")
    print(json.dumps({"status": "passed", "verified_files": counts}, indent=2))


if __name__ == "__main__":
    main()
