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
    diagnostics = diagnostic_sources = 0
    for directory in sorted((ROOT / "results/diagnostics").glob("*")):
        if not directory.is_dir() or not (directory / "diagnostics.json").is_file():
            continue
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"Diagnostics manifest is missing: {directory}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("mode") != "formal":
            raise ValueError(f"Invalid diagnostics manifest: {manifest}")
        verify(ROOT, {"path": "scripts/diagnose_generations.py",
                      "sha256": payload["analysis_script_sha256"]})
        for item in payload["files"]:
            verify(directory, item)
        required = {"diagnostics.json", "group_checks.csv", "arm_diagnostics.csv",
                    "sequence_diagnostics.csv", "examples.csv", "cross_generation_seed_duplicates.csv",
                    "all_adapted_pooled_duplicates.csv", "reported_pair_verification.json", "README.md"}
        paths = [item["path"] for item in payload["files"]]
        if len(paths) != len(set(paths)) or set(paths) != required:
            raise ValueError(f"Incomplete or repeated diagnostics outputs: {directory}")
        report = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
        if not report.get("complete") or report.get("completed_groups") != 12:
            raise ValueError(f"Formal diagnostics are incomplete: {directory}")
        if report["analysis_script_sha256"] != payload["analysis_script_sha256"]:
            raise ValueError(f"Diagnostics script hashes disagree: {directory}")
        if payload.get("source_root") != "results/formal":
            raise ValueError(f"Unexpected diagnostics source bundle: {directory}")
        sources = {item["path"]: item["sha256"] for item in payload["sources"]}
        if (len(sources) != len(payload["sources"]) or len(sources) != 36
                or sources != report["source_files_sha256"]):
            raise ValueError(f"Diagnostics source bindings disagree: {directory}")
        for item in payload["sources"]:
            verify(ROOT / "results/formal", item)
        paired = json.loads((directory / "reported_pair_verification.json").read_text(encoding="utf-8"))
        if (paired["source_files_sha256"] != sources
                or paired["analysis_script_sha256"] != payload["analysis_script_sha256"]
                or paired["published_default_diagnostics_sha256"] != hashlib.sha256(
                    (directory / "diagnostics.json").read_bytes()).hexdigest()
                or paired["checks"]["global_nearest_search_reexecuted"] is not False):
            raise ValueError(f"Reported-pair verification bindings disagree: {directory}")
        diagnostics += len(paths)
        diagnostic_sources += len(sources)
    counts["diagnostics"] = diagnostics
    counts["diagnostic_sources"] = diagnostic_sources
    summary_directory = ROOT / "results/experiment-summary"
    if (summary_directory / "summary.json").is_file():
        manifest = summary_directory / "manifest.json"
        if not manifest.is_file():
            raise ValueError("Experiment-summary manifest is missing")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not payload.get("files"):
            raise ValueError("Invalid experiment-summary manifest")
        paths = [item["path"] for item in payload["files"]]
        if len(paths) != len(set(paths)) or not {"summary.json", "summary.csv", "report.md"} <= set(paths):
            raise ValueError("Incomplete or repeated experiment-summary outputs")
        for item in payload["files"]:
            verify(summary_directory, item)
        summary = json.loads((summary_directory / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("formal_status") != "complete" or summary.get("formal_training_runs") != 3
                or summary.get("missing_expected_stages") != []
                or ("missing" in summary and summary["missing"] != [])
                or summary.get("formal_validation_issues") not in ({}, [])):
            raise ValueError("Formal experiment-summary completion checks failed")
        bindings = payload["report_row_sources"]
        original_paths = [item["original_path"] for item in bindings]
        mapped_paths = [item["path"] for item in bindings]
        rows = [(row["source_json"], row["source_sha256"]) for row in summary["rows"]]
        bound = [(item["original_path"], item["sha256"]) for item in bindings]
        if (len(original_paths) != len(set(original_paths)) or len(mapped_paths) != len(set(mapped_paths))
                or len(rows) != len(set(rows)) or len(rows) != len(bound) or set(rows) != set(bound)):
            raise ValueError("Experiment-summary row sources have omissions, repetitions or hash mismatches")
        for item in bindings:
            verify(ROOT, item)
        counts["experiment_summary"] = len(paths)
        counts["experiment_summary_row_sources"] = len(bindings)
    if not counts.get("pilot"):
        raise ValueError("Published pilot evidence is missing")
    print(json.dumps({"status": "passed", "verified_files": counts}, indent=2))


if __name__ == "__main__":
    main()
