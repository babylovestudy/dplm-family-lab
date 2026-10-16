"""Verify and restore the licensed, frozen raw inputs without network access.

Run with --prepare to recompute HMM filtering, exact clustering, splits and audit.
Different existing raw files or frozen datasets are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_split_sha256(manifest):
    """Hash scientific inputs/assignments; exclude wall-clock and report fields."""
    payload = {"signature": manifest["signature"],
               "sequences": sorted(manifest["sequences"], key=lambda record: record["id"])}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Snapshot path escapes its directory: {relative}")
    return path


def checked(path, expected):
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Snapshot checksum mismatch: {path.name}")


def restore(snapshot, destination, prepare=False, project_root=None, config_path=None):
    snapshot, destination = Path(snapshot).resolve(), Path(destination).resolve()
    project_root = Path(project_root or snapshot.parents[1]).resolve()
    destination_existed_before = destination.exists()
    processed_existed_before = (destination / "data/processed").exists()
    manifest = load_json(snapshot / "manifest.json")
    files = manifest["files"]
    # Preflight every source and existing destination before any file is copied.
    for item in files:
        source = inside(snapshot, item["path"])
        checked(source, item["sha256"])
        if source.stat().st_size != item["bytes"]:
            raise ValueError(f"Snapshot length mismatch: {item['path']}")
        target = inside(destination, item["restore_to"])
        if target.exists() and (not target.is_file() or file_sha256(target) != item["sha256"]):
            raise FileExistsError(f"Refusing to replace different existing raw input: {target}")
    processed = inside(destination, "data/processed")
    frozen = processed / "split_manifest.json"
    expected = manifest["reference"]
    if frozen.exists() and canonical_split_sha256(load_json(frozen)) != expected["canonical_split_sha256"]:
        raise FileExistsError("Refusing to modify a different frozen processed dataset; select a fresh --destination")
    for name, digest in expected["split_fasta_sha256"].items():
        target = processed / name
        if target.exists() and (not target.is_file() or file_sha256(target) != digest):
            raise FileExistsError(f"Refusing to replace a different existing split: {target}")
    restored = []
    for item in files:
        source = inside(snapshot, item["path"])
        target = inside(destination, item["restore_to"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        checked(target, item["sha256"])
        restored.append(item["restore_to"])
    report = {"schema_version": 1, "kind": "licensed_snapshot_restore",
              "snapshot_manifest_sha256": file_sha256(snapshot / "manifest.json"),
              "restored_files": restored, "network_used_by_restore": False,
              "full_prepare_executed": False,
              "destination_existed_before": destination_existed_before,
              "processed_directory_existed_before": processed_existed_before,
              "original_working_manifest_sha256": expected["original_working_manifest_sha256"],
              "expected_canonical_split_sha256": expected["canonical_split_sha256"]}
    if prepare:
        try:
            import yaml
            sys.path.insert(0, str(project_root / "src"))
            from dplm_family_lab.data import prepare as prepare_data, audit
            from dplm_family_lab.similarity import METRIC
        except ImportError as error:
            raise RuntimeError("Offline preparation needs installed base dependencies plus HMMER or pyhmmer; see README.md") from error
        config = yaml.safe_load(Path(config_path or project_root / "configs/pf00194.yaml").read_text(encoding="utf-8-sig"))
        for key in ("family", "query", "min_length", "max_length", "min_hmm_coverage", "identity", "coverage", "split", "seed"):
            if config["data"][key] != expected["signature"][key]:
                raise ValueError(f"Configuration changes the frozen scientific protocol: data.{key}")
        if METRIC != expected["signature"]["metric"]:
            raise ValueError("The implementation's exact alignment metric differs from the snapshot")
        # Only a private configuration copy's filesystem locations change.
        config["data"]["raw_dir"] = str(destination / "data/raw")
        config["data"]["processed_dir"] = str(processed)
        config["hmm"]["path"] = str(destination / "data/raw/PF00194.hmm")
        started = time.perf_counter()
        preparation = prepare_data(config, download=False)
        audited = audit(config)
        elapsed = time.perf_counter() - started
        rebuilt = load_json(frozen)
        canonical = canonical_split_sha256(rebuilt)
        if canonical != expected["canonical_split_sha256"]:
            raise ValueError("Recomputed sequence selection or whole-component split differs from the frozen snapshot")
        hashes = {}
        for name, digest in expected["split_fasta_sha256"].items():
            checked(processed / name, digest)
            hashes[name] = file_sha256(processed / name)
        if not audited["passed"]:
            raise ValueError("Recomputed cross-split audit failed")
        versions = {}
        for package in ("numpy", "biopython", "pyhmmer"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        report.update(full_prepare_executed=True, prepare_and_audit_seconds=elapsed,
                      canonical_split_sha256=canonical, canonical_split_matches=True,
                      split_fasta_sha256=hashes, split_sizes=preparation["split_sizes"],
                      retained_sequences=preparation["hmm_pass"], components=preparation["clusters"]["clusters"],
                      cross_split_audit=audited, software_versions=versions,
                      recomputed_working_manifest_sha256=file_sha256(frozen),
                      byte_manifest_note="Working manifests contain measured timing and creation timestamps. Scientific split equality is checked by canonical split SHA-256 and exact FASTA file SHA-256, not by claiming a new report has the old byte hash.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, help="defaults to this clone; use a fresh directory for independent recomputation")
    parser.add_argument("--prepare", action="store_true", help="run the actual CPU pipeline and audit, then verify all split hashes")
    parser.add_argument("--verify-only", action="store_true", help="check packaged file hashes without copying")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project-root", type=Path, help="clone containing src/ and configs/; defaults to this clone")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    snapshot = Path(__file__).resolve().parent
    project = args.project_root.resolve() if args.project_root else snapshot.parents[1]
    try:
        if args.verify_only and args.prepare:
            raise ValueError("--verify-only and --prepare are mutually exclusive")
        if args.verify_only:
            manifest = load_json(snapshot / "manifest.json")
            for item in manifest["files"]:
                checked(inside(snapshot, item["path"]), item["sha256"])
            result = {"kind": "snapshot_integrity_check", "passed": True,
                      "files": len(manifest["files"]), "manifest_sha256": file_sha256(snapshot / "manifest.json")}
        else:
            result = restore(snapshot, args.destination or project, args.prepare, project, args.config)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
