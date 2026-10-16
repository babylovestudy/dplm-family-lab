"""Safety and integrity checks for the public frozen-input restore helper."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SNAPSHOT = Path(__file__).resolve().parents[1] / "results/dataset-snapshot"
SPEC = importlib.util.spec_from_file_location("dataset_snapshot_restore", SNAPSHOT / "restore.py")
RESTORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESTORE)


def test_fresh_restore_and_identical_files_are_reusable(tmp_path):
    destination = tmp_path / "fresh"
    first = RESTORE.restore(SNAPSHOT, destination)
    assert not first["destination_existed_before"]
    assert not first["processed_directory_existed_before"]
    assert not first["network_used_by_restore"]
    second = RESTORE.restore(SNAPSHOT, destination)
    assert second["destination_existed_before"]
    for item in RESTORE.load_json(SNAPSHOT / "manifest.json")["files"]:
        assert RESTORE.file_sha256(destination / item["restore_to"]) == item["sha256"]


def test_different_raw_input_aborts_before_any_other_copy(tmp_path):
    destination = tmp_path / "fresh"
    different = destination / "data/raw/provenance.json"
    different.parent.mkdir(parents=True)
    different.write_text("unrelated input\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different existing raw input"):
        RESTORE.restore(SNAPSHOT, destination)
    assert different.read_text(encoding="utf-8") == "unrelated input\n"
    assert not (destination / "data/raw/uniprot.fasta").exists()
    assert not (destination / "data/raw/PF00194.hmm").exists()


def test_different_frozen_split_aborts_before_any_raw_copy(tmp_path):
    destination = tmp_path / "fresh"
    frozen = destination / "data/processed/split_manifest.json"
    frozen.parent.mkdir(parents=True)
    manifest = {"signature": {}, "sequences": []}
    frozen.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileExistsError, match="different frozen processed dataset"):
        RESTORE.restore(SNAPSHOT, destination)
    assert RESTORE.load_json(frozen) == manifest
    assert not (destination / "data/raw").exists()


def test_canonical_split_hash_excludes_report_but_covers_assignments():
    original = {
        "signature": {"identity": 0.8, "seed": 42},
        "sequences": [
            {"id": "B", "split": "train", "sha256": "bbb"},
            {"id": "A", "split": "val", "sha256": "aaa"},
        ],
        "report": {"elapsed": 1},
        "created_at": "old",
    }
    rerun = deepcopy(original)
    rerun["report"] = {"elapsed": 99}
    rerun["created_at"] = "new"
    rerun["sequences"].reverse()
    assert RESTORE.canonical_split_sha256(original) == RESTORE.canonical_split_sha256(rerun)
    rerun["sequences"][0]["split"] = "test"
    assert RESTORE.canonical_split_sha256(original) != RESTORE.canonical_split_sha256(rerun)


def test_restore_cannot_follow_processed_directory_outside_destination(tmp_path):
    destination = tmp_path / "fresh"
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "data").mkdir(parents=True)
    try:
        (destination / "data/processed").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is unavailable on this platform")
    with pytest.raises(ValueError, match="escapes its directory"):
        RESTORE.restore(SNAPSHOT, destination)
    assert not list(outside.iterdir())
    assert not (destination / "data/raw").exists()
