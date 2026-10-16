import json

import pytest

from dplm_family_lab.cli import demo, main
from dplm_family_lab.common import SequenceRecord, read_fasta, sha256_file, write_fasta, write_json


def test_fasta_roundtrip_and_duplicate_identifiers(tmp_path):
    path = tmp_path / "input.fasta"
    records = [SequenceRecord("protein1", "ACDE" * 30, "source description")]
    write_fasta(records, path)
    assert read_fasta(path) == records
    path.write_text(">same\nACDE\n>same\nFGHI\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate FASTA"):
        read_fasta(path)


def test_invalid_fasta_and_finite_json(tmp_path):
    path = tmp_path / "input.fasta"
    path.write_text("ACDE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before first"):
        read_fasta(path)
    with pytest.raises(ValueError):
        write_json(tmp_path / "report.json", {"loss": float("nan")})
    assert not (tmp_path / "report.json").exists()


def test_demo_is_explicitly_synthetic_and_refuses_overwrite(tmp_path):
    report = demo(str(tmp_path / "demo"))
    assert report["accepted_records"] == 2
    assert report["model_training"] == "not run"
    assert len(sha256_file(tmp_path / "demo" / "clean.fasta")) == 64
    assert json.loads((tmp_path / "demo" / "report.json").read_text())["kind"].startswith("synthetic")
    with pytest.raises(FileExistsError):
        demo(str(tmp_path / "demo"))


def test_cli_reports_actionable_error(tmp_path, capsys):
    assert main(["fetch", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert "ERROR:" in capsys.readouterr().err
