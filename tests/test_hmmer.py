from pathlib import Path
from types import SimpleNamespace

import pytest

from dplm_family_lab.common import SequenceRecord
from dplm_family_lab.hmmer import parse_domtblout, read_hmm_metadata, score_sequences


def make_hmm(path: Path):
    path.write_text("HMMER3/f\nNAME  TEST\nACC   PF00194.27\nLENG  8\nGA    27 27;\nHMM test\n//\n")
    return path


def row(name, start=1, end=8, full=40, domain=30):
    return f"{name} - 8 TEST PF00194.27 8 1e-12 {full} 0 1 1 1e-10 1e-10 {domain} 0 {start} {end} 1 8 1 8 0.99 description\n"


def test_ga_and_single_domain_coverage(tmp_path):
    metadata = read_hmm_metadata(make_hmm(tmp_path / "model.hmm"))
    results = parse_domtblout(row("full") + row("split", 1, 4) + row("split", 5, 8)
                              + row("lowseq", full=26) + row("lowdomain", domain=26), metadata)
    assert results["full"]["family_pass"]
    assert results["split"]["ga_pass"]
    assert not results["split"]["family_pass"]  # Separate domains are never merged.
    assert not results["lowseq"]["ga_pass"]
    assert not results["lowdomain"]["ga_pass"]
    assert metadata["ga_sequence"] == metadata["ga_domain"] == 27


def test_external_runner_preserves_misses_and_passes_cut_ga(tmp_path, monkeypatch):
    import dplm_family_lab.hmmer as module

    monkeypatch.setattr(module.shutil, "which", lambda _: "/fake/hmmsearch")

    def run(command, **kwargs):
        assert "--cut_ga" in command
        Path(command[command.index("--domtblout") + 1]).write_text(row("hit"))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    results = score_sequences([SequenceRecord("hit", "ACDEFGHI"), SequenceRecord("miss", "CCCCCCCC")],
                              make_hmm(tmp_path / "model.hmm"), backend="external")
    assert results["hit"]["family_pass"]
    assert not results["miss"]["family_pass"]
    assert results["miss"]["hmm_coverage"] == 0


def test_rejects_uncurated_model(tmp_path):
    path = tmp_path / "bad.hmm"
    path.write_text("HMMER3/f\nACC PF00194.27\nLENG 8\n//\n")
    with pytest.raises(ValueError, match="GA"):
        read_hmm_metadata(path)
