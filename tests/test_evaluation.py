import csv
from pathlib import Path

import pytest
import test_data

from dplm_family_lab.common import SequenceRecord, read_fasta, write_fasta
from dplm_family_lab import evaluation

prepared = test_data.prepared  # Shared offline fixture; no network or model downloads.


def test_all_valid_denominator_duplicates_and_rows(prepared, tmp_path, monkeypatch):
    train = read_fasta(prepared["evaluation"]["train_fasta"])[0]
    fasta = tmp_path / "generated.fasta"
    write_fasta([SequenceRecord("exact", train.sequence), SequenceRecord("duplicate", train.sequence),
                 SequenceRecord("novel", "YYYYYYYYY"), SequenceRecord("invalid", "AXXXXXXX")], fasta)

    def scores(records, *args, **kwargs):
        return {r.id: {"family_pass": r.id != "novel", "ga_pass": r.id != "novel",
                       "hmm_coverage": 0.0 if r.id == "novel" else 1.0,
                       "sequence_score": None, "domain_score": None} for r in records}

    monkeypatch.setattr(evaluation, "score_sequences", scores)
    output = tmp_path / "eval"
    result = evaluation.evaluate(prepared, fasta, output)
    assert result["counts"]["emitted_fasta_records"] == 4
    assert result["counts"]["valid_amino_acid_sequences"] == 3
    assert result["metrics"]["family_pass_rate"]["value"] == pytest.approx(2 / 3)
    assert result["metrics"]["exact_training_match_rate"]["value"] == pytest.approx(2 / 3)
    assert result["metrics"]["excess_duplicate_rate"]["value"] == pytest.approx(1 / 3)
    rows = list(csv.DictReader((output / "per_sequence.csv").open()))
    assert len(rows) == 4
    assert rows[1]["duplicate_of_earlier"] == "True"
    assert rows[2]["within_training_length"] == "False"
    assert len(list(csv.DictReader((output / "pairwise_diversity.csv").open()))) == 3
    assert evaluation.evaluate(prepared, fasta, tmp_path / "eval2")["metrics"] == result["metrics"]


def test_evaluation_requires_matching_training_hmm(prepared, tmp_path):
    path = Path(prepared["hmm"]["path"])
    path.write_text(path.read_text().replace("27 27", "28 28"))
    with pytest.raises(ValueError, match="HMM differs"):
        evaluation.evaluate(prepared, tmp_path / "unused.fasta", tmp_path / "eval")


def test_evaluation_csv_and_metrics_identical_with_eight_workers(prepared, tmp_path, monkeypatch):
    fasta = tmp_path / "parallel.fasta"
    sequences = ["ACDEFGHI", "CCCCCCCC", "YYYYYYYYY", "AAAAAAAA", "ACAAAAAA",
                 "DDDDDDDD", "EEEEEEEE", "GGGGGGGG", "ACDEFGHI", "AXXXXXXX"]
    write_fasta([SequenceRecord(f"generated_{i}", sequence) for i, sequence in enumerate(sequences)], fasta)

    def scores(records, *args, **kwargs):
        return {r.id: {"family_pass": False, "ga_pass": False, "hmm_coverage": 0.0,
                       "sequence_score": None, "domain_score": None} for r in records}

    monkeypatch.setattr(evaluation, "score_sequences", scores)
    prepared["evaluation"].update(workers=1, nearest_cache=str(tmp_path / "serial_nearest.sqlite"))
    serial = evaluation.evaluate(prepared, fasta, tmp_path / "serial")
    prepared["evaluation"].update(workers=8, nearest_cache=str(tmp_path / "parallel_nearest.sqlite"))
    parallel = evaluation.evaluate(prepared, fasta, tmp_path / "parallel")
    assert serial["metrics"] == parallel["metrics"]
    assert serial["counts"] == parallel["counts"]
    assert parallel["nearest_search"]["workers_used"] == 8
    for name in ("per_sequence.csv", "pairwise_diversity.csv"):
        assert (tmp_path / "serial" / name).read_bytes() == (tmp_path / "parallel" / name).read_bytes()
    restarted = evaluation.evaluate(prepared, fasta, tmp_path / "restarted")
    assert restarted["metrics"] == parallel["metrics"]
    assert restarted["nearest_search"]["computed_queries"] == 0
