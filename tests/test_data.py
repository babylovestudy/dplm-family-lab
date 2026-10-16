import json
from pathlib import Path

import pytest

from dplm_family_lab.common import SequenceRecord, write_fasta
from dplm_family_lab import data


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    hmm = raw / "PF00194.hmm"
    hmm.write_text("HMMER3/f\nNAME TEST\nACC PF00194.27\nLENG 8\nGA 27 27;\nHMM test\n//\n")
    write_fasta([SequenceRecord(str(i), aa * 8) for i, aa in enumerate("ACDEFG")], raw / "uniprot.fasta")
    config = {"data": {"family": "PF00194", "query": "test", "raw_dir": str(raw),
                       "processed_dir": str(out), "min_length": 8, "max_length": 8,
                       "min_hmm_coverage": 0.8, "identity": 0.8, "coverage": 0.8,
                       "split": [0.8, 0.1, 0.1], "seed": 42},
              "hmm": {"path": str(hmm), "binary": "hmmsearch"},
              "train": {"train_fasta": str(out / "train.fasta")},
              "evaluation": {"train_fasta": str(out / "train.fasta"), "hmm_path": str(hmm),
                             "bootstrap_samples": 20, "seed": 42}}

    def scores(records, *args, **kwargs):
        return {r.id: {"family_pass": True, "ga_pass": True, "hmm_coverage": 1.0,
                       "sequence_score": 40.0, "domain_score": 30.0, "domains": []} for r in records}

    monkeypatch.setattr(data, "score_sequences", scores)
    data.prepare(config, download=False)
    return config


def test_clean_filter_dedup_preserves_aliases():
    records = [SequenceRecord("z", "ACDE"), SequenceRecord("a", "acde"),
               SequenceRecord("bad", "ACDX"), SequenceRecord("short", "AC")]
    kept, counts, aliases = data.clean_records(records, 4, 8)
    assert [(r.id, r.sequence) for r in kept] == [("a", "ACDE")]
    assert counts == {"input": 4, "invalid_alphabet": 1, "outside_length": 1,
                      "exact_duplicates": 1, "retained": 1}
    assert aliases == {"a": ["z"]}


def test_prepare_freezes_and_audits(prepared):
    config = prepared
    out = Path(config["data"]["processed_dir"])
    before = (out / "split_manifest.json").read_bytes()
    result = data.prepare(config, download=False)
    assert sum(result["split_sizes"].values()) == 6
    assert all(result["split_sizes"].values())
    assert data.audit(config)["passed"]
    assert (out / "split_manifest.json").read_bytes() == before
    reference = json.loads((out / "training_ref.json").read_text())
    assert reference["source"]["source"] == "local_offline"
    config["data"]["seed"] = 99
    with pytest.raises(ValueError, match="Frozen"):
        data.prepare(config, download=False)


def test_audit_rejects_sequence_mutation(prepared):
    path = Path(prepared["data"]["processed_dir"]) / "test.fasta"
    path.write_text(path.read_text().replace("GG", "AA").replace("AA", "CC"))
    with pytest.raises(ValueError, match="checksum"):
        data.audit(prepared)


def test_restart_recovers_interrupted_reference_creation(prepared):
    output = Path(prepared["data"]["processed_dir"])
    reference = output / "training_ref.json"
    expected = reference.read_bytes()
    reference.unlink()
    data.prepare(prepared, download=False)
    assert reference.read_bytes() == expected


def test_split_never_splits_components():
    clusters = {"a": "large", "b": "large", "c": "second", "d": "third"}
    first, sizes = data.assign_splits(clusters, seed=42)
    assert first["a"] == first["b"]
    assert all(sizes.values())
    assert data.assign_splits(dict(reversed(list(clusters.items()))), seed=42)[0] == first
    with pytest.raises(ValueError, match="three"):
        data.assign_splits({"a": "only", "b": "only"})
