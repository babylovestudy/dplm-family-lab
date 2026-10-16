import sqlite3

import pytest

from dplm_family_lab.common import SequenceRecord, sha256_file, write_fasta
from dplm_family_lab import similarity


def _fingerprint(tmp_path, name, records):
    path = tmp_path / name
    write_fasta(records, path)
    return sha256_file(path)


def test_spawn_eight_workers_matches_serial_and_exhaustive_ties(tmp_path):
    references = [SequenceRecord("z_exact", "ACDEFGHIKL"),
                  SequenceRecord("a_exact", "ACDEFGHIKL"),
                  SequenceRecord("c", "CCCCCCCCCC"),
                  SequenceRecord("b", "AAAACCCCCC")]
    queries = ["ACDEFGHIKL", "ACDEFGHIKA", "ACDEFGHIKLAA", "CCCCCCCCCC", "AAAACCCCCC",
               "AAAACCCCAA", "DDDDDDDDDD", "ACDEFGHI", "ACDEFGHIKL"]
    fingerprint = _fingerprint(tmp_path, "references.fasta", references)
    serial, serial_stats = similarity.nearest_training_matches(
        queries, references, fingerprint, tmp_path / "serial.sqlite", workers=1)
    parallel, parallel_stats = similarity.nearest_training_matches(
        queries, references, fingerprint, tmp_path / "parallel.sqlite", workers=8)
    assert serial == parallel
    assert serial_stats["computed_queries"] == parallel_stats["computed_queries"] == 8
    assert parallel_stats["workers_used"] == 8
    assert serial["ACDEFGHIKL"] == (1.0, "a_exact")
    for query in set(queries):
        scored = [(similarity.global_similarity(query, r.sequence).identity, r.id) for r in references]
        highest = max(score for score, _ in scored)
        assert serial[query] == (highest, min(name for score, name in scored if score == highest))


def test_final_cache_is_invalidated_by_complete_training_fasta(tmp_path):
    first = [SequenceRecord("old", "AAAAAAAA")]
    second = [SequenceRecord("new", "CCCCCCCC")]
    cache = tmp_path / "shared.sqlite"
    one, _ = similarity.nearest_training_matches(["AAAAAAAA"], first,
        _fingerprint(tmp_path, "first.fasta", first), cache)
    two, stats = similarity.nearest_training_matches(["AAAAAAAA"], second,
        _fingerprint(tmp_path, "second.fasta", second), cache)
    assert one["AAAAAAAA"] == (1.0, "old")
    assert two["AAAAAAAA"] == (0.0, "new")
    assert stats["cache_hits"] == 0
    with sqlite3.connect(cache) as connection:
        assert connection.execute("SELECT count(*) FROM nearest").fetchone()[0] == 2


def test_interrupted_query_restart_retains_committed_results(tmp_path, monkeypatch):
    references = [SequenceRecord("a", "AAAAAAAA"), SequenceRecord("c", "CCCCCCCC")]
    fingerprint = _fingerprint(tmp_path, "training.fasta", references)
    original = similarity._nearest_exact
    calls = 0

    def interrupted(sequence, training, aligner):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(sequence, training, aligner)

    monkeypatch.setattr(similarity, "_nearest_exact", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        similarity.nearest_training_matches(["AAAAAAAC", "CCCCCCCA"], references,
            fingerprint, tmp_path / "restart.sqlite")
    monkeypatch.setattr(similarity, "_nearest_exact", original)
    result, stats = similarity.nearest_training_matches(["AAAAAAAC", "CCCCCCCA"], references,
        fingerprint, tmp_path / "restart.sqlite", workers=8)
    assert len(result) == 2
    assert stats["cache_hits"] == 1
    assert stats["computed_queries"] == 1


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_workers_must_be_positive_integer(tmp_path, workers):
    with pytest.raises(ValueError, match="positive integer"):
        similarity.nearest_training_matches([], [], "a" * 64, tmp_path / "unused.sqlite", workers)
