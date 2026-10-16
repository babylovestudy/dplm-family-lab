from itertools import product
import random

import pytest

from dplm_family_lab.common import SequenceRecord
from dplm_family_lab.similarity import (PairwiseCache, connected_clusters, could_pass,
                                       cross_split_audit, global_similarity, _candidate_rows)


def test_global_identity_includes_terminal_gaps():
    result = global_similarity("ACDE", "ACDEAA")
    assert result.identity == pytest.approx(4 / 6)
    assert result.coverage_a == 1
    assert result.coverage_b == pytest.approx(4 / 6)
    assert result == global_similarity("ACDE", "ACDEAA")


def test_cache_reversal_and_restart(tmp_path):
    path = tmp_path / "pairs.sqlite"
    with PairwiseCache(path) as cache:
        result = cache.compare("ACDE", "ACDEAA")
    with PairwiseCache(path) as cache:
        reversed_result = cache.compare("ACDEAA", "ACDE")
    assert result.identity == reversed_result.identity
    assert result.coverage_a == reversed_result.coverage_b


def test_lossless_qgram_filter_exhaustive_small_sequences():
    seqs = ["".join(x) for n in range(1, 6) for x in product("AC", repeat=n)]
    for a in seqs:
        for b in seqs:
            for identity in (0.5, 0.8, 1.0):
                if global_similarity(a, b).passes(identity, 0.8):
                    assert could_pass(a, b, identity, 0.8), (a, b, identity)


def test_transitive_components_not_representative_only(tmp_path):
    # A-B and B-C meet 80%, while A-C does not. All must stay together.
    records = [SequenceRecord("a", "AAAAAAAAAA"), SequenceRecord("b", "CAAAAAAAAA"),
               SequenceRecord("c", "CCAAAAAAAA"), SequenceRecord("d", "DDDDDDDDDD")]
    clusters, stats = connected_clusters(records, 0.9, 0.8, tmp_path / "cache.sqlite")
    assert clusters["a"] == clusters["b"] == clusters["c"]
    assert clusters["d"] != clusters["a"]
    assert stats["clusters"] == 2
    audit = cross_split_audit(records, {"a": "train", "b": "test", "c": "test", "d": "val"},
                              0.9, 0.8, tmp_path / "cache.sqlite")
    assert not audit["passed"]
    assert {"a", "b"} == {audit["violations"][0]["a"], audit["violations"][0]["b"]}


def test_vectorized_filter_and_components_match_exhaustive_alignment(tmp_path):
    rng = random.Random(7)
    sequences = ["".join(rng.choices("ACDEFG", k=12)) for _ in range(12)]
    sequences += [sequences[0][:-1] + "A", sequences[0][:-2] + "AA", sequences[1] + "AA"]
    records = [SequenceRecord(str(i), s) for i, s in enumerate(sequences)]
    candidates = {i: set(js) for i, js in _candidate_rows(records, 0.8, 0.8)}
    expected = [{i} for i in range(len(records))]
    for i, a in enumerate(records):
        for j in range(i + 1, len(records)):
            b = records[j]
            assert (j in candidates[i]) == could_pass(a.sequence, b.sequence, 0.8, 0.8)
            if global_similarity(a.sequence, b.sequence).passes(0.8, 0.8):
                merged = expected[i] | expected[j]
                for member in merged:
                    expected[member] = merged
    actual, _ = connected_clusters(records, cache_path=tmp_path / "exhaustive.sqlite")
    for i in range(len(records)):
        for j in range(len(records)):
            assert (actual[str(i)] == actual[str(j)]) == (j in expected[i])
