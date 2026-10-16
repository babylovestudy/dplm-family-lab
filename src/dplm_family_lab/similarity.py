"""Exact global alignment metrics and whole-component sequence isolation.

The worst case is O(N² L²) time; SQLite retains completed comparisons across
restarts. Length and q-gram filters are lossless upper bounds, not approximate
homology searches. No MMseqs candidate search is used to claim a strict audit.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sqlite3
from typing import Callable

import numpy as np

METRIC = "global-v1:match=1,mismatch=-1,open=-2,extend=-0.5;identity=matches/all_columns;coverage=paired_residues/full_length"


@dataclass(frozen=True)
class Similarity:
    identity: float
    coverage_a: float
    coverage_b: float
    matches: int
    columns: int
    paired_residues: int

    def passes(self, identity: float, coverage: float) -> bool:
        return (self.identity + 1e-12 >= identity
                and self.coverage_a + 1e-12 >= coverage
                and self.coverage_b + 1e-12 >= coverage)


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _aligner():
    try:
        from Bio.Align import PairwiseAligner
    except ImportError as exc:
        raise RuntimeError("Exact alignment requires biopython; install the base project dependencies.") from exc
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    return aligner


def global_similarity(a: str, b: str, aligner=None) -> Similarity:
    """Score the first deterministic optimal global alignment, terminal gaps included.

    Coverage counts columns containing a residue on BOTH sides, divided by each
    full sequence length. Identity includes mismatches and every gap column in
    its denominator. Tied optimal alignments use Biopython's first alignment.
    """
    if not a or not b:
        raise ValueError("Cannot align an empty sequence")
    if a == b:
        return Similarity(1.0, 1.0, 1.0, len(a), len(a), len(a))
    # Canonical orientation makes tie handling independent of call order.
    reverse = (sequence_hash(a), a) > (sequence_hash(b), b)
    aa, bb = (b, a) if reverse else (a, b)
    alignment = (aligner or _aligner()).align(aa, bb)[0]
    coordinates = alignment.coordinates
    matches = paired = columns = 0
    for i in range(coordinates.shape[1] - 1):
        a0, a1 = int(coordinates[0, i]), int(coordinates[0, i + 1])
        b0, b1 = int(coordinates[1, i]), int(coordinates[1, i + 1])
        da, db = a1 - a0, b1 - b0
        columns += max(da, db)
        if da and db:
            if da != db:
                raise RuntimeError("Unexpected unequal aligned block")
            paired += da
            matches += sum(x == y for x, y in zip(aa[a0:a1], bb[b0:b1]))
    ca, cb = paired / len(aa), paired / len(bb)
    if reverse:
        ca, cb = cb, ca
    return Similarity(matches / columns, ca, cb, matches, columns, paired)


class PairwiseCache:
    """A content-addressed cache; IDs can change without invalidating comparisons."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS pairs (metric TEXT, a TEXT, b TEXT, result TEXT, PRIMARY KEY(metric,a,b))")
        self.aligner = _aligner()
        self.pending = 0

    def compare(self, a: str, b: str) -> Similarity:
        ha, hb = sequence_hash(a), sequence_hash(b)
        reverse = ha > hb
        lo, hi = sorted((ha, hb))
        row = self.connection.execute("SELECT result FROM pairs WHERE metric=? AND a=? AND b=?", (METRIC, lo, hi)).fetchone()
        if row:
            value = Similarity(**json.loads(row[0]))
        else:
            value = global_similarity(b, a, self.aligner) if reverse else global_similarity(a, b, self.aligner)
            self.connection.execute("INSERT OR REPLACE INTO pairs VALUES (?,?,?,?)", (METRIC, lo, hi, json.dumps(asdict(value))))
            self.pending += 1
            if self.pending >= 500:
                self.connection.commit()
                self.pending = 0
        if reverse:
            return Similarity(value.identity, value.coverage_b, value.coverage_a,
                              value.matches, value.columns, value.paired_residues)
        return value

    def close(self):
        self.connection.commit()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_NEAREST_REFERENCES = None
_NEAREST_ALIGNER = None


def _initialize_nearest_worker(references):
    """Spawn initializer: retain CPU references and one aligner per process."""
    global _NEAREST_REFERENCES, _NEAREST_ALIGNER
    _NEAREST_REFERENCES = references
    _NEAREST_ALIGNER = _aligner()


def _nearest_exact(sequence: str, references, aligner) -> tuple[float, str]:
    best_identity, best_id = -1.0, None
    for reference_id, reference_sequence in references:
        upper = min(len(sequence), len(reference_sequence)) / max(len(sequence), len(reference_sequence))
        if upper + 1e-12 < best_identity:
            continue
        identity = global_similarity(sequence, reference_sequence, aligner).identity
        if identity > best_identity or (identity == best_identity and (best_id is None or reference_id < best_id)):
            best_identity, best_id = identity, reference_id
    return best_identity, best_id


def _nearest_worker(sequence: str) -> tuple[float, str]:
    # Tasks transmit only a generated sequence. Workers never open SQLite,
    # import training code, or initialize a GPU runtime.
    if _NEAREST_REFERENCES is None or _NEAREST_ALIGNER is None:
        raise RuntimeError("Nearest-neighbor worker was not initialized")
    return _nearest_exact(sequence, _NEAREST_REFERENCES, _NEAREST_ALIGNER)


def nearest_training_matches(sequences, training, training_sha256: str,
                             cache_path: str | Path, workers: int = 1) -> tuple[dict, dict]:
    """Exact final nearest matches with parent-only, restartable result caching.

    The scientific metric and lexicographically smallest reference-ID tie break
    are identical for serial and spawn workers. Cache keys include METRIC, the
    complete training FASTA checksum (including IDs), and the query sequence
    checksum. A completed query is committed immediately, so an interrupted
    evaluation resumes without retaining millions of pairwise SQLite rows.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("evaluation.workers must be a positive integer")
    if len(training_sha256) != 64 or any(c not in "0123456789abcdef" for c in training_sha256):
        raise ValueError("A lowercase SHA-256 of the complete training FASTA is required")
    references = [(record.id, record.sequence) for record in training]
    if not references or any(not sequence for _, sequence in references):
        raise ValueError("Nearest-neighbor search needs nonempty training references")
    reference_ids = {name for name, _ in references}
    if len(reference_ids) != len(references):
        raise ValueError("Training reference IDs must be unique")
    queries = list(dict.fromkeys(sequences))
    if any(not sequence for sequence in queries):
        raise ValueError("Nearest-neighbor queries must be nonempty")
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("CREATE TABLE IF NOT EXISTS nearest (metric TEXT, training_sha256 TEXT, query_sha256 TEXT, identity REAL, reference_id TEXT, PRIMARY KEY(metric,training_sha256,query_sha256))")
    connection.commit()
    results, missing = {}, []
    stats = {"unique_queries": len(queries), "cache_hits": 0, "computed_queries": 0,
             "workers_requested": workers, "workers_used": 0,
             "training_fasta_sha256": training_sha256, "cache_path": str(path)}

    def retain(sequence, result):
        identity, reference_id = result
        if not math.isfinite(identity) or not 0 <= identity <= 1 or reference_id not in reference_ids:
            raise ValueError("Invalid cached or computed nearest-neighbor result")
        results[sequence] = (float(identity), reference_id)
        connection.execute("INSERT OR REPLACE INTO nearest VALUES (?,?,?,?,?)",
                           (METRIC, training_sha256, sequence_hash(sequence), identity, reference_id))
        connection.commit()
        stats["computed_queries"] += 1

    try:
        for sequence in queries:
            row = connection.execute("SELECT identity,reference_id FROM nearest WHERE metric=? AND training_sha256=? AND query_sha256=?",
                                     (METRIC, training_sha256, sequence_hash(sequence))).fetchone()
            if row is None:
                missing.append(sequence)
                continue
            identity, reference_id = row
            if not math.isfinite(identity) or not 0 <= identity <= 1 or reference_id not in reference_ids:
                raise ValueError("Invalid nearest-neighbor cache entry; use a new cache path")
            results[sequence] = (float(identity), reference_id)
            stats["cache_hits"] += 1
        stats["workers_used"] = min(workers, len(missing))
        if workers == 1 or len(missing) <= 1:
            aligner = _aligner() if missing else None
            for sequence in missing:
                retain(sequence, _nearest_exact(sequence, references, aligner))
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(missing)),
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_initialize_nearest_worker,
                                     initargs=(references,)) as pool:
                futures = {pool.submit(_nearest_worker, sequence): sequence for sequence in missing}
                for future in as_completed(futures):
                    retain(futures[future], future.result())
    finally:
        connection.close()
    # Restore caller query order regardless of worker completion order.
    return {sequence: results[sequence] for sequence in queries}, stats


def _qgrams(sequence: str, k: int = 2) -> Counter:
    return Counter(sequence[i:i + k] for i in range(max(0, len(sequence) - k + 1)))


def could_pass(a: str, b: str, identity: float, coverage: float,
               qa: Counter | None = None, qb: Counter | None = None) -> bool:
    """Lossless necessary conditions for a qualifying global alignment.

    If identity >= t, alignment edit columns are at most floor(min_len*(1-t)/t).
    One edit destroys at most k shared k-grams. The multiset q-gram lower bound
    therefore rejects impossible pairs without choosing an alignment.
    """
    shortest, longest = sorted((len(a), len(b)))
    if shortest / longest + 1e-12 < max(identity, coverage):
        return False
    if identity <= 0 or shortest < 2:
        return True
    edits = math.floor(shortest * (1 - identity) / identity + 1e-9)
    required = longest - 1 - 2 * edits
    if required <= 0:
        return True
    qa = qa if qa is not None else _qgrams(a)
    qb = qb if qb is not None else _qgrams(b)
    return sum((qa & qb).values()) >= required


def _candidate_rows(records, identity, coverage):
    """Vectorize only mathematically necessary q-gram/length conditions.

    This changes the implementation cost of the exact filter, not its results.
    Counts use int32 to support arbitrary sequence lengths without overflow.
    """
    grams = [_qgrams(r.sequence) for r in records]
    vocabulary = sorted({word for bag in grams for word in bag})
    columns = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(records), len(vocabulary)), dtype=np.int32)
    for i, bag in enumerate(grams):
        for word, count in bag.items():
            matrix[i, columns[word]] = count
    lengths = np.asarray([len(r.sequence) for r in records], dtype=np.int64)
    for i in range(len(records)):
        other = np.arange(i + 1, len(records))
        shortest = np.minimum(lengths[i], lengths[other])
        longest = np.maximum(lengths[i], lengths[other])
        allowed = shortest / longest + 1e-12 >= max(identity, coverage)
        candidates = other[allowed]
        if len(candidates) and identity > 0:
            edits = np.floor(np.minimum(lengths[i], lengths[candidates]) * (1 - identity) / identity + 1e-9).astype(np.int64)
            required = np.maximum(lengths[i], lengths[candidates]) - 1 - 2 * edits
            common = np.minimum(matrix[i], matrix[candidates]).sum(axis=1)
            candidates = candidates[(common >= required) | (np.minimum(lengths[i], lengths[candidates]) < 2)]
        yield i, candidates.tolist()


def connected_clusters(records, identity: float = 0.8, coverage: float = 0.8,
                       cache_path: str | Path = ".cache/pairs.sqlite",
                       progress: Callable[[dict], None] | None = None) -> tuple[dict[str, str], dict]:
    if not (0 < identity <= 1 and 0 < coverage <= 1):
        raise ValueError("Identity and coverage must be in (0,1]")
    ordered = sorted(records, key=lambda r: r.id)
    if len({r.id for r in ordered}) != len(ordered):
        raise ValueError("Sequence IDs must be unique")
    parent = list(range(len(ordered)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    stats = {"pairs": 0, "bounded_out": 0, "aligned": 0, "edges": 0,
             "within_component_skips": 0,
             "metric": METRIC, "identity": identity, "coverage": coverage}
    with PairwiseCache(cache_path) as cache:
        for i, candidates in _candidate_rows(ordered, identity, coverage):
            a = ordered[i]
            stats["pairs"] += len(ordered) - i - 1
            stats["bounded_out"] += len(ordered) - i - 1 - len(candidates)
            for j in candidates:
                b = ordered[j]
                if find(i) == find(j):
                    # Existing paths already establish connectivity. A missing
                    # redundant edge cannot alter the component partition.
                    stats["within_component_skips"] += 1
                    continue
                stats["aligned"] += 1
                if cache.compare(a.sequence, b.sequence).passes(identity, coverage):
                    stats["edges"] += 1
                    parent[find(j)] = find(i)
            if progress and (i + 1) % 25 == 0:
                progress({**stats, "rows_completed": i + 1, "rows_total": len(ordered)})
    components = {}
    for i, record in enumerate(ordered):
        components.setdefault(find(i), []).append(record.id)
    assignments = {}
    for members in components.values():
        cluster = "cluster_" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        assignments.update({member: cluster for member in members})
    stats["clusters"] = len(components)
    stats["largest_cluster"] = max((len(c) for c in components.values()), default=0)
    return assignments, stats


def cross_split_audit(records, splits: dict[str, str], identity: float, coverage: float,
                      cache_path: str | Path) -> dict:
    ordered = sorted(records, key=lambda r: r.id)
    if set(splits) != {r.id for r in ordered}:
        raise ValueError("Split assignments do not cover exactly the supplied sequences")
    violations, examined, aligned = [], 0, 0
    with PairwiseCache(cache_path) as cache:
        for i, candidates in _candidate_rows(ordered, identity, coverage):
            a = ordered[i]
            examined += sum(splits[a.id] != splits[b.id] for b in ordered[i + 1:])
            for j in candidates:
                b = ordered[j]
                if splits[a.id] == splits[b.id]:
                    continue
                aligned += 1
                result = cache.compare(a.sequence, b.sequence)
                if result.passes(identity, coverage):
                    violations.append({"a": a.id, "b": b.id, **asdict(result)})
    return {"passed": not violations, "cross_split_pairs": examined, "aligned": aligned,
            "violations": violations, "metric": METRIC,
            "identity": identity, "coverage": coverage}
