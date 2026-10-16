"""Read recorded evidence without running or changing any experiment.

The report is a metadata summary, not a rerun or a claim that the full protocol
has finished. Tiny fixtures and short runs can never establish a formal result.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path, PurePosixPath
import re

from .backend import DPLM_150M_REVISION, DPLM_150M_SHA256, UPSTREAM_COMMIT
from .common import sha256_file, utc_now, write_json


FIELDS = [
    "source_type", "stage", "status", "arm", "source_json", "source_sha256",
    "model_source", "model_revision", "upstream_commit", "training_seed",
    "generation_seed", "step", "max_steps", "emitted_sequences", "valid_sequences",
    "requested_sequences", "ga_pass_rate", "family_pass_rate", "overall_family_yield",
    "excess_duplicate_rate", "exact_training_match_rate", "mean_pairwise_distance",
    "mean_nearest_training_identity", "denoising_loss", "elapsed_seconds",
    "peak_gpu_allocated_bytes", "data_sha256", "notes",
]

LIMITATIONS = [
    "This summary reads existing files; it does not rerun or independently authenticate an experiment.",
    "Tiny/synthetic fixtures, smoke tests and pilot runs are not formal 150M results.",
    "HMM GA/coverage measures family sequence compatibility, not activity, folding, expression or stability.",
    "HMM rates are conditional on valid emitted sequences; overall yield additionally requires the requested count.",
    "Sequence bootstrap intervals describe sampling uncertainty, not independent training replicates; related sequences and only three training seeds limit inference.",
    "The 80/80 adaptation split does not rule out distant homology or foundation-model pretraining overlap.",
    "Fixed-noise weighted denoising CE is not exact sequence likelihood or perplexity.",
    "Training elapsed time is for the reported session, not necessarily all resumed sessions or total billing.",
]


def _read(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("result must be a JSON object")
    return value


def _display(root, value):
    path = Path(str(value))
    if path.is_absolute():
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return str(value).replace("\\", "/")


def _reference(root, value):
    """Resolve project references, including paths preserved from a remote run."""
    if not value:
        return None
    text = str(value).replace("\\", "/")
    candidates = [Path(text) if Path(text).is_absolute() else root / text]
    for anchor in ("runs/", "outputs/", "data/"):
        if text.startswith(anchor):
            candidates.append(root / text)
        elif f"/{anchor}" in text:
            candidates.append(root / anchor / text.split(f"/{anchor}", 1)[1])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and resolved.exists():
            return resolved
    return None


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _metric(payload, name):
    value = payload.get("metrics", {}).get(name)
    return _number(value.get("value") if isinstance(value, dict) else value)


def _marked(root, values, marker):
    words = re.split(r"[^a-z0-9]+", " ".join(_display(root, v) for v in values).lower())
    return marker in words


def _path_stage(root, paths, provenance):
    values = [*paths, provenance.get("source", "")]
    if provenance.get("tiny_fixture") is True or _marked(root, values, "tiny"):
        return "tiny"
    if _marked(root, values, "smoke"):
        return "smoke"
    if _marked(root, values, "pilot"):
        return "pilot"
    return None


def _official_source_chain(provenance):
    """Check recorded original identity, never the adapted weight hash.

    This is a consistency check of metadata, not authentication of weights.
    Adapted exports retain checkpoint_sha256 for the original pretrained file;
    their loaded_weights_sha256 correctly refers to different, adapted weights.
    """
    return (
        provenance.get("tiny_fixture") is not True
        and provenance.get("source") == "airkingbd/dplm_150m"
        and provenance.get("revision") == DPLM_150M_REVISION
        and provenance.get("checkpoint_sha256") == DPLM_150M_SHA256
        and provenance.get("upstream_commit") == UPSTREAM_COMMIT
        and provenance.get("artifact_role", "pretrained_source") in {"pretrained_source", "adapted_export"}
        and (provenance.get("artifact_role", "pretrained_source") != "pretrained_source"
             or provenance.get("loaded_weights_sha256", DPLM_150M_SHA256) == DPLM_150M_SHA256)
    )


def _configured_official_source(expected):
    """Recognize the explicit project cache or a local cache with a sidecar.

    The standard cache namespace can be checked from archived metadata even
    when the large weights were not copied with a result bundle. An arbitrary
    path is not made equivalent by a matching model name alone.
    """
    source = str(expected.get("source", "")).replace("\\", "/").rstrip("/")
    if source == "airkingbd/dplm_150m":
        return True
    parts = PurePosixPath(source).parts
    if not source or ".." in parts or "tiny" in re.split(r"[^a-z0-9]+", source.lower()):
        return False
    sidecar = Path(source) / "family_lab_provenance.json"
    if sidecar.is_file():
        try:
            metadata = _read(sidecar)
        except (OSError, ValueError):
            return False
        return (_official_source_chain(metadata)
                and metadata.get("artifact_role", "pretrained_source") == "pretrained_source")
    return ((len(parts) >= 2 and parts[-2:] == ("model_cache", "dplm_150m"))
            or (len(parts) >= 3 and parts[-3:] == (
                "models--airkingbd--dplm_150m", "snapshots", DPLM_150M_REVISION)))


def _real_identity(provenance, config):
    expected = config.get("model", {})
    return (
        _official_source_chain(provenance)
        and _configured_official_source(expected)
        and expected.get("revision") == DPLM_150M_REVISION
        and expected.get("upstream_commit") == UPSTREAM_COMMIT
        and (_number(provenance.get("parameter_count")) or 0) >= 100_000_000
    )


def _verified_export_identity(run_directory):
    """Verify small identity records without reading the large weight file.

    The CPU seal binds the serialized weights to the original tensor digest.
    Recorded generation hashes must then match this independently sealed file
    identity; matching a directory name is insufficient.
    """
    run = Path(run_directory)
    export = run / "export_best"
    paths = {"identity": export / "export_identity.json", "report": run / "training_report.json",
             "roundtrip": export / "roundtrip.json", "provenance": export / "family_lab_provenance.json"}
    values = {name: _read(path) for name, path in paths.items()}
    identity, report, roundtrip, provenance = (values[name] for name in ("identity", "report", "roundtrip", "provenance"))
    if identity.get("schema_version") != 1 or identity.get("weights_file") not in {
        "pytorch_model.bin", "model.safetensors"
    }:
        raise ValueError("adapted export identity has an unsupported schema or weight file")
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("weights_file_sha256", ""))):
        raise ValueError("adapted export identity has no valid weights-file SHA-256")
    if (roundtrip.get("passed") is not True or report.get("export_roundtrip") != roundtrip
            or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("tensor_sha256", "")))
            or identity.get("tensor_sha256") != roundtrip.get("weights_sha256")):
        raise ValueError("adapted export tensor identity differs from the original training/roundtrip proof")
    for field, name in (("training_report_sha256", "report"), ("roundtrip_json_sha256", "roundtrip"),
                        ("provenance_json_sha256", "provenance")):
        if identity.get(field) != sha256_file(paths[name]):
            raise ValueError(f"adapted export identity source SHA differs: {field}")
    if (not _official_source_chain(provenance) or provenance.get("artifact_role") != "adapted_export"
            or (provenance.get("training_checkpoint_sha256") is not None
                and identity.get("training_checkpoint_sha256") != provenance["training_checkpoint_sha256"])):
        raise ValueError("adapted export identity has an invalid original-model/checkpoint source chain")
    return identity, list(paths.values())


def _training_stage(root, path, payload, saved_config, config):
    provenance = payload.get("provenance", {})
    marked = _path_stage(root, [path, payload.get("output_dir", "")], provenance)
    target = _number(payload.get("max_steps"))
    if target is None:
        target = _number(saved_config.get("train", {}).get("max_steps"))
    step = _number(payload.get("step"))
    if marked:
        return marked, target
    if target is not None and target <= 20:
        return "smoke", target
    if target is not None and target <= 100:
        return "pilot", target
    expected = int(config.get("train", {}).get("max_steps", 1000))
    complete = payload.get("status") == "complete" and target is not None and step == target
    if (complete and target >= max(101, expected) and _real_identity(provenance, config)
            and payload.get("proof", {}).get("backbone_updated") is True):
        return "formal", target
    return "incomplete" if target and step is not None and step < target else "unverified", target


def _source(root, path, payload, kind):
    provenance = payload.get("provenance", {})
    row = {field: None for field in FIELDS}
    row.update({
        "source_type": kind, "status": payload.get("status", "recorded"),
        "source_json": _display(root, path), "source_sha256": sha256_file(path),
        "model_source": provenance.get("source"), "model_revision": provenance.get("revision"),
        "upstream_commit": provenance.get("upstream_commit"), "notes": "",
    })
    return row


def _linked_training(root, model, training):
    reference = str(model or "").replace("\\", "/")
    for directory, row in sorted(training.items(), key=lambda item: -len(item[0])):
        relative = directory.replace("\\", "/")
        if reference == relative or reference.startswith(relative + "/") or f"/{relative}/" in reference:
            return row
    resolved = _reference(root, model)
    if resolved:
        for directory, row in training.items():
            if resolved.is_relative_to(root / directory):
                return row
    return None


def _add_manifest_source(root, path, sources):
    key = _display(root, path)
    sources[key] = {"path": key, "sha256": sha256_file(path)}


def _format(value):
    if value is None or value == "":
        return "pending"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_format(v) for v in row) + " |" for row in rows),
    ])


def _formal_completion(root, config, rows, sources):
    """Verify the fixed 3x1000 / 12x256 protocol against durable artifacts.

    The status journal alone is insufficient: every expected result must also
    have a parsed row, matching journal metadata and consistent source hashes.
    This does not recompute alignments, HMM scores, model weights or losses.
    """
    seeds, generation_seeds = (42, 43, 44), (101, 102, 103)
    arms = ("baseline", *(f"train_seed_{seed}" for seed in seeds))
    expected = ["experiment_status", "data_audit", "data_snapshot"]
    expected += [f"train_{seed}" for seed in seeds]
    expected += [f"test_denoising_{arm}" for arm in arms]
    expected += [f"{kind}_{arm}_{seed}" for arm in arms for seed in generation_seeds
                 for kind in ("generate", "evaluate")]
    problems = {stage: [] for stage in expected}
    row_index = {row["source_json"]: row for row in rows}

    def problem(stage, message):
        problems[stage].append(message)

    def load(path, stage):
        try:
            value = _read(path)
            _add_manifest_source(root, path, sources)
            return value
        except (OSError, ValueError) as exc:
            problem(stage, f"{_display(root, path)}: {exc}")
            return {}

    def hash_matches(path, expected_hash, stage):
        if not path or not path.is_file() or not expected_hash:
            problem(stage, "required file or recorded hash is missing")
            return False
        if sha256_file(path) != expected_hash:
            problem(stage, f"source hash differs: {_display(root, path)}")
            return False
        return True

    def parsed(path, kind, stage):
        row = row_index.get(_display(root, path))
        if not row or row["source_type"] != kind:
            problem(stage, "required parsed result row is missing")
        return row

    status_path = root / "outputs/formal/experiment_status.json"
    journal = load(status_path, "experiment_status")
    if journal.get("status") != "complete" or journal.get("mode") != "formal":
        problem("experiment_status", "formal journal is not complete")
    completed = {}
    entries = journal.get("completed", [])
    if not isinstance(entries, list):
        problem("experiment_status", "completed stages must be a list")
        entries = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("result"), dict):
            problem("experiment_status", "invalid completed-stage metadata")
            continue
        name = entry.get("stage")
        if not isinstance(name, str):
            problem("experiment_status", "stage name must be a string")
            continue
        if name in completed:
            problem("experiment_status", f"duplicate stage: {name}")
        completed[name] = entry["result"]

    def journal_matches(stage, value):
        if not value or completed.get(stage) != value:
            problem(stage, "journal stage is missing or differs from its durable result")

    if completed.get("data_audit", {}).get("passed") is not True:
        problem("data_audit", "journal has no successful data audit")
    tc = config.get("train", {})
    processed = Path(config.get("data", {}).get("processed_dir", "data/processed"))
    processed = processed if processed.is_absolute() else root / processed
    reference = load(processed / "training_ref.json", "data_snapshot")
    split_manifest = load(processed / "split_manifest.json", "data_snapshot")
    hash_matches(processed / "split_manifest.json", reference.get("split_manifest_sha256"), "data_snapshot")
    for filename, digest in split_manifest.get("files", {}).items():
        hash_matches(_reference(root, processed / filename), digest, "data_snapshot")
    data_paths = {name: _reference(root, tc.get(f"{name}_fasta")) for name in ("train", "val", "test")}
    data_hashes = {name: sha256_file(path) if path and path.is_file() else None
                   for name, path in data_paths.items()}
    for name, digest in data_hashes.items():
        if digest is None:
            problem("data_snapshot", f"{name} split file is missing")
    if data_hashes["train"] != reference.get("train_fasta_sha256") or data_hashes["train"] is None:
        problem("data_snapshot", "training reference hash differs")
    hash_matches(_reference(root, config.get("hmm", {}).get("path")), reference.get("hmm_sha256"), "data_snapshot")

    parent = Path(tc.get("output_dir", "runs/pf00194/seed_42")).parent
    run_rows, export_identities = {}, {}
    for seed in seeds:
        stage = f"train_{seed}"
        path = root / parent / f"seed_{seed}" / "training_report.json"
        payload = load(path, stage)
        saved_config = load(path.parent / "config.json", stage)
        row = parsed(path, "training", stage)
        if (not row or row["stage"] != "formal" or row["training_seed"] != seed
                or row["step"] != 1000 or row["max_steps"] != 1000):
            problem(stage, "requires the matching seed and a completed formal 1000-step run")
        else:
            run_rows[seed] = row
        if saved_config.get("model") != config.get("model"):
            problem(stage, "saved model source/revision configuration differs from the report protocol")
        for name in ("train", "val"):
            if payload.get("data_sha256", {}).get(f"{name}_fasta") != data_hashes[name] or data_hashes[name] is None:
                problem(stage, f"{name} data hash differs")
        roundtrip = payload.get("export_roundtrip", {})
        exported_roundtrip = load(path.parent / "export_best/roundtrip.json", stage)
        if (roundtrip.get("passed") is not True
                or _number(roundtrip.get("max_logit_error_cpu_fp32")) is None
                or roundtrip.get("max_logit_error_cpu_fp32", -1) < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(roundtrip.get("weights_sha256", "")))
                or exported_roundtrip != roundtrip):
            problem(stage, "verified export metadata is missing")
        try:
            identity, identity_paths = _verified_export_identity(path.parent)
            export_identities[seed] = identity
            for identity_path in identity_paths:
                _add_manifest_source(root, identity_path, sources)
        except (OSError, ValueError) as error:
            problem(stage, f"adapted export file identity is not verified: {error}")
        journal_matches(stage, payload)

    def model_matches(payload, arm):
        if not _real_identity(payload.get("provenance", {}), config):
            return False
        model = payload.get("model")
        if arm == "baseline":
            # The formal baseline arm may record an offline HF cache path.
            # Its original pinned HF identity is already checked above.
            return _configured_official_source({"source": model}) and _linked_training(root, model, {
                _display(root, root / parent / f"seed_{s}"): row for s, row in run_rows.items()
            }) is None
        seed = int(arm.removeprefix("train_seed_"))
        linked = _linked_training(root, model, {
            _display(root, root / parent / f"seed_{s}"): row for s, row in run_rows.items()
        })
        identity = export_identities.get(seed, {})
        return (linked is not None and linked["training_seed"] == seed
                and bool(identity.get("weights_file_sha256"))
                and payload.get("provenance", {}).get("loaded_weights_sha256") == identity["weights_file_sha256"])

    schedules = {}
    for arm in arms:
        stage = f"test_denoising_{arm}"
        filename = "denoising_baseline.json" if arm == "baseline" else "denoising_adapted.json"
        path = root / "outputs/formal" / arm / "test_denoising" / filename
        payload = load(path, stage)
        row = parsed(path, "denoising", stage)
        if (not row or payload.get("split") != "test" or not model_matches(payload, arm)
                or payload.get("source_sha256") != data_hashes["test"] or data_hashes["test"] is None
                or payload.get("noise_seed") != tc.get("validation_seed", 9173) + 1
                or _number(payload.get("weighted_masked_denoising_loss")) is None
                or (_number(payload.get("masked_residues")) or 0) <= 0):
            problem(stage, "test model, data, noise seed or denoising metadata differs")
        journal_matches(stage, payload)
        for seed in generation_seeds:
            generation_stage, evaluation_stage = f"generate_{arm}_{seed}", f"evaluate_{arm}_{seed}"
            directory = root / "outputs/formal" / arm / f"seed_{seed}"
            manifest = load(directory / "manifest.json", generation_stage)
            schedule = load(directory / "length_schedule.json", generation_stage)
            shared = _reference(root, manifest.get("length_schedule_path"))
            hash_matches(shared, manifest.get("length_schedule_sha256"), generation_stage)
            if shared:
                shared_schedule = load(shared, generation_stage)
                if shared_schedule != schedule:
                    problem(generation_stage, "saved and shared length schedules differ")
            fasta = directory / "sequences.fasta"
            hash_matches(fasta, manifest.get("fasta_sha256"), generation_stage)
            if (manifest.get("status") != "complete" or manifest.get("seed") != seed
                    or manifest.get("sequences") != 256 or not model_matches(manifest, arm)
                    or schedule.get("seed") != seed or schedule.get("total_sequences") != 256
                    or len(schedule.get("lengths", [])) != 256
                    or schedule.get("source_sha256") != data_hashes["train"]
                    or manifest.get("max_iter") != config.get("generate", {}).get("max_iter", 100)
                    or manifest.get("batch_size") != config.get("generate", {}).get("batch_size", 1)
                    or manifest.get("sampling_strategy") != "gumbel_argmax"):
                problem(generation_stage, "model, seed, sampling parameters or 256-sequence budget differs")
            minimum = config.get("data", {}).get("min_length", 1)
            maximum = config.get("data", {}).get("max_length", 256)
            if any(not isinstance(length, int) or isinstance(length, bool)
                   or not minimum <= length <= maximum for length in schedule.get("lengths", [])):
                problem(generation_stage, "length schedule contains invalid or out-of-range lengths")
            if arm == "baseline":
                schedules[seed] = (manifest.get("length_schedule_sha256"), schedule)
            elif schedules.get(seed) != (manifest.get("length_schedule_sha256"), schedule):
                problem(generation_stage, "length schedule is not paired to its baseline")
            journal_matches(generation_stage, manifest)

            path = directory / "evaluation/metrics.json"
            metrics = load(path, evaluation_stage)
            row = parsed(path, "generation", evaluation_stage)
            if (not row or row["generation_seed"] != seed
                    or row["stage"] != ("baseline" if arm == "baseline" else "formal")
                    or _reference(root, metrics.get("input_fasta")) != fasta
                    or metrics.get("input_sha256") != manifest.get("fasta_sha256")
                    or metrics.get("counts", {}).get("emitted_fasta_records") != 256):
                problem(evaluation_stage, "parsed evaluation or input FASTA differs")
            for key in ("train_fasta_sha256", "hmm_sha256", "split_manifest_sha256"):
                if not reference.get(key) or metrics.get("training_reference", {}).get(key) != reference.get(key):
                    problem(evaluation_stage, f"frozen reference differs: {key}")
            valid_count = _number(metrics.get("counts", {}).get("valid_amino_acid_sequences"))
            if valid_count is None or not 0 <= valid_count <= 256:
                problem(evaluation_stage, "valid-sequence denominator is missing or invalid")
            for key in ("ga_pass_rate", "family_pass_rate", "excess_duplicate_rate",
                        "exact_training_match_rate", "mean_pairwise_distance", "mean_nearest_training_identity"):
                if not isinstance(metrics.get("metrics", {}).get(key), dict) or "value" not in metrics["metrics"][key]:
                    problem(evaluation_stage, f"metric metadata is missing: {key}")
                else:
                    value = _metric(metrics, key)
                    required_count = 2 if key == "mean_pairwise_distance" else 1
                    if ((valid_count is not None and valid_count >= required_count and value is None)
                            or (value is not None and not 0 <= value <= 1)):
                        problem(evaluation_stage, f"metric value is invalid for its denominator: {key}")
            journal_matches(evaluation_stage, metrics)
    missing = [stage for stage in expected if problems[stage]]
    has_progress = status_path.exists() or any(row["stage"] == "formal" for row in rows)
    return {
        "formal_status": "complete" if not missing else "partial" if has_progress else "pending",
        "missing_expected_stages": missing,
        "formal_validation_issues": {stage: problems[stage] for stage in missing},
        "verified_expected_stages": [stage for stage in expected if not problems[stage]],
    }


def build_report(config: dict, output_dir="artifacts/report") -> dict:
    """Summarize runs/ and outputs/ relative to the current project directory.

    Only the selected report directory is written. A formal training label
    requires a completed recorded budget, the pinned real-model identity and
    backbone-update evidence. Missing provenance remains unverified.
    """
    root = Path.cwd().resolve()
    output = Path(output_dir).resolve()
    candidates = set()
    for folder in (root / "runs", root / "outputs"):
        if folder.exists():
            candidates.update(p.resolve() for p in folder.rglob("*.json")
                              if not p.resolve().is_relative_to(output))
    rows, warnings, sources, training, export_identities = [], [], {}, {}, {}
    sidecar = Path(str(config.get("model", {}).get("source", ""))) / "family_lab_provenance.json"
    if sidecar.is_file():
        _add_manifest_source(root, sidecar.resolve(), sources)
    payloads = []
    for path in sorted(candidates):
        if path.name not in {"training_report.json", "metrics.json"} and not path.name.startswith(("denois", "test")):
            continue
        try:
            payload = _read(path)
            if path.name == "training_report.json":
                kind = "training"
            elif path.name == "metrics.json" and isinstance(payload.get("metrics"), dict):
                kind = "generation"
            elif payload.get("split") == "test" and "weighted_masked_denoising_loss" in payload:
                kind = "denoising"
            else:
                continue
            payloads.append((path, payload, kind))
            _add_manifest_source(root, path, sources)
        except (OSError, ValueError) as exc:
            warnings.append(f"{_display(root, path)}: {exc}")

    for path, payload, kind in payloads:
        if kind != "training":
            continue
        saved_config = {}
        config_path = path.parent / "config.json"
        if config_path.is_file():
            try:
                saved_config = _read(config_path)
                _add_manifest_source(root, config_path, sources)
            except (OSError, ValueError) as exc:
                warnings.append(f"{_display(root, config_path)}: {exc}")
        row = _source(root, path, payload, kind)
        stage, target = _training_stage(root, path, payload, saved_config, config)
        row.update({
            "stage": stage, "arm": "adapted", "step": _number(payload.get("step")),
            "max_steps": target, "training_seed": saved_config.get("train", {}).get("seed"),
            "denoising_loss": _number(payload.get("best_validation_loss")),
            "elapsed_seconds": _number(payload.get("elapsed_seconds_this_session")),
            "peak_gpu_allocated_bytes": _number(payload.get("peak_gpu_allocated_bytes")),
            "data_sha256": json.dumps(payload.get("data_sha256", {}), sort_keys=True),
            "notes": "validation loss; elapsed time is this session only",
        })
        if stage == "unverified":
            row["notes"] += "; formal budget/model identity/update proof is incomplete"
        rows.append(row)
        training[_display(root, path.parent)] = row
        if stage in {"formal", "pilot", "smoke"} and _real_identity(payload.get("provenance", {}), config):
            try:
                identity, identity_paths = _verified_export_identity(path.parent)
                export_identities[row["source_json"]] = identity
                for identity_path in identity_paths:
                    _add_manifest_source(root, identity_path, sources)
            except (OSError, ValueError) as error:
                warnings.append(f"{row['source_json']}: adapted export identity is not verified: {error}")

    for path, payload, kind in payloads:
        if kind == "training":
            continue
        manifest, manifest_path = {}, None
        if kind == "generation":
            fasta = _reference(root, payload.get("input_fasta"))
            if fasta and (fasta.parent / "manifest.json").is_file():
                manifest_path = fasta.parent / "manifest.json"
            elif (path.parent / "manifest.json").is_file():
                manifest_path = path.parent / "manifest.json"
            if manifest_path:
                try:
                    manifest = _read(manifest_path)
                    _add_manifest_source(root, manifest_path, sources)
                except (OSError, ValueError) as exc:
                    warnings.append(f"{_display(root, manifest_path)}: {exc}")
        evidence = manifest if manifest else payload
        row = _source(root, path, evidence, kind)
        model = evidence.get("model", payload.get("model"))
        linked = _linked_training(root, model, training)
        provenance = evidence.get("provenance", {})
        marked = _path_stage(root, [path, model or ""], provenance)
        baseline = model == config.get("model", {}).get("source") or (
            bool(model) and linked is None and _real_identity(provenance, config)
            and _marked(root, [path], "baseline")
        )
        stage = marked or (linked["stage"] if linked else "baseline" if baseline else "unverified")
        if stage == "baseline" and not _real_identity(provenance, config):
            stage = "unverified"
        row.update({
            "stage": stage, "arm": "baseline" if baseline else "adapted" if linked else "unknown",
            "training_seed": linked.get("training_seed") if linked else None,
            "generation_seed": evidence.get("seed") if kind == "generation" else None,
            "step": linked.get("step") if linked else None,
            "max_steps": linked.get("max_steps") if linked else None,
        })
        if linked and stage in {"formal", "pilot", "smoke"}:
            identity = export_identities.get(linked["source_json"], {})
            if (not identity.get("weights_file_sha256")
                    or provenance.get("loaded_weights_sha256") != identity["weights_file_sha256"]):
                row["stage"] = "unverified"
                row["notes"] = "adapted loaded weights do not match the sealed training export"
        if kind == "generation":
            counts = payload.get("counts", {})
            requested = _number(manifest.get("requested_sequences"))
            schedule_path = _reference(root, manifest.get("length_schedule_path"))
            if schedule_path is None and manifest_path:
                sibling = manifest_path.parent / "length_schedule.json"
                schedule_path = sibling if sibling.is_file() else None
            if schedule_path:
                try:
                    schedule = _read(schedule_path)
                    requested = _number(schedule.get("total_sequences"))
                    _add_manifest_source(root, schedule_path, sources)
                except (OSError, ValueError) as exc:
                    warnings.append(f"{_display(root, schedule_path)}: {exc}")
            passed = _number(counts.get("family_pass"))
            row.update({
                "emitted_sequences": _number(counts.get("emitted_fasta_records")),
                "valid_sequences": _number(counts.get("valid_amino_acid_sequences")),
                "requested_sequences": requested,
                "overall_family_yield": passed / requested if passed is not None and requested and requested > 0 else None,
                "elapsed_seconds": _number(manifest.get("elapsed_seconds")),
                "peak_gpu_allocated_bytes": _number(manifest.get("peak_gpu_allocated_bytes")),
                "data_sha256": json.dumps(payload.get("training_reference", {}), sort_keys=True),
                "notes": "; ".join(filter(None, [row["notes"], payload.get("denominator", "denominator unavailable; see source")])),
            })
            for metric in ("ga_pass_rate", "family_pass_rate", "excess_duplicate_rate", "exact_training_match_rate",
                           "mean_pairwise_distance", "mean_nearest_training_identity"):
                row[metric] = _metric(payload, metric)
            if not manifest:
                row["notes"] += "; no generation manifest: model and resource provenance unavailable"
            expected_count = int(config.get("generate", {}).get("total_sequences", 256))
            if row["stage"] == "formal":
                if not _real_identity(provenance, config):
                    row["stage"] = "unverified"
                    row["notes"] += "; generation model identity is not verified"
                elif requested is not None and requested < expected_count:
                    row["stage"] = "pilot"
                    row["notes"] += "; generation count below the formal configured budget"
                elif (requested is None or row["emitted_sequences"] != requested
                      or manifest.get("status") != "complete"):
                    row["stage"] = "incomplete"
                    row["notes"] += "; generation budget/completion is incomplete"
        else:
            row.update({
                "denoising_loss": _number(payload.get("weighted_masked_denoising_loss")),
                "data_sha256": payload.get("source_sha256"),
                "notes": "; ".join(filter(None, [row["notes"], "fixed-noise test denoising CE; not exact perplexity"])),
            })
        rows.append(row)

    formal = sum(row["stage"] == "formal" and row["source_type"] == "training" for row in rows)
    completion = _formal_completion(root, config, rows, sources)
    result = {
        "schema_version": 1, "generated_at_utc": utc_now(),
        "status": "available" if rows else "pending",
        **completion,
        "formal_training_runs": formal, "recorded_rows": len(rows),
        "rows": rows, "sources": list(sources.values()), "warnings": warnings,
        "limitations": LIMITATIONS,
        "artifacts": {"summary_csv": str(output / "summary.csv"), "report": str(output / "report.md"),
                      "manifest": str(output / "summary.json")},
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    evidence_rows = [[r["source_type"], r["stage"], r["source_json"], r["status"],
                      f"{_format(r['step'])}/{_format(r['max_steps'])}" if r["source_type"] == "training" else "—",
                      r["model_revision"], r["elapsed_seconds"],
                      r["peak_gpu_allocated_bytes"] / (1024 ** 3) if r["peak_gpu_allocated_bytes"] is not None else None]
                     for r in rows]
    metric_rows = [[r["source_json"], r["valid_sequences"], r["ga_pass_rate"], r["family_pass_rate"], r["overall_family_yield"],
                    r["excess_duplicate_rate"], r["mean_pairwise_distance"], r["mean_nearest_training_identity"],
                    r["denoising_loss"]] for r in rows]
    parts = ["# DPLM Family Lab：已有结果汇总", "",
             f"结果状态：**{result['status']}**；正式协议状态：**{result['formal_status']}**（{formal}次正式训练已记录）。",
             "", "只汇总磁盘上实际存在的结果，不把tiny、smoke或pilot升级为正式实验；缺失值显示pending。",
             "只有正式状态记录、三次1000步训练、12组256条配对生成评估与四组测试去噪的元数据和hash一致，才标记complete。",
             "", "## 正式协议尚缺阶段", "",
             ", ".join(result["missing_expected_stages"]) if result["missing_expected_stages"] else "无；预期阶段及来源一致性检查已通过。",
             "", "## 运行与成本", "",
             _table(["类型", "阶段", "来源", "状态", "步数/目标", "实际权重revision", "秒/本会话", "峰值GiB"], evidence_rows)
             if rows else "pending：尚无可汇总的训练、生成评估或测试去噪结果。",
             "", "## 指标", "",
             _table(["来源", "有效N", "GA率", "HMM家族率", "请求总体产出率", "重复率", "配对距离", "最近训练identity", "去噪CE"], metric_rows)
             if rows else "pending", "",
             "精确训练匹配率及完整字段见summary.csv；各来源JSON保留原始区间与分母，不在此把区间重新解释成显著性。",
             "", "## 统计与生物学边界", "", *(f"- {note}" for note in LIMITATIONS),
             "", "## 来源与SHA-256", "",
             _table(["来源", "SHA-256"], [[s["path"], s["sha256"]] for s in sources.values()]) if sources else "pending"]
    if warnings:
        parts.extend(["", "## 未读取或未验证项", "", *(f"- {warning}" for warning in warnings)])
    (output / "report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    write_json(output / "summary.json", result)
    return result
