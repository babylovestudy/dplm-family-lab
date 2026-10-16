"""Read recorded evidence without running or changing any experiment.

The report is a metadata summary, not a rerun or a claim that the full protocol
has finished. Tiny fixtures and short runs can never establish a formal result.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

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
    for anchor in ("runs/", "outputs/"):
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


def _real_identity(provenance, config):
    expected = config.get("model", {})
    return (
        provenance.get("tiny_fixture") is not True
        and bool(expected.get("source"))
        and provenance.get("source") == expected.get("source")
        and bool(expected.get("revision"))
        and provenance.get("revision") == expected.get("revision")
        and (_number(provenance.get("parameter_count")) or 0) >= 100_000_000
    )


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
    rows, warnings, sources, training = [], [], {}, {}
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
        baseline = model == config.get("model", {}).get("source")
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
                "notes": payload.get("denominator", "denominator unavailable; see source"),
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
                "notes": "fixed-noise test denoising CE; not exact perplexity",
            })
        rows.append(row)

    formal = sum(row["stage"] == "formal" and row["source_type"] == "training" for row in rows)
    result = {
        "schema_version": 1, "generated_at_utc": utc_now(),
        "status": "available" if rows else "pending",
        "formal_status": "partial" if formal else "pending",
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
             f"结果状态：**{result['status']}**；正式训练证据：**{result['formal_status']}**（{formal}次已记录）。",
             "", "只汇总磁盘上实际存在的结果，不把tiny、smoke或pilot升级为正式实验；缺失值显示pending。",
             "正式结果出现也不自动代表三训练种子×三生成种子的完整协议已结束。", "", "## 运行与成本", "",
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
