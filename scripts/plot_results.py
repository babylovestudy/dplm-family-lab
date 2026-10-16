"""Plot measured fixed-noise validation and generation results without inference tests.

Usage: python scripts/plot_results.py --mode pilot
       python scripts/plot_results.py --mode formal --allow-incomplete

Requires the optional analysis dependencies. Only completed, measured input
files are used; no training loss is substituted for validation loss.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys


LOSS = "weighted_masked_denoising_loss"
COLORS = {"baseline": "#62676D", "train_seed_42": "#0072B2",
          "train_seed_43": "#009E73", "train_seed_44": "#D55E00"}
MARKERS = {101: "o", 102: "s", 103: "^"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path, root):
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {path.name}: {error}") from error


def finite(value, label, probability=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite measured value: {label}")
    if probability and not 0 <= value <= 1:
        raise ValueError(f"Measured value outside [0,1]: {label}")
    return float(value)


def load_inputs(root, mode, allow_incomplete):
    training_seeds = [42] if mode == "pilot" else [42, 43, 44]
    sampling_seeds = [101] if mode == "pilot" else [101, 102, 103]
    expected_steps, expected_count = (100, 32) if mode == "pilot" else (1000, 256)
    sources, issues, curves, generations = {}, [], [], []

    def source(path, kind):
        key = relative(path, root)
        sources[key] = {"path": key, "sha256": sha256(path), "kind": kind}

    for seed in training_seeds:
        name = f"pilot_seed_{seed}" if mode == "pilot" else f"seed_{seed}"
        run = root / "runs/pf00194" / name
        paths = {"baseline": run / "baseline_validation.json", "log": run / "training.jsonl",
                 "report": run / "training_report.json"}
        absent = [relative(p, root) for p in paths.values() if not p.is_file()]
        if absent:
            issues.extend(f"Missing: {path}" for path in absent)
            continue
        report = read_json(paths["report"])
        if report.get("status") != "complete" or report.get("step", 0) < expected_steps:
            issues.append(f"Incomplete training run: {relative(run, root)}")
            continue
        noise_digest = report.get("validation_noise_sha256", "")
        validation_digest = report.get("data_sha256", {}).get("val_fasta", "")
        if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
               for value in (noise_digest, validation_digest)):
            raise ValueError(f"Missing fixed-noise or validation-data SHA-256 provenance: {relative(paths['report'], root)}")
        baseline = finite(read_json(paths["baseline"]).get(LOSS), f"seed {seed} baseline")
        rows = []
        try:
            for line_number, line in enumerate(paths["log"].read_text(encoding="utf-8-sig").splitlines(), 1):
                if line.strip():
                    rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid completed training log {relative(paths['log'], root)}, line {line_number}") from error
        validation = []
        for row in rows:
            if "validation" not in row:
                continue
            step = row.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise ValueError(f"Invalid validation step in {relative(paths['log'], root)}")
            validation.append((step, finite(row["validation"].get(LOSS), f"seed {seed} validation step {step}")))
        if not validation or max(step for step, _ in validation) < expected_steps:
            issues.append(f"No completed {expected_steps}-step validation curve: {relative(run, root)}")
            continue
        if len({step for step, _ in validation}) != len(validation):
            raise ValueError(f"Duplicate validation steps in {relative(paths['log'], root)}")
        curves.append({"seed": seed, "baseline": baseline, "points": sorted(validation),
                       "noise_sha256": report.get("validation_noise_sha256"),
                       "validation_fasta_sha256": report.get("data_sha256", {}).get("val_fasta")})
        for kind, path in paths.items():
            source(path, f"training_{kind}")
    for field in ("noise_sha256", "validation_fasta_sha256"):
        values = {curve[field] for curve in curves if curve[field]}
        if len(values) > 1:
            raise ValueError(f"Training curves use different {field}; fixed-noise comparison is invalid")
    arms = ["baseline"] + [f"train_seed_{seed}" for seed in training_seeds]
    for arm in arms:
        for seed in sampling_seeds:
            path = root / "outputs" / mode / arm / f"seed_{seed}" / "evaluation/metrics.json"
            if not path.is_file():
                issues.append(f"Missing: {relative(path, root)}")
                continue
            record = read_json(path)
            counts = record.get("counts", {})
            if counts.get("emitted_fasta_records") != expected_count:
                issues.append(f"Wrong or incomplete sampling budget: {relative(path, root)}; expected {expected_count}")
                continue
            valid = counts.get("valid_amino_acid_sequences", 0)
            if not isinstance(valid, int) or valid <= 0:
                issues.append(f"No valid emitted sequences: {relative(path, root)}")
                continue
            metrics = record.get("metrics", {})
            values = {key: finite(metrics.get(key, {}).get("value"), f"{arm}/{seed}/{key}", True)
                      for key in ("family_pass_rate", "mean_nearest_training_identity", "mean_pairwise_distance")}
            if metrics["family_pass_rate"].get("n") != valid:
                raise ValueError(f"Family-rate denominator differs from valid emitted count: {relative(path, root)}")
            passed = counts.get("family_pass")
            if not isinstance(passed, int) or not 0 <= passed <= valid or abs(passed / valid - values["family_pass_rate"]) > 1e-12:
                raise ValueError(f"Family-rate numerator differs from emitted counts: {relative(path, root)}")
            generations.append({"arm": arm, "sampling_seed": seed, "valid_sequences": valid, **values})
            generations[-1]["family_pass_count"] = passed
            source(path, "generation_evaluation")
    if issues and not allow_incomplete:
        excerpt = "\n".join(issues[:15])
        if len(issues) > 15:
            excerpt += f"\n... and {len(issues) - 15} more missing/incomplete inputs."
        raise ValueError(f"Required {mode} results are incomplete:\n{excerpt}\nUse --allow-incomplete to label and plot completed inputs only.")
    if not curves or not generations:
        raise ValueError("At least one completed training curve and one completed generation evaluation are required for the two figures")
    return {"mode": mode, "curves": curves, "generations": generations, "issues": issues,
            "sources": list(sources.values()), "arms": arms, "sampling_seeds": sampling_seeds,
            "expected_steps": expected_steps, "expected_sequences_per_sampling_seed": expected_count}


def arm_label(arm):
    return "Pretrained baseline" if arm == "baseline" else f"Training seed {arm.rsplit('_', 1)[1]}"


def plot_figures(data, output, root, dpi):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib. Install the project analysis extra first.") from error
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.titleweight": "bold"})
    output.mkdir(parents=True, exist_ok=True)
    status = data["mode"].upper() + (" | INCOMPLETE" if data["issues"] else "")
    output_files = []

    def save(figure, stem):
        for suffix in ("png", "svg"):
            path = output / f"{data['mode']}_{stem}.{suffix}"
            figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight",
                           metadata={"Title": f"DPLM family lab: {stem} ({status})"})
            output_files.append({"path": relative(path, root), "sha256": sha256(path)})
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    baselines = [curve["baseline"] for curve in data["curves"]]
    common_baseline = max(baselines) - min(baselines) <= 1e-12
    if common_baseline:
        axis.axhline(baselines[0], color=COLORS["baseline"], linestyle="--", linewidth=1.6,
                     label=f"Pretrained baseline ({baselines[0]:.4f})")
    for curve in data["curves"]:
        color = COLORS[f"train_seed_{curve['seed']}"]
        if not common_baseline:
            axis.axhline(curve["baseline"], color=color, linestyle="--", alpha=0.65,
                         label=f"Seed {curve['seed']} baseline ({curve['baseline']:.4f})")
        points = [(0, curve["baseline"])] + curve["points"]
        axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=4,
                  color=color, linewidth=1.8, label=f"Training seed {curve['seed']}")
    axis.set(title=f"Fixed-noise validation denoising loss | {status}",
             xlabel="Optimizer step", ylabel="Weighted masked denoising CE (validation)")
    axis.grid(alpha=0.18)
    axis.legend(fontsize=9)
    caption = "Validation values only; the fixed validation noise bank is held constant. Lower is better.\nWeighted masked denoising CE is not exact sequence likelihood or perplexity."
    if data["mode"] == "pilot":
        caption += "\nPILOT: one 100-step training run; this is a workflow check."
    if data["issues"]:
        caption += f"\nINCOMPLETE: {len(data['issues'])} required inputs missing or unfinished; only completed runs are shown."
    figure.text(0.1, 0.025, caption, fontsize=8.5, va="bottom", color="#444444")
    figure.subplots_adjust(bottom=0.25, top=0.88, left=0.11)
    save(figure, "validation_loss")

    grouped = defaultdict(list)
    for record in data["generations"]:
        grouped[record["arm"]].append(record)
    arms = [arm for arm in data["arms"] if grouped[arm]]
    figure, (family_axis, tradeoff_axis) = plt.subplots(1, 2, figsize=(12.2, 6.4))
    for index, arm in enumerate(arms):
        records = sorted(grouped[arm], key=lambda row: row["sampling_seed"])
        color = COLORS[arm]
        offsets = [0.0] if len(records) == 1 else [-0.13 + 0.26 * i / (len(records) - 1) for i in range(len(records))]
        for offset, record in zip(offsets, records):
            marker = MARKERS[record["sampling_seed"]]
            family_axis.scatter(index + offset, record["family_pass_rate"] * 100,
                                color=color, marker=marker, s=48, alpha=0.85, zorder=3)
            tradeoff_axis.scatter(record["mean_nearest_training_identity"], record["mean_pairwise_distance"],
                                  color=color, marker=marker, s=45, alpha=0.8, zorder=3)
        mean_rate = sum(r["family_pass_rate"] for r in records) / len(records)
        family_axis.plot([index - 0.23, index + 0.23], [mean_rate * 100] * 2, color=color, linewidth=2.5)
        if data["mode"] == "pilot" and len(records) == 1:
            record = records[0]
            family_axis.annotate(f"{record['family_pass_count']}/{record['valid_sequences']} ({mean_rate * 100:.1f}%)",
                                 (index, mean_rate * 100), xytext=(0, 10), textcoords="offset points",
                                 ha="center", fontsize=9, color=color)
        mean_nearest = sum(r["mean_nearest_training_identity"] for r in records) / len(records)
        mean_diversity = sum(r["mean_pairwise_distance"] for r in records) / len(records)
        tradeoff_axis.scatter(mean_nearest, mean_diversity, color=color, marker="D", s=83,
                              edgecolors="black", linewidths=0.7, zorder=4)
    family_axis.set_xticks(range(len(arms)), [arm_label(arm).replace(" baseline", "\nbaseline").replace(" seed ", "\nseed ") for arm in arms])
    family_axis.set(title="A  Family compatibility", ylabel="GA + domain-coverage pass rate (%)", ylim=(-3, 103))
    family_axis.grid(axis="y", alpha=0.18)
    seed_handles = [Line2D([], [], color="#555555", marker=MARKERS[seed], linestyle="None",
                           label=f"Sampling seed {seed}") for seed in data["sampling_seeds"]]
    seed_handles.append(Line2D([], [], color="#333333", linewidth=2, label="Arm mean"))
    family_axis.legend(handles=seed_handles, fontsize=7.5, loc="best")
    tradeoff_axis.set(title="B  Similarity and diversity", xlabel="Mean nearest-training identity",
                      ylabel="Mean pairwise diversity (1 - identity)", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    tradeoff_axis.grid(alpha=0.18)
    tradeoff_axis.legend(handles=[Line2D([], [], color=COLORS[arm], marker="D", linestyle="None", label=arm_label(arm)) for arm in arms], fontsize=7.5, loc="best")
    figure.suptitle(f"Generation outcomes | {status}", fontsize=14, fontweight="bold")
    if data["mode"] == "pilot":
        caption = "PILOT: 32 requested sequences per arm, one sampling seed (101), one 100-step training run.\n"
    else:
        caption = "Points: sampling seeds 101-103; 256 sequences requested per sampling seed. Lines/diamonds: arm means.\nThree sampling seeds per arm are not three independent training runs; each training seed is shown separately.\n"
    caption += "Rates use all valid emitted sequences, including duplicates; HMM GA and >=80% single-domain coverage measure family compatibility.\nNearest identity and diversity describe a trade-off; biochemical activity, folding and expression remain untested. No p-values are inferred."
    if data["issues"]:
        caption += f"\nINCOMPLETE: {len(data['issues'])} required inputs missing or unfinished; means use the available completed sampling seeds."
    figure.text(0.075, 0.025, caption, fontsize=8.1, va="bottom", color="#444444")
    figure.subplots_adjust(bottom=0.29, top=0.86, left=0.08, right=0.98, wspace=0.28)
    save(figure, "generation_metrics")
    manifest = {"schema_version": 1, "mode": data["mode"], "incomplete": bool(data["issues"]),
                "plot_script_sha256": sha256(__file__),
                "issues": data["issues"], "sources": data["sources"], "outputs": output_files,
                "training_seeds_plotted": [curve["seed"] for curve in data["curves"]],
                "generation_cells_plotted": data["generations"],
                "expected_sampling_seeds": data["sampling_seeds"],
                "expected_sequences_per_sampling_seed": data["expected_sequences_per_sampling_seed"],
                "matplotlib_version": matplotlib.__version__,
                "notes": ["Training loss is never plotted as validation loss.",
                          "Sampling seeds within an arm do not constitute independent training replicates.",
                          "No significance test or biochemical-function inference is performed."]}
    manifest_path = output / f"manifest_{data['mode']}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"outputs": output_files, "manifest": relative(manifest_path, root), "incomplete": bool(data["issues"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("formal", "pilot"), default="formal")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        if args.dpi < 72:
            raise ValueError("--dpi must be at least 72")
        root = args.root.resolve()
        output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
        data = load_inputs(root, args.mode, args.allow_incomplete)
        print(json.dumps(plot_figures(data, output, root, args.dpi), indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
