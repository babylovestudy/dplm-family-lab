"""Synthetic operators test scheduling only; no model or biological results."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading

import pytest

from dplm_family_lab import data, evaluation, experiment, training
from dplm_family_lab.common import write_json


def config(workers=8):
    return {"train": {"output_dir": "runs/pf00194/seed_42", "max_steps": 1000,
                      "validation_seed": 9173, "test_fasta": "test.fasta"},
            "evaluation": {"workers": workers}, "generate": {},
            "model": {"source": "synthetic-scheduling-test-only"}}


def operators(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    events = []
    monkeypatch.setattr(data, "audit", lambda _: {"passed": True, "synthetic_test": True})
    monkeypatch.setattr(experiment, "_training_run", lambda cfg: {"status": "complete", "seed": cfg["train"]["seed"]})
    monkeypatch.setattr(training, "evaluate_denoising", lambda cfg, model_path=None: {"synthetic_test": True})

    def generate(cfg, output, model_path=None):
        output = Path(output)
        events.append(("generate", output.parent.name, cfg["generate"]["seed"], cfg["generate"]["total_sequences"]))
        output.mkdir(parents=True, exist_ok=True)
        (output / "sequences.fasta").write_text(">synthetic_test_only\nACDE\n")
        manifest = {"synthetic_test": True, "sequences": cfg["generate"]["total_sequences"]}
        write_json(output / "manifest.json", manifest)
        return manifest

    def evaluate(cfg, fasta_path, output_dir):
        path = Path(fasta_path)
        events.append(("evaluate", path.parent.parent.name, cfg["generate"]["seed"]))
        result = {"synthetic_test": True, "source": path.as_posix()}
        write_json(Path(output_dir) / "metrics.json", result)
        return result

    monkeypatch.setattr(experiment, "_generation_run", generate)
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    return events, generate, evaluate


def wrapped_job(evaluate):
    def job(cfg, fasta_path, output_dir):
        return {"metrics": evaluate(cfg, fasta_path, output_dir),
                "execution": {"synthetic_test": True, "torch_imported": False}}
    return job


def test_overlap_generates_while_evaluation_runs_and_waits_for_last(monkeypatch, tmp_path):
    events, generate, evaluate = operators(monkeypatch, tmp_path)
    first_started, release_first = threading.Event(), threading.Event()
    last_started, release_last = threading.Event(), threading.Event()
    journal_writers = set()
    snapshots = []
    original_config = config()
    before = copy.deepcopy(original_config)

    def record_json(path, value):
        if Path(path).name == "experiment_status.json":
            journal_writers.add(threading.get_ident())
            snapshots.append(copy.deepcopy(value))
        write_json(path, value)

    def generate_overlap(cfg, output, model_path=None):
        if Path(output).as_posix().endswith("baseline/seed_102"):
            assert first_started.wait(5), "Generation 102 must overlap evaluation 101"
            release_first.set()
        return generate(cfg, output, model_path)

    def job(cfg, fasta_path, output_dir):
        path = Path(fasta_path).parent.as_posix()
        if path.endswith("baseline/seed_101"):
            first_started.set()
            assert release_first.wait(5)
        if path.endswith("train_seed_44/seed_103"):
            last_started.set()
            assert release_last.wait(5)
        return wrapped_job(evaluate)(cfg, fasta_path, output_dir)

    monkeypatch.setattr(experiment, "write_json", record_json)
    monkeypatch.setattr(experiment, "_generation_run", generate_overlap)
    monkeypatch.setattr(experiment, "_cpu_evaluation_job", job)
    monkeypatch.setattr(experiment, "_new_evaluation_executor", lambda: ThreadPoolExecutor(max_workers=1))
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(experiment.run_experiment, original_config)
        assert last_started.wait(10)
        current = json.loads(Path("outputs/formal/experiment_status.json").read_text())
        assert current["status"] == "running"
        assert not future.done()
        release_last.set()
        result = future.result(timeout=10)
    assert result["status"] == "complete"
    assert original_config == before
    assert len(journal_writers) == 1
    assert all(snapshot["status"] != "complete" for snapshot in snapshots[:-1])
    assert len([entry for entry in result["completed"] if entry["stage"].startswith("evaluate_")]) == 12
    assert all(task["status"] == "complete" for task in result["execution"]["evaluation_tasks"])
    assert [(event[1], event[2]) for event in events if event[0] == "generate"] == [
        (arm, seed) for arm in ("baseline", "train_seed_42", "train_seed_43", "train_seed_44") for seed in (101, 102, 103)]
    assert all(event[3] == 256 for event in events if event[0] == "generate")


def test_failed_evaluation_retains_generated_evidence_and_never_completes(monkeypatch, tmp_path):
    _, generate, evaluate = operators(monkeypatch, tmp_path)
    second_generated, evaluation_failed = threading.Event(), threading.Event()

    def generate_overlap(cfg, output, model_path=None):
        result = generate(cfg, output, model_path)
        if Path(output).as_posix().endswith("baseline/seed_102"):
            second_generated.set()
            assert evaluation_failed.wait(5)
        return result

    def failed_job(cfg, fasta_path, output_dir):
        if Path(fasta_path).parent.as_posix().endswith("baseline/seed_101"):
            assert second_generated.wait(5)
            evidence = Path(output_dir) / "failure.json"
            write_json(evidence, {"synthetic_failure_test": True})
            evaluation_failed.set()
            raise RuntimeError("synthetic evaluation failure")
        return wrapped_job(evaluate)(cfg, fasta_path, output_dir)

    monkeypatch.setattr(experiment, "_generation_run", generate_overlap)
    monkeypatch.setattr(experiment, "_cpu_evaluation_job", failed_job)
    monkeypatch.setattr(experiment, "_new_evaluation_executor", lambda: ThreadPoolExecutor(max_workers=1))
    with pytest.raises(RuntimeError, match="CPU evaluation failed"):
        experiment.run_experiment(config())
    status = json.loads(Path("outputs/formal/experiment_status.json").read_text())
    assert status["status"] == "failed"
    assert "finished_at" not in status
    assert status["failed_stages"][0]["stage"] == "evaluate_baseline_101"
    assert Path("outputs/formal/baseline/seed_102/manifest.json").exists()
    assert Path("outputs/formal/baseline/seed_101/evaluation/failure.json").exists()
    # After fixing the operator the same durable generation namespace can resume.
    monkeypatch.setattr(experiment, "_cpu_evaluation_job", wrapped_job(evaluate))
    result = experiment.run_experiment(config())
    assert result["status"] == "complete"
    assert Path("outputs/formal/baseline/seed_101/evaluation/failure.json").exists()


def test_generation_failure_still_records_completed_pending_evaluation(monkeypatch, tmp_path):
    _, generate, evaluate = operators(monkeypatch, tmp_path)
    release = threading.Event()

    def generate_then_fail(cfg, output, model_path=None):
        if Path(output).as_posix().endswith("baseline/seed_102"):
            release.set()
            raise RuntimeError("synthetic generation failure")
        return generate(cfg, output, model_path)

    def job(cfg, fasta_path, output_dir):
        assert release.wait(5)
        return wrapped_job(evaluate)(cfg, fasta_path, output_dir)

    monkeypatch.setattr(experiment, "_generation_run", generate_then_fail)
    monkeypatch.setattr(experiment, "_cpu_evaluation_job", job)
    monkeypatch.setattr(experiment, "_new_evaluation_executor", lambda: ThreadPoolExecutor(max_workers=1))
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        experiment.run_experiment(config())
    status = json.loads(Path("outputs/formal/experiment_status.json").read_text())
    assert status["status"] == "failed"
    assert any(entry["stage"] == "evaluate_baseline_101" for entry in status["completed"])


@pytest.mark.parametrize("pilot,workers", [(True, 8), (False, 1)])
def test_pilot_and_single_worker_keep_original_sequential_schedule(monkeypatch, tmp_path, pilot, workers):
    events, _, _ = operators(monkeypatch, tmp_path)

    def forbidden_pool():
        raise AssertionError("Sequential protocol must not create an evaluation process")

    monkeypatch.setattr(experiment, "_new_evaluation_executor", forbidden_pool)
    result = experiment.run_experiment(config(workers), pilot=pilot)
    assert result["status"] == "complete"
    assert result["execution"]["evaluation_strategy"] == "sequential"
    for i, event in enumerate(events):
        if event[0] == "generate":
            assert events[i + 1][:3] == ("evaluate", event[1], event[2])
            assert event[3] == (32 if pilot else 256)
