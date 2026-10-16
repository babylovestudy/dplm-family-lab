# ruff: noqa: E402
from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from dplm_family_lab.backend import create_tiny_fixture
from dplm_family_lab.common import SequenceRecord, read_fasta, seed_everything, write_fasta
from dplm_family_lab.sampling import _batch_seed, generate, length_schedule


def test_run_seeds_do_not_reuse_adjacent_batch_streams():
    streams = [{_batch_seed(seed, index) for index in range(256)} for seed in (101, 102, 103)]
    assert all(len(stream) == 256 for stream in streams)
    assert not (streams[0] & streams[1] or streams[0] & streams[2] or streams[1] & streams[2])


def test_seeded_generation_schedule_and_fixed_segments(tmp_path):
    torch.set_num_threads(1)
    seed_everything(4)
    source = create_tiny_fixture(tmp_path / "weights")
    fasta = tmp_path / "train.fasta"
    write_fasta(
        [SequenceRecord("x", "ACDEFGHIKLMNPQ"), SequenceRecord("y", "RSTVWYACDEFGHIKLM")], fasta
    )
    config = {
        "model": {"source": str(source), "allow_tiny_fixture": True},
        "train": {"device": "cpu"},
        "generate": {
            "output_dir": str(tmp_path / "arm_a"),
            "lengths_source": str(fasta),
            "lengths_file": str(tmp_path / "shared.json"),
            "seed": 101,
            "total_sequences": 4,
            "batch_size": 1,
            "max_iter": 3,
            "fixed_segments": [{"start": 1, "sequence": "ACD"}],
        },
    }
    first = generate(config)
    second = generate(config, output_dir=tmp_path / "arm_b")
    assert first["length_schedule_sha256"] == second["length_schedule_sha256"]
    assert first["fasta_sha256"] == second["fasta_sha256"]
    rows = read_fasta(first["fasta"])
    assert len(rows) == 4 and all(row.sequence[1:4] == "ACD" for row in rows)
    assert [len(row.sequence) for row in rows] == length_schedule(config)[0]["lengths"]
    with pytest.raises(FileExistsError):
        generate(config)
    changed = deepcopy(config)
    changed["generate"]["seed"] = 102
    with pytest.raises(ValueError, match="signature"):
        length_schedule(changed)


def test_generation_rejects_out_of_bounds_segments(tmp_path):
    torch.set_num_threads(1)
    source = create_tiny_fixture(tmp_path / "weights")
    fasta = tmp_path / "source.fasta"
    write_fasta([SequenceRecord("x", "ACDEFGHIK")], fasta)
    config = {
        "model": {"source": str(source), "allow_tiny_fixture": True},
        "train": {"device": "cpu"},
        "generate": {
            "output_dir": str(tmp_path / "generated"),
            "lengths_source": str(fasta),
            "lengths_file": str(tmp_path / "schedule.json"),
            "total_sequences": 1,
            "max_iter": 2,
            "fixed_segments": [{"start": 8, "sequence": "ACD"}],
        },
    }
    with pytest.raises(ValueError, match="outside"):
        generate(config)
