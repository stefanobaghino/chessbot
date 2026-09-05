"""Checkpoint/resume and window gating of train/train.py."""

import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import train


def make_data(path: Path, n: int = 3000) -> None:
    rng = np.random.default_rng(1)
    pieces = np.zeros((n, 64), dtype=np.uint8)
    for i in range(n):
        squares = rng.choice(64, size=8, replace=False)
        pieces[i, squares] = rng.integers(1, 13, size=8)
    np.savez(path, pieces=pieces, stm=rng.integers(0, 2, size=n).astype(np.uint8), score=rng.normal(0, 200, size=n))


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "train" / "train.py"), *args], capture_output=True, text=True, check=False)


def test_resume_continues_from_the_checkpoint(tmp_path: Path) -> None:
    data = tmp_path / "d.npz"
    make_data(data)
    out = str(tmp_path / "net.bin")
    base = [str(data), out, "--hidden", "8", "--batch", "512", "--threads", "1"]
    two = run(base + ["--epochs", "2"])
    assert two.returncode == 0, two.stderr
    assert "epoch 2/2" in two.stdout
    ckpt = Path(out + ".ckpt")
    assert ckpt.exists()
    # A different run (more epochs) must not silently pick up this checkpoint.
    four = run(base + ["--epochs", "4"])
    assert four.returncode != 0 and "different run" in four.stderr
    # The same command again resumes after epoch 2 and has nothing left to do.
    again = run(base + ["--epochs", "2"])
    assert again.returncode == 0 and "nothing left to do" in again.stdout, again.stdout + again.stderr
    # Resuming a partial run: fake a checkpoint after epoch 1 of 2 and check epoch 2 runs once.
    import torch

    ck = torch.load(ckpt, weights_only=False)
    ck["epoch"] = 1
    torch.save(ck, ckpt)
    resumed = run(base + ["--epochs", "2"])
    assert resumed.returncode == 0, resumed.stderr
    assert "resumed from" in resumed.stdout and "epoch 2/2" in resumed.stdout and "epoch 1/2 train" not in resumed.stdout


def test_window_stops_before_an_epoch_that_would_not_fit(tmp_path: Path) -> None:
    data = tmp_path / "d.npz"
    make_data(data)
    out = str(tmp_path / "net.bin")
    hour = dt.datetime.now().astimezone().hour
    closed = f"{(hour + 2) % 24}-{(hour + 3) % 24}"  # a window that is not open right now
    res = run([str(data), out, "--hidden", "8", "--epochs", "1", "--batch", "512", "--threads", "1", "--window", closed])
    assert res.returncode == 3, res.stdout + res.stderr
    assert "would not finish inside" in res.stdout
    assert not Path(out + ".ckpt").exists()


def test_window_allows_uses_the_epoch_estimate() -> None:
    at = dt.datetime(2026, 9, 5, 20, 30).astimezone()
    assert train.window_allows((9, 21), 20 * 60, at)
    assert not train.window_allows((9, 21), 40 * 60, at)
    assert not train.window_allows((9, 21), 0, dt.datetime(2026, 9, 5, 8, 59).astimezone())
    assert train.parse_window(None) is None
    assert train.parse_window("9-21") == (9, 21)
