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


def test_holdout_loss_is_reported_and_only_improvements_are_exported(tmp_path: Path) -> None:
    data, held = tmp_path / "d.npz", tmp_path / "h.npz"
    make_data(data)
    make_data(held, n=300)
    out = str(tmp_path / "net.bin")
    res = run([str(data), out, "--hidden", "8", "--epochs", "3", "--batch", "512", "--threads", "1", "--holdout", str(held)])
    assert res.returncode == 0, res.stderr
    losses = [float(line.split("holdout ")[1].split()[0]) for line in res.stdout.splitlines() if line.startswith("epoch ")]
    assert len(losses) == 3
    # An export follows exactly the epochs that improved on the best holdout loss so far
    # (the printed losses are rounded, so a tie may hide a tiny improvement).
    best, strict, loose = float("inf"), 0, 0
    for loss in losses:
        strict += loss < best
        loose += loss <= best
        best = min(best, loss)
    assert strict <= res.stdout.count("exported ") <= loose
    import torch

    assert abs(torch.load(out + ".ckpt", weights_only=False)["best"] - best) < 1e-5


def test_dedup_keeps_the_first_copy_and_drops_excluded_positions() -> None:
    import dedup

    rng = np.random.default_rng(1)
    pieces = rng.integers(0, 13, size=(6, 64)).astype(np.uint8)
    stm = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
    score = np.arange(6, dtype=np.int16)
    pieces[3], stm[3] = pieces[0], stm[0]  # a repeat of position 0 with another label
    pieces[4], stm[4] = pieces[0], 1 - stm[0]  # same pieces, other side to move: distinct
    _, _, kept = dedup.dedup(pieces, stm, score)
    assert kept.tolist() == [0, 1, 2, 4, 5]
    _, _, kept = dedup.dedup(pieces, stm, score, exclude=(pieces[[5, 1]], stm[[5, 1]]))
    assert kept.tolist() == [0, 2, 4]
