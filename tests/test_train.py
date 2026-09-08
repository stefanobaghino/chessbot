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


def test_feature_indexing_matches_the_engine() -> None:
    """The same cases are pinned in engine/src/nnue.rs (feature_index_matches_trainer)."""
    # Square numbering: a1 = 0 .. h8 = 63; codes: white P..K = 1..6, black = 7..12.
    def row(placements: dict[int, int]) -> np.ndarray:
        p = np.zeros(64, dtype=np.uint8)
        for sq, code in placements.items():
            p[sq] = code
        return p

    g1, f3, e8, d1, c2, a7, h4, e1 = 6, 21, 60, 3, 10, 48, 31, 4
    pieces = np.stack([
        row({g1: 6, f3: 2, e8: 12}),  # white to move, white king g1, knight f3
        row({e8: 12, d1: 5, e1: 6}),  # black to move, black king e8, white queen d1
        row({c2: 6, a7: 7, e8: 12}),  # white to move, white king c2, black pawn a7
        row({h4: 12, e1: 6}),  # black to move, black king h4
    ])
    stm = np.array([0, 1, 0, 1], dtype=np.uint8)
    us_i, us_o, _, _, ob = train.features(pieces, stm)
    us = [set(us_i[us_o[i] : (us_o[i + 1] if i + 1 < len(us_o) else len(us_i))].tolist()) for i in range(4)]
    assert 64 + 18 in us[0]
    assert 768 + 384 + 256 + 60 in us[1]
    assert 2 * 768 + 384 + 48 in us[2]
    assert 3 * 768 + 320 + 32 in us[3]
    assert ob.tolist() == [0, 0, 0, 0]
    full = np.zeros((1, 64), dtype=np.uint8)
    full[0, :32] = np.array([1, 2, 3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [7, 8, 9, 10, 11, 12, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])
    assert train.features(full, np.zeros(1, dtype=np.uint8))[4].tolist() == [3]


def test_export_writes_the_versioned_header(tmp_path: Path) -> None:
    net = train.Net(8)
    out = tmp_path / "n.bin"
    train.export(net, str(out))
    raw = out.read_bytes()
    assert np.frombuffer(raw[:16], dtype=np.int32).tolist() == [train.FORMAT, 8, train.KING_BUCKETS, train.OUT_BUCKETS]
    rows = train.KING_BUCKETS * 768
    assert len(raw) == 16 + 2 * (rows * 8 + 8 + train.OUT_BUCKETS * 16) + 4 * train.OUT_BUCKETS
