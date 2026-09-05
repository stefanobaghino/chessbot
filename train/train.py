"""Train a (768 -> N)x2 -> 1 NNUE with SCReLU and export quantised weights.

Usage: train.py data.npz out.bin [--hidden 256] [--epochs 20] [--batch 16384] [--lr 1e-3]
                [--checkpoint out.bin.ckpt] [--window 9-21]

The full training state (weights, optimiser, LR schedule, RNG, data split) is saved to the
checkpoint after every epoch and picked up again by the same command, so a run can be
stopped and resumed. With --window H1-H2 an epoch is only started when it is expected to
finish inside that daily window (estimate: the previous epoch's duration), and the process
exits with status 3 otherwise; run the same command again after the window opens.
"""
import argparse
import datetime as dt
import os
import sys
import time

import numpy as np
import torch
from torch import nn

SCALE = 400.0
QA = 255
QB = 64


class Net(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.ft = nn.EmbeddingBag(768, hidden, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(2 * hidden, 1)
        nn.init.uniform_(self.ft.weight, -0.05, 0.05)

    def forward(self, us_idx, us_off, them_idx, them_off):
        a = self.ft(us_idx, us_off) + self.ft_bias
        b = self.ft(them_idx, them_off) + self.ft_bias
        h = torch.cat([a, b], dim=1).clamp(0.0, 1.0)
        h = h * h
        return self.out(h).squeeze(1)

    def clip(self):
        with torch.no_grad():
            self.ft.weight.clamp_(-127 / QA * 1.0, 127 / QA * 1.0) if False else None
            self.ft.weight.clamp_(-32767 / QA, 32767 / QA)
            self.ft_bias.clamp_(-32767 / QA, 32767 / QA)
            self.out.weight.clamp_(-32767 / QB, 32767 / QB)


def features(pieces: np.ndarray, stm: np.ndarray):
    """Build perspective feature indices for a batch.

    Feature index = rel_color * 384 + piece_type * 64 + rel_square, where rel_color is 0
    for the perspective's own pieces and squares are flipped for the black perspective.
    """
    n = pieces.shape[0]
    occ = pieces > 0
    sq = np.broadcast_to(np.arange(64, dtype=np.int64), (n, 64))
    code = pieces.astype(np.int64) - 1  # 0..11
    color = code // 6  # 0 white, 1 black
    ptype = code % 6
    stm_col = stm.astype(np.int64)[:, None]
    # us perspective
    rel_color_us = (color != stm_col).astype(np.int64)
    rel_sq_us = np.where(stm_col == 0, sq, sq ^ 56)
    idx_us = rel_color_us * 384 + ptype * 64 + rel_sq_us
    # them perspective
    rel_color_them = (color == stm_col).astype(np.int64)
    rel_sq_them = np.where(stm_col == 0, sq ^ 56, sq)
    idx_them = rel_color_them * 384 + ptype * 64 + rel_sq_them
    counts = occ.sum(axis=1)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (
        torch.from_numpy(idx_us[occ]),
        torch.from_numpy(offsets),
        torch.from_numpy(idx_them[occ]),
        torch.from_numpy(offsets.copy()),
    )


def export(net: Net, path: str):
    w0 = (net.ft.weight.detach().numpy() * QA).round().clip(-32767, 32767).astype(np.int16)
    b0 = (net.ft_bias.detach().numpy() * QA).round().clip(-32767, 32767).astype(np.int16)
    w1 = (net.out.weight.detach().numpy().reshape(-1) * QB).round().clip(-32767, 32767).astype(np.int16)
    b1 = np.array([net.out.bias.item() * QA * QB]).round().astype(np.int32)
    with open(path, "wb") as f:
        f.write(np.array([net.ft.weight.shape[1]], dtype=np.int32).tobytes())
        f.write(w0.tobytes())
        f.write(b0.tobytes())
        f.write(w1.tobytes())
        f.write(b1.tobytes())
    print(f"exported {path}: hidden={w0.shape[1]} w0 range [{w0.min()},{w0.max()}] w1 range [{w1.min()},{w1.max()}] b1={b1[0]}")


def parse_window(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    start, _, end = spec.partition("-")
    return int(start), int(end)


def window_allows(window: tuple[int, int], est_seconds: float, now: dt.datetime | None = None) -> bool:
    """True if an epoch estimated at est_seconds, started now, ends inside today's window."""
    now = now or dt.datetime.now().astimezone()  # local wall clock, the window is a local rule
    start = now.replace(hour=window[0], minute=0, second=0, microsecond=0)
    end = now.replace(hour=window[1], minute=0, second=0, microsecond=0)
    return start <= now and now + dt.timedelta(seconds=est_seconds) <= end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="comma-separated list of .npz files")
    ap.add_argument("out")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", default=None, help="initialise the weights from this .pt file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--checkpoint", default=None, help="full-state checkpoint, default <out>.ckpt")
    ap.add_argument("--window", default=None, help="hours H1-H2 inside which epochs may run, e.g. 9-21")
    args = ap.parse_args()
    ckpt_path = args.checkpoint or args.out + ".ckpt"
    window = parse_window(args.window)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    parts = [np.load(f) for f in args.data.split(",")]
    pieces = np.concatenate([d["pieces"] for d in parts])
    stm = np.concatenate([d["stm"] for d in parts])
    score = np.concatenate([d["score"] for d in parts]).astype(np.float32)
    del parts
    if args.limit:
        pieces, stm, score = pieces[: args.limit], stm[: args.limit], score[: args.limit]
    n = len(score)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = min(200_000, n // 20)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"{len(train_idx)} train / {n_val} val positions")

    net = Net(args.hidden)
    if args.resume:
        net.load_state_dict(torch.load(args.resume))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.3)
    first_epoch, est = 0, 0.0
    key = {"data": args.data, "hidden": args.hidden, "epochs": args.epochs, "n": n}
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        if ck["key"] != key:
            sys.exit(f"{ckpt_path} was written by a different run ({ck['key']} != {key}); remove it to start over")
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["rng"]
        train_idx = ck["train_idx"]
        first_epoch, est = ck["epoch"], ck["epoch_seconds"]
        print(f"resumed from {ckpt_path} after epoch {first_epoch}/{args.epochs}", flush=True)
        if first_epoch >= args.epochs:
            print("nothing left to do", flush=True)
            return

    def batch_loss(idx):
        us_i, us_o, th_i, th_o = features(pieces[idx], stm[idx])
        target = torch.sigmoid(torch.from_numpy(score[idx]) / SCALE)
        pred = torch.sigmoid(net(us_i, us_o, th_i, th_o))
        return ((pred - target) ** 2).mean()

    for epoch in range(first_epoch, args.epochs):
        if window and not window_allows(window, est):
            print(f"epoch {epoch + 1}/{args.epochs} would not finish inside {args.window} (est {est / 60:.0f} min); stopping", flush=True)
            sys.exit(3)
        t0 = time.time()
        rng.shuffle(train_idx)
        net.train()
        total, nb = 0.0, 0
        for start in range(0, len(train_idx), args.batch):
            idx = np.sort(train_idx[start : start + args.batch])
            loss = batch_loss(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            net.clip()
            total += loss.item()
            nb += 1
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = np.mean([batch_loss(np.sort(val_idx[s : s + args.batch])).item() for s in range(0, n_val, args.batch)])
        print(f"epoch {epoch + 1}/{args.epochs} train {total / nb:.5f} val {vl:.5f} lr {sched.get_last_lr()[0]:.1e} {time.time() - t0:.0f}s", flush=True)
        torch.save(net.state_dict(), args.out + ".pt")
        export(net, args.out)
        est = time.time() - t0
        state = {"key": key, "epoch": epoch + 1, "epoch_seconds": est, "net": net.state_dict(),
                 "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                 "train_idx": train_idx}
        torch.save(state, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)


if __name__ == "__main__":
    main()
