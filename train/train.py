"""Train a (4x768 -> N)x2 -> 4 NNUE with SCReLU and export quantised weights.

Each perspective indexes its features by its own king's bucket (KING_BUCKET, files e-h
mirrored onto a-d) and the output layer is chosen by piece count (OUT_BUCKETS bands), both
matching engine/src/nnue.rs; the exported header (FORMAT, hidden, king buckets, out
buckets) lets the engine reject a net built for another layout.

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
FORMAT = 2
KING_BUCKETS = 4
OUT_BUCKETS = 4
KING_BUCKET = np.array([0, 0, 1, 1, 1, 1, 0, 0] + [2] * 8 + [3] * 48, dtype=np.int64)
WHITE_KING, BLACK_KING = 6, 12  # piece codes in the npz files


class Net(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.ft = nn.EmbeddingBag(KING_BUCKETS * 768, hidden, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        bound = 1.0 / (2 * hidden) ** 0.5  # nn.Linear's default initialisation
        self.out_w = nn.Parameter(torch.empty(OUT_BUCKETS, 2 * hidden).uniform_(-bound, bound))
        self.out_b = nn.Parameter(torch.empty(OUT_BUCKETS).uniform_(-bound, bound))
        nn.init.uniform_(self.ft.weight, -0.05, 0.05)

    def forward(self, us_idx, us_off, them_idx, them_off, ob):
        a = self.ft(us_idx, us_off) + self.ft_bias
        b = self.ft(them_idx, them_off) + self.ft_bias
        h = torch.cat([a, b], dim=1).clamp(0.0, 1.0)
        h = h * h
        return (h * self.out_w[ob]).sum(dim=1) + self.out_b[ob]

    def clip(self):
        with torch.no_grad():
            self.ft.weight.clamp_(-32767 / QA, 32767 / QA)
            self.ft_bias.clamp_(-32767 / QA, 32767 / QA)
            self.out_w.clamp_(-32767 / QB, 32767 / QB)


def perspective(rel_king: np.ndarray, rel_color: np.ndarray, ptype: np.ndarray, rel_sq: np.ndarray):
    """Feature indices of one perspective given its own king's relative square.

    Index = king_bucket * 768 + rel_color * 384 + piece_type * 64 + rel_square, where
    rel_color is 0 for the perspective's own pieces; squares are already flipped for the
    black perspective and get mirrored (file ^ 7) when the king stands on files e-h.
    """
    mirror = np.where((rel_king & 7) >= 4, 7, 0)[:, None]
    return KING_BUCKET[rel_king][:, None] * 768 + rel_color * 384 + ptype * 64 + (rel_sq ^ mirror)


def features(pieces: np.ndarray, stm: np.ndarray):
    """Build perspective feature indices and output buckets for a batch."""
    n = pieces.shape[0]
    occ = pieces > 0
    sq = np.broadcast_to(np.arange(64, dtype=np.int64), (n, 64))
    code = pieces.astype(np.int64) - 1  # 0..11
    color = code // 6  # 0 white, 1 black
    ptype = code % 6
    stm_col = stm.astype(np.int64)[:, None]
    white = stm_col[:, 0] == 0
    wk = np.argmax(pieces == WHITE_KING, axis=1)  # 0 when absent (synthetic test data)
    bk = np.argmax(pieces == BLACK_KING, axis=1)
    # us perspective
    rel_color_us = (color != stm_col).astype(np.int64)
    rel_sq_us = np.where(stm_col == 0, sq, sq ^ 56)
    idx_us = perspective(np.where(white, wk, bk ^ 56), rel_color_us, ptype, rel_sq_us)
    # them perspective
    rel_color_them = (color == stm_col).astype(np.int64)
    rel_sq_them = np.where(stm_col == 0, sq ^ 56, sq)
    idx_them = perspective(np.where(white, bk ^ 56, wk), rel_color_them, ptype, rel_sq_them)
    counts = occ.sum(axis=1)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ob = np.clip((counts - 2) // (32 // OUT_BUCKETS), 0, OUT_BUCKETS - 1)
    return (
        torch.from_numpy(idx_us[occ]),
        torch.from_numpy(offsets),
        torch.from_numpy(idx_them[occ]),
        torch.from_numpy(offsets.copy()),
        torch.from_numpy(ob),
    )


def export(net: Net, path: str):
    w0 = (net.ft.weight.detach().numpy() * QA).round().clip(-32767, 32767).astype(np.int16)
    b0 = (net.ft_bias.detach().numpy() * QA).round().clip(-32767, 32767).astype(np.int16)
    w1 = (net.out_w.detach().numpy() * QB).round().clip(-32767, 32767).astype(np.int16)
    b1 = (net.out_b.detach().numpy() * QA * QB).round().astype(np.int32)
    with open(path, "wb") as f:
        f.write(np.array([FORMAT, w0.shape[1], KING_BUCKETS, OUT_BUCKETS], dtype=np.int32).tobytes())
        f.write(w0.tobytes())
        f.write(b0.tobytes())
        f.write(w1.tobytes())
        f.write(b1.tobytes())
    print(f"exported {path}: hidden={w0.shape[1]} w0 range [{w0.min()},{w0.max()}] w1 range [{w1.min()},{w1.max()}] b1={b1.tolist()}")


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
    key = {"data": args.data, "hidden": args.hidden, "epochs": args.epochs, "n": n,
           "format": FORMAT, "king_buckets": KING_BUCKETS, "out_buckets": OUT_BUCKETS}
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
        us_i, us_o, th_i, th_o, ob = features(pieces[idx], stm[idx])
        target = torch.sigmoid(torch.from_numpy(score[idx]) / SCALE)
        pred = torch.sigmoid(net(us_i, us_o, th_i, th_o, ob))
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
