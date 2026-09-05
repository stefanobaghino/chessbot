# chessbot

A chess engine and Lichess bot built to run unattended on a Raspberry Pi 5.

- `engine/` — Rust UCI engine: alpha-beta search (PVS, transposition table, null move,
  late move reductions, quiescence with static exchange evaluation) and a tapered
  hand-crafted evaluation (PeSTO piece-square tables plus king safety, pawn structure,
  mobility). Move generation comes from the `cozy-chess` crate.
- `bot/` — Python Lichess client (berserk + python-chess) that accepts challenges and
  plays them with the engine.
- `scripts/` — match runner and analysis helpers built on fastchess and Stockfish.

## Build the engine

```
cd engine && cargo build --release
./target/release/chessbot-engine bench 10
```

## Release

Every commit is validated by the `pre-commit` hook (`scripts/validate.sh`: engine
build, tests, bench, NNUE self-check, ruff, pytest). Releases are deliberate:
on `main`, run

```
scripts/release.sh
```

It validates `HEAD` again, signs the next `v*` tag, pushes `main` and the tag,
and publishes a GitHub Release whose notes cover every commit since the previous
tag (`MANUAL:` lines in commit messages are hoisted to the top). The `pre-push`
hook refuses to push an untagged `main`, so this is the only way commits reach
GitHub. Logs land in `ci/logs/`, the last outcome in `ci/last_result`.

## Measure strength

`scripts/match.sh [elo] [games] [tc] [concurrency] [name]` plays the engine against
Stockfish limited with `UCI_LimitStrength`/`UCI_Elo` using the UHO opening book and
writes a PGN and log under `matches/`.

```
scripts/match.sh 2000 100 10+0.1 3 goal
```

`scripts/relabel_chunks.sh data/x.fens data/x_d10 10` relabels positions with Stockfish at
depth 10 in resumable 50k chunks (`x_d10_<i>.npz`, pass them comma-separated to
`train/train.py`), starting a chunk only if it can finish before 21:00; rerun the same
command after 09:00 to resume. `train/fens_diff.py` picks the positions of self-play batches
that have not been labelled yet. `train/train.py` saves its full state to `<out>.ckpt` after
every epoch and resumes from it when run again with the same arguments; with `--window 9-21`
it exits with status 3 instead of starting an epoch that would end after 21:00.
`scripts/train_net6.sh` is the net6 job built on both: a no-op until every relabel chunk
exists, otherwise it trains (or resumes) inside the window.
`scripts/install_timers.sh` installs both as persistent daily user timers (09:00 and 09:05,
`CPUQuota=200%`, i.e. half of the four cores) that survive a reboot; `--uninstall` removes them.

`scripts/blunders.py matches/<run>.pgn` lists the moves that lost the most in each
lost game, `scripts/evalsym.py` checks that the static evaluation is colour-symmetric.

### Sharing the machine with the live bot

The Lichess bot runs on cores 0-1 (`CPUAffinity=0 1` in its systemd unit). `spar.sh`
and `match.sh` pin their games to cores 2-3 with `taskset` (override with `SPAR_CPUS`);
sparring engines alternate moves, so two concurrent games fill the two cores.
Background jobs go into cgroups created once per boot with `sudo scripts/cgroups.sh`:
`quiet` (one core) for relabelling and data generation, `train` (two cores) for
training, both restricted to cores 2-3. `match.sh` freezes them while timed games run.
