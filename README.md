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

## Measure strength

`scripts/match.sh [elo] [games] [tc] [concurrency] [name]` plays the engine against
Stockfish limited with `UCI_LimitStrength`/`UCI_Elo` using the UHO opening book and
writes a PGN and log under `matches/`.

```
scripts/match.sh 2000 100 10+0.1 3 goal
```

`scripts/blunders.py matches/<run>.pgn` lists the moves that lost the most in each
lost game, `scripts/evalsym.py` checks that the static evaluation is colour-symmetric.

## CI/CD and releases

Continuous delivery runs on this machine, not on a hosted CI. Enable the hooks once per
clone with `git config core.hooksPath scripts/hooks`.

- `scripts/hooks/post-commit` starts `scripts/ci.sh` in the background after every commit
  on `main` (log in `ci/ci.log`, per-release logs in `ci/logs/`).
- `scripts/ci.sh` takes each commit since the last `v*` tag, in order, exports it with
  `git archive`, runs `scripts/validate.sh` (release build for this CPU, `cargo test`,
  `bench`, NNUE self-check, ruff, pytest), then `scripts/package.sh`, creates a signed
  tag with the next patch version, pushes the commit and tag, and publishes a GitHub
  Release whose notes are the commit message. A failing commit stops the pipeline until a
  fixing commit lands; it never becomes a release.
- `scripts/hooks/pre-push` refuses to push commits to `main` that were not released this
  way, so `origin/main` only ever contains validated commits.
- Development happens on branches in separate worktrees (`git worktree add ../chessbot-<topic>
  -b <topic> main`), never directly in the `main` checkout, so agents working in parallel do
  not disturb each other. Land a branch with a fast-forward merge from the `main` checkout
  (`git merge --ff-only <topic>`); `scripts/hooks/post-merge` then releases each new commit.
  In a worktree, symlink `.venv`, `data` and `matches` to the main checkout to share them;
  never replace those directories in the main checkout itself, they hold the only copies
  of the virtualenv, training data and match results.

A service manager must signal only the bot's main process on stop (systemd:
`KillMode=mixed`); the engine child runs in its own process group and is re-spawned if
it dies mid-game, but killing it needlessly costs search time.
- Release assets: `chessbot-<tag>-linux-aarch64.tar.gz` (`bin/chessbot-engine`, `bot/`,
  `requirements.txt`, `VERSION`) and `SHA256SUMS`. Put a line starting with `MANUAL:` in a
  commit body when deploying it needs a manual step; it is hoisted to the top of the notes.
- The engine reports its version and embedded net in the `uci` banner, e.g.
  `id name chessbot-engine v0.1.0 net:d17c329e3df9`.

## Play on Lichess

1. Create a fresh Lichess account (it must have played no games) and generate a
   personal API token with the `bot:play`, `challenge:read` and `challenge:write` scopes.
2. Put it in `.env` as `LICHESS_TOKEN=lip_...` (the file is git-ignored).
3. Upgrade the account once: `.venv/bin/python scripts/upgrade_to_bot.py`.
4. Run `./run_bot.sh`. The bot accepts standard-chess challenges at bullet, blitz,
   rapid and classical time controls.

## Results

Engine at commit `63124ea` vs Stockfish 15.1 at `UCI_Elo` 2000, 100 games, 10+0.1,
UHO book, Raspberry Pi 5: 60 wins, 18 losses, 22 draws (71%), Elo +156 ± 65.
