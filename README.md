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
