"""Classify why games were lost: for each lost game find the first move that turned a
holdable position into a lost one (Stockfish at fixed depth), then say whether our engine
would have avoided it with a deeper search (search miss) or still plays it (eval miss).

Usage: scripts/losses.py <pgn> [our_name] [sf_depth] [workers] [engine_path]
"""
import re
import sys
from collections import Counter
from multiprocessing import Pool

import chess
import chess.engine
import chess.pgn

PGN = sys.argv[1]
US = sys.argv[2] if len(sys.argv) > 2 else "chessbot"
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 14
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ENGINE = sys.argv[5] if len(sys.argv) > 5 else "engine/target/release/chessbot-engine"
HOLDABLE, LOST, BIG = -100, -250, 150
COMMENT = re.compile(r"([+-]?[\d.]+|[+-]M\d+)/(\d+) ([\d.]+)s")


def phase(board):
    npm = sum(len(board.pieces(p, c)) * v for c in chess.COLORS
              for p, v in ((chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9)))
    if board.fullmove_number <= 12:
        return "opening"
    return "endgame" if npm <= 14 else "middlegame"


def parse_comment(c):
    m = COMMENT.search(c or "")
    if not m:
        return None, None, None
    s, d, t = m.groups()
    score = None if s.startswith(("+M", "-M")) else round(float(s) * 100)
    return score, int(d), float(t)


def analyse(game_text):
    game = chess.pgn.read_game(__import__("io").StringIO(game_text))
    our = chess.WHITE if game.headers["White"] == US else chess.BLACK
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    own = chess.engine.SimpleEngine.popen_uci(ENGINE)
    board = game.board()
    node = game
    evals = []  # (move_no, san, uci, sf_before, sf_after, our_score, our_depth, secs, phase, fen)
    prev_after = None
    clock = 10.0
    while node.variations:
        nxt = node.variations[0]
        mv = nxt.move
        if board.turn == our:
            before = prev_after if prev_after is not None else \
                sf.analyse(board, chess.engine.Limit(depth=DEPTH))["score"].pov(our).score(mate_score=10000)
            ph, fen, no, san = phase(board), board.fen(), board.fullmove_number, board.san(mv)
            board.push(mv)
            after = sf.analyse(board, chess.engine.Limit(depth=DEPTH))["score"].pov(our).score(mate_score=10000)
            sc, d, t = parse_comment(nxt.comment)
            clock = clock - (t or 0) + 0.1
            evals.append((no, san, mv.uci(), before, after, sc, d, t, ph, fen, clock))
            prev_after = None
        else:
            board.push(mv)
            prev_after = sf.analyse(board, chess.engine.Limit(depth=DEPTH))["score"].pov(our).score(mate_score=10000)
        node = nxt
    # the move on which the position first became lost (SF), else the largest single drop
    crit = next((e for e in evals if e[3] > LOST and e[4] <= LOST), None)
    kind = "turned-lost"
    if crit is None:
        crit = max(evals, key=lambda e: e[3] - e[4]) if evals else None
        kind = "largest-drop"
    verdict = {}
    if crit:
        no, san, uci, before, after, sc, d, t, ph, fen, clk = crit
        drop = before - after
        b = chess.Board(fen)
        deep = own.play(b, chess.engine.Limit(depth=min((d or 10) + 6, 26)))
        deeper_avoids = deep.move.uci() != uci
        if drop < BIG:
            cause = "squeeze"          # no single big error: outplayed gradually
        elif deeper_avoids:
            cause = "search-miss"      # more depth finds it: horizon / pruning
        else:
            cause = "eval-miss"        # even deeper search plays it: evaluation wrong
        verdict = {"kind": kind, "move": f"{no}.{san}", "sf": f"{before:+d}->{after:+d}", "ours": sc, "depth": d,
                   "secs": t, "clock": round(clk, 1), "phase": ph, "deeper": deep.move.uci(), "cause": cause, "fen": fen}
    sf.quit(); own.quit()
    return game.headers.get("Round"), "W" if our else "B", len(evals), verdict


if __name__ == "__main__":
    games = []
    with open(PGN) as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            our = chess.WHITE if g.headers["White"] == US else chess.BLACK
            r = g.headers["Result"]
            if (r == "0-1" and our) or (r == "1-0" and not our):
                games.append(str(g))
    print(f"{len(games)} lost games, SF depth {DEPTH}, {WORKERS} workers", flush=True)
    causes, phases, kinds = Counter(), Counter(), Counter()
    with Pool(WORKERS) as pool:
        for rnd, col, n, v in pool.imap_unordered(analyse, games):
            if not v:
                continue
            causes[v["cause"]] += 1; phases[v["phase"]] += 1; kinds[v["kind"]] += 1
            print(f"round {rnd} ({col}) {v['kind']:12} {v['move']:9} sf {v['sf']:12} ours {v['ours']} d{v['depth']} "
                  f"{v['secs']}s clk {v['clock']} {v['phase']:10} deeper={v['deeper']} -> {v['cause']}", flush=True)
    print("\ncauses:", dict(causes)); print("phases:", dict(phases)); print("kinds:", dict(kinds))
