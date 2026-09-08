"""scripts/sparsum.py combines fastchess batch results."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sparsum

LOG = """Started game 3 of 4 (new vs old)
Games: 2, Wins: 1, Losses: 0, Draws: 1, Points: 1.5 (75.00 %)
Elo: 191.02 +/- nan
Games: 4, Wins: {w}, Losses: {l}, Draws: {d}, Points: 2.0 (50.00 %)
Finished match
"""


def test_final_line_wins_and_even_score_is_zero_elo() -> None:
    assert sparsum.final_wld(LOG.format(w=1, l=1, d=2)) == (1, 1, 2)
    assert sparsum.final_wld("no results") == (0, 0, 0)
    s, e, err = sparsum.elo(1, 1, 2)
    assert (s, e) == (0.5, 0.0) and err > 0


def test_elo_matches_fastchess_for_a_real_batch() -> None:
    # fastchess reported +26.11 +/- 33.63 for 56/41/103.
    _, e, err = sparsum.elo(56, 41, 103)
    assert abs(e - 26.11) < 0.05
    assert abs(err - 33.63) < 1.0


def test_cli_sums_batches(tmp_path: Path) -> None:
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text(LOG.format(w=2, l=1, d=1))
    b.write_text(LOG.format(w=1, l=2, d=1))
    out = subprocess.run([sys.executable, str(ROOT / "scripts/sparsum.py"), str(a), str(b)], capture_output=True, text=True, check=True).stdout
    assert "combined: 8 games W3 L3 D2 score 50.00% Elo +0.0" in out
