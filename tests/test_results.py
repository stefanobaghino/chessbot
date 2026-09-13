"""scripts/results.py summarises the per-game ledger of #64."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import results

LEDGER = """2026-09-12T20:00:00+02:00\tv0.1.51\tg1\tblitz\t180+2\trated\twhite\ta\t2500\t2600\twin\tmate\t60
2026-09-12T21:00:00+02:00\tv0.1.51\tg2\tbullet\t60+1\trated\tblack\tb\t2550\t2540\tloss\tresign\t80
2026-09-13T01:00:00+02:00\tv0.1.52\tg3\tblitz\t180+2\trated\twhite\tc\t-\t2610\tdraw\tdraw\t100
2026-09-13T02:00:00+02:00\tv0.1.52\tg4\tblitz\t180+2\tcasual\tblack\td\t2400\t2610\t-\taborted\t0
"""


def test_summary_by_version_day_and_speed(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_text(LEDGER)
    rows = results.read(str(path))
    assert len(rows) == 4
    by_version = results.summarise(rows, "version")
    assert [r["version"] for r in by_version] == ["v0.1.51", "v0.1.52"]
    first, second = by_version
    assert (first["games"], first["wins"], first["losses"], first["draws"], first["score"]) == (2, 1, 1, 0, 0.5)
    assert first["avg_opp"] == 2525 and (first["rating_first"], first["rating_last"]) == (2600, 2540)
    assert second["games"] == 2 and second["score"] == 0.5 and second["avg_opp"] == 2400  # the aborted game has no result
    assert [r["day"] for r in results.summarise(rows, "day")] == ["2026-09-12", "2026-09-13"]
    speeds = {r["speed"]: r for r in results.summarise(rows, "speed")}
    assert speeds["blitz"]["games"] == 3 and speeds["bullet"]["rating_first"] == 2540
    assert len(results.read(str(path), since="2026-09-13")) == 2
