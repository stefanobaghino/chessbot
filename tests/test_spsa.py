import datetime as dt
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("spsa", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "spsa.py")
spsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spsa)

TABLE = {"rfp_margin": {"default": 75, "min": 30, "max": 150, "step": 10}, "razor_margin": {"default": 250, "min": 100, "max": 500, "step": 25}}


def test_perturbations_straddle_the_current_values_and_stay_in_range():
    values = {"rfp_margin": 145.0, "razor_margin": 250.0}
    delta = {"rfp_margin": 1, "razor_margin": -1}
    plus = spsa.perturbed(values, TABLE, delta, 1.0, +1)
    minus = spsa.perturbed(values, TABLE, delta, 1.0, -1)
    assert plus == {"rfp_margin": 150, "razor_margin": 225}  # clamped at the top, one step down
    assert minus == {"rfp_margin": 135, "razor_margin": 275}


def test_update_moves_towards_the_winning_side_and_shrinks_with_the_iteration():
    values = {"rfp_margin": 75.0, "razor_margin": 250.0}
    delta = {"rfp_margin": 1, "razor_margin": -1}
    a0, c0 = spsa.coefficients(0, 100)
    a50, c50 = spsa.coefficients(50, 100)
    assert 0 < a50 < a0 and 0 < c50 < c0 <= 1.0
    moved = spsa.update(values, TABLE, delta, a0, c0, diff=0.5)  # plus scored 75%
    assert moved["rfp_margin"] > 75.0 and moved["razor_margin"] < 250.0
    assert abs(moved["rfp_margin"] - 75.0 - a0 * 10 * 0.5) < 1e-9
    unmoved = spsa.update(values, TABLE, delta, a0, c0, diff=0.0)
    assert unmoved == values


def test_parse_points_takes_the_final_fastchess_report():
    log = "Games: 4, Wins: 1, Losses: 1, Draws: 2, Points: 2.0 (50.00 %)\nmore\nGames: 8, Wins: 3, Losses: 2, Draws: 3, Points: 4.5 (56.25 %)\n"
    assert spsa.parse_points(log) == (4.5, 8)
    assert spsa.parse_points("nothing") is None


def test_window_refuses_iterations_that_would_end_after_the_window():
    tz = dt.timezone(dt.timedelta(hours=2))

    def at(h, m):
        return dt.datetime(2026, 9, 7, h, m, tzinfo=tz)

    assert spsa.window_allows((9, 21), 600, at(10, 0))
    assert not spsa.window_allows((9, 21), 600, at(20, 55))
    assert not spsa.window_allows((9, 21), 0, at(8, 59))
    assert spsa.window_allows(None, 10**6, at(23, 0))
