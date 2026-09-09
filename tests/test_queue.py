"""scripts/queue.sh serialises jobs on the sparring cores and gates them on the window."""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "scripts" / "queue.sh"


def next_start(when: str, est: int) -> str:
    now = int(datetime.fromisoformat(when).timestamp())
    out = subprocess.run(
        [QUEUE, "--window", "--est", str(est), "--next-start", "--", "true"],
        env={**os.environ, "QUEUE_NOW": str(now)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return datetime.fromtimestamp(int(out)).astimezone().strftime("%Y-%m-%d %H:%M")


def test_window_waits_for_a_start_that_ends_before_nine_pm() -> None:
    assert next_start("2026-09-09 08:00", 200) == "2026-09-09 09:00"
    assert next_start("2026-09-09 12:00", 200) == "2026-09-09 12:00"
    # The bug this replaces: 18:32 is inside the window but a 200-minute job is not.
    assert next_start("2026-09-09 18:32", 200) == "2026-09-10 09:00"
    assert next_start("2026-09-09 22:00", 10) == "2026-09-10 09:00"


def test_second_job_waits_for_the_first(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "QUEUE_LOCK": str(tmp_path / "lock"),
        "QUEUE_LOG": str(tmp_path / "queue.log"),
    }
    first = subprocess.Popen(
        [QUEUE, "--name", "a", "--", "sh", "-c", "echo a-start; sleep 1.5"],
        env=env,
        stdout=subprocess.PIPE,
    )
    time.sleep(0.3)
    second = subprocess.run(
        [QUEUE, "--name", "b", "--", "sh", "-c", "exit 3"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.wait() == 0
    assert second.returncode == 3
    assert "waits for the cores lock" in second.stdout
    lines = (tmp_path / "queue.log").read_text().splitlines()
    kinds = [line.split()[2] for line in lines]
    assert kinds == ["start", "end", "start", "end"]
    assert lines[1].endswith("end a rc=0") and lines[3].endswith("end b rc=3")
