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
    assert "waits for" in second.stdout  # its turn, or the lock if the first is still exiting
    lines = (tmp_path / "queue.log").read_text().splitlines()
    kinds = [line.split()[2] for line in lines]
    assert kinds == ["start", "end", "start", "end"]
    assert lines[1].endswith("end a rc=0") and lines[3].endswith("end b rc=3")


def queue_env(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "QUEUE_LOCK": str(tmp_path / "lock"),
        "QUEUE_LOG": str(tmp_path / "queue.log"),
        "QUEUE_POLL": "0.05",
    }


def test_jobs_start_in_submission_order(tmp_path: Path) -> None:
    env = queue_env(tmp_path)
    jobs = []
    for name in "abcde":
        jobs.append(subprocess.Popen([QUEUE, "--name", name, "--", "sleep", "0.3"], env=env, stdout=subprocess.DEVNULL))
        time.sleep(0.15)
    assert [job.wait() for job in jobs] == [0] * 5
    starts = [line.split()[3] for line in (tmp_path / "queue.log").read_text().splitlines() if line.split()[2] == "start"]
    assert starts == ["a:", "b:", "c:", "d:", "e:"]
    assert list((tmp_path / "lock.d").iterdir()) == []


def test_stale_ticket_of_a_dead_process_is_removed(tmp_path: Path) -> None:
    env = queue_env(tmp_path)
    tickets = tmp_path / "lock.d"
    tickets.mkdir()
    stale = tickets / ("0" * 19 + "-999999999")
    stale.touch()
    (tickets / (stale.name + ".ready")).touch()
    result = subprocess.run([QUEUE, "--name", "x", "--", "true"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "waits for its turn" not in result.stdout
    assert list(tickets.iterdir()) == []


def test_list_shows_queued_jobs_in_order(tmp_path: Path) -> None:
    env = queue_env(tmp_path)
    tickets = tmp_path / "lock.d"
    tickets.mkdir()
    first = subprocess.Popen([QUEUE, "--name", "a", "--", "sleep", "1"], env=env, stdout=subprocess.DEVNULL)
    time.sleep(0.3)
    second = subprocess.Popen([QUEUE, "--name", "b", "--", "sh", "-c", "true"], env=env, stdout=subprocess.DEVNULL)
    time.sleep(0.3)
    # A dead job's ticket younger than the live ones: no waiter looks at it, --list cleans it.
    stale = tickets / f"{time.time_ns():019d}-999999999"
    stale.write_text("old: sleep 1\n")
    listing = subprocess.run([QUEUE, "--list"], env=env, capture_output=True, text=True, check=True).stdout.splitlines()
    assert [line.split()[0] for line in listing] == ["running", "ready", "stale"]
    assert listing[0].endswith("a: sleep 1") and listing[1].endswith("b: sh -c true") and listing[2].endswith("old: sleep 1")
    assert f"pid {first.pid}" in listing[0] and f"pid {second.pid}" in listing[1]
    assert first.wait() == 0 and second.wait() == 0
    assert not stale.exists()
    assert subprocess.run([QUEUE, "--list"], env=env, capture_output=True, text=True, check=True).stdout == "queue: empty\n"


def test_now_releases_a_job_waiting_for_its_window(tmp_path: Path) -> None:
    env = queue_env(tmp_path)
    # A job that can never fit the window waits for the next 09:00 whatever the time of day.
    job = subprocess.Popen(
        [QUEUE, "--window", "--est", "100000", "--name", "w", "--", "sh", "-c", "echo start=$WINDOW_START"],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.5)
    assert subprocess.run([QUEUE, "--list"], env=env, capture_output=True, text=True, check=True).stdout.startswith("waiting")
    nothing = subprocess.run([QUEUE, "--now", "other"], env=env, capture_output=True, text=True, check=False)
    assert nothing.returncode == 1 and "nothing to release" in nothing.stdout
    released = subprocess.run([QUEUE, "--now", "w"], env=env, capture_output=True, text=True, check=False)
    assert released.returncode == 0 and f"released w (pid {job.pid})" in released.stdout
    out, _ = job.communicate(timeout=10)
    assert job.returncode == 0
    assert "waits until" in out and "released early" in out and "start=0" in out
    log = (tmp_path / "queue.log").read_text().splitlines()
    assert [line.split()[2] for line in log] == ["release", "start", "end"]
    assert list((tmp_path / "lock.d").iterdir()) == []
