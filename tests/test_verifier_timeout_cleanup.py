import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import pytest

from agent_bench.tools.verify_single import _read_partial_progress
from sandbox.verifier.verifier import _run_verify_in_session, _terminate_verify_process


def _wait_with_child(verifyrequest, result_queue):
    # Ignore TERM to prove that cleanup also kills a stubborn descendant.
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(300)"
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "ready"
    result_queue.put(child.pid)
    time.sleep(300)


def _alive_non_zombie(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = f"/proc/{pid}/stat"
    if os.path.exists(status):
        try:
            return open(status, encoding="utf-8").read().split(") ", 1)[1][0] != "Z"
        except FileNotFoundError:
            return False
    return True


@pytest.mark.skipif(os.name != "posix", reason="POSIX sessions and process groups")
def test_timeout_kills_worker_and_stubborn_compiler_descendant():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_run_verify_in_session, args=(_wait_with_child, None, queue))
    process.start()
    child_pid = None
    try:
        child_pid = queue.get(timeout=30)
        assert os.getsid(process.pid) == process.pid
        assert os.getpgid(child_pid) == process.pid
        _terminate_verify_process(process)
        assert not process.is_alive()
        for _ in range(50):
            if not _alive_non_zombie(child_pid):
                break
            time.sleep(0.1)
        assert not _alive_non_zombie(child_pid)
    finally:
        if process.is_alive():
            _terminate_verify_process(process)
        if child_pid and _alive_non_zombie(child_pid):
            os.kill(child_pid, 9)
        queue.close()


def test_timeout_progress_preserves_partial_counts_and_last_start(tmp_path):
    name = "some_op"
    report = tmp_path / f"test_report_{name}.json"
    with open(f"{report}.partial.jsonl", "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"success": True}) + "\n")
        stream.write(json.dumps({"success": False}) + "\n")
        stream.write('{"success":')  # A terminated writer may leave a partial line.
    last = {"function": "test_op", "params": {"n": 128}}
    with open(f"{report}.started.jsonl", "w", encoding="utf-8") as stream:
        stream.write(json.dumps(last) + "\n")
    result = _read_partial_progress(tmp_path, name)
    assert result == {"timed_out": True, "partial_results_available": True,
                      "completed_tests": 2, "completed_passed_tests": 1,
                      "completed_failed_tests": 1, "last_started_test": last}


def test_timeout_progress_distinguishes_missing_checkpoint(tmp_path):
    assert _read_partial_progress(tmp_path, "sort") == {
        "timed_out": True, "partial_results_available": False,
    }
