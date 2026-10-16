"""Exercise the real subprocess protocol without recursively running this suite."""

import json
import os
from pathlib import Path
import subprocess
import sys

from evaluation.pytest_worker import redact


ROOT = Path(__file__).resolve().parents[1]


def run_worker(tmp_path, source, *, environment=None):
    test_file = tmp_path / "test_example.py"
    test_file.write_text(source, encoding="utf-8")
    events_file = tmp_path / "events.jsonl"
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", **(environment or {})}
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.pytest_worker", "--events", str(events_file), str(test_file)],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=30,
    )
    events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()]
    return result, events


def test_worker_streams_collection_started_and_final_results(tmp_path):
    process, events = run_worker(tmp_path, """
import pytest
def test_passes():
    assert 2 + 2 == 4
def test_fails():
    assert False, 'meaningful failure'
@pytest.mark.skip(reason='optional dependency')
def test_skipped():
    pass
""")
    assert process.returncode == 1
    collected = next(event for event in events if event["event"] == "collected")
    assert len(collected["tests"]) == 3
    results = [event for event in events if event["event"] == "result"]
    assert [item["status"] for item in results] == ["Passed", "Failed", "Skipped"]
    assert all(item["elapsed_seconds"] >= 0 for item in results)
    assert "meaningful failure" in results[1]["error"]
    for item in results:
        assert events.index({"event": "started", "name": item["name"]}) < events.index(item)


def test_worker_reports_setup_and_teardown_errors(tmp_path):
    process, events = run_worker(tmp_path, """
import pytest
@pytest.fixture
def broken_setup():
    raise RuntimeError('setup unavailable')
@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError('cleanup unavailable')
def test_setup(broken_setup):
    pass
def test_teardown(broken_teardown):
    pass
""")
    assert process.returncode == 1
    results = [event for event in events if event["event"] == "result"]
    assert [item["status"] for item in results] == ["Error", "Error"]
    assert "setup unavailable" in results[0]["error"]
    assert "cleanup unavailable" in results[1]["error"]


def test_worker_reports_collection_error(tmp_path):
    process, events = run_worker(tmp_path, "raise RuntimeError('cannot import test module')\n")
    assert process.returncode == 2
    failures = [event for event in events if event["event"] == "result"]
    assert len(failures) == 1
    assert failures[0]["status"] == "Error"
    assert "cannot import test module" in failures[0]["error"]


def test_worker_success_preserves_zero_exit_code(tmp_path):
    process, events = run_worker(tmp_path, "def test_success():\n    pass\n")
    assert process.returncode == 0, process.stdout + process.stderr
    assert [event["status"] for event in events if event["event"] == "result"] == ["Passed"]


def test_worker_redacts_credentials_in_events_and_output(tmp_path):
    secret = "worker-secret-very-sensitive"
    process, events = run_worker(
        tmp_path,
        "import os\ndef test_failure():\n    raise RuntimeError(os.environ['LLM_API_KEY'])\n",
        environment={"LLM_API_KEY": secret},
    )
    assert process.returncode == 1
    assert secret not in json.dumps(events)
    assert secret not in process.stdout + process.stderr
    assert "[REDACTED]" in next(event for event in events if event["event"] == "result")["error"]


def test_redact_handles_common_unconfigured_credentials(monkeypatch):
    monkeypatch.setenv("SERVICE_PASSWORD", "known-secret")
    message = 'known-secret api_key="new-secret" Authorization: Bearer opaque-token sk-123456789abcdef https://person:password@example.org/path'
    result = redact(message)
    for secret in ("known-secret", "new-secret", "opaque-token", "sk-123456789abcdef", "person:password"):
        assert secret not in result
    assert "example.org/path" in result
