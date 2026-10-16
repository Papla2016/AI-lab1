"""Run pytest in a disposable process and stream a small JSONL event protocol.

Usage: ``python -m evaluation.pytest_worker --events events.jsonl [test paths]``.
The parent owns process lifetime and cancellation; this worker never starts other
test runs or imports the application's model backend.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, TextIO

import pytest


_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTHORIZATION|CREDENTIAL", re.I)
_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:[\w-]*(?:api[_-]?key|token|secret|password|passwd)|authorization)"
    r"\s*[=:]\s*)(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|(?:Bearer\s+)?[^\s,;}]+)"
)
_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,})\b")
_BASIC_AUTH = re.compile(r"(https?://)[^\s/@]+:[^\s/@]+@", re.I)


def redact(value: str) -> str:
    """Remove configured secrets and common credential forms from diagnostics.

    Only environment variables whose *names* indicate credentials are read;
    neither their values nor an environment dump are emitted.
    """
    result = str(value)
    secrets = {
        item for name, item in os.environ.items()
        if _SECRET_NAME.search(name) and item
    }
    for item in sorted(secrets, key=len, reverse=True):
        result = result.replace(item, "[REDACTED]")
    result = _ASSIGNMENT.sub(lambda match: match[1] + "[REDACTED]", result)
    result = _TOKEN.sub("[REDACTED]", result)
    return _BASIC_AUTH.sub(r"\1[REDACTED]@", result)


class EventPlugin:
    """Translate pytest reports, including setup/teardown failures, to events."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        self.reports: dict[str, list[Any]] = {}

    def emit(self, event: str, **data: Any) -> None:
        # Redact string values before serialization, preserving valid JSON even
        # when an exception contains quotes, control characters or Unicode.
        def clean(value: Any) -> Any:
            if isinstance(value, str):
                return redact(value)
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            return value

        self.stream.write(json.dumps(clean({"event": event, **data}), ensure_ascii=False) + "\n")
        self.stream.flush()

    def pytest_collection_finish(self, session: Any) -> None:
        self.emit("collected", tests=[item.nodeid for item in session.items])

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.emit(
                "result", name=report.nodeid or "collection", status="Error",
                elapsed_seconds=0.0, error=report.longreprtext,
            )

    def pytest_runtest_logstart(self, nodeid: str, location: Any) -> None:
        self.emit("started", name=nodeid)

    def pytest_runtest_logreport(self, report: Any) -> None:
        reports = self.reports.setdefault(report.nodeid, [])
        reports.append(report)
        if report.when != "teardown":
            return

        if any(item.failed and item.when != "call" for item in reports):
            status = "Error"
        elif any(item.failed for item in reports):
            status = "Failed"
        elif any(item.skipped for item in reports):
            status = "Skipped"
        else:
            status = "Passed"
        errors = [item.longreprtext for item in reports if item.failed or item.skipped]
        self.emit(
            "result", name=report.nodeid, status=status,
            elapsed_seconds=sum(item.duration for item in reports),
            error="\n\n".join(errors),
        )
        del self.reports[report.nodeid]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("paths", nargs="*", help="Pytest paths; defaults to tests")
    args = parser.parse_args(argv)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    # pytest output can also include exception messages containing credentials.
    # Events flush immediately; the human-readable summary is redacted as one
    # string after completion so chunk boundaries cannot split secret values.
    output = io.StringIO()
    with args.events.open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = pytest.main([*(args.paths or ["tests"]), "-q"], plugins=[EventPlugin(stream)])
    sys.stdout.write(redact(output.getvalue()))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
