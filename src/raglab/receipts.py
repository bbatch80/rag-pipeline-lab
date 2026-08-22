"""Run receipts: every pipeline command ends with a legible summary and a
non-zero exit on any failure."""

import sys
import time

RULE = "─" * 56


class Receipt:
    def __init__(self, command: str):
        self.command = command
        self._started = time.monotonic()
        self._rows: list[tuple[str, str]] = []
        self._failures: list[str] = []

    def add(self, label: str, value: object) -> None:
        self._rows.append((label, str(value)))

    def fail(self, message: str) -> None:
        self._failures.append(message)

    @property
    def ok(self) -> bool:
        return not self._failures

    def emit(self) -> int:
        elapsed = time.monotonic() - self._started
        width = max((len(label) for label, _ in self._rows), default=0)
        print(f"\n{RULE}")
        print(f"run receipt — {self.command}")
        print(RULE)
        for label, value in self._rows:
            print(f"  {label.ljust(width)}  {value}")
        for message in self._failures:
            print(f"  FAIL  {message}")
        status = "OK" if self.ok else f"FAILED ({len(self._failures)})"
        print(f"  status: {status}  ({elapsed:.1f}s)")
        print(RULE)
        return 0 if self.ok else 1

    def finish(self) -> None:
        sys.exit(self.emit())
