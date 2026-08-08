"""Bounded shutdown for Isaac Sim command-line utilities."""

from __future__ import annotations

import os
import threading
from typing import Protocol


class ClosableApplication(Protocol):
    def close(self) -> None: ...


def close_simulation_app(
    application: ClosableApplication,
    *,
    timeout_seconds: float = 15.0,
    forced_exit_code: int = 0,
) -> None:
    """Close Isaac Sim, forcing process teardown if the container blocks forever.

    Isaac Sim 5.1 on the audited server completes simulation work but can block in
    ``application.close()``. The watchdog preserves normal close behavior when it
    works and uses ``os._exit`` only after the requested timeout. Callers must pass
    a non-zero ``forced_exit_code`` while handling an active exception so failures
    cannot be converted into successful exits.
    """
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    close_finished = threading.Event()

    def watchdog() -> None:
        if not close_finished.wait(timeout_seconds):
            print(
                f"[WARN] Isaac Sim close exceeded {timeout_seconds:.1f}s; forcing process teardown.",
                flush=True,
            )
            os._exit(forced_exit_code)

    threading.Thread(target=watchdog, name="isaac-close-watchdog", daemon=True).start()
    application.close()
    close_finished.set()
