"""Simple timer utility for measuring code execution time."""

from __future__ import annotations

import time


class Timer:
    """Context manager that measures elapsed wall-clock time.

    The timer records the start and end timestamps, stores the elapsed duration, and
    optionally prints a message whenever the context exits.
    """

    def __init__(self, message: str | None = None):
        """Create a timer with an optional human-readable label.

        Args:
            message: Optional label printed when the timer exits.
        """
        self.message = message
        self.start_time: float | None = None
        self.elapsed = 0.0

    def __enter__(self):
        """Start measuring once the context is entered.

        Returns:
            The timer instance itself so it can be used in a ``with`` block.
        """
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record elapsed time at the end of the context.

        Args:
            exc_type: Exception type raised in the context, if any.
            exc_val: Exception instance raised in the context, if any.
            exc_tb: Traceback for the exception, if any.

        Returns:
            ``False`` so exceptions are not suppressed by the context manager.
        """
        if self.start_time is None:
            self.elapsed = 0.0
            return False

        self.elapsed = time.perf_counter() - self.start_time
        if self.message:
            print(f"{self.message}: {self.elapsed:.2f}s")
        return False
