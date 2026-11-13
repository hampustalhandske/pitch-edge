"""Exceptions shared by the research harness."""

from __future__ import annotations


class ProtocolError(ValueError):
    """Raised when an action would break the research protocol (unregistered or edited hypothesis,
    config outside the registered grid, unregistered metric, inconsistent trial row, unknown review
    kind, ...)."""


class LedgerLockTimeout(TimeoutError):
    """Raised when the ledger file stayed locked by another process for longer than the retry budget."""
