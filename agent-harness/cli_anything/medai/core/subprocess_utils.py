from __future__ import annotations

from typing import Any


def subprocess_text(value: Any) -> str:
    """Normalize TimeoutExpired output, which may be bytes even with text=True."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if value is None:
        return ""
    return str(value)
