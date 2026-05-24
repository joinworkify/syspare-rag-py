"""Generic retry helpers for transient Vertex/Google API failures."""

from __future__ import annotations

import random
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

# Class-name signals that almost always indicate transient infrastructure issues.
RETRIABLE_EXCEPTION_NAMES = frozenset(
    {
        "ResourceExhausted",       # 429 quota
        "ServiceUnavailable",      # 503
        "DeadlineExceeded",        # 504
        "InternalServerError",     # 500
        "Aborted",                 # 409
        "RetryError",
        "TooManyRequests",
        # gRPC channel errors
        "_InactiveRpcError",
        "_MultiThreadedRendezvous",
    }
)

# Message substrings that suggest a retry is worth attempting (e.g. transient
# upstream errors wrapped in a generic Exception).
RETRIABLE_MESSAGE_TOKENS: Iterable[str] = (
    "429",
    "500",
    "503",
    "504",
    "rate limit",
    "quota",
    "resource exhausted",
    "deadline exceeded",
    "service unavailable",
    "internal error",
    "temporarily",
    "try again",
    "timed out",
    "timeout",
    "broken pipe",
    "connection reset",
    "recvmsg",
)


def is_retriable(exc: BaseException) -> bool:
    """Return True if `exc` looks like a transient failure worth retrying."""
    name = type(exc).__name__
    if name in RETRIABLE_EXCEPTION_NAMES:
        return True
    msg = str(exc).lower()
    return any(token in msg for token in RETRIABLE_MESSAGE_TOKENS)


def retry_call(
    fn: Callable[..., T],
    *args,
    max_attempts: int = 5,
    base_delay: float = 1.5,
    max_delay: float = 60.0,
    label: str = "call",
    **kwargs,
) -> T:
    """Invoke `fn(*args, **kwargs)` with exponential backoff on transient errors.

    Non-retriable exceptions propagate immediately. On the final attempt, the
    last exception is re-raised so callers see the original stack trace.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts or not is_retriable(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, 0.5)
            print(
                f"[retry:{label}] attempt {attempt}/{max_attempts} hit "
                f"{type(exc).__name__}: {exc}. Sleeping {delay:.1f}s before retry...",
                flush=True,
            )
            time.sleep(delay)
    # Unreachable, but keeps type checker happy.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"retry loop for {label} exited without a result")
