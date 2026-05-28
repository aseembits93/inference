from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_WORKFLOW_STREAM_FLUSH_ACTIVE: ContextVar[bool] = ContextVar(
    "workflow_stream_flush_active",
    default=False,
)


def is_workflow_stream_flush_active() -> bool:
    return _WORKFLOW_STREAM_FLUSH_ACTIVE.get()


@contextmanager
def workflow_stream_flush_context() -> Iterator[None]:
    token = _WORKFLOW_STREAM_FLUSH_ACTIVE.set(True)
    try:
        yield
    finally:
        _WORKFLOW_STREAM_FLUSH_ACTIVE.reset(token)
