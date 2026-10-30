"""Frame identity for blocks returning delayed stream predictions."""

from typing import Any, Dict, List, Optional, Tuple


class StreamPipelineBatchResult(list):
    """A batch result carrying submission and response identities outside its data.

    Args:
        outputs: Ordinary block outputs, possibly containing unresolved futures.
        context_id: Identity of the frame submitted by this invocation.
        response_context_id: Identity of the returned frame, or None for warmup.
        response_indices: Batch indices belonging to the returned frame.

    Notes:
        A synchronous fallback returns the current context ID. A delayed result
        returns an earlier ID. Warmup outputs must not reach downstream blocks.
        Flush results use the same ID for submission and response.
    """

    def __init__(
        self,
        outputs: List[Dict[str, Any]],
        *,
        context_id: str,
        response_context_id: Optional[str],
        response_indices: List[Tuple[int, ...]],
    ) -> None:
        super().__init__(outputs)
        self.context_id = context_id
        self.response_context_id = response_context_id
        self.response_indices = response_indices
