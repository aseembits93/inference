"""Retain frame-local execution data across a delayed model invocation."""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import networkx as nx
from roboflow_workflows.execution_engine.entities.stream_pipeline import (
    StreamPipelineBatchResult,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import CompiledWorkflow
from roboflow_workflows.execution_engine.v1.executor.execution_data_manager.manager import (
    ExecutionDataManager,
)


@dataclass
class _PendingRun:
    data: ExecutionDataManager
    context_id: Optional[str] = None
    complete: bool = True


class _StreamPipelineExecutionState:
    """One stream's bounded queue; never shared between engine instances."""

    @classmethod
    def _init(
        cls, workflow: CompiledWorkflow
    ) -> Optional["_StreamPipelineExecutionState"]:
        candidates = []
        for name, step in workflow.steps.items():
            if any(
                callable(probe := getattr(step.step, method, None)) and probe()
                for method in ("is_stream_pipelined", "can_activate_stream_pipeline")
            ):
                candidates.append(name)

        if not candidates:
            return None

        if len(candidates) != 1:
            raise RuntimeError("Stream pipeline execution supports one pipelined step")

        state = cls(workflow, step_name=candidates[0])
        return state

    def __init__(self, workflow: CompiledWorkflow, *, step_name: str) -> None:
        self.step = workflow.steps[step_name].step
        self.selector = f"$steps.{step_name}"
        self.downstream = nx.descendants(workflow.execution_graph, self.selector)
        self.pending: deque[_PendingRun] = deque()
        self.contexts: dict[str, _PendingRun] = {}
        self.depth = 0

    def _finish_submission(
        self, data: ExecutionDataManager
    ) -> Optional[ExecutionDataManager]:
        result = data._stream_pipeline_outputs.pop(self.selector, None)
        current = _PendingRun(data=data)
        if result is not None:
            if result.context_id in self.contexts:
                raise RuntimeError("Stream submission reused a pending frame identity")

            current.context_id = result.context_id
            current.complete = False
            self.contexts[result.context_id] = current
            if result.response_context_id is not None:
                self._register_response(result)

        depth = max(0, int(self.step.stream_pipeline_depth()))
        if depth > 1:
            raise RuntimeError("Stream pipeline execution supports a one-frame delay")

        # Keep the output delay stable through skipped steps and synchronous
        # fallbacks, so the stream runner's frame queue stays aligned as well.
        self.depth = max(self.depth, depth)
        self.pending.append(current)
        if len(self.pending) <= self.depth:
            return None

        data = self._take_next()
        return data

    def _register_response(self, result: StreamPipelineBatchResult) -> None:
        target = self.contexts.get(result.response_context_id)
        if target is None or target.complete:
            raise RuntimeError("Stream response did not match a pending workflow frame")

        target.data.register_simd_step_output(
            step_selector=self.selector,
            indices=result.response_indices,
            outputs=list(result),
        )
        target.complete = True

    def _take_next(self) -> Optional[ExecutionDataManager]:
        if not self.pending:
            return None

        target = self.pending[0]
        if not target.complete:
            # Also handles a skipped model invocation: finish the previous
            # frame without running any upstream block again.
            for _, result in self.step.flush_stream_pipeline_outputs():
                if not isinstance(result, StreamPipelineBatchResult):
                    raise RuntimeError(
                        "Stream flush omitted the response frame identity"
                    )
                self._register_response(result)

        if not target.complete:
            raise RuntimeError(
                "Stream flush did not complete the pending workflow frame"
            )

        self.pending.popleft()
        self.contexts.pop(target.context_id, None)
        return target.data

    def _clear(self) -> None:
        self.pending.clear()
        self.contexts.clear()
        self.depth = 0
