"""Delayed results must resume the entire originating workflow frame."""

from concurrent.futures import Future
from typing import Literal

import networkx as nx
import pytest
from roboflow_workflows.errors import StepExecutionError
from roboflow_workflows.execution_engine.entities.base import (
    JsonField,
    OutputDefinition,
    WorkflowBatchInput,
    WorkflowParameter,
)
from roboflow_workflows.execution_engine.entities.stream_pipeline import (
    StreamPipelineBatchResult,
)
from roboflow_workflows.execution_engine.entities.types import WILDCARD_KIND
from roboflow_workflows.execution_engine.profiling.core import NullWorkflowsProfiler
from roboflow_workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    CompiledWorkflow,
    DynamicStepInputDefinition,
    InitialisedStep,
    InputNode,
    NodeCategory,
    NodeInputCategory,
    OutputNode,
    ParameterSpecification,
    ParsedWorkflowDefinition,
    StepNode,
)
from roboflow_workflows.execution_engine.v1.executor.core import (
    _run_workflow,
    flush_stream_pipeline_workflow,
)
from roboflow_workflows.execution_engine.v1.executor.stream_pipeline import (
    _StreamPipelineExecutionState,
)
from roboflow_workflows.prototypes.block import WorkflowBlock, WorkflowBlockManifest


class _Manifest(WorkflowBlockManifest):
    type: Literal["test/frame@v1"]

    @classmethod
    def accepts_batch_input(cls):
        return True

    @classmethod
    def describe_outputs(cls):
        return [OutputDefinition(name="value")]


class _Crop(WorkflowBlock):
    def __init__(self):
        self.calls = 0
        self.skip = set()

    @classmethod
    def get_manifest(cls):
        return _Manifest

    def run(self, images):
        self.calls += 1
        return [
            {"value": None if image in self.skip else f"crop({image})"}
            for image in images
        ]


class _DelayedModel(_Crop):
    def __init__(self):
        super().__init__()
        self.pending = None
        self.synchronous = set()
        self.bad_identity = False

    def can_activate_stream_pipeline(self):
        return True

    def stream_pipeline_depth(self):
        return 1

    def run(self, images):
        self.calls += 1
        if len(images) != 1:
            return [{"value": image} for image in images]
        context = str(self.calls)
        value = images[0]
        if value in self.synchronous:
            response = (context, value)
        else:
            response, self.pending = self.pending, (context, value)
        future = Future()
        future.set_result(response[1] if response else "warmup")
        return StreamPipelineBatchResult(
            [{"value": future}],
            context_id=context,
            response_context_id=(
                "unknown" if self.bad_identity else response[0] if response else None
            ),
            response_indices=[(0,)],
        )

    def flush_stream_pipeline_outputs(self):
        if self.pending is None:
            return []
        context, value = self.pending
        self.pending = None
        return [
            (
                [(0,)],
                StreamPipelineBatchResult(
                    [{"value": value}],
                    context_id=context,
                    response_context_id=context,
                    response_indices=[(0,)],
                ),
            )
        ]

    def close_stream_pipeline(self):
        self.pending = None


class _Render(_Crop):
    def run(self, images, predictions, auxiliary, tag):
        self.calls += 1
        return [
            {"value": (image, prediction, aux, tag)}
            for image, prediction, aux in zip(images, predictions, auxiliary)
        ]


def _workflow():
    crop, model, render = _Crop(), _DelayedModel(), _Render()
    blocks = {
        "crop": crop,
        "model": model,
        "aux1": _Crop(),
        "aux2": _Crop(),
        "aux3": _Crop(),
        "render": render,
    }
    wiring = {
        "crop": {"images": "$inputs.image"},
        "model": {"images": "$steps.crop.value"},
        "aux1": {"images": "$inputs.image"},
        "aux2": {"images": "$steps.aux1.value"},
        "aux3": {"images": "$steps.aux2.value"},
        "render": {
            "images": "$steps.crop.value",
            "predictions": "$steps.model.value",
            "auxiliary": "$steps.aux3.value",
            "tag": "$inputs.tag",
        },
    }
    lineage = ["<workflow_input>"]
    graph = nx.DiGraph()
    inputs = [
        WorkflowBatchInput(type="WorkflowBatchInput", name="image"),
        WorkflowParameter(type="WorkflowParameter", name="tag"),
    ]
    for inp in inputs:
        selector = f"$inputs.{inp.name}"
        graph.add_node(
            selector,
            node_compilation_output=InputNode(
                node_category=NodeCategory.INPUT_NODE,
                name=inp.name,
                selector=selector,
                data_lineage=lineage if inp.name == "image" else [],
                input_manifest=inp,
            ),
        )
    steps = {}
    for name, block in blocks.items():
        manifest = _Manifest(type="test/frame@v1", name=name)
        selector = f"$steps.{name}"
        parameters = {}
        for key, source in wiring[name].items():
            category = NodeInputCategory.BATCH_STEP_OUTPUT
            if source.startswith("$inputs"):
                category = (
                    NodeInputCategory.NON_BATCH_INPUT_PARAMETER
                    if key == "tag"
                    else NodeInputCategory.BATCH_INPUT_PARAMETER
                )
            parameters[key] = DynamicStepInputDefinition(
                parameter_specification=ParameterSpecification(parameter_name=key),
                category=category,
                data_lineage=[] if key == "tag" else lineage,
                selector=source,
            )
            graph.add_edge(
                source if source.startswith("$inputs") else source.rsplit(".", 1)[0],
                selector,
            )
        graph.add_node(
            selector,
            node_compilation_output=StepNode(
                node_category=NodeCategory.STEP_NODE,
                name=name,
                selector=selector,
                data_lineage=lineage,
                step_manifest=manifest,
                input_data=parameters,
                batch_oriented_parameters=set(parameters) - {"tag"},
                step_execution_dimensionality=1,
            ),
        )
        steps[name] = InitialisedStep(
            block_specification=BlockSpecification(
                block_source="test",
                identifier="test/frame@v1",
                block_class=type(block),
                manifest_class=_Manifest,
            ),
            manifest=manifest,
            step=block,
        )
    output = JsonField(type="JsonField", name="result", selector="$steps.render.value")
    graph.add_node(
        "$outputs.result",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name="result",
            selector="$outputs.result",
            data_lineage=lineage,
            output_manifest=output,
            kind=[WILDCARD_KIND],
        ),
    )
    graph.add_edge("$steps.render", "$outputs.result")
    workflow = CompiledWorkflow(
        workflow_definition=ParsedWorkflowDefinition(
            version="1.0",
            inputs=inputs,
            steps=[s.manifest for s in steps.values()],
            outputs=[output],
        ),
        execution_graph=graph,
        steps=steps,
        input_substitutions=[],
        workflow_json={},
        init_parameters={},
        kinds_serializers={},
        kinds_deserializers={},
    )
    return workflow, crop, model, render


def _run(workflow, state, frame):
    return _run_workflow(
        workflow=workflow,
        runtime_parameters={"image": [frame], "tag": f"tag-{frame}"},
        max_concurrent_steps=2,
        kinds_serializers={},
        profiler=NullWorkflowsProfiler.init(),
        defer_stream_pipeline_flush=state is not None,
        stream_pipeline_state=state,
    )


def _flush(workflow, state):
    # Deliberately wrong runtime parameters: retained upstream state must win.
    return flush_stream_pipeline_workflow(
        workflow=workflow,
        runtime_parameters={"image": [999], "tag": "wrong"},
        max_concurrent_steps=2,
        kinds_serializers={},
        profiler=NullWorkflowsProfiler.init(),
        stream_pipeline_state=state,
    )


def _expected(frame):
    return [
        {
            "result": (
                f"crop({frame})",
                f"crop({frame})",
                f"crop(crop(crop({frame})))",
                f"tag-{frame}",
            )
        }
    ]


@pytest.mark.parametrize("frames", [1, 5])
def test_delayed_predictions_keep_crop_parameters_and_long_independent_branch(frames):
    workflow, crop, model, render = _workflow()
    state = _StreamPipelineExecutionState._init(workflow)
    for frame in range(1, frames + 1):
        result = _run(workflow, state, frame)
        assert result == ([] if frame == 1 else _expected(frame - 1))
        assert len(state.pending) == len(state.contexts) == 1
        assert render.calls == frame - 1
    assert _flush(workflow, state) == _expected(frames)
    assert crop.calls == model.calls == render.calls == frames
    assert not state.pending and not state.contexts
    assert _flush(workflow, state) == []


def test_synchronous_fallback_does_not_replace_the_pending_frame():
    workflow, crop, model, render = _workflow()
    model.synchronous = {"crop(2)"}
    state = _StreamPipelineExecutionState._init(workflow)
    assert _run(workflow, state, 1) == []
    assert _run(workflow, state, 2) == _expected(1)
    assert _run(workflow, state, 3) == _expected(2)
    assert _flush(workflow, state) == _expected(3)
    assert crop.calls == render.calls == 3


def test_skipped_model_flushes_previous_frame_and_preserves_the_skipped_frame():
    workflow, crop, model, render = _workflow()
    crop.skip = {2}
    state = _StreamPipelineExecutionState._init(workflow)
    assert _run(workflow, state, 1) == []
    assert _run(workflow, state, 2) == _expected(1)
    assert _run(workflow, state, 3) == [{"result": None}]
    assert _flush(workflow, state) == _expected(3)
    assert crop.calls == 3
    assert model.calls == render.calls == 2


def test_regular_run_finishes_model_before_rendering():
    workflow, crop, model, render = _workflow()
    for frame in (1, 2):
        assert _run(workflow, None, frame) == _expected(frame)
    assert crop.calls == model.calls == render.calls == 2


def test_unknown_response_identity_fails_before_rendering():
    workflow, _, model, render = _workflow()
    state = _StreamPipelineExecutionState._init(workflow)
    assert _run(workflow, state, 1) == []
    model.bad_identity = True
    with pytest.raises(RuntimeError, match="pending workflow frame"):
        _run(workflow, state, 2)
    assert render.calls == 0
    assert not state.pending and not state.contexts
    assert model.pending is None


def test_engine_owns_frame_state_and_releases_it_on_close():
    from roboflow_workflows.execution_engine.v1.core import ExecutionEngineV1

    workflow, _, model, _ = _workflow()
    engine = ExecutionEngineV1(
        compiled_workflow=workflow,
        max_concurrent_steps=2,
        prevent_local_images_loading=True,
        profiler=NullWorkflowsProfiler.init(),
    )
    assert (
        engine.run({"image": [1], "tag": "tag-1"}, defer_stream_pipeline_flush=True)
        == []
    )
    assert engine.stream_pipeline_depth() == 1
    with pytest.raises(RuntimeError, match="Flush pending"):
        engine.run({"image": [2], "tag": "tag-2"})
    assert engine.flush_stream_pipeline({"image": [999], "tag": "wrong"}) == _expected(
        1
    )
    assert engine.flush_stream_pipeline({"image": [999], "tag": "wrong"}) == []
    assert (
        engine.run({"image": [2], "tag": "tag-2"}, defer_stream_pipeline_flush=True)
        == []
    )
    engine.close()
    assert not engine._stream_pipeline_state.pending
    assert not engine._stream_pipeline_state.contexts
    assert model.pending is None
    assert engine.stream_pipeline_depth() == 0


def test_multiple_pipeline_candidates_fail_before_executing():
    workflow, crop, _, render = _workflow()
    crop.can_activate_stream_pipeline = lambda: True
    with pytest.raises(RuntimeError, match="one pipelined step"):
        _StreamPipelineExecutionState._init(workflow)
    assert crop.calls == render.calls == 0


def test_inactive_pipeline_returns_current_frame_without_buffering():
    workflow, _, model, _ = _workflow()
    model.synchronous = {"crop(1)", "crop(2)"}
    model.stream_pipeline_depth = lambda: 0
    state = _StreamPipelineExecutionState._init(workflow)
    for frame in (1, 2):
        assert _run(workflow, state, frame) == _expected(frame)
        assert not state.pending and not state.contexts
    assert state.depth == 0


def test_tail_failure_releases_saved_state_and_model_queue():
    workflow, _, model, render = _workflow()
    state = _StreamPipelineExecutionState._init(workflow)
    assert _run(workflow, state, 1) == []

    def fail(**kwargs):
        raise RuntimeError("render failed")

    render.run = fail
    with pytest.raises(StepExecutionError, match="render failed"):
        _flush(workflow, state)
    assert not state.pending and not state.contexts
    assert model.pending is None


def test_batched_synchronous_fallback_preserves_each_frames_batch_indices():
    workflow, _, _, _ = _workflow()
    state = _StreamPipelineExecutionState._init(workflow)
    assert _run(workflow, state, 1) == []
    result = _run_workflow(
        workflow=workflow,
        runtime_parameters={"image": [2, 3], "tag": "batched"},
        max_concurrent_steps=2,
        kinds_serializers={},
        profiler=NullWorkflowsProfiler.init(),
        defer_stream_pipeline_flush=True,
        stream_pipeline_state=state,
    )
    assert result == _expected(1)
    result = _run(workflow, state, 4)
    assert result == [
        {"result": ("crop(2)", "crop(2)", "crop(crop(crop(2)))", "batched")},
        {"result": ("crop(3)", "crop(3)", "crop(crop(crop(3)))", "batched")},
    ]
    assert _flush(workflow, state) == _expected(4)
