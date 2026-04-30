# Roboflow Inference on Jetson — Static Analysis of Parallel-Stream Throughput

> Scoping memo that motivated `perf/optimize-rfdetr-seg-plus-is-seg-dataclasses`.
> Kept for reference; the optimizations it identified are implemented on that branch.

## Engine Configuration (RESOLVED)

The `trt_config.json` shipped alongside the `rfdetr-seg-nano` engine is available in-repo as a test asset:

- `/home/ubuntu/inference/inference_models/tests/integration_tests/models/assets/models/rfdetr-seg-nano-t4-trt/unpacked/trt_config.json`:
  `{"static_batch_size": 1, "dynamic_batch_size_min": null, "dynamic_batch_size_opt": null, "dynamic_batch_size_max": null}`
- `inference_config.json` in the same directory: `training_input_size: 312×312`, stretch resize, ImageNet normalization, `dynamic_spatial_size_supported: false`.
- Engine file size: ~187 MB.

**What this means.** The shipped engine is **hard-coded to batch size 1**. The cross-stream batching path in `trt.py:607-675` cannot engage — with `max_batch_size=1`, any N-frame batch is split into N serial single-frame forwards in a Python loop (`trt.py:644-675`), each holding the `threading.Lock()` at `rfdetr_instance_segmentation_trt.py:253`. This confirms the static-analysis hypothesis: Opportunity #1 is real and blocked by engine config, not code.

**Target hardware from the engine blob: Tesla T4 (sm_75, Turing).** Confirmed by dumping the TRT engine inspector JSON on `rfdetr-seg-nano-t4-trt/unpacked/engine.plan` — every kernel tactic in the engine is prefixed `sm75_xmma_*` (e.g., `sm75_xmma_gemm_f16f16_f16f16_f16_...`). `hardware_compatibility_level` is `NONE`, meaning the engine is locked to sm_75 and will not load on Orin NX (sm_87, Ampere). The `-t4-trt` filename suffix matches: this package is T4-only.

**Implication for Orin NX deployment.** There is **no on-device rebuild path invoked at runtime** — `RUNS_ON_JETSON` appears in the codebase only at `inference/core/env.py:701` and `video_source.py:184` (for V4L2 capture-backend selection). The TRT compile entry points in `inference_models/development/compile_rfdetr.py` and `compilation/engine_builder.py` are developer scripts, not runtime hooks. Therefore, the Orin NX deployment must be pulling a **separate, Orin-built `.plan`** that is not present in this repo checkout. The `trt_config.json` we can see (`static_batch_size: 1`) is almost certainly the same config used for the Orin build, but this cannot be confirmed without reading the Orin model cache directly.

**Other caveats:**
- Training input size shown is 312×312, not 560×560 as in `development/compile_rfdetr.py:22`. The compile script is a different variant (base, not nano). Nano-seg genuinely runs at 312×312 — so inference is on a substantially smaller tensor than the 560×560 assumption in earlier notes.
- This is a test asset snapshot (2026-04-21). The production Orin package could carry a different config.

## Target Hardware

**Jetson Orin NX** (confirmed by Brad). Implications for the analysis below:
- Single NVDEC engine — easily handles 4× 1080p30 H.264/H.265 on paper, so NVDEC is a real lever (§1).
- 2 DLA cores available (unlike Xavier NX's 2, Nano's 0), but DLA value for RF-DETR seg is still gated by the transformer head (§8).
- 8 GB or 16 GB unified LPDDR5, shared with CPU — makes zero-copy / mapped-memory particularly attractive and makes large-batch engines memory-constrained (§3, §6).
- 6 or 8 Cortex-A78AE cores — CPU decode + CPU preprocessing + Python GIL is a harder ceiling here than on AGX Orin (§1, §2, §7).
- Current observed per-stream FPS and target FPS: **TBD — to be established by runtime measurement.**

## Executive Summary

- **Current architecture (one paragraph).** `InferencePipeline` spawns one decode thread per `VideoSource` using OpenCV `cv2.VideoCapture` (CPU decode). A multiplexer collects frames from N sources into a single batch list and hands them to a single inference thread, which invokes a compiled Workflow. The one model block (`RFDetrForInstanceSegmentationTRT`) holds a single TensorRT engine, a single `IExecutionContext`, and a `threading.Lock()` that serializes every forward pass. Pre-process runs on one CUDA stream, the forward runs on a second, post-process on a third, each followed by a `synchronize()`. With N parallel streams, only the decode stage is actually parallel; model execution is strictly sequential.
- **Top 3 throughput opportunities, ranked.**
  1. **Batched forward across streams.** High-confidence. N streams arrive as a list but batch size 1 engines are the default observed path; moving from N sequential forwards to one batched forward is the biggest potential lever. (`_infer_from_trt_engine_with_batch_size_boundaries`, `trt.py:588-604`).
  2. **CPU video decode + CPU-side preprocessing.** High-confidence from code, size pending runtime evidence. Decode is pure `cv2.VideoCapture` (`video_source.py:140-142`); the numpy preprocessing path does `cv2.cvtColor`/`cv2.resize` on CPU before H2D copy (`pre_processing.py:973,1052-1053`). On Jetson this competes with the Python GIL and inference.
  3. **Per-stage `stream.synchronize()` scaffolding.** Medium-confidence hypothesis. Each of pre/forward/post ends with a blocking sync (`rfdetr_instance_segmentation_trt.py:243, 293` plus the lock at 253); event-based chaining could overlap stages across frames.
- **Negatives worth flagging early:** No NVDEC, GStreamer, DeepStream, VPI, or DLA code paths anywhere in the repo (see §1, §8, §10). INT8 flag is supported in builder but no calibration code is committed (§3).

## Current Implementation Map

Tracing one frame from a single stream:

1. A `VideoSource` is constructed around `CV2VideoFrameProducer` (`inference/core/interfaces/camera/video_source.py:136-142`). On Jetson camera devices it uses `cv2.CAP_V4L2`; otherwise the default FFmpeg backend. No GStreamer/NVDEC pipeline string is ever passed.
2. Each source runs a dedicated decode thread started at `video_source.py:635` (`_consume_video`), pushing decoded CPU `ndarray` frames into a per-source `frames_buffer` Queue.
3. `InferencePipeline._generate_frames()` (`inference_pipeline.py:1031-1044`) delegates to `multiplex_videos()`, which calls `VideoSourcesManager.retrieve_frames_from_sources()` (`camera/utils.py:143-175`). That method loops over sources serially, appending whatever frame is ready within `batch_collection_timeout`.
4. The inference thread (`inference_pipeline.py:914-922`) runs `predictions = self._on_video_frame(video_frames)` on the batch, then pushes `(predictions, video_frames)` onto a dispatch queue.
5. `_on_video_frame` is a partial wrapping `WorkflowRunner.run_workflow` (`model_handlers/workflows.py:10-60`), which wraps each frame as `{"type": "numpy_object", "value": ndarray, ...}` and calls `execution_engine.run(...)`.
6. The Workflow is compiled once at pipeline init (`inference_pipeline.py` init_with_workflow path; compilation cached at `workflows/execution_engine/v1/compiler/core.py:60-69`). The single-block Workflow executes the RF-DETR seg block, which calls `model.infer([...])`.
7. Inside `RFDetrForInstanceSegmentationTRT`: `pre_process` (line 225) → `forward` (246) → `post_process` (268). Forward acquires `self._lock` and pushes the primary CUDA context (line 253-254). Under the hood, `_execute_trt_engine` either replays a cached CUDA graph (`trt.py:706-713`) or sets dynamic input shape and runs `execute_async_v3` (`trt.py:716-740`).

**What changes with N parallel streams.** Decode remains N-way parallel (N OS threads, CPU). The multiplexer serializes retrieval into one list. The inference thread processes that list as one workflow invocation; the SIMD-batch workflow path (`workflows/execution_engine/v1/executor/core.py:317-378`) passes the full list to the block's single `run()` call. The block, however, receives a list of numpy arrays, and preprocessing loops over them one-by-one on CPU (`pre_processing.py:843-865`) before concatenating tensors. The TRT forward is thus the only place where batching across streams could pay off, and only if the engine was built with `dynamic_batch_size_max ≥ N`.

## Findings by Category

### 1. Stream ingestion and decode path

- **What the code does:** All decoding goes through `cv2.VideoCapture(video)` or `cv2.VideoCapture(video, cv2.CAP_V4L2)` (`video_source.py:140-142`). There is no GStreamer pipeline string, no PyAV, no DeepStream, no Jetson Multimedia API usage anywhere in `inference/` or `inference_models/`. `retrieve()` returns a host `ndarray` (`video_source.py:150-151`).
- **Why it matters:** On Jetson, FFmpeg-backed `VideoCapture` does CPU H.264/H.265 decode by default, bypassing NVDEC. For N 30fps streams this is N separate CPU decoders fighting the GIL alongside inference.
- **Confidence:** verified from code for the decoder path; "FFmpeg CPU decode" is the default runtime behavior of OpenCV unless a GStreamer pipeline is passed, which it never is here.
- **Estimated opportunity:** medium-to-large — NVDEC on Orin can decode many 1080p30 streams at negligible CPU cost.

### 2. Preprocessing pipeline

- **What the code does:** The numpy path (`pre_process_numpy_images_list`, `pre_processing.py:843-865`) calls `pre_process_numpy_image` per image in a Python `for` loop, concatenating tensors at the end. Within each iteration, grayscale/contrast transforms and resize use `cv2.cvtColor` (`:973`) and `cv2.resize` (`:1052`, `:1104`, `:1254`) on the CPU. Only after the resize does the code do `torch.from_numpy(...).to(target_device)` (`:1053, :1105, :1269`) — one H2D copy per image.
- **Why it matters:** On N streams this is N CPU resizes plus N H2D copies per multiplex cycle. For RF-DETR nano seg at 560×560 input, this is non-trivial but small per-frame; it becomes the bottleneck when decode+preprocess time exceeds the available Python runtime budget between forwards.
- **Confidence:** verified from code.
- **Estimated opportunity:** medium — batching resize on GPU (e.g., via torchvision, VPI, or a single concat-then-resize) would remove the per-image Python overhead.

### 3. TensorRT engine instantiation and batching (CRITICAL)

- **Single context, lock-serialized.** `engine.create_execution_context()` is called once in `from_pretrained` (`rfdetr_instance_segmentation_trt.py:159`), stored as `self._execution_context`. Every `forward()` acquires `self._lock = threading.Lock()` (set at `:216`) before invoking the engine (`:253-265`). This lock is the choke point for parallel streams — even if the workflow hands the block a list of N frames, the forward is still a single lock-protected call.
- **Dynamic shapes and batching are supported but splitting/padding happens automatically.** `_infer_from_trt_engine_with_batch_size_boundaries` (`trt.py:607-675`) pads a batch up to `min_batch_size` with zeros (`:619-631`) or splits anything larger than `max_batch_size` into serial sub-batches in a Python for loop (`:644-675`). The engine build profile reads `dynamic_batch_size_min/opt/max` from `trt_config.json` (`model_packages.py:146-196`).
- **CUDA graphs, when enabled, require fixed shapes.** A separate `graph_context = engine.create_execution_context()` is created per captured shape (`trt.py:759`, comment at `:756-758`). The replay path (`:706-713`) is the fast path and avoids the per-call `set_input_shape`/`set_tensor_address` work at `:716-722`.
- **Precision.** Builder supports FP32/FP16/INT8 (`engine_builder.py:80-87`). The RF-DETR compile script at `inference_models/development/compile_rfdetr.py:22` requests FP16/FP32 for 560×560 inputs, with 15 GB workspace (`:11`). No committed INT8 calibration code.
- **Why it matters:** With `max_batch_size ≥ N`, N streams could be one forward. But if the `.plan` shipped to the Jetson was built with `static_batch_size=1` or `max=1`, the code at `trt.py:644-675` degrades to a Python loop of single-frame forwards. The practical throughput for parallel streams therefore depends entirely on what `trt_config.json` ships inside the RF-DETR seg model package — which we cannot see from this repo alone.
- **Confidence:** single-context + global lock is verified from code; actual batch capabilities of the shipped `.plan` are a runtime artifact we cannot read.
- **Estimated opportunity:** large — if the engine is batch-1, rebuilding with a dynamic profile unlocks the single biggest lever.

### 4. CUDA stream usage and async execution

- **What the code does:** Three CUDA streams per model instance — one inference stream on the class (`rfdetr_instance_segmentation_trt.py:217`), plus per-thread pre- and post-process streams via `threading.local()` (`:218`, `:297-310`). The forward uses `execute_async_v3(stream_handle=stream.cuda_stream)` (`trt.py:740`). However, each stage ends with an explicit `stream.synchronize()` (`:243`, `:293`, `trt.py:712`).
- **Why it matters:** The streams are real, but the synchronize barriers between stages mean pre-process finishes → sync → forward → sync → post → sync, serially. Within a single forward call, the async execution is in principle overlapping with any prior post-process work on a different stream; in practice the `synchronize()` at end of each stage blocks until completion and the downstream stage waits.
- **Confidence:** verified from code for the sync points; whether they are required for correctness or just defensive is a code-reading judgment — the post-process sync at `:293` is plausibly redundant because consumers of the returned `InstanceDetections` will cause their own syncs.
- **Estimated opportunity:** small-to-medium — cross-frame pipelining would require a redesign of the model handler; the per-frame overhead is likely tens of milliseconds total but won't multiply with N.

### 5. Postprocessing

- **What the code does:** RF-DETR seg post-processing runs entirely on GPU tensors (`inference_models/models/rfdetr/common.py:45-129`): sigmoid, top-k, gather on masks, cxcywh→xyxy. `align_instance_segmentation_results` is GPU-based. No `.cpu()`, `.numpy()`, or `.item()` calls inside the post_process method at `rfdetr_instance_segmentation_trt.py:268-294`. Final `.cpu()` transfers happen at the workflow boundary, not inside the model.
- **Why it matters:** Post-processing is already well-structured. It is not a major target.
- **Confidence:** verified from code.
- **Estimated opportunity:** small.

### 6. Memory management

- **What the code does:** CUDA graphs reuse `input_buffer`/`output_buffers` allocated at capture time (`trt.py:761, 779-789`), cached in an LRU keyed by input shape (`TRTCudaGraphCache`, `trt.py:83-272`, default size 8). The non-graph path, however, allocates a fresh output tensor every forward (`trt.py:732-736`: `result = torch.empty(tuple(output_tensor_shape), ...)`). No `cudaHostAlloc` with mapped flag, no `cudaMallocManaged`, no pinned-memory pool — the only pinning comes from PyTorch's default allocator.
- **Why it matters:** Jetson's physically unified memory means CPU→GPU copies are logical, not over PCIe, but Torch still does a staged copy by default. Zero-copy of the decoded frame into a GPU-addressable buffer would cut a full frame-sized H2D per stream per frame. This repo has no such path.
- **Confidence:** verified from code for absence of zero-copy.
- **Estimated opportunity:** medium on Jetson; same change on discrete GPU would be small.

### 7. Threading / process model in InferencePipeline

- **What the code does:** N decode threads (one per `VideoSource`) → shared queues → one inference thread (`inference_pipeline.py:906-922`) → one dispatch thread. Workflow block execution can use a `ThreadPoolExecutor` but for a single-block workflow there is nothing to parallelize.
- **Why it matters:** This topology is actually good for batching — the multiplexer already produces a per-iteration list of frames, one per active source. The missed opportunity is downstream: the model handler serializes that list through one lock.
- **Confidence:** verified from code.
- **Estimated opportunity:** small in isolation (the topology is already set up to feed a batch); the value comes from pairing with batched inference (§3).

### 8. DLA usage

- **What the code does:** Nothing. `grep` for `DLA`, `useDLACore`, `setDeviceType`, `kDLA` across `inference_models/` returns no matches (confirmed by the TRT exploration). The only builder config beyond precision is `hardware_compatibility_level = trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY` (`engine_builder.py:91`).
- **Why it matters:** RF-DETR's transformer blocks almost certainly cannot target DLA, but the CNN backbone could if split. No such path exists.
- **Confidence:** verified from code (absence).
- **Estimated opportunity:** unknown — likely small for seg because of the transformer head, and would require engine-graph surgery.

### 9. Workflow / single-block overhead

- **What the code does:** Workflow compilation is cached (`workflows/execution_engine/v1/compiler/core.py:60-69`). Per frame, however, `assemble_runtime_parameters` pydantic-validates inputs (`runtime_input_assembler.py:16-42`), fresh `ExecutionCache`/`DynamicBatchesManager`/`BranchingManager` instances are created in `ExecutionDataManager.init()` (`execution_data_manager/manager.py:48-66`), and step-input assembly traverses the compound input graph. For a single-block SIMD workflow the block is called once per batch with the full list, so the overhead is amortized across all N frames in the batch.
- **Why it matters:** Python orchestration is per-*batch* not per-*frame*, which is a relief at higher stream counts. But per-batch it is still measurable.
- **Confidence:** verified from code.
- **Estimated opportunity:** small — would only show up at very high N or very fast inference (where Python overhead approaches forward time).

### 10. Jetson-specific code paths

- **What the code does:** Jetson detection exists (`inference/core/devices/utils.py:68-86`, reading `/proc/device-tree/serial-number`) and is used for device-id reporting and the V4L2 capture-backend choice (`video_source.py:183-188`). The adaptive buffer-filling strategy is the default for streams (`video_source.py:932` per agent report). No `nvpmodel`, `jetson_clocks`, JetPack version check, VPI library usage, or DeepStream integration anywhere.
- **Why it matters:** There is essentially no Jetson-specific optimization beyond "use V4L2 for USB cameras." RF-DETR on Jetson runs the same code path as on desktop CUDA.
- **Confidence:** verified from code.
- **Estimated opportunity:** medium — Jetson-specific hooks (NVDEC, VPI, mapped-memory frames) are untouched ground.

## Top Opportunities, Ranked

1. **Batched TRT forward across streams.** Feed the multiplexer's N-frame list as a single batched tensor to one `execute_async_v3`. *Impact (static-analysis estimate): 1.5–2x on N=4 if the engine supports it.* **Runtime evidence needed:** the `trt_config.json` shipped in the RF-DETR seg model package (look at `static_batch_size` and `dynamic_batch_size_max`), and a `trtexec --loadEngine --shapes` run at batch 1 vs 4 to confirm that amortized kernel time scales sublinearly.
2. **GPU video decode via NVDEC / GStreamer pipeline string into `VideoCapture`.** Replace `cv2.VideoCapture(video)` at `video_source.py:140-142` with a GStreamer appsink pipeline that uses `nvv4l2decoder`/`nvvidconv` on Jetson. *Impact: 10–30% CPU headroom at N=4, 30fps 1080p; also removes GIL contention with the inference thread.* **Runtime evidence:** `tegrastats` during a 4-stream run showing CPU saturation; Nsight Systems trace showing decode threads blocking inference.
3. **GPU-side batched preprocessing (skip the per-image Python loop).** The current `pre_process_numpy_images_list` loop (`pre_processing.py:843-865`) with its per-image `cv2.resize` → `from_numpy` → `.to(device)` chain is a serial CPU gauntlet. *Impact: 1.1–1.3x at N=4.* **Runtime evidence:** Nsight kernel timeline showing gaps between H2D copies; CPU sampling profiler (py-spy) showing time in `cv2.resize` and `to_numpy`.
4. **Eliminate mid-pipeline `stream.synchronize()` barriers.** The three stage-ending syncs (`rfdetr_instance_segmentation_trt.py:243, 293`, `trt.py:712`) plus the global lock mean no overlap across frames. *Impact: 5–20% at N=4, depending on how close pre/post times are to forward time.* **Runtime evidence:** Nsight Systems trace showing gaps between kernels on the inference stream.
5. **Zero-copy or pinned-memory input buffer on Jetson.** The non-CUDA-graph path allocates output tensors every forward (`trt.py:732-736`); the input also transits a staged copy. *Impact: 5–15% at N=4.* **Runtime evidence:** Nsight memory trace showing H2D/D2H transfer time relative to compute.

## What Static Analysis Cannot Tell Us

- **The actual batch capabilities of the shipped `.plan` file.** `dynamic_batch_size_max` lives in the model package `trt_config.json`, not in this repo. *Resolves with:* reading `trt_config.json` from the cached model package directory, or `trtexec --loadEngine=engine.plan --verbose` dumping profile ranges.
- **Where wall-clock time is actually being spent.** Decode vs preprocess vs forward vs post vs Python overhead. *Resolves with:* Nsight Systems CPU+GPU timeline for a 4-stream run.
- **Whether `execute_async_v3` kernels on the three streams actually overlap, or serialize on a single engine queue.** TRT execution contexts cannot run two enqueues concurrently on one context regardless of stream. *Resolves with:* Nsight GPU kernel timeline showing kernel concurrency or lack thereof.
- **Whether CUDA graph capture is actually in use for the RF-DETR seg workload.** The cache is set up (`establish_trt_cuda_graph_cache`) but activation depends on `disable_cuda_graphs` and input-shape stability. *Resolves with:* a log line or trace confirming graph replay vs full enqueue.
- **Whether the OpenCV build on this Jetson is compiled with GStreamer support.** Without GStreamer in OpenCV, no pipeline-string backdoor is possible. *Resolves with:* `cv2.getBuildInformation()` on the target.
- **Python GIL contention between decode threads and the inference thread.** *Resolves with:* py-spy sampled profile during a 4-stream run.
- **Whether the `nvpmodel` / `jetson_clocks` state on the deployment unit has the GPU clocked to max.** *Resolves with:* `sudo nvpmodel -q` and `sudo jetson_clocks --show`.

## Open Questions for Brad / Pawel

1. **[OPEN — answer not yet known] What is the `.plan` file's dynamic shape profile** for RF-DETR nano seg as currently shipped from the inference-models cache? Specifically: what are `dynamic_batch_size_min/opt/max` in the shipped `trt_config.json`? This changes which optimization is first. Can be answered by reading the cached model package on an Orin NX that has already pulled the engine.
2. **[PARTIALLY RESOLVED — code supports both patterns, production choice still unknown]** The codebase supports two deployment patterns:
   - **Pattern A (single process, multiple streams):** `InferencePipeline.init_with_workflow(video_reference=[...])` — `inference_pipeline.py:90-91` accepts a list; docs at `docs/workflows/video_processing/overview.md:71` demonstrate it. Example in-repo: `development/stream_interface/yolo_world_demo.py:19`. One model instance, one lock, one TRT context.
   - **Pattern B (multiple processes):** Enterprise Stream Manager spawns one `InferencePipelineManager(Process)` per pipeline (`inference/enterprise/stream_management/manager/inference_pipeline_manager.py:44`, spawned from `manager/app.py:146`). Each process independent: own engine, own context, no shared lock. On Jetson without MPS, contexts time-slice the GPU.
   - **Impact on opportunities:** batched-forward (#1) only helps Pattern A; CUDA MPS and per-process engine-sharing tricks only help Pattern B; NVDEC decode, stage-sync removal, and GPU preprocessing help both. Needs product confirmation which is in use for the Orin NX deployment.
3. **[ANSWERED] Target Jetson:** Orin NX. Captured in the "Target Hardware" section above.
4. **[TO BE MEASURED] Actual observed per-stream FPS today on Orin NX, and expected FPS at 2x.** Needs a profiling run; gives us the budget to compare against the static-analysis estimates above.
5. **[OPEN — answer not yet known] Was the `.plan` built on-device via the `RUNS_ON_JETSON` compile path, or shipped prebuilt?** Relevant because engines built with different `max_batch` may exist in the model cache, and rebuilding with a new profile is a different task from swapping an engine. Folds into Q1 — both resolve by inspecting what's in the model cache on a live Orin NX.
