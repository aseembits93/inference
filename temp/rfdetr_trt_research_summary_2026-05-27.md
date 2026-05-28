# RF-DETR TRT Research Summary

Date: 2026-05-27
Status: living document, updated as experiments continue

This document summarizes the RF-DETR TRT work completed so far in this branch.
It includes:

- the original parity and workflow-surface fixes,
- the exact-path optimizations that were kept,
- the profiling results that motivated the later work,
- the forward-focused experiments (builder/tactic sweeps, ONNX rewrites, TRT plugins, native plugins),
- benchmark numbers, and
- correctness results.

This is intended as a review document, not a changelog. Numbers below are the
load-bearing results from the work, with explicit notes where a number was later
discarded as noisy or as a harness artifact.

Unless otherwise noted, newer experiments should be appended here rather than
tracked only in terminal output. This file is the review surface for ongoing
RF-DETR TRT work.

## Scope

The work started with two concrete goals:

1. Establish exact parity for RF-DETR instance segmentation with all fast-path
   flags enabled.
2. Increase throughput on the real workflow benchmark:
   [development/stream_interface/rfdetr_nano_seg_trt_workflow.py](/home/ubuntu/inference/development/stream_interface/rfdetr_nano_seg_trt_workflow.py:1)
   on `vehicles_312px.mp4`.

The fast-path configuration used throughout most of this work:

```bash
RFDETR_TRITON_POSTPROC=true
INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=true
RFDETR_PIPELINE_DEPTH=2
ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=true
ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=true
```

## Important Caveats

- `temp/detection_parity_full.py` and `temp/workflow_parity_stream.py` are
  correctness gates, not throughput benchmarks.
- Some early workflow-speed conclusions were invalid because the parity harness
  was doing heavy serialization, rasterization, and pickling work.
- `nsys` slows the workflow path dramatically; the trace is still valid for
  hotspot analysis, but not for reporting unprofiled FPS.
- This host does not support `nsys` sampled CPU stacks or CPU context-switch
  tracing, so "CPU+GPU" profiling here means CUDA/NVTX plus OS-runtime wait
  timelines, not sampled CPU stacks.
- Exact-path workflow FPS has real machine variance. Same code has produced
  roughly `~580-620 FPS` depending on session load and harness. The most stable
  exact-path milestone before the current forward-kernel experiments was the
  `620.10 FPS` median run described below.
- On 2026-05-27, the benchmark script
  [development/stream_interface/rfdetr_nano_seg_trt_workflow.py](/home/ubuntu/inference/development/stream_interface/rfdetr_nano_seg_trt_workflow.py:1)
  was found to be **over-reporting FPS** because it did not call
  `torch.cuda.synchronize()` after `pipeline.join()`. The workflow sink receives
  a lazy GPU-backed `LazyWorkflowSVDetections` payload, so `join()` could return
  before the final queued CUDA work had drained. Historical `~580-620 FPS`
  figures from that script are therefore join-only optimistic numbers, not true
  end-to-end throughput numbers.
- After fixing that benchmark bug, the corrected real throughput on this host is
  much lower but still strong:
  - clean `main` / no fast flags: `65.00 FPS` median over 3 runs
  - current exact fast path: `245.97 FPS` median over 3 runs
  - real validated speedup: about `3.78x`

## Core Tooling and Correctness Gates

### Kept / improved tools

- [temp/detection_parity_full.py](/home/ubuntu/inference/temp/detection_parity_full.py:1)
  - updated to force all fast flags on for base/candidate runs,
  - compare `main` vs working tree,
  - count Triton preproc/postproc call paths,
  - exercise pipeline depth through the adapter path,
  - support separate base/candidate model IDs.
- [temp/workflow_parity_stream.py](/home/ubuntu/inference/temp/workflow_parity_stream.py:1)
  - updated for exact workflow parity under the real stream surface,
  - later updated to support separate base/candidate model IDs,
  - later updated to count graph-path preproc correctly.
- [temp/repeat_workflow_benchmark.py](/home/ubuntu/inference/temp/repeat_workflow_benchmark.py:1)
  - added to stabilize workflow measurements by reporting repeated-run median.
- [temp/direct_stream_parity.py](/home/ubuntu/inference/temp/direct_stream_parity.py:1)
  - added later as a cheaper gate on the direct `InferencePipeline.init(...)`
    surface for local TRT engine screening.
- [temp/profile_rfdetr_trt_forward_only.py](/home/ubuntu/inference/temp/profile_rfdetr_trt_forward_only.py:1)
  - added to isolate TRT forward without workflow / postproc noise.
- [temp/profile_rfdetr_workflow_range.py](/home/ubuntu/inference/temp/profile_rfdetr_workflow_range.py:1)
  - added to capture only the steady-state callback range with
    `cudaProfilerStart/Stop`.
- [temp/profile_rfdetr_workflow_hotspots.py](/home/ubuntu/inference/temp/profile_rfdetr_workflow_hotspots.py:1)
  - added to instrument the exact workflow path and measure Python-side / graph
    capture overhead directly.

### Baseline correctness gates that held

- Workflow parity against `main` on the exact current path repeatedly passed:
  - `shift +0`
  - `538/538` exact frame matches
  - `1901/1901` matched detections
  - `1901/1901` pixel-identical masks
- COCO parity against `main` on the exact current path repeatedly passed:
  - `1500/1500` records
  - `8036/8036` matched detections
  - `0` count-mismatch images
  - `0` class disagreements

Representative exact-path artifact pairs:

- Workflow:
  - [/tmp/workflow_parity_base_ring4default.pkl](/tmp/workflow_parity_base_ring4default.pkl)
  - [/tmp/workflow_parity_candidate_ring4default.pkl](/tmp/workflow_parity_candidate_ring4default.pkl)
- COCO:
  - [/tmp/det_parity_base_ring4default.pkl](/tmp/det_parity_base_ring4default.pkl)
  - [/tmp/det_parity_candidate_ring4default.pkl](/tmp/det_parity_candidate_ring4default.pkl)

## Initial Parity Work

### `temp/detection_parity_full.py`

Initial parity work was on the direct COCO path with all flags on. The script
was modified to:

- compare `main` vs working tree,
- force all fast flags in both runs,
- use `InferenceModelsInstanceSegmentationAdapter`,
- record RLE masks and Triton call counts,
- handle older checkouts that do not have the newer pipeline sentinel.

One branch bug was also fixed on the current branch:

- [inference/core/models/inference_models_adapters.py](/home/ubuntu/inference/inference/core/models/inference_models_adapters.py:1)
  - added missing `import os` so the depth-2 path would not crash.

### First full COCO parity result

Full 1500-image run:

- `main`
  - pipeline depth `1`
  - Triton preproc calls `0/1500`
  - Triton postproc calls `0/1500`
- current branch
  - pipeline depth `2`
  - Triton preproc calls `1500/1500`
  - Triton postproc calls `1500/1500`
- detections: `8037 / 8037`
- matched at box IoU > 0.5: `8037`
- count-mismatch images: `0`
- class disagreements: `0`
- mean box IoU: `0.999932`
- mean/max score delta: `1.424e-08 / 1.192e-07`
- mean/min mask IoU: `0.999627 / 0.0`
- pixel-identical masks: `8018 / 8037`

Artifacts:

- [/tmp/det_parity_full_base.pkl](/tmp/det_parity_full_base.pkl)
- [/tmp/det_parity_full_candidate.pkl](/tmp/det_parity_full_candidate.pkl)

### Non-identical mask investigation

The first scary mask result was a script artifact:

- all `19` "non-identical" cases were raw COCO RLE `counts` mismatches only,
- decoded masks were bit-identical,
- `3` cases were empty-vs-empty masks and the script reported their IoU as
  `0.0` because it computed `0/0`.

Conclusion:

- true pixel parity was effectively exact,
- the under-report came from raw compressed RLE comparison, not real mask drift.

## Workflow Surface Fixes

### Missing flush / one-frame delay

The depth-2 workflow path initially had a missing flush / one-frame tail issue.

Root cause:

- `InferencePipeline` never drained stateful handlers at stream end,
- workflow path and direct model path both dropped the final buffered result.

Fixes landed in:

- [inference/core/interfaces/stream/inference_pipeline.py](/home/ubuntu/inference/inference/core/interfaces/stream/inference_pipeline.py:1)
- [inference/core/interfaces/stream/model_handlers/roboflow_models.py](/home/ubuntu/inference/inference/core/interfaces/stream/model_handlers/roboflow_models.py:1)
- [inference/core/interfaces/stream/model_handlers/workflows.py](/home/ubuntu/inference/inference/core/interfaces/stream/model_handlers/workflows.py:1)
- [inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v3.py](/home/ubuntu/inference/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v3.py:1)
- plus `ModelManager.flush()` and a focused unit test.

Result after fix:

- workflow parity became strict `shift +0`
- direct model smoke path became `538` callbacks, `538` non-empty outputs
- workflow parity: `1907/1907` matched detections, pixel-identical masks
- workflow demo throughput around that point: `88.49 FPS`

## Profiling: Early Pre/Postproc and TRT Baseline

Before the forward-only pivot, `nsys` and `ncu` were used to establish whether
the Triton kernels were actually executing and what fraction of time they took.

### Runtime path verification

Verified on the workflow path with all fast flags on:

- Triton preproc entrypoint hit `538` times
- Triton postproc entrypoint hit `538` times
- adapter `_pipeline_depth == 2`
- async path signature was correct:
  - `1` priming return
  - `537` deferred returns
  - `1` final flush

### Early `nsys` / `ncu` findings

Representative early artifacts:

- `/tmp/profiles/rfdetr_workflow/workflow_cpu_gpu.nsys-rep`
- `/tmp/profiles/rfdetr_workflow/ncu_preproc.ncu-rep`
- `/tmp/profiles/rfdetr_workflow/ncu_postproc.ncu-rep`
- `/tmp/profiles/rfdetr_workflow/ncu_trt_gemm_top1.ncu-rep`
- `/tmp/profiles/rfdetr_workflow/ncu_trt_gemm_mha.ncu-rep`
- `/tmp/profiles/rfdetr_workflow/ncu_trt_gemm_fused.ncu-rep`

Representative readout:

- Triton postproc kernel:
  - `538` launches
  - `482.6 us` average
  - about `6.3%` of total GPU kernel time
- Triton preproc kernel:
  - `538` launches
  - `18.0 us` average
  - about `0.2%` of total GPU kernel time

Top TRT kernels were already larger:

- top GEMM about `18.6%`
- MHA GEMM about `13.5%`
- fused GEMM about `10.9%`

Representative `ncu` on those TRT kernels:

- top GEMM:
  - `67.46 us`
  - compute `45.59%`
  - memory `27.57%`
  - achieved occupancy `16.93%`
  - regs/thread `166`
  - dynamic shared memory `16.38 KB`
- MHA GEMM:
  - `104.45 us`
  - compute `35.51%`
  - memory `25.11%`
  - achieved occupancy `20.35%`
  - regs/thread `245`
  - dynamic shared memory `24.58 KB`

Conclusion from the beginning:

- the real long-term ceiling was TRT forward compute,
- not Triton preproc/postproc,
- and those TRT kernels were occupancy / launch-shape limited, not DRAM-bound.

## Exact Path Optimizations That Were Kept

This section lists the exact-path changes that survived measurement and parity.

### 1. CUDA graph and zero-copy integration

Key files:

- [inference_models/inference_models/models/common/trt.py](/home/ubuntu/inference/inference_models/inference_models/models/common/trt.py:1)
- [inference_models/inference_models/models/rfdetr/rfdetr_instance_segmentation_trt.py](/home/ubuntu/inference/inference_models/inference_models/models/rfdetr/rfdetr_instance_segmentation_trt.py:1)

Major kept ideas:

- zero-copy CUDA-graph replay path,
- graph input buffer reuse tied to fast-preprocess slots,
- output handoff/event fixes,
- later preprocess folded into the same captured graph,
- later eager graph-ring capture for both rotating slots on first eligible call.

Representative milestone:

- after the zero-copy path and real graph replay correctness fix:
  - workflow graphs off: `84.35 FPS`
  - workflow graphs on: `202.73 FPS`
  - parity held

### 2. Dense Triton postproc redesign

Key file:

- [inference_models/inference_models/models/rfdetr/triton_fullpostproc.py](/home/ubuntu/inference/inference_models/inference_models/models/rfdetr/triton_fullpostproc.py:1)

Important kept changes:

- split non-RLE postproc into:
  - exact top-k selection kernel
  - mask postproc kernel
- later vectorized the selection kernel
- later tuned the default `_TOPK_QUERY_BLOCK`
- later matched graph ring depth to the pipeline behavior

Representative milestones:

- split postproc:
  - graphs on around `216-218 FPS`
  - graphs off about `94.54 FPS`
- vectorized selection:
  - graphs on `219.27 FPS`
- final exact default tuning:
  - `_TOPK_QUERY_BLOCK=12`
  - ring depth default `max(3, 2 * RFDETR_PIPELINE_DEPTH)` -> depth-2 uses `4`

### 3. Workflow / adapter exact-path cleanup

Key files:

- [inference/core/models/inference_models_adapters.py](/home/ubuntu/inference/inference/core/models/inference_models_adapters.py:1)
- [inference/core/entities/responses/inference.py](/home/ubuntu/inference/inference/core/entities/responses/inference.py:1)
- [inference/core/workflows/core_steps/common/utils.py](/home/ubuntu/inference/inference/core/workflows/core_steps/common/utils.py:1)
- [inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v3.py](/home/ubuntu/inference/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v3.py:1)

Kept ideas:

- depth-2 deferred-count propagation on the async adapter path,
- workflow-local `sv.Detections` fast path,
- skipping expensive parent/root coordinate copies when already aligned,
- lazy workflow-friendly detection structures,
- eliminating redundant object conversions.

Representative gain:

- deferred-count + rooted-output fast path:
  - clean workflow runs around `243.65-247.28 FPS`
  - prior stable band was about `226.9 FPS`

### 4. Runtime config / graph-ring exact-path cleanup

Key file:

- [inference_models/inference_models/models/rfdetr/rfdetr_instance_segmentation_trt.py](/home/ubuntu/inference/inference_models/inference_models/models/rfdetr/rfdetr_instance_segmentation_trt.py:1)

Kept ideas:

- cache prepared runtime config once per confidence / geometry / remap shape,
- slot-specific graph-ring storage on `_FastPathState`,
- later eager capture all graph-ring states on the first eligible call.

Measured effects:

- eager graph capture of all states:
  - same-session repeat median moved from about `327.03 FPS` to `464.56 FPS`
  - single clean workflow run hit `508.24 FPS`
- runtime config cache:
  - materially reduced per-frame bookkeeping
  - only noise-level movement on total FPS

### 5. Stream handling cleanup

Key file:

- [inference_models/inference_models/models/common/trt.py](/home/ubuntu/inference/inference_models/inference_models/models/common/trt.py:1)

Kept change:

- use explicit `torch.cuda.set_stream()` / restore instead of the
  `torch.cuda.stream(...)` context manager in `infer_from_trt_engine()`,
  while keeping `record_stream()` intact.

Measured effect:

- exact-path workflow median improved from about `500.81 FPS` to `504.26 FPS`

### 6. Final exact-path default before the forward-only experiments

Representative exact-path milestone:

- ring depth default changed so depth-2 uses graph ring depth `4`
- repeated workflow benchmark:
  - `585.68`
  - `625.11`
  - `620.10`
  - median `620.10 FPS`
- clean direct workflow run:
  - `614.97 FPS`

This was the key exact milestone before the later forward-kernel experiments.

## Benchmark Timeline (Representative)

These are the representative workflow numbers that mattered. They are not every
single run, but they cover the major exact-path milestones.

| Phase | Benchmark result | Correctness |
| --- | --- | --- |
| Early actual workflow path after harness correction | `main 77.83 FPS`, branch `87.85 FPS` | exactness not yet final on stream tail |
| Missing flush fixed | about `88.49 FPS` | workflow parity fixed to `shift +0` |
| Zero-copy CUDA-graph replay | graphs on `202.73 FPS`, off `84.35 FPS` | exact |
| Split dense Triton postproc | graphs on `216-218 FPS` | exact |
| Vectorized selection kernel | `219.27 FPS` | exact |
| Deferred-count async adapter fix | about `243-247 FPS` | exact |
| Workflow CPU/object cleanup | about `246-248 FPS` | exact |
| Lazy workflow / graph-enabled path | about `303-308 FPS` | exact |
| Ring depth `3` default | about `319.72 FPS` median | exact |
| Full graph (preproc+TRT+postproc) | about `326.20 FPS` median | exact |
| Eager capture of all graph-ring states | `464.56 FPS` median, `508.24 FPS` single run | exact |
| Runtime config cache + stream cleanup | about `500-504 FPS` median | exact |
| Ring depth `4` default for depth-2 | `620.10 FPS` median, `614.97 FPS` single run | exact |

## Current Exact Production-State Profile

### Fresh `nsys` CPU+GPU profile requested for the `~620 FPS` path

The most recent profiling request was to capture a fresh `nsys` profile on the
current exact path instead of continuing immediately on the forward-plugin line.

Artifacts:

- non-node capture:
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620/workflow_range.nsys-rep](/tmp/profiles/rfdetr_workflow_cpu_gpu_620/workflow_range.nsys-rep)
- node-expanded capture:
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_range.nsys-rep](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_range.nsys-rep)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_range_export.sqlite](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_range_export.sqlite)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_gpu_kern_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_gpu_kern_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_api_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_api_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_osrt_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_osrt_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_gpu_mem_time_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_gpu_mem_time_sum.csv)

Later, after the CPU+GPU profiling fix was in place and the native plugin line
had been resumed, I regenerated the exact-path captures on the same
range-limited workflow surface:

- same-session exact-path repeat benchmark before the regenerated profile:
  - `593.27`
  - `584.49`
  - `585.17 FPS`
  - median `585.17 FPS`
- fresh node-expanded capture:
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_range.nsys-rep](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_range.nsys-rep)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_range_export.sqlite](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_range_export.sqlite)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_gpu_kern_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_gpu_kern_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_api_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_api_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_osrt_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_osrt_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_gpu_mem_time_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_gpu_mem_time_sum.csv)
- fresh graph-granularity capture:
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_range.nsys-rep](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_range.nsys-rep)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_range_export.sqlite](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_range_export.sqlite)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_stats_cuda_api_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_stats_cuda_api_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_stats_osrt_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_stats_osrt_sum.csv)

Host limitation:

- CPU IP/backtrace sampling unsupported on this host
- CPU context-switch tracing unsupported on this host
- so the usable CPU side is the OS-runtime wait timeline, not sampled stacks

### Host fix for `nsys` CPU+GPU profiling

Later in the investigation, the reason CPU+GPU profiling was "not working" on
this host became explicit:

- `nsys status --environment` as the normal user showed:
  - `Linux Kernel Paranoid Level = 4`
  - `Linux perf_event_open syscall available: Fail`
  - `CPU Profiling Environment (process-tree): Fail`
- root profiling already worked, so the blocker was not Nsight Systems itself.
- the blocker was the host kernel perf policy for unprivileged users.

Fix applied:

- runtime:
  - `kernel.perf_event_paranoid = 2`
  - `kernel.kptr_restrict = 0`
- persisted in:
  - [99-nsys-profiling.conf](/etc/sysctl.d/99-nsys-profiling.conf:1)

After the fix:

- `nsys status --environment` as the normal user now reports:
  - `Linux perf_event_open syscall available: OK`
  - `Sampling trigger event available: OK`
  - `CPU Profiling Environment (process-tree): OK`
- `CPU Profiling Environment (system-wide)` is still `Fail` for the normal user,
  so system-wide CPU tracing still requires root on this host.
- a short real verification run no longer emitted the earlier CPU warnings and
  its SQLite export contained scheduling data.

Verification artifacts:

- verification trace:
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_verify/workflow_range.nsys-rep](/tmp/profiles/rfdetr_workflow_cpu_gpu_verify/workflow_range.nsys-rep)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_verify/workflow_range_export.sqlite](/tmp/profiles/rfdetr_workflow_cpu_gpu_verify/workflow_range_export.sqlite)

Representative verification evidence:

- `SCHED_EVENTS` rows present in the exported SQLite: `140`
- CPU warnings about unsupported sampling/context-switch tracing disappeared in
  the verification run

Practical conclusion:

- process-tree CPU+GPU `nsys` profiling is now fixed for the normal user
- system-wide CPU profiling still needs root on this host

Fresh regenerated CPU+GPU evidence:

- node-expanded regenerated capture:
  - `SCHED_EVENTS` rows: `14876`
  - `cudaDeviceSynchronize`: `602.6 ms`, `1` call
  - `cudaGraphLaunch_v10000`: `306.2 ms`, `398` calls
  - HtoD memcpy: `18.36 ms`, `398` copies
  - top GPU kernels:
    1. TRT GEMM `16.2%`
    2. `_gemm_mha_v2_0x7daddb359f728ff2e600188f192f4549` `10.9%`
    3. TRT GEMM `10.4%`
    4. TRT fused GEMM `9.4%`
- graph-granularity regenerated capture:
  - `SCHED_EVENTS` rows: `15244`
  - `cudaDeviceSynchronize`: `793.7 ms`, `1` call
  - `cudaGraphLaunch_v10000`: `107.7 ms`, `397` calls
  - kernel/mem summaries are intentionally empty there because `--cuda-graph-trace=graph`
    records the graph launch as a whole instead of node-level GPU kernels

The regenerated captures do not change the conclusion: the exact path is still
down to one real graph launch per hot callback, CPU schedule data is now
present, and the remaining ceiling is still TRT forward compute.

Important `nsys` API summary from the node-expanded steady-state range:

- `cudaDeviceSynchronize`: `559.2 ms`, `1` call
- `cudaGraphLaunch_v10000`: `309.0 ms`, `406` calls
- `cudaStreamIsCapturing_v10000`: `1.0 ms`, `406` calls

GPU memory summary:

- host-to-device memcpy:
  - `18.37 ms` total
  - `406` copies
- CUDA memset:
  - `8.08 ms` total

Interpretation:

- the exact path is already down to one real graph launch per hot callback,
- HtoD traffic is small,
- the remaining time is still the TRT forward kernels.

Top GPU kernels from the same trace:

1. `sm75_xmma_gemm_f16f16_f16f16_f16_nn...` -> `15.7%`
2. `_gemm_mha_v2_0x7daddb359f728ff2e600188f192f4549` -> `10.4%`
3. `sm75_xmma_gemm_f16f16_f16f16_f16_nn...` -> `10.1%`
4. `sm75_xmma_gemm_f16f16_f16f32_f32..._fused` -> `8.8%`

Largest custom RF-DETR kernels in the same exact workflow trace:

- `rfdetr_topk_partial_triton_kernel` -> `1.3%`
- `rfdetr_topk_merge_finalize_triton_kernel` -> `0.7%`
- `fused_resize_normalize_kernel` -> `0.4%`
- `rfdetr_mask_postproc_triton_kernel` -> `0.4%`

Conclusion from the current exact-path profile:

- the remaining ceiling is still TRT forward compute,
- not preproc/postproc,
- not host copies,
- and not graph-launch count anymore.

### Why the `~620 FPS` exact result is plausible

The `620.10 FPS` number corresponds to about `1.61 ms/frame` on the real
workflow surface, not a synchronous per-frame `model(frame)` latency
measurement.

The exact-path reasons that this is plausible are:

1. The benchmark surface is already highly amortized:
   - one-block workflow only,
   - `312 x 312` input,
   - depth-2 async pipeline,
   - full graph-captured fast path for preprocess + TRT + dense Triton
     postprocess.
2. The path is overlap-driven, not sum-of-kernels-driven:
   - TRT already spreads work across multiple CUDA streams,
   - the workflow path overlaps producer / consumer behavior through the
     depth-2 stream surface,
   - the postproc graph ring depth was raised to `4`, which removed graph-state
     reuse pressure on the depth-2 path.
3. Host launch overhead was collapsed:
   - graph-granularity `nsys`: `cudaGraphLaunch_v10000` was only `107.7 ms`
     across `397` hot launches, about `0.271 ms` per hot callback,
   - HtoD memcpy on the node-expanded trace was only `18.36 ms` across `398`
     copies, about `0.046 ms` per hot callback.
4. Triton pre/post kernels are no longer first-order costs:
   - in the node-expanded exact trace, the largest custom RF-DETR kernel,
     `rfdetr_topk_partial_triton_kernel`, is only `1.2-1.3%` of total GPU
     kernel time,
   - the steady-state ceiling is still TRT forward GEMM/MHA.

The important interpretation rule is that the visible GPU kernel totals in the
node-expanded `nsys` trace must **not** be summed as if they were serialized
per-frame latency. In the regenerated node-expanded exact trace, the top 19
kernel buckets alone sum to about `3926 ms` total, or about `9.86 ms` per hot
callback if divided naively by `398`. That is much larger than the measured
workflow frame time because:

- node-expanded graph tracing inflates runtime relative to the unprofiled run,
- TRT launches many kernels on multiple streams,
- graph replay and the depth-2 pipeline overlap work across callbacks,
- throughput is governed by the slowest overlapped stage on the steady-state
  critical path, not by the arithmetic sum of all visible kernels.

So the correct first-principles model for the `~620 FPS` result is:

- small host cost per hot callback (`~0.27 ms` graph-launch view),
- tiny HtoD cost (`~0.05 ms`),
- custom RF-DETR kernels reduced to a low single-digit percentage of GPU time,
- throughput ultimately limited by overlapped TRT forward compute,
- enough graph-ring depth (`4`) to keep that overlapped path from stalling on
  buffer reuse.

This also explains why later same-session reruns in the regenerated profiling
session were lower (`585.17 FPS` median) without any code change: the exact path
did not fundamentally change, but the benchmark is sensitive to session load,
trace overhead, and machine variance.

## Forward-Only Research Summary

After the exact-path postproc / workflow work plateaued, the focus shifted to
model forward.

### Forward-only profiling conclusion

Tool:

- [temp/profile_rfdetr_trt_forward_only.py](/home/ubuntu/inference/temp/profile_rfdetr_trt_forward_only.py:1)

Representative artifacts:

- [/tmp/profiles/rfdetr_forward_only_current/forward_graph.nsys-rep](/tmp/profiles/rfdetr_forward_only_current/forward_graph.nsys-rep)
- [/tmp/profiles/rfdetr_forward_only_current_refresh_ncu/unfiltered_probe_full.ncu-rep](/tmp/profiles/rfdetr_forward_only_current_refresh_ncu/unfiltered_probe_full.ncu-rep)
- [/tmp/profiles/rfdetr_forward_only_current_refresh_ncu/mha_focus.ncu-rep](/tmp/profiles/rfdetr_forward_only_current_refresh_ncu/mha_focus.ncu-rep)

Representative `ncu`:

- top GEMM:
  - `67.81 us`
  - compute `45.74%`
  - DRAM `10.19%`
  - `166` regs/thread
  - achieved occupancy `17.03%`
  - `0.45` waves/SM
- MHA GEMM:
  - `104.10 us`
  - compute `35.95%`
  - DRAM `6.09%`
  - `245` regs/thread
  - achieved occupancy `20.40%`
  - `0.82` waves/SM

Core forward conclusion:

- dominant kernels remain TRT GEMM/MHA,
- they are occupancy-limited rather than memory-bandwidth-limited,
- so broad "make Triton kernels faster" work was not the right frontier.

## Forward-Side Experiments

### 1. Builder / tactic-source sweeps

Tooling:

- [inference_models/development/compilation/engine_builder.py](/home/ubuntu/inference/inference_models/development/compilation/engine_builder.py:1)
- [inference_models/development/compilation/core.py](/home/ubuntu/inference/inference_models/development/compilation/core.py:1)
- [temp/sweep_rfdetr_seg_trt_variant.py](/home/ubuntu/inference/temp/sweep_rfdetr_seg_trt_variant.py:1)

What was tried:

- aux-stream count sweeps,
- timing iterations,
- tactic-source restrictions (`CUBLAS`, `CUBLAS_LT`, `EDGE_MASK_CONVOLUTIONS`,
  `JIT_CONVOLUTIONS`),
- timing-cache replay and hybrid cache blends.

Best broad builder-side lead:

- `opt3_tsrc_cublas`
  - same-session workflow median: `331.25 FPS`
  - shipped comparison in that session: `324.11 FPS`

Why rejected:

- workflow parity vs shipped failed:
  - `1901 / 1903` detections
  - `2` count-mismatch frames
- COCO smoke failed:
  - `274 / 275` detections
  - extra class `75` detection on `000000001503.jpg`

Later timing-cache leads:

- `top5`
  - workflow median around `619.56 FPS` vs `612.22 FPS` control in one session
  - still failed direct async parity on `52` frames
  - `2` real duplicate frames: `342`, `495`
- `drop4`
  - workflow median `621.94 FPS` vs `612.22 FPS`
  - workflow parity failed much harder: `1901 / 1905`, `12` mismatch frames

Conclusion:

- timing-cache / tactic sweeps found faster kernels,
- but none produced a parity-safe forward engine.

### 2. Exact ONNX graph rewrites

Tooling:

- [temp/patch_rfdetr_rank3_linear_flatten_matmul.py](/home/ubuntu/inference/temp/patch_rfdetr_rank3_linear_flatten_matmul.py:1)
- later narrower canonical-attention rewriters and screening scripts

What was tried:

- rank-3 `MatMul+Add -> Flatten -> MatMul -> Add -> Reshape`
  rewrites across decoder / encoder families,
- canonical self-attention rewrites for decoder layers,
- encoder attention canonical rewrites for hotspot layers.

Representative results:

- full decoder family:
  - workflow median improved from about `319.90 FPS` to `324.99 FPS`
  - parity failed
- decoder cross-attn family:
  - also showed speed signal
  - parity failed
- canonical decoder layer-3 self-attn rewrite:
  - ONNX exact on real frames
  - TRT candidate still drifted on workflow parity

Conclusion:

- exact ONNX rewrites could move TRT onto faster kernels,
- but TRT lowering around the rewrite was not confined enough to preserve
  exactness.

### 3. Built-in TRT QKV attention plugin

Tooling:

- [temp/patch_rfdetr_selfattn_to_qkv_plugin.py](/home/ubuntu/inference/temp/patch_rfdetr_selfattn_to_qkv_plugin.py:1)

Important correction:

- the TRT plugin expects head-interleaved packed Q/K/V,
- the early plugin attempt was wrong because it used plain `concat(Q, K, V)`,
- after fixing the packing, the plugin path became "close" instead of
  catastrophically wrong.

Best plugin lead:

- corrected decoder layers `2+3` plugin with output/residual/norm FP32 rescue
  was the cleanest version

Representative results:

- same-session baseline: `614.81 FPS`
- best corrected plugin lead: `608.10 FPS`
- async mismatches: `52`
- real count-mismatch frames: `236`, `398`

COCO gate on the best plugin lead:

- 50-image smoke:
  - `274 / 275` detections
  - mismatch image `000000001503.jpg`
  - extra class `75` detection, score `0.5146`, box `[1, 171, 101, 216]`

Conclusion:

- the built-in plugin became much closer after packing was fixed,
- but it was still not exact enough and not faster than the current exact path.

### 4. TensorRT Python plugin path

Tooling:

- [temp/trt_exact_projection_matmul_plugin.py](/home/ubuntu/inference/temp/trt_exact_projection_matmul_plugin.py:1)
- [temp/build_rfdetr_python_plugin_matmul_variant.py](/home/ubuntu/inference/temp/build_rfdetr_python_plugin_matmul_variant.py:1)

What was shown:

- TRT 10.12 on this host exposes the experimental Python plugin path,
- JIT and AOT plugin builds both work in principle.

Why rejected:

- JIT plugin workflow runs were slower (`~596-598 FPS`) and exited with heap
  corruption,
- AOT plugin builds succeeded but runtime launch was unstable / invalid.

Conclusion:

- the Python-plugin route was technically possible but not stable enough to use
  as the forward replacement surface.

### 5. Native projection-matmul plugin

Tooling:

- [temp/native_projection_matmul_plugin.cpp](/home/ubuntu/inference/temp/native_projection_matmul_plugin.cpp:1)
- [temp/build_rfdetr_native_plugin_matmul_variant.py](/home/ubuntu/inference/temp/build_rfdetr_native_plugin_matmul_variant.py:1)
- [temp/native_identity_copy_plugin.cpp](/home/ubuntu/inference/temp/native_identity_copy_plugin.cpp:1)

What was proven:

- native TRT V3 plugins compile and register correctly on this host,
- plugin math was correct on a small TRT probe:
  - FP32 `max_abs ~ 9.5e-7`
  - FP16 `max_abs = 0`

Why rejected:

- real RF-DETR placements were slower or destabilized surrounding scheduling:
  - baseline workflow median: `599.07 FPS`
  - decoder layer0 projection plugin: `591.03 FPS`
  - encoder fc2 projection plugin: `425.51 FPS`

Conclusion:

- native plugin route is viable technically,
- but standalone projection matmuls are not the right replacement boundary.

## Current Native Attention-Core Plugin Line

This is the newest forward-side line and the current active custom-kernel path.

### Motivation

Engine inspector work showed the real hotspot boundary is the attention core:

- `query/key/value` projection matmuls come first,
- then the hotspot `_gemm_mha_v2_*`,
- then the attention output-dense matmul.

That means the right replacement boundary is:

- projected `Q/K/V` tensors in,
- context tensor out,
- not a standalone projection/output matmul.

Relevant artifact:

- [/tmp/rfdetr_current_engine_inspector.json](/tmp/rfdetr_current_engine_inspector.json)

### Tooling

- [temp/native_encoder_attention_core_plugin.cpp](/home/ubuntu/inference/temp/native_encoder_attention_core_plugin.cpp:1)
- [temp/build_rfdetr_native_attention_core_variant.py](/home/ubuntu/inference/temp/build_rfdetr_native_attention_core_variant.py:1)

### What was built

The native plugin:

- packs projected Q/K/V into head-major contiguous layout,
- runs:
  - batched `QK^T`
  - row-wise softmax
  - batched `P*V`
- unpacks back to the original output layout,
- uses `cublasGemmStridedBatchedEx`.

### Synthetic validation

Synthetic attention-core TRT probe:

- build succeeded
- runtime succeeded
- `max_abs = 0.00048828125` vs torch reference

This proved the attention-core plugin boundary is executable and approximately
numerically valid.

### Two-layer encoder replacement (`layers 4/5`)

Candidate:

- [/tmp/rfdetr-seg-nano-native-attncore-l45](/tmp/rfdetr-seg-nano-native-attncore-l45)

Same-session workflow benchmark:

- baseline current exact path: median `583.15 FPS`
- candidate `l45`: median `592.29 FPS`

Single workflow hotspot run:

- candidate: `565.98 FPS`
- baseline in same harness: `569.32 FPS`

So the speed signal exists, but it is not huge and is harness-sensitive.

Correctness gate on the cheap direct async surface:

- direct async parity vs baseline:
  - `538` frames compared
  - `126` mismatched frames

Conclusion:

- `l45` is faster than baseline in repeated workflow runs,
- but it is far too noisy to be considered parity-safe.

### Single-layer encoder replacements

Built:

- [/tmp/rfdetr-seg-nano-native-attncore-l4](/tmp/rfdetr-seg-nano-native-attncore-l4)
- [/tmp/rfdetr-seg-nano-native-attncore-l5](/tmp/rfdetr-seg-nano-native-attncore-l5)

Quick one-run screens:

- `l4`: `601.89 FPS`
- `l5`: `595.61 FPS`

So `l4` is the better single-layer lead.

Cheap direct async parity on `l4`:

- `538` frames compared
- `126` mismatched frames

This already disqualifies the current `l4` replacement as an exact engine lead.

`l5` was built and benchmarked, but full parity screening was not pursued after
`l4` showed the same scale of drift as `l45`.

### All-layer encoder attention-core replacement

Candidate:

- [/tmp/rfdetr-seg-nano-native-attncore-all12](/tmp/rfdetr-seg-nano-native-attncore-all12)

This was the natural next escalation after the corrected workflow benchmark and
the stream-overlap work, because the current exact engine’s TRT layer profiler
still shows encoder attention core as the hottest repeated logical block:

- `mha_gemm`: `1.3400 ms`, `15.6%`
- twelve `_gemm_mha_v2_myl2_*` layers at roughly `0.104-0.109 ms` each

Current exact TRT layer profiler on the direct path:

- total: `8.5896 ms`
- `other_matmul`: `22.9%`
- `other`: `18.7%`
- `mha_gemm`: `15.6%`
- `attention_other`: `15.6%`
- `encoder_mlp_fc2`: `12.2%`

Why the all-layer replacement was rejected:

- corrected workflow benchmark run did not return in a reasonable wall time and
  was killed as a bad lead
- direct forward-only probe with the plugin actually loaded via `LD_PRELOAD`
  was much slower than baseline:
  - baseline current exact engine: `154.79 FPS`
  - all-layer attention-core plugin: `94.09 FPS`

Conclusion:

- the native attention-core line is now effectively exhausted as a speed lead
  in its current implementation
- even replacing the whole repeated encoder attention-core family is slower
  than the current exact engine on the direct forward surface
- this makes “improve the current path by replacing `_gemm_mha_v2` with the
  current native plugin implementation” a dead end, not just a parity problem

### Segmentation-head `pwconv1` native matmul replacement

Candidate build:

- [/tmp/rfdetr-seg-nano-native-seghead-pwconv1-opt0](/tmp/rfdetr-seg-nano-native-seghead-pwconv1-opt0)

What was targeted:

- `/segmentation_head/blocks.0/pwconv1/MatMul`
- `/segmentation_head/blocks.1/pwconv1/MatMul`
- `/segmentation_head/blocks.2/pwconv1/MatMul`
- `/segmentation_head/blocks.3/pwconv1/MatMul`

Why this family was chosen:

- on the current exact TRT layer profile, these are the largest untouched late
  repeated matmul layers after the encoder-attention and encoder-`fc2` work
  had already been ruled out
- together they account for roughly `0.35 ms` of the direct TRT layer time

Result:

- normal builder settings stalled badly in TensorRT build search
- the only bounded build that completed cleanly was `builder_optimization_level=0`
- direct forward-only probe on that conservative engine, with the native plugin
  actually loaded via `LD_PRELOAD`, was far slower than baseline:
  - baseline current exact engine: `154.79 FPS`
  - segmentation-head native matmul candidate: `53.68 FPS`

Conclusion:

- the standalone native matmul replacement path is also a dead end for this
  late segmentation-head family
- because it is already much slower on the direct forward surface, no workflow
  or parity budget was spent on it

### Current conclusion on native attention-core plugins

- The boundary is right.
- The native plugin is technically viable.
- The current implementation is not parity-safe.
- Replacing even one encoder attention-core site currently drifts badly on the
  direct async stream surface.

### Latest attention-core refinement: split-scale match

The original plugin used a single fused `1 / sqrt(d)` factor on the `QK^T`
GEMM. The source ONNX attention block does not do that. It computes
`d^-1/4`, multiplies `Q` by that factor, multiplies `K` by that factor, and
then performs `QK^T`.

That difference is mathematically equivalent but not numerically identical in
FP16, so the plugin was updated to match the ONNX operation order exactly.

Result on the direct async screen for `l4`:

- before split-scale match: `126` mismatched frames
- after split-scale match: `110` mismatched frames

So the scaling-order fix is real, but it is not sufficient by itself to make
the plugin parity-safe.

This is the current starting point for the next attention-core refinement pass.

### Split-scale follow-up: FP32 compute/workspace path

After the split-scale match, the next higher-fidelity test was to let the
plugin run the attention core through an FP32 path for half inputs:

- FP32 GEMM compute mode on half inputs
- later a full float-workspace path for packed `Q/K/V`, scores, and output
  accumulation inside the plugin

Result on the direct async `l4` screen:

- split-scale path: `110` mismatched frames
- FP32 compute / full-float plugin path: still `110` mismatched frames

So the remaining drift is not explained by the obvious GEMM-accumulation mode
alone. The next refinement has to target some other detail of the attention-core
replacement, not just "use more FP32 inside the plugin."

### Boundary comparator: standalone `fp32` plugin is exact, `fp16` path is not

I added [temp/compare_native_attention_core_boundary.py](/home/ubuntu/inference/temp/compare_native_attention_core_boundary.py:1)
to compare the native encoder attention-core plugin directly against the source
ONNX boundary on a real bad frame.

On frame `236`, encoder layer `4`, the standalone plugin boundary results were:

- `plugin_fp32`: `max_abs=4.29e-06`, `mean_abs=8e-08`, `p99=6e-07`
- `plugin_fp16`: `max_abs=515.19`, `mean_abs=64.79`, `p99=512.46`
- `plugin_fp16_fullfp32path`: `max_abs=515.08`, `mean_abs=64.38`, `p99=512.45`

This is the strongest localization result on the native attention-core line so
far:

- the replacement boundary and math are effectively exact in pure `fp32`
- the current `fp16` plugin path is catastrophically wrong at the layer output
  itself
- the earlier workflow/parity drift is therefore not just "small downstream TRT
  retuning noise"; the half-precision plugin surface is still fundamentally bad

### FP32-constrained engine follow-up for native encoder attention-core `l4`

I then updated
[temp/build_rfdetr_native_attention_core_variant.py](/home/ubuntu/inference/temp/build_rfdetr_native_attention_core_variant.py:1)
so it can apply TRT plugin-layer precision constraints and tried to force the
single-layer `l4` attention-core plugin to run as `fp32` inside the real `fp16`
engine.

New candidates:

- [/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32prefer](/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32prefer)
- [/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32obey](/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32obey)

Cheap bad-frame screen against the exact current engine on frames
`28, 85, 179, 236, 242, 270, 338, 391, 431, 437`:

- `pluginfp32prefer`: `8 / 10` mismatches
- `pluginfp32obey`: `7 / 10` mismatches

The real count mismatches were still present in both constrained variants, most
notably:

- frame `179`: `4 -> 5`
- frame `236`: `4 -> 3`
- frame `242`: `3 -> 4`
- frame `391`: `5 -> 4`
- frame `431`: `4 -> 3`

So simply forcing the plugin layer to `fp32` through TRT precision constraints
does not rescue the real engine path. Either TRT is still not actually keeping
that replacement on the clean `fp32` boundary we validated in isolation, or the
remaining error is deeper than plugin-layer precision selection alone.

Follow-up inspection and full-screen result made that answer clearer.

Engine inspector:

- the original `l4` plugin engine keeps the plugin fully half-typed:
  - plugin inputs: `Half`
  - plugin output: `Half`
- the `pluginfp32obey` engine really does insert reformats around the plugin:
  - three `Half -> Float` reformat copy nodes into
    `/backbone/.../layer.4/attention/attention/AttentionCorePlugin`
  - plugin inputs: `Float`
  - plugin output: `Float`
  - one `Float -> Half` reformat copy node after the plugin

So the constrained engine is not silently ignoring the precision request. It is
actually running the plugin on float tensors inside the engine.

The full direct async screen on that constrained engine was still worse than the
earlier split-scale baseline:

- candidate:
  - [/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32obey](/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32obey)
- `538` frames compared
- `131` mismatches

This is worse than the earlier unconstrained split-scale `l4` path (`110`
mismatches). So the float-constrained plugin path is not a rescue; it is a
deprioritized branch now.

### Engine-boundary trace for the native encoder attention-core `l4` plugin

The next useful step was to stop treating the plugin path as a black box and
trace the actual boundary inside the real engine. I updated
[temp/build_rfdetr_native_attention_core_variant.py](/home/ubuntu/inference/temp/build_rfdetr_native_attention_core_variant.py:1)
so `--append-plugin-outputs` now exposes not just the plugin output, but also
the plugin inputs:

- `query/Add_output_0`
- `key/Add_output_0`
- `value/Add_output_0`
- `Reshape_3_output_0`

I also added
[temp/trace_native_attention_engine_boundary.py](/home/ubuntu/inference/temp/trace_native_attention_engine_boundary.py:1),
which:

- runs ORT on the same frame and same intermediate outputs,
- runs the real TRT engine with those outputs exposed,
- recomputes the attention core in torch from the TRT `Q/K/V`,
- compares all of those tensors directly.

Key result on frame `236`, layer `4`, using:

- engine:
  [/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32obey-debugqkv/engine.plan](/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32obey-debugqkv/engine.plan)
- plugin:
  [/tmp/rfprobe_native_plugin/libRfProbeEncoderAttentionCore.so](/tmp/rfprobe_native_plugin/libRfProbeEncoderAttentionCore.so)

The real engine-side drift is:

- TRT vs ORT `query/Add_output_0`
  - `max_abs = 0.08095`
  - `mean_abs = 0.00399`
  - `p99 = 0.02061`
- TRT vs ORT `key/Add_output_0`
  - `max_abs = 0.06775`
  - `mean_abs = 0.00377`
  - `p99 = 0.01828`
- TRT vs ORT `value/Add_output_0`
  - `max_abs = 0.07488`
  - `mean_abs = 0.00256`
  - `p99 = 0.01259`
- TRT vs ORT plugin output `Reshape_3_output_0`
  - `max_abs = 0.02785`
  - `mean_abs = 0.00120`
  - `p99 = 0.00615`

But the decisive comparison is local to the plugin boundary:

- ORT attention formula vs ORT output
  - `max_abs = 4.29e-06`
  - `mean_abs = 8e-08`
- torch attention recomputed on the **TRT engine's own `Q/K/V`** vs the TRT
  plugin output
  - `max_abs = 7.2e-07`
  - `mean_abs = 2e-08`
- torch attention on TRT `Q/K/V` vs ORT output
  - `max_abs = 0.02785`
  - `mean_abs = 0.00120`

That is the cleanest localization result on this native plugin line so far:

- the plugin's actual attention-core math is already effectively exact on the
  engine's own `Q/K/V`,
- the remaining error at this boundary comes **before** the plugin, in the
  incoming TRT-projected `Q/K/V`,
- so further polishing the attention-core plugin itself is unlikely to rescue
  exactness.

This materially changes the branch priority:

- the native encoder attention-core plugin is no longer the most plausible
  direct fix,
- the next coherent replacement boundary is now the upstream
  `query/key/value MatMul -> Add -> attention core` block, not just the
  attention core alone.

### Combined native `Q/K/V MatMul + attention-core` probe on encoder layer `4`

To test that new boundary directly, I added:

- [temp/build_rfdetr_native_encoder_full_variant.py](/home/ubuntu/inference/temp/build_rfdetr_native_encoder_full_variant.py:1)

This builder composes two native plugins on the same encoder attention block:

- exact projection-matmul plugins on:
  - `query/MatMul`
  - `key/MatMul`
  - `value/MatMul`
- the native encoder attention-core plugin from:
  - `query/Add_output_0`
  - `key/Add_output_0`
  - `value/Add_output_0`
  - to `Reshape_3_output_0`

Debug candidate:

- [/tmp/rfdetr-seg-nano-native-fullattn-l4-debug](/tmp/rfdetr-seg-nano-native-fullattn-l4-debug)

Using the same boundary tracer on frame `236`, layer `4`:

- TRT vs ORT `query/Add_output_0`
  - `max_abs = 0.08257`
  - `mean_abs = 0.00420`
  - `p99 = 0.02164`
- TRT vs ORT `key/Add_output_0`
  - `max_abs = 0.05927`
  - `mean_abs = 0.00394`
  - `p99 = 0.01889`
- TRT vs ORT `value/Add_output_0`
  - `max_abs = 0.04550`
  - `mean_abs = 0.00267`
  - `p99 = 0.01310`
- TRT vs ORT `Reshape_3_output_0`
  - `max_abs = 0.03790`
  - `mean_abs = 0.00114`
  - `p99 = 0.00619`

And again, the local attention math stayed effectively exact on the TRT engine's
own `Q/K/V`:

- torch attention on TRT `Q/K/V` vs TRT plugin output
  - `max_abs = 1.19e-06`
  - `mean_abs = 2e-08`

So even after replacing the projection matmuls and the attention core together,
the incoming `Q/K/V` were **not** materially pulled back to the reference in a
way that helped the layer output. On the same frame, the raw `dets` tensor was
also effectively unchanged from the earlier attention-core-only candidate:

- `output=dets frame=236`
  - `max_abs = 0.561699`
  - `mean_abs = 0.024412`
  - `p99 = 0.359790`

That makes this combined `l4` native full-attention probe a dead end for now.
It does not improve the internal boundary enough to justify a non-debug engine
build or full parity budget.

Practical implication:

- the error source is earlier than the local `Q/K/V MatMul + attention-core`
  replacement on this encoder block,
- so the next coherent native replacement boundary would have to expand further
  upstream (for example the full encoder attention block including its input
  normalization / residual path), not just the `Q/K/V` projections or the
  attention core.

### Mixed-type half interface bug on the standalone plugin path

The boundary work also isolated a separate bug inside the plugin's half I/O
surface itself.

Using the same real frame and same layer-4 `Q/K/V`, I compared three standalone
paths:

- exact float plugin on half-rounded `Q/K/V`
- half-I/O plugin with `RFPROBE_ENCODER_ATTN_FULL_FP32_PATH=1`
- source ONNX boundary reference

Result:

- exact float plugin on half-rounded `Q/K/V` vs reference:
  - `max_abs = 0.00406`
  - `mean_abs = 9.81e-05`
- half-I/O plugin with full-float internal path vs reference:
  - `max_abs = 515.08`
  - `mean_abs = 64.38`
- the two plugin outputs differ by the same huge amount

That proves the catastrophic standalone error is not caused by normal half
rounding of `Q/K/V`. It is a real bug in the plugin's mixed half-I/O path.

This does **not** automatically make the native attention-core branch viable,
because the real float-constrained engine path was still worse than the simpler
split-scale baseline. But it does localize one concrete implementation bug if
the branch is revisited later.

### Current working hypothesis on the attention-core drift

The current native attention-core plugin is structurally aligned with the source
ONNX attention block:

- encoder layer attention reshapes use `[1, 677, 6, 64]`,
- the attention scale is `1 / sqrt(64)`,
- the plugin boundary preserves projected `Q/K/V` and replaces only the
  `QK^T -> softmax -> PV` core.

That means the remaining mismatch is unlikely to be a simple boundary mistake.
The most likely causes are:

- precision / accumulation differences inside the plugin core,
- operation-order differences relative to the original TRT lowering,
- or layout / packing differences that are numerically close but not exact
  enough at the workflow threshold surface.

This is the active line of investigation after the current document update.

## Files Added for Forward Research

This is not every file touched in the repo, but it is the important research
tooling added during the forward-focused work.

- [temp/profile_rfdetr_trt_forward_only.py](/home/ubuntu/inference/temp/profile_rfdetr_trt_forward_only.py:1)
- [temp/profile_rfdetr_seg_trt_layers.py](/home/ubuntu/inference/temp/profile_rfdetr_seg_trt_layers.py:1)
- [temp/profile_rfdetr_workflow_range.py](/home/ubuntu/inference/temp/profile_rfdetr_workflow_range.py:1)
- [temp/profile_rfdetr_workflow_hotspots.py](/home/ubuntu/inference/temp/profile_rfdetr_workflow_hotspots.py:1)
- [temp/repeat_workflow_benchmark.py](/home/ubuntu/inference/temp/repeat_workflow_benchmark.py:1)
- [temp/direct_stream_parity.py](/home/ubuntu/inference/temp/direct_stream_parity.py:1)
- [temp/build_rfdetr_trt_timingcache_hybrid.py](/home/ubuntu/inference/temp/build_rfdetr_trt_timingcache_hybrid.py:1)
- [temp/build_rfdetr_trt_layer_constrained_variant.py](/home/ubuntu/inference/temp/build_rfdetr_trt_layer_constrained_variant.py:1)
- [temp/patch_rfdetr_rank3_linear_flatten_matmul.py](/home/ubuntu/inference/temp/patch_rfdetr_rank3_linear_flatten_matmul.py:1)
- [temp/patch_rfdetr_selfattn_to_qkv_plugin.py](/home/ubuntu/inference/temp/patch_rfdetr_selfattn_to_qkv_plugin.py:1)
- [temp/trt_exact_projection_matmul_plugin.py](/home/ubuntu/inference/temp/trt_exact_projection_matmul_plugin.py:1)
- [temp/build_rfdetr_python_plugin_matmul_variant.py](/home/ubuntu/inference/temp/build_rfdetr_python_plugin_matmul_variant.py:1)
- [temp/native_projection_matmul_plugin.cpp](/home/ubuntu/inference/temp/native_projection_matmul_plugin.cpp:1)
- [temp/native_identity_copy_plugin.cpp](/home/ubuntu/inference/temp/native_identity_copy_plugin.cpp:1)
- [temp/native_encoder_attention_core_plugin.cpp](/home/ubuntu/inference/temp/native_encoder_attention_core_plugin.cpp:1)
- [temp/build_rfdetr_native_attention_core_variant.py](/home/ubuntu/inference/temp/build_rfdetr_native_attention_core_variant.py:1)
- [temp/compare_native_attention_core_boundary.py](/home/ubuntu/inference/temp/compare_native_attention_core_boundary.py:1)

## Current State

### Best exact production path

The current exact production path remains the best validated state.

Corrected workflow benchmark on the real script after adding final
`torch.cuda.synchronize()`:

- clean detached `main` worktree, no fast flags:
  - `65.06`
  - `64.63`
  - `65.00`
  - median `65.00 FPS`
- current exact path with all fast flags:
  - `248.24`
  - `245.97`
  - `243.99`
  - median `245.97 FPS`
  - about `3.78x` faster than the clean `main` baseline
- single sanity rerun on the current exact path after later stream-count work:
  - `251.88 FPS`

Later same-session baselines during the native attention-core work were lower
because of machine variance and different harness conditions:

- exact workflow median in one later session: `583.15 FPS`
- exact workflow hotspot one-run baseline in the same session: `569.32 FPS`

### Benchmark script correction: final FPS must include GPU drain after `join()`

The benchmark script
[development/stream_interface/rfdetr_nano_seg_trt_workflow.py](/home/ubuntu/inference/development/stream_interface/rfdetr_nano_seg_trt_workflow.py:1)
was corrected to:

- call `torch.cuda.synchronize()` after `pipeline.join()`
- report `gpu_drain_ms` so the gap between callback completion and true GPU
  completion is visible

Why this matters:

- `InferencePipeline.join()` only waits for the inference and dispatch threads
  to finish
- the workflow sink was receiving a dict with:
  - key: `predictions`
  - value type:
    `inference.core.models.inference_models_adapters.LazyWorkflowSVDetections`
- so the old benchmark was counting frames when callback payloads were ready,
  not when the last queued GPU work had actually completed

Direct validation on the exact current path with all fast flags on:

- ad-hoc validation script:
  - `join_only`: `543.81 FPS`
  - `after_sync`: `251.85 FPS`
  - extra final GPU drain: `1146.88 ms`
- patched benchmark script:
  - progress prints still show the old optimistic callback rate, rising to
    about `583.66 FPS`
  - final corrected line:
    - `frames=538`
    - `elapsed=2.15s`
    - `fps=250.57`
    - `gpu_drain_ms=1230.54`

So the old `~620 FPS` benchmark numbers from that script should now be treated
as invalid for true end-to-end throughput. They were measuring callback issue
rate on a lazy GPU-backed workflow surface, not full completion rate.

### Outer replay-stream experiment: ring reuse vs real stream overlap

To test whether the prior speed signal was really coming from graph-ring reuse
or from actual outer CUDA-stream overlap, the combined dense-graph path now has
an opt-in knob:

```bash
RFDETR_TRITON_POSTPROC_GRAPH_STREAM_COUNT=<N>
```

Important detail:

- when unset, the exact path keeps the historical default behavior of one
  distinct replay stream per captured graph state
- when set, the captured graph states share only `N` replay streams, so
  `N=1` means no outer replay-stream parallelism

Corrected workflow benchmark matrix on the current exact path:

| Ring depth | Replay stream count | 3-run median FPS | Interpretation |
| --- | --- | ---: | --- |
| `1` | `1` | `226.92` | minimal graph-state reuse, no outer replay overlap |
| `2` | `1` | `229.04` | more reusable states alone buys almost nothing |
| `4` | `1` | `228.93` | ring depth alone is still basically flat |
| `4` | `2` | `249.03` | first clear gain from outer replay-stream overlap |
| `4` | `4` | `252.56` | slightly better than `2`, close to current exact default band |

Load-bearing conclusion from that matrix:

- graph-state ring depth by itself is **not** the source of the real
  corrected throughput gain once all states are forced onto one replay stream
- the real throughput lift comes from allowing at least `2` outer replay
  streams, which lets low-occupancy work overlap enough to move end-to-end
  completion throughput from about `229 FPS` to about `249-253 FPS`
- going from `2` to `4` replay streams helps only a little on this host
  (`249.03 -> 252.56 FPS` median), so the outer-stream benefit saturates early

Matched `nsys` range profiles were captured for the two most important cases:

- `ring4/stream1`:
  - [/tmp/profiles/rfdetr_streamcmp_ring4_stream1/workflow_range.nsys-rep](/tmp/profiles/rfdetr_streamcmp_ring4_stream1/workflow_range.nsys-rep)
  - [/tmp/profiles/rfdetr_streamcmp_ring4_stream1/workflow_range.sqlite](/tmp/profiles/rfdetr_streamcmp_ring4_stream1/workflow_range.sqlite)
  - [/tmp/profiles/rfdetr_streamcmp_ring4_stream1/workflow_stats_cuda_api_sum.csv](/tmp/profiles/rfdetr_streamcmp_ring4_stream1/workflow_stats_cuda_api_sum.csv)
  - [/tmp/profiles/rfdetr_streamcmp_ring4_stream1/workflow_stats_cuda_gpu_kern_sum.csv](/tmp/profiles/rfdetr_streamcmp_ring4_stream1/workflow_stats_cuda_gpu_kern_sum.csv)
- `ring4/stream4`:
  - [/tmp/profiles/rfdetr_streamcmp_ring4_stream4/workflow_range.nsys-rep](/tmp/profiles/rfdetr_streamcmp_ring4_stream4/workflow_range.nsys-rep)
  - [/tmp/profiles/rfdetr_streamcmp_ring4_stream4/workflow_range.sqlite](/tmp/profiles/rfdetr_streamcmp_ring4_stream4/workflow_range.sqlite)
  - [/tmp/profiles/rfdetr_streamcmp_ring4_stream4/workflow_stats_cuda_api_sum.csv](/tmp/profiles/rfdetr_streamcmp_ring4_stream4/workflow_stats_cuda_api_sum.csv)
  - [/tmp/profiles/rfdetr_streamcmp_ring4_stream4/workflow_stats_cuda_gpu_kern_sum.csv](/tmp/profiles/rfdetr_streamcmp_ring4_stream4/workflow_stats_cuda_gpu_kern_sum.csv)

These traces give the first-principles explanation for the FPS change:

- `ring4/stream1`
  - kernel span: `1890.62 ms`
  - total summed kernel time: `1846.75 ms`
  - concurrency ratio (`sum / span`): about `0.98x`
  - dominant stream:
    - stream `31`: `1529.12 ms` of kernel work
  - `cudaGraphLaunch_v10000`: `440` calls, `942.56 ms` total, about `2.14 ms` avg

- `ring4/stream4`
  - kernel span: `1515.00 ms`
  - total summed kernel time: `4251.67 ms`
  - concurrency ratio (`sum / span`): about `2.81x`
  - four dominant streams:
    - stream `43`: `873.82 ms`
    - stream `31`: `870.94 ms`
    - stream `39`: `862.20 ms`
    - stream `35`: `860.79 ms`
  - `cudaGraphLaunch_v10000`: `392` calls, `286.11 ms` total, about `0.73 ms` avg

Interpretation:

- with `stream1`, the GPU is effectively close to serialized on the outer
  replay surface; total kernel work is almost the same as wall-clock kernel
  span
- with `stream4`, the GPU is running about `2.8x` more total kernel work than
  wall-clock kernel span, which is direct evidence of real overlap across
  replay streams
- the throughput gain is therefore **not** coming from graph-ring reuse alone;
  it comes from giving low-occupancy work enough independent outer streams to
  overlap
- the interesting side effect is that individual kernels become slower under
  the higher-overlap configuration because of contention, but end-to-end
  throughput still rises because the work is no longer serialized

One follow-up exact A/B was also run to check whether the historical default
"one replay stream per graph state" should be capped explicitly after the
overlap study:

- current default exact path (`stream_count` unset): median `250.19 FPS`
- explicit `stream_count=4`: median `250.58 FPS`
- explicit `stream_count=2`: median `246.89 FPS`

Conclusion:

- capping the default replay stream count at `4` does **not** buy a meaningful
  exact-path improvement over the historical default on this host
- the stream-count knob is useful for analysis, but there is no strong enough
  exact win here to promote another default scheduling change

The correctness gates on the exact path still hold:

- workflow parity exact against `main`
- full COCO parity exact against `main`

### Best non-exact forward leads found so far

- tactic / timing-cache engine families:
  - up to roughly `+7-10 FPS`
  - always parity-breaking
- native encoder attention-core plugin `l45`:
  - roughly `+9 FPS` in one repeated workflow session
  - direct async parity badly broken
- native encoder attention-core plugin `l4`:
  - best single-layer quick screen at `601.89 FPS`
  - direct async parity also badly broken

### Exact ONNX rewrite: segmentation-head linear family

After the native-plugin branches stalled, I returned to exact ONNX graph-shape
rewrites on the remaining rewriteable `other_matmul` family. I widened the temp
rewriter [patch_rfdetr_rank3_linear_flatten_matmul.py](/home/ubuntu/inference/temp/patch_rfdetr_rank3_linear_flatten_matmul.py:1)
so it now supports both rank-3 and rank-4 linear layers by rewriting:

- `Shape -> Slice(prefix) -> Flatten -> MatMul -> Add -> Concat -> Reshape`

This made it possible to screen the remaining segmentation-head matmul family.

`pwconv1` only:

- patched nodes:
  - `/segmentation_head/blocks.0/pwconv1/MatMul`
  - `/segmentation_head/blocks.1/pwconv1/MatMul`
  - `/segmentation_head/blocks.2/pwconv1/MatMul`
  - `/segmentation_head/blocks.3/pwconv1/MatMul`
- candidate:
  - [/tmp/rfdetr-seg-nano-seghead-flatmatmul/local_fp16_opt3_seghead_pwconv1_flat](/tmp/rfdetr-seg-nano-seghead-flatmatmul/local_fp16_opt3_seghead_pwconv1_flat)
- direct forward-only gate:
  - baseline current exact engine: `145.44 FPS`
  - candidate: `142.31-154.12 FPS`
- conclusion:
  - effectively flat / noisy on the cheapest gate

`query_features_*` only:

- patched nodes:
  - `/segmentation_head/query_features_block/layers.0/MatMul`
  - `/segmentation_head/query_features_block/layers.2/MatMul`
  - `/segmentation_head/query_features_proj/MatMul`
- candidate:
  - [/tmp/rfdetr-seg-nano-queryfeatures-flatmatmul/local_fp16_opt3_queryfeatures_flat](/tmp/rfdetr-seg-nano-queryfeatures-flatmatmul/local_fp16_opt3_queryfeatures_flat)
- direct forward-only gate:
  - baseline current exact engine: `145.44 FPS`
  - candidate: `146.16 FPS`
- conclusion:
  - also effectively flat

Combined segmentation-head family:

- patched nodes:
  - the four `pwconv1` matmuls above
  - the three `query_features_*` matmuls above
- candidate:
  - [/tmp/rfdetr-seg-nano-seghead-all-flatmatmul/local_fp16_opt3_seghead_all_flat](/tmp/rfdetr-seg-nano-seghead-all-flatmatmul/local_fp16_opt3_seghead_all_flat)
- direct forward-only:
  - baseline current exact engine: `145.44 FPS`
  - candidate: `151.85-154.12 FPS`
- corrected workflow benchmark:
  - baseline current exact path: `247.61 FPS` median over 3 runs
  - candidate: `249.28 FPS` median over 3 runs

So this was the first exact ONNX rewrite in this branch with a real speed
signal.

But the cheap parity gate disqualified it:

- full direct async-stream screen:
  - `52` mismatches over `538` frames
- mismatch breakdown:
  - `46` are 1-pixel box drifts
  - `4` are pure ordering mismatches
  - `2` are real count-mismatch frames: `342` and `495`

That is the same bad-frame class as earlier non-exact timing-cache leads.

Layer-profiler evidence on the combined-family candidate explains why it was
tempting but not promotable:

- current exact engine total TRT layer time: `8.5753 ms`
- combined-family candidate total TRT layer time: `6.1459 ms`
- bucket shifts:
  - `other_matmul`: `1.9587 -> 1.3958 ms`
  - `mha_gemm`: `1.3453 -> 0.8837 ms`
  - `encoder_mlp_fc2`: `1.0471 -> 0.7157 ms`
  - segmentation head: `0.8102 -> 0.5945 ms`

So this is another case where an exact local graph rewrite induces a broader
TRT retune. The speedup is real, but it is not confined to the intended
subgraph, and the resulting engine is not parity-safe.

## Relaxed correctness metric: exact class + box drift <= 5 px

At this point the strict exact-parity gate is probably too pessimistic for the
remaining forward-side TRT leads, so the temp screening tools were widened to
support a relaxed geometric metric:

- class ID must match exactly
- box match ignores ordering
- every box edge must stay within `5` px (`max(|Δx1|, |Δy1|, |Δx2|, |Δy2|) <= 5`)

This was implemented in:

- [temp/direct_stream_parity.py](/home/ubuntu/inference/temp/direct_stream_parity.py:1)
  - new `--box-drift-px`, default `5`
- [temp/workflow_parity_stream.py](/home/ubuntu/inference/temp/workflow_parity_stream.py:1)
  - reports `relaxed frame matches` and `relaxed matched`
- [temp/detection_parity_full.py](/home/ubuntu/inference/temp/detection_parity_full.py:1)
  - reports `relaxed matched` and `relaxed full-match images`

### Current relaxed-gate leaders

Baseline exact path on the corrected workflow benchmark:

- current exact path: `247.61 FPS` median over 3 runs

#### Timing-cache subset cluster (`top1` .. `top10`)

The relaxed workflow screen showed that a large timing-cache subset cluster
collapses to the **same** remaining duplicate-count frames:

- `top1`, `top3`, `top4`, `top5`, `top6`, `top7`, `top8`, `top9`, `top10`,
  and `top45` all reduce to:
  - frame `342`
  - frame `495`

So the ranking question becomes pure throughput. Single corrected workflow runs
for that cluster were:

- `top1`: `248.28 FPS`
- `top3`: `247.53 FPS`
- `top4`: `249.74 FPS`
- `top5`: `254.24 FPS`
- `top6`: `245.91 FPS`
- `top7`: `253.06 FPS`
- `top8`: `253.59 FPS`
- `top9`: `246.55 FPS`
- `top10`: `253.95 FPS`
- `top45`: `246.41 FPS`

The best clean 3-run medians from that cluster were:

- `top7_engine`: `254.38 FPS`
- `top10_engine`: `253.90 FPS`
- `top5_engine`: `253.25 FPS`

So `top7_engine` is the current relaxed leader.

Candidate:

- [/tmp/rfdetr-seg-nano-tcache-hybrid/top7_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top7_engine)

Corrected workflow benchmark:

- `254.38 FPS` median over 3 runs
- gain over current exact path: about `+6.77 FPS`

Relaxed workflow screen on `vehicles_312px.mp4`:

- `2 / 538` mismatched frames
- both are real duplicate-count frames:
  - `342`
  - `495`

Relaxed COCO smoke (`50` images):

- base / candidate detections: `274 / 275`
- `relaxed matched`: `274`
- `relaxed full-match images`: `49 / 50`

Relaxed COCO full run (`1500` images):

- base / candidate detections: `8036 / 8041`
- `relaxed matched`: `8027`
- `relaxed full-match images`: `1458 / 1500`
- `count-mismatch images`: `21`
- unmatched detections are overwhelmingly threshold-edge:
  - score bins from the `21` count-mismatch images:
    - `<0.45`: `20`
    - `0.45-0.5`: `1`
  - the workflow duplicates remain the harder mismatch surface; the broader
    COCO count drift is mostly low-confidence toggling near the `0.4`
    threshold

So under the relaxed metric this is currently the **fastest live lead**.

#### Canonical decoder layer-3 rewrite

Candidate:

- [/tmp/rfdetr-seg-nano-trt-sweep/local_fp16_opt3_selfattn_canonical_full_l3_tsrc_cublas](/tmp/rfdetr-seg-nano-trt-sweep/local_fp16_opt3_selfattn_canonical_full_l3_tsrc_cublas)

Corrected workflow benchmark:

- `248.73 FPS` median over 3 runs
- gain over current exact path: about `+1.12 FPS`

Relaxed workflow screen:

- `1 / 538` mismatched frame
- remaining miss:
  - `236`
  - one missing borderline class-`2` detection

Relaxed COCO smoke (`50` images):

- base / candidate detections: `274 / 275`
- `relaxed matched`: `274`
- `relaxed full-match images`: `49 / 50`

So this is the **cleanest relaxed lead**, but it is materially slower than the
timing-cache subset leader.

#### Combined segmentation-head rewrite

Candidate:

- [/tmp/rfdetr-seg-nano-seghead-all-flatmatmul/local_fp16_opt3_seghead_all_flat](/tmp/rfdetr-seg-nano-seghead-all-flatmatmul/local_fp16_opt3_seghead_all_flat)

Corrected workflow benchmark:

- `249.28 FPS` median over 3 runs
- gain over current exact path: about `+1.67 FPS`

Relaxed workflow screen:

- `2 / 538` mismatched frames
- same duplicate-count class as the old non-exact leads:
  - `342`
  - `495`

This makes it less attractive than `top7_engine` on speed and no better on the
relaxed workflow surface.

#### `top5` decoder-2 canonical FP32 rescue

Candidate:

- [/tmp/rfdetr-seg-nano-forward-rewrites/encoder45_decoder2_top5_decoder2canon_fp32](/tmp/rfdetr-seg-nano-forward-rewrites/encoder45_decoder2_top5_decoder2canon_fp32)

Corrected workflow benchmark:

- `251.28 FPS` median over 3 runs
- runs: `251.28`, `250.94`, `252.36`

Relaxed workflow screen:

- `2 / 538` mismatched frames
- it fixes frame `342`, but introduces `441`
- remaining mismatches:
  - `441`
  - `495`

This is a weaker tradeoff than `top7_engine`: slower by about `3.1 FPS` and no
cleaner on the relaxed metric.

#### `top7_drop*` timing-cache neighborhood

I screened the whole `top7_drop1..7` family under the relaxed workflow metric.
Every single drop variant collapses back to the same two duplicate-count
workflow mismatches:

- `342`
- `495`

So dropping any one entry from `top7` does **not** fix the relaxed workflow
surface.

One-run corrected workflow speeds:

- `top7_drop1_engine`: `247.67 FPS`
- `top7_drop2_engine`: `253.42 FPS`
- `top7_drop3_engine`: `250.80 FPS`
- `top7_drop4_engine`: `251.48 FPS`
- `top7_drop5_engine`: `251.72 FPS`
- `top7_drop6_engine`: `251.96 FPS`
- `top7_drop7_engine`: `247.67 FPS`

None of them beat plain `top7_engine`, so the immediate `top7` neighborhood is
closed.

#### `top10` full relaxed COCO comparison

Candidate:

- [/tmp/rfdetr-seg-nano-tcache-hybrid/top10_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top10_engine)

Corrected workflow benchmark:

- `253.90 FPS` median over 3 runs

Relaxed workflow screen:

- same `2 / 538` duplicate-count frames as `top7_engine`
  - `342`
  - `495`

Full relaxed COCO (`1500` images):

- base / candidate detections: `8036 / 8041`
- `relaxed matched`: `8026`
- `relaxed full-match images`: `1457 / 1500`
- `count-mismatch images`: `21`
- `mean / max box drift`: `0.167 / 4.000`

This is slightly slower and slightly less accurate than `top7_engine`, so it
does **not** overtake the current relaxed leader.

#### `top5` full relaxed COCO comparison

Candidate:

- [/tmp/rfdetr-seg-nano-tcache-hybrid/top5_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top5_engine)

Corrected workflow benchmark:

- `253.25 FPS` median over 3 runs

Relaxed workflow screen:

- same `2 / 538` duplicate-count frames as `top7_engine`
  - `342`
  - `495`

Full relaxed COCO (`1500` images):

- base / candidate detections: `8036 / 8041`
- `relaxed matched`: `8026`
- `relaxed full-match images`: `1457 / 1500`
- `count-mismatch images`: `21`
- `mean / max box drift`: `0.167 / 4.000`

This is effectively the same relaxed COCO surface as `top10_engine`, but still
slower than `top7_engine`. So the timing-cache subset cluster is settled:
`top7_engine` remains the best relaxed high-FPS lead in that family.

#### Canonical decoder layer-3 rewrite on full relaxed COCO

Candidate:

- [/tmp/rfdetr-seg-nano-trt-sweep/local_fp16_opt3_selfattn_canonical_full_l3_tsrc_cublas](/tmp/rfdetr-seg-nano-trt-sweep/local_fp16_opt3_selfattn_canonical_full_l3_tsrc_cublas)

Corrected workflow benchmark:

- `248.73 FPS` median over 3 runs

Relaxed workflow screen:

- `1 / 538` mismatched frame
- only frame `236`

Full relaxed COCO (`1500` images):

- base / candidate detections: `8036 / 8037`
- `relaxed matched`: `8021`
- `relaxed full-match images`: `1450 / 1500`
- `count-mismatch images`: `29`
- `mean / max box drift`: `0.167 / 4.000`

So the canonical layer-3 rewrite is cleaner on the workflow clip, but it is
meaningfully worse than `top7_engine` on the broader relaxed COCO gate and much
slower. It is not the best relaxed tradeoff.

#### `top7` duplicate-suppression probe

I also checked whether the remaining `top7` relaxed COCO misses were mostly just
duplicate boxes that could be removed with a tiny class-aware dedupe.

Result on the full `1500`-image relaxed COCO artifacts:

- count-mismatch images: `21`
- count-mismatch images with same-class duplicate boxes within `1 px`: `0`

So the broad COCO drift is **not** a duplicate-box problem. The duplicate-count
behavior is isolated to the two workflow frames `342` and `495`. A simple
dedupe could clean the workflow screen, but it would not materially improve the
full COCO relaxed metric.

#### Fresh `nsys` / `ncu` compare: exact baseline vs `top7_engine`

Fresh current-session corrected workflow benchmarks:

- exact baseline: `246.08 FPS` median
- `top7_engine`: `251.06 FPS` median

Fresh forward-only direct harness:

- exact baseline: `209.93 FPS`
- `top7_engine`: `219.84 FPS`

Forward-only `nsys` artifacts:

- exact:
  - [/tmp/profiles/rfdetr_forward_top7_compare/exact/forward_graph.nsys-rep](/tmp/profiles/rfdetr_forward_top7_compare/exact/forward_graph.nsys-rep)
  - [/tmp/profiles/rfdetr_forward_top7_compare/exact/forward_graph.sqlite](/tmp/profiles/rfdetr_forward_top7_compare/exact/forward_graph.sqlite)
- `top7`:
  - [/tmp/profiles/rfdetr_forward_top7_compare/top7/forward_graph.nsys-rep](/tmp/profiles/rfdetr_forward_top7_compare/top7/forward_graph.nsys-rep)
  - [/tmp/profiles/rfdetr_forward_top7_compare/top7/forward_graph.sqlite](/tmp/profiles/rfdetr_forward_top7_compare/top7/forward_graph.sqlite)

Key finding:

- summed GPU kernel time is essentially flat
  - exact: `7.600780 ms`
  - `top7`: `7.599177 ms`
- but wall time over `200` forward-only iterations still drops by about
  `26.7 ms`, or about `133.5 us/iter`

So the `top7` win is a **small TRT tactic improvement on the critical path**,
not a big reduction in total GPU work.

The most visible tactic swaps are:

- fused GEMM:
  - exact: `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32..._fused`
  - `top7`: `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x256x32..._fused`
- execute GEMM:
  - exact: `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x64x64...execute_kernel_trt`
  - `top7`: `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize128x64x64...execute_kernel_trt`

Targeted `ncu` artifacts:

- exact fused:
  - [/tmp/profiles/rfdetr_forward_top7_compare/exact/ncu_fused_exact.ncu-rep](/tmp/profiles/rfdetr_forward_top7_compare/exact/ncu_fused_exact.ncu-rep)
- `top7` fused:
  - [/tmp/profiles/rfdetr_forward_top7_compare/top7/ncu_fused_top7.ncu-rep](/tmp/profiles/rfdetr_forward_top7_compare/top7/ncu_fused_top7.ncu-rep)
- exact execute:
  - [/tmp/profiles/rfdetr_forward_top7_compare/exact/ncu_exec_exact.ncu-rep](/tmp/profiles/rfdetr_forward_top7_compare/exact/ncu_exec_exact.ncu-rep)
- `top7` execute:
  - [/tmp/profiles/rfdetr_forward_top7_compare/top7/ncu_exec_top7.ncu-rep](/tmp/profiles/rfdetr_forward_top7_compare/top7/ncu_exec_top7.ncu-rep)

Selected metrics:

- fused kernel
  - exact: `81.632 us`, regs/thread `240`, smem `17.408 KB`, waves/SM `0.90`,
    SM throughput `48.28%`, DRAM `9.91%`
  - `top7`: `85.504 us`, regs/thread `230`, smem `33.792 KB`, waves/SM `0.90`,
    SM throughput `46.93%`, DRAM `9.16%`
- execute kernel
  - exact: `32.128 us`, regs/thread `186`, smem `24.576 KB`, waves/SM `0.45`,
    SM throughput `31.64%`, DRAM `11.83%`
  - `top7`: `31.904 us`, regs/thread `168`, smem `24.576 KB`, waves/SM `0.45`,
    SM throughput `32.08%`, DRAM `11.82%`

The useful conclusion is that `top7` is still winning through **small
occupancy-limited TRT tactic swaps** inside the same GEMM family, not through a
new custom kernel or lower total GPU work.

#### `top7.cache` rebuilt under `CUBLAS + EDGE + JIT`

Candidate:

- [/tmp/rfdetr-seg-nano-tcache-hybrid/top7_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top7_tsrc_cublas_engine)

This keeps the same `top7.cache`, but rebuilds under the narrower tactic
surface:

- `CUBLAS`
- `EDGE_MASK_CONVOLUTIONS`
- `JIT_CONVOLUTIONS`

Corrected workflow benchmark:

- one-run screen: `253.40 FPS`
- clean 3-run repeat: `252.22 FPS` median
- runs: `252.22`, `250.87`, `254.77`

Relaxed workflow screen:

- same `2 / 538` mismatched frames as `top7_engine`
  - `342`
  - `495`

Full relaxed COCO (`1500` images):

- base / candidate detections: `8036 / 8041`
- `relaxed matched`: `8026`
- `relaxed full-match images`: `1457 / 1500`
- `count-mismatch images`: `21`

This is slightly faster than plain `top7_engine` on the corrected workflow
benchmark, while keeping the same relaxed workflow surface. It is only slightly
worse than `top7_engine` on the broad relaxed COCO gate (`1457` vs `1458`
full-match images), so it is the current fastest relaxed candidate.

#### `top8` / `top10` rebuilt under `CUBLAS + EDGE + JIT`

I then pushed the same tactic-surface idea to the nearest cache neighbors:

- [/tmp/rfdetr-seg-nano-tcache-hybrid/top8_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top8_tsrc_cublas_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top10_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top10_tsrc_cublas_engine)

Cheap corrected workflow screens:

- `top8_tsrc_cublas_engine`: `254.16 FPS`
- `top10_tsrc_cublas_engine`: `255.01 FPS`

Relaxed workflow screen for both:

- same `2 / 538` mismatched frames
  - `342`
  - `495`

Clean 3-run corrected workflow benchmark for `top10_tsrc_cublas_engine`:

- runs: `254.64`, `254.11`, `253.37`
- median: `254.11 FPS`

Full relaxed COCO (`1500` images) for `top10_tsrc_cublas_engine`:

- base / candidate detections: `8036 / 8041`
- `relaxed matched`: `8026`
- `relaxed full-match images`: `1457 / 1500`
- `count-mismatch images`: `21`

So `top10_tsrc_cublas_engine` becomes the new best relaxed-speed lead:

- faster than `top7_tsrc_cublas_engine`
- same relaxed workflow surface
- same broad relaxed COCO surface

I also checked nearby builder levels on this exact candidate:

- [/tmp/rfdetr-seg-nano-tcache-hybrid/top10_tsrc_cublas_opt2_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top10_tsrc_cublas_opt2_engine): `250.63 FPS`
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top10_tsrc_cublas_opt4_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top10_tsrc_cublas_opt4_engine): `251.06 FPS`

Both are slower than the `opt3` build, so the local builder sweep around this
candidate is closed.

#### Local refinement around `top10_tsrc_cublas`

I refined the local neighborhood around `top10_tsrc_cublas` with the same
tactic surface:

- [/tmp/rfdetr-seg-nano-tcache-hybrid/top9_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top9_tsrc_cublas_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top11_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top11_tsrc_cublas_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top12_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top12_tsrc_cublas_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top15_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top15_tsrc_cublas_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top20_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top20_tsrc_cublas_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top25_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top25_tsrc_cublas_engine)

Cheap corrected workflow screens:

- `top9_tsrc_cublas_engine`: `254.81 FPS`
- `top11_tsrc_cublas_engine`: `255.27 FPS`
- `top12_tsrc_cublas_engine`: `254.63 FPS`
- `top15_tsrc_cublas_engine`: `252.85 FPS`
- `top20_tsrc_cublas_engine`: `252.87 FPS`
- `top25_tsrc_cublas_engine`: `252.76 FPS`

This makes the local optimum look clearly centered around `top11`, with
performance already falling off again by `top12` and beyond.

Relaxed workflow screen for `top11_tsrc_cublas_engine`:

- same `2 / 538` mismatched frames as the rest of the `top7/top10` family
  - `342`
  - `495`

Clean 3-run corrected workflow benchmark for `top11_tsrc_cublas_engine`:

- runs: `255.42`, `252.53`, `255.22`
- median: `255.22 FPS`

Full relaxed COCO (`1500` images) for `top11_tsrc_cublas_engine`:

- base / candidate detections: `8036 / 8041`
- `relaxed matched`: `8026`
- `relaxed full-match images`: `1457 / 1500`
- `count-mismatch images`: `21`

So `top11_tsrc_cublas_engine` is the new best relaxed candidate:

- faster than `top10_tsrc_cublas_engine`
- same relaxed workflow surface
- same broad relaxed COCO surface

#### `top11 + 13` and `top11 + 13 + 15`

I then tested whether the next tiny timing-cache entries could improve the
`top11_tsrc_cublas` local optimum without changing the relaxed correctness
surface.

Built candidates:

- [/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_tsrc_cublas_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus14_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus14_tsrc_cublas_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_engine)

The extra cache entries involved are tiny by measured cache timing gain:

- index `13`: key `0xc3a62d2d931e109e1cbd1f2351eda209`, gain `0.0001088 ms`
- index `14`: key `0x4646a92dad6e9de7ce82d276c604d46b`, gain `0.0001023 ms`
- index `15`: key `0xb0714bff5e954dba57437bdb85677d00`, gain `0.0000954 ms`

Same-session clean 3-run corrected workflow benchmarks:

- exact baseline `rfdetr-seg-nano`: `247.65 FPS`
- `top11_tsrc_cublas_engine`: `253.52 FPS`
- `top11_plus13_tsrc_cublas_engine`: `254.83 FPS`
- `top11_plus13_plus14_tsrc_cublas_engine`: `253.89 FPS`
- `top11_plus13_plus15_tsrc_cublas_engine`: `255.04 FPS`

So `+13` was a real improvement over plain `top11`, `+14` gave that back, and
`+13+15` became the best point in this local neighborhood.

Relaxed workflow screen for `top11_plus13_tsrc_cublas_engine`:

- same `2 / 538` mismatched frames
  - `342`
  - `495`

Full relaxed COCO (`1500` images) for `top11_plus13_tsrc_cublas_engine`:

- base / candidate detections: `8036 / 8041`
- `relaxed matched`: `8027`
- `relaxed full-match images`: `1458 / 1500`
- `count-mismatch images`: `21`

Relaxed workflow screen for `top11_plus13_plus15_tsrc_cublas_engine`:

- same `2 / 538` mismatched frames
  - `342`
  - `495`

Full relaxed COCO (`1500` images) for `top11_plus13_plus15_tsrc_cublas_engine`:

- base / candidate detections: `8036 / 8041`
- `relaxed matched`: `8027`
- `relaxed full-match images`: `1458 / 1500`
- `count-mismatch images`: `21`

So `top11_plus13_plus15_tsrc_cublas_engine` is the current best relaxed-speed
lead:

- fastest corrected workflow benchmark in this family
- same relaxed workflow surface as the prior leaders
- better broad relaxed COCO than `top10_tsrc_cublas_engine` /
  `top11_tsrc_cublas_engine`

#### Tactic-surface sweep on `top11 + 13 + 15`

I then treated `top11_plus13_plus15_tsrc_cublas_engine` as the new cache-level
winner and checked whether the TRT tactic surface around it could be improved
further.

Built variants:

- [/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_only_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_only_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_cudnn_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_cudnn_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_edge_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_edge_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_jit_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_jit_engine)
- [/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_cudnn_edge_jit_engine](/tmp/rfdetr-seg-nano-tcache-hybrid/top11_plus13_plus15_tsrc_cublas_cudnn_edge_jit_engine)

Cheap one-run corrected workflow screen:

- `CUBLAS` only: `247.42 FPS`
- `CUBLAS + CUDNN`: `249.24 FPS`
- `CUBLAS + EDGE`: `252.04 FPS`
- `CUBLAS + JIT`: `213.57 FPS`
- `CUBLAS + EDGE + JIT` (current winner): `255.04 FPS` median from the prior
  clean 3-run batch
- `CUBLAS + CUDNN + EDGE + JIT`: `256.40 FPS`

So the only tactic-surface variant that cleared the current winner was the one
that **added CUDNN on top of the existing `CUBLAS + EDGE + JIT` mix**.

Clean 3-run corrected workflow benchmark for
`top11_plus13_plus15_tsrc_cublas_cudnn_edge_jit_engine`:

- runs: `256.70`, `253.98`, `255.69`
- median: `255.69 FPS`

Relaxed workflow screen:

- same `2 / 538` mismatched frames
  - `342`
  - `495`

Full relaxed COCO (`1500` images):

- base / candidate detections: `8036 / 8041`
- `relaxed matched`: `8027`
- `relaxed full-match images`: `1458 / 1500`
- `count-mismatch images`: `21`

So `top11_plus13_plus15_tsrc_cublas_cudnn_edge_jit_engine` becomes the new best
relaxed candidate:

- slightly faster than `top11_plus13_plus15_tsrc_cublas_engine`
- same relaxed workflow surface
- same broad relaxed COCO surface

### Current relaxed ranking

If the relaxed metric is accepted, the ordering at the moment is:

1. `top11_plus13_plus15_tsrc_cublas_cudnn_edge_jit_engine`
   - fastest corrected workflow benchmark in the relaxed set
   - `255.69 FPS` median in the current session
   - same `2 / 538` relaxed workflow mismatches as the `top7/top10/top11`
     family
   - `1458 / 1500` relaxed COCO full-match images
2. `top11_plus13_plus15_tsrc_cublas_engine`
   - slightly slower
   - `255.04 FPS` median in the current session
   - same `2 / 538` relaxed workflow mismatches
   - `1458 / 1500` relaxed COCO full-match images
3. `top11_plus13_tsrc_cublas_engine`
   - fastest corrected workflow benchmark in the relaxed set
   - `254.83 FPS` median in the current session
   - same `2 / 538` relaxed workflow mismatches
   - `1458 / 1500` relaxed COCO full-match images
4. `top11_tsrc_cublas_engine`
   - slightly slower
   - `253.52 FPS` median
   - same `2 / 538` relaxed workflow mismatches
   - `1457 / 1500` relaxed COCO full-match images
5. `top10_tsrc_cublas_engine`
   - slightly slower
   - `254.11 FPS` median
   - same `2 / 538` relaxed workflow mismatches
   - `1457 / 1500` relaxed COCO full-match images
6. `top7_tsrc_cublas_engine`
   - slightly slower
   - same `2 / 538` relaxed workflow mismatches
   - `1457 / 1500` relaxed COCO full-match images
7. `top7_engine`
   - slightly slower
   - `2 / 538` relaxed workflow mismatches
   - `49 / 50` relaxed COCO smoke full-match images
   - `1458 / 1500` relaxed COCO full-match images
   - nearby `top7_drop*` variants do not improve either speed or relaxed
     workflow parity
   - `top5_engine` and `top10_engine` are both slightly worse on speed and full
     relaxed COCO
8. canonical decoder layer-3 rewrite
   - cleaner workflow screen (`1 / 538`)
   - much smaller speedup
   - worse full relaxed COCO (`1450 / 1500`)
9. combined segmentation-head rewrite
   - small speedup
   - same `2` relaxed workflow mismatch frames as `top7_engine`
10. `top5` decoder-2 canonical FP32 rescue
   - fixes `342`
   - introduces `441`
   - slower than `top7_engine`

## Practical Conclusion

The research has converged on a fairly clear picture:

1. The current exact path is already highly optimized.
2. The remaining ceiling is TRT forward GEMM/MHA, not Triton pre/postproc and
   not workflow glue.
3. Broad TRT tactic or timing-cache sweeps can find faster kernels, but they do
   not preserve exactness.
4. Under the relaxed metric, the best current lead is a **narrow** timing-cache
   plus tactic-surface combination, not a broad rewrite.
5. The current best relaxed candidate is
   `top11_plus13_plus15_tsrc_cublas_cudnn_edge_jit_engine`:
   - `255.69 FPS` median on the corrected workflow benchmark
   - `1458 / 1500` relaxed full-match COCO images
   - only `2 / 538` relaxed workflow mismatch frames (`342`, `495`)
6. Native plugin replacement is technically possible on this host.
7. The correct forward replacement boundary is the attention core, not isolated
   matmuls.
8. The current native attention-core plugin implementation is not exact enough
   to promote.

## Recommended Next Steps

If work continues, the best next options are:

1. Refine the native attention-core plugin implementation itself:
   - investigate where the first drift appears on the direct async surface,
   - compare plugin output to the baseline around a single replaced encoder
     layer on a handful of mismatch frames,
   - only continue if the drift looks local and correctable.
2. If the plugin line keeps drifting at the single-layer level, stop that branch
   and return to exact TRT-side tactic confinement around the current engine's
   hot `_gemm_mha_v2` / GEMM family.
3. Do not spend more time on:
   - broad timing-cache blends,
   - broad tactic-source sweeps,
   - built-in TRT QKV pluginization,
   - standalone projection-matmul / fc2 plugins.

## Quick Reference: Important Artifacts

### Exact-path correctness

- workflow parity:
  - [/tmp/workflow_parity_base_ring4default.pkl](/tmp/workflow_parity_base_ring4default.pkl)
  - [/tmp/workflow_parity_candidate_ring4default.pkl](/tmp/workflow_parity_candidate_ring4default.pkl)
- COCO parity:
  - [/tmp/det_parity_base_ring4default.pkl](/tmp/det_parity_base_ring4default.pkl)
  - [/tmp/det_parity_candidate_ring4default.pkl](/tmp/det_parity_candidate_ring4default.pkl)

### Current exact-path profiling

- steady-state `nsys`:
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_range.nsys-rep](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_range.nsys-rep)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_range_export.sqlite](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_range_export.sqlite)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_gpu_kern_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_gpu_kern_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_api_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_api_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_osrt_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_osrt_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_gpu_mem_time_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_node/workflow_stats_fixed_cuda_gpu_mem_time_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_range.nsys-rep](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_range.nsys-rep)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_range_export.sqlite](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_range_export.sqlite)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_gpu_kern_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_gpu_kern_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_api_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_api_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_osrt_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_osrt_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_gpu_mem_time_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_node/workflow_stats_cuda_gpu_mem_time_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_range.nsys-rep](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_range.nsys-rep)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_range_export.sqlite](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_range_export.sqlite)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_stats_cuda_api_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_stats_cuda_api_sum.csv)
  - [/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_stats_osrt_sum.csv](/tmp/profiles/rfdetr_workflow_cpu_gpu_620_regen_graph/workflow_stats_osrt_sum.csv)

### Current native attention-core candidates

- two-layer candidate:
  - [/tmp/rfdetr-seg-nano-native-attncore-l45](/tmp/rfdetr-seg-nano-native-attncore-l45)
- single-layer candidates:
  - [/tmp/rfdetr-seg-nano-native-attncore-l4](/tmp/rfdetr-seg-nano-native-attncore-l4)
  - [/tmp/rfdetr-seg-nano-native-attncore-l5](/tmp/rfdetr-seg-nano-native-attncore-l5)
  - [/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32prefer](/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32prefer)
  - [/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32obey](/tmp/rfdetr-seg-nano-native-attncore-l4-pluginfp32obey)
- standalone boundary comparator:
  - [temp/compare_native_attention_core_boundary.py](/home/ubuntu/inference/temp/compare_native_attention_core_boundary.py:1)
