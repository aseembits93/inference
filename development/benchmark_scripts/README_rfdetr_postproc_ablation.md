# RFDETR-seg post-process: torch.compile vs Triton fullpost ablation

Experiment: can `torch.compile` match the perf of the Triton fullpost kernel
(`triton_rfdetr_fullpost`, W2 in `triton_fullpostproc.py`) on the post-TRT
path?

Run: `python benchmark_rfdetr_postproc_ablation.py`

## Results (T4, 300 queries, 91 classes, 78x78 masks, ~25 survivors, 720x1280 output)

| Variant             | Median per-iter | Speedup vs eager | Kernel launches / iter |
| ------------------- | --------------: | ---------------: | ---------------------: |
| eager (baseline)    | 3.85 ms         | 1.00x            | 60                     |
| compiled (naive)    | 3.75 ms         | 1.03x            | 33                     |
| compiled_fixed      | 3.57 ms         | 1.08x            | 14                     |
| compiled_hybrid     | 1.72 ms         | 2.25x            | 14                     |
| triton fullpost     | 1.56 ms         | 2.47x            | 3.4                    |

Kernel-launch counts come from nsys (`nsys stats --report gpukernsum`) on a
70-iter run.

## Why naive `torch.compile` fails

Dynamo breaks the graph on every boolean-mask index (`confidence[keep_thr]`
-> `aten.nonzero`, data-dependent output shape). Even with
`capture_dynamic_output_shape_ops=True`, the post-proc partitions into 7
cudagraph regions because of the mask-gather and `counter.item()`-like CPU
dependencies. The filter chain's small tensors stay eager and there is
nothing to fuse — hence ~1x speedup.

## What compiled_hybrid does

1. Compile only the shape-static prefix (sigmoid + argmax + remap + bbox
   denorm). No boolean indexing in this region — compiles cleanly with
   `fullgraph=True`.
2. In eager Python: `keep.nonzero()`, sort by confidence, gather the mask
   rows, and `F.interpolate` only those.

Two things matter here:

- **`F.interpolate` beats `TVF.resize`.** The real path uses
  `torchvision.transforms.functional.resize`, which defaults to an
  antialiased kernel (`upsample_gen2d_aa_out_frame`) that is ~2x slower
  than plain bilinear `upsample_bilinear2d_out_frame` at this size. The
  hybrid variant drops antialiasing.
- **Compacting before upsample keeps mask work proportional to survivors.**
  25 mask rows to 720x1280 is 1.3 ms eager / 0.5 ms compiled. 300 rows is
  15.5 ms eager / 3.5 ms compiled.

## What the Triton fullpost can do that torch.compile can't

1. **Over-launch + on-GPU early exit.** Mask kernel launches a grid sized
   for all 300 queries; each program reads the atomic counter on-GPU and
   returns if its slot index >= n_survivors. This avoids the CPU-blocking
   `counter.item()` that would otherwise sit between the two kernels.
   No torch op maps to this pattern.
2. **Atomic-counter compaction.** `tl.atomic_add(counter_ptr, 1)` reserves
   a unique output slot per surviving query, replacing the
   `nonzero -> sort -> gather` chain with one RMW.
3. **Combined struct-of-ints output.** One int32 buffer with
   `[x1, y1, x2, y2, conf_bits, cls]` per slot (conf bit-reinterpreted as
   int32 and decoded host-side), instead of four separate tensors.

## nsys dominant kernels

- **eager**: `upsample_gen2d_aa_out_frame` = 70.7% of kernel time.
- **compiled_hybrid**: `upsample_bilinear2d_out_frame` on ~25 masks
  = 29.6%; uint8 `direct_copy_kernel` = 41.9%.
- **triton**: `_rfdetr_fullpost_mask_kernel_compact` = 99.5%.

## Takeaway

The practical 90% path: compile the filter+bbox prefix and swap
`TVF.resize` for `F.interpolate` (dropping antialiasing) in the upsample.
That's ~10 lines of torch and gets 2.25x vs 2.47x Triton.

The last ~10% requires the atomic-counter + over-launch pattern, which is
fundamentally Triton territory.
