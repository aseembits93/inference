#!/usr/bin/env python3
"""Find the first layer at which FP16-TRT output diverges from PyTorch.

Marks a user-specified tensor from the FP16 network as an additional output,
builds a new engine, runs on fixed input, and compares against PyTorch
intermediate output via hooks. Binary-search the divergence point.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import torch
import tensorrt as trt

ONNX_PATH = Path(
    "./sam3_onnx_exports/sam3_vision_backbone_fp16_native.onnx"
)


def build_with_extra_outputs(extra_output_names: list[str]) -> tuple:
    """Build engine with additional tensors marked as outputs."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    # Build a name lookup
    tensor_by_name = {}
    for i in range(network.num_layers):
        l = network.get_layer(i)
        for j in range(l.num_outputs):
            out = l.get_output(j)
            tensor_by_name[out.name] = out

    for name in extra_output_names:
        if name in tensor_by_name:
            t = tensor_by_name[name]
            network.mark_output(t)
            print(f"  marked extra output: {name}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("Build failed")
        return None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(serialized))
    return engine


def main() -> int:
    # Get candidate output names (a few from each block)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(ONNX_PATH, "rb") as f:
        parser.parse(f.read())

    # Pick one tensor per block at the block output
    import re
    candidates = []
    pat = re.compile(r"/blocks\.(\d+)/")
    for i in range(network.num_layers):
        l = network.get_layer(i)
        name = l.name or ""
        m = pat.search(name)
        if not m:
            continue
        # Pick "norm2/LayerNormalization_output_0" which is near block exit
        if l.type == trt.LayerType.NORMALIZATION and "norm2" in name:
            out = l.get_output(0)
            candidates.append(out.name)

    print(f"Found {len(candidates)} candidate block-exit tensors")
    # Deduplicate per block
    by_block = {}
    for c in candidates:
        m = pat.search(c)
        if m:
            n = int(m.group(1))
            by_block[n] = c
    probe_names = [by_block[i] for i in sorted(by_block)[:]]
    print(f"Probing {len(probe_names)} block-exit tensors")

    engine = build_with_extra_outputs(probe_names)
    if engine is None:
        return 1

    # Run TRT and capture each block's output
    ctx = engine.create_execution_context()
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(nm))
        ctx.set_input_shape(nm, shape) if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT else None

    dtype_map = {
        trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
    }
    bufs = {}
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(nm))
        dt = dtype_map[engine.get_tensor_dtype(nm)]
        buf = torch.empty(shape, dtype=dt, device="cuda")
        bufs[nm] = buf
        ctx.set_tensor_address(nm, buf.data_ptr())

    # Fill input deterministically
    torch.manual_seed(42)
    x = torch.randn(1, 3, 1008, 1008, device="cuda", dtype=torch.float32)
    bufs["samples"].copy_(x.to(bufs["samples"].dtype))

    stream = torch.cuda.current_stream().cuda_stream
    ctx.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    # Now run PyTorch with hooks on block outputs
    # same-directory imports
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    from export_sam3_backbone_v2 import patch_vitdet_rope_v2

    rf = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    backbone = rf.model.backbone.eval()
    patch_vitdet_rope_v2(backbone)

    # Collect block-exit outputs in PyTorch
    trunk = backbone.vision_backbone.trunk
    block_outputs: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook(module, inp, out):
            block_outputs[idx] = out.detach().clone()
        return hook

    hooks = []
    for i, blk in enumerate(trunk.blocks):
        # Hook norm2 (the layer we marked in TRT)
        hooks.append(blk.norm2.register_forward_hook(make_hook(i)))

    with torch.inference_mode():
        _ = backbone.forward_image(x)

    for h in hooks:
        h.remove()

    print(f"\nBlock output divergences (TRT FP16 vs PT FP32):")
    print(f"{'block':>6} {'TRT range':>25} {'PT range':>25} {'ratio(std)':>10} {'cos':>6}")
    for bi in sorted(by_block):
        tname = by_block[bi]
        if tname not in bufs:
            continue
        trt_t = bufs[tname].float().cpu()
        if bi not in block_outputs:
            continue
        pt_t = block_outputs[bi].float().cpu()
        # TRT shape from ONNX may be [1, H*W, C] while PT is [1, H, W, C]
        if trt_t.numel() == pt_t.numel():
            trt_f = trt_t.flatten()
            pt_f = pt_t.flatten()
        else:
            # Shape mismatch; just compare stats
            trt_f = trt_t.flatten()[:pt_t.numel()]
            pt_f = pt_t.flatten()[:trt_t.numel()]
        ts, ps = trt_t.std().item(), pt_t.std().item()
        ratio = ts / max(ps, 1e-9)
        if trt_t.numel() == pt_t.numel():
            cos = (trt_f @ pt_f / (trt_f.norm() * pt_f.norm() + 1e-12)).item()
        else:
            cos = float("nan")
        print(f"  {bi:>4d}  [{trt_t.min():>7.2f}..{trt_t.max():>7.2f}] "
              f"[{pt_t.min():>7.2f}..{pt_t.max():>7.2f}]  {ratio:>9.3f}  {cos:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
