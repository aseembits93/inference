#!/usr/bin/env python3
"""Build HF SAM3 TRT engine with the vision backbone's attention path
pinned to FP32 (analogous to our SAM3-repo `fp16_attn_hard` preset).

Hypothesis: the ~8-point HF-TRT score compression and per-mask IoU drop
accumulate inside the 32 ViT attention blocks. Pinning every attention
layer to FP32 (q_proj, k_proj, v_proj, o_proj, QK^T, softmax, attn*V,
and all the small RoPE/scale ops between them) should recover quality
while letting MLPs and layer norms run FP16.

Scope: every layer whose name starts with
  /sam3/vision_encoder/backbone/layers.<k>/attention/
stays pinned FP32. Everything else — including MLP, layer norms, neck,
DETR encoder/decoder, text encoder — runs FP16.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import tensorrt as trt

ONNX_PATH = Path(
    "./sam3_hf_onnx_full/sam3_full.onnx"
)
ENGINE_PATH = Path(
    "./sam3_hf_onnx_full/sam3_hf_fp16_attn_fp32.engine"
)

ATTN_RE = re.compile(
    r"^/sam3/vision_encoder/backbone/layers\.\d+/attention/"
)


def main() -> int:
    if not ONNX_PATH.exists():
        print(f"ERROR: {ONNX_PATH} missing.")
        return 1

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing {ONNX_PATH} ...")
    t0 = time.perf_counter()
    if not parser.parse_from_file(str(ONNX_PATH)):
        for i in range(parser.num_errors):
            print(f"  {parser.get_error(i)}")
        return 2
    print(f"  parsed in {time.perf_counter() - t0:.1f}s")

    SKIP_TYPES = {
        trt.LayerType.SHAPE, trt.LayerType.CAST, trt.LayerType.CONSTANT,
        trt.LayerType.SLICE, trt.LayerType.GATHER, trt.LayerType.SHUFFLE,
        trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY,
    }

    forced = 0
    per_block = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name or ""
        if not ATTN_RE.match(name):
            continue
        if layer.type in SKIP_TYPES:
            continue
        if layer.num_outputs == 0:
            continue
        out0 = layer.get_output(0)
        if out0.dtype not in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
            continue
        # Block index
        m = re.match(r"^/sam3/vision_encoder/backbone/layers\.(\d+)/", name)
        if m:
            per_block[int(m.group(1))] = per_block.get(int(m.group(1)), 0) + 1
        try:
            layer.precision = trt.DataType.FLOAT
            for j in range(layer.num_outputs):
                out = layer.get_output(j)
                if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                    layer.set_output_type(j, trt.DataType.FLOAT)
            forced += 1
        except Exception:
            pass

    print(f"Pinned {forced} attention layers to FP32 across {len(per_block)} blocks")
    if per_block:
        samp = sorted(per_block.items())[:3] + sorted(per_block.items())[-3:]
        print(f"  (example counts: {samp})")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    print("\nBuilding engine ...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0
    if serialized is None:
        print("ERROR: OBEY build failed, retrying with PREFER ...")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        t0 = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        dt = time.perf_counter() - t0
        if serialized is None:
            print("ERROR: build failed with PREFER too")
            return 3

    blob = bytes(serialized)
    print(f"  built in {dt:.1f}s ({len(blob) / 1e6:.1f} MB)")
    with open(ENGINE_PATH, "wb") as f:
        f.write(blob)
    print(f"Engine: {ENGINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
