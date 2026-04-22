#!/usr/bin/env python3
"""Build HF SAM3 TRT engine pinning EVERYTHING non-backbone to FP32.

If decoder-only pinning didn't help, maybe the problem is upstream in the
DETR encoder, vision neck/FPN, or dot_product_scoring. Try pinning the
entire post-backbone graph and see how much correctness is recovered.

Scope: keep FP16 only in /sam3/vision_encoder/backbone/ and
/sam3/text_encoder/. Everything else -> FP32.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import tensorrt as trt

ONNX_PATH = Path(
    "./sam3_hf_onnx_full/sam3_full.onnx"
)
ENGINE_PATH = Path(
    "./sam3_hf_onnx_full/sam3_hf_fp16_backbone_only.engine"
)

# Layers inside these prefixes stay FP16. Everything else gets pinned FP32.
KEEP_FP16_PREFIXES = (
    "/sam3/vision_encoder/backbone/",
    "/sam3/text_encoder/",
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
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name or ""
        # Skip if we want FP16 here
        if any(name.startswith(p) for p in KEEP_FP16_PREFIXES):
            continue
        if layer.type in SKIP_TYPES:
            continue
        if layer.num_outputs == 0:
            continue
        out0 = layer.get_output(0)
        if out0.dtype not in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
            continue
        try:
            layer.precision = trt.DataType.FLOAT
            for j in range(layer.num_outputs):
                out = layer.get_output(j)
                if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                    layer.set_output_type(j, trt.DataType.FLOAT)
            forced += 1
        except Exception as e:
            pass

    print(f"Pinned {forced} non-backbone layers to FP32")

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
