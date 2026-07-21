#!/usr/bin/env python3
"""Build HF SAM3 TRT engine with the DETR decoder + mask/bbox heads pinned
to FP32 while the vision backbone and text encoder stay FP16.

Hypothesis: HF-TRT's correctness regression (78% recall, 8pt score drift)
is driven by FP16 accumulation in the DETR decoder, not the vision
backbone. Pinning only the decoder path keeps most of the FP16 speedup
while recovering detection quality.

Components to pin (total ~3369 nodes out of ~15000):
  /sam3/detr_decoder/*       (3135 nodes) - 6 decoder layers + heads
  /sam3/mask_decoder/*        (197 nodes) - mask upsampling/refinement
  /sam3/box_head/*              (8 nodes) - final bbox regression
  /sam3/dot_product_scoring/*  (29 nodes) - similarity scoring

Vision backbone, text encoder, detr_encoder stay FP16.
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
    "./sam3_hf_onnx_full/sam3_hf_fp16_decoder_fp32.engine"
)

PIN_PREFIXES = (
    "/sam3/detr_decoder/",
    "/sam3/mask_decoder/",
    "/sam3/box_head/",
    "/sam3/dot_product_scoring/",
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

    # Collect layers whose names start with any PIN_PREFIX.
    # Skip layer types that produce indices/shape (TRT rejects FP32 precision
    # overrides on them) — same skip list we used for the repo engine.
    SKIP_TYPES = {
        trt.LayerType.SHAPE,
        trt.LayerType.CAST,
        trt.LayerType.CONSTANT,
        trt.LayerType.SLICE,
        trt.LayerType.GATHER,
        trt.LayerType.SHUFFLE,
        trt.LayerType.CONCATENATION,
        trt.LayerType.IDENTITY,
    }

    forced = 0
    skipped_idx_dtype = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name or ""
        if not any(name.startswith(p) for p in PIN_PREFIXES):
            continue
        if layer.type in SKIP_TYPES:
            continue
        if layer.num_outputs == 0:
            continue
        out0 = layer.get_output(0)
        if out0.dtype not in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
            skipped_idx_dtype += 1
            continue
        try:
            layer.precision = trt.DataType.FLOAT
            for j in range(layer.num_outputs):
                out = layer.get_output(j)
                if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                    layer.set_output_type(j, trt.DataType.FLOAT)
            forced += 1
        except Exception as e:
            # Log once per unique error type
            print(f"  warn: could not pin {name} ({layer.type}): {e}")

    print(f"Pinned {forced} decoder/head layers to FP32 "
          f"(skipped {skipped_idx_dtype} with non-FP output types)")

    print("Inputs:",
          [(network.get_input(i).name, network.get_input(i).dtype, network.get_input(i).shape)
           for i in range(network.num_inputs)])
    print("Outputs:",
          [(network.get_output(i).name, network.get_output(i).dtype, network.get_output(i).shape)
           for i in range(network.num_outputs)])

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    # OBEY is strict (build fails if no tactic exists). Try that first; if
    # build fails, retry with PREFER which lets TRT fall back.
    strict = True
    if strict:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    else:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    print("\nBuilding engine ...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0
    if serialized is None:
        print("ERROR: OBEY build returned None -- retrying with PREFER ...")
        # Reset config and retry with PREFER
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        t0 = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        dt = time.perf_counter() - t0
        if serialized is None:
            print("ERROR: build failed even with PREFER")
            return 3

    blob = bytes(serialized)
    print(f"  built in {dt:.1f}s ({len(blob) / 1e6:.1f} MB)")
    with open(ENGINE_PATH, "wb") as f:
        f.write(blob)
    print(f"Engine: {ENGINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
