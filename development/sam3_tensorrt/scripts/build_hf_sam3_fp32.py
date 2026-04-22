#!/usr/bin/env python3
"""Build a pure FP32 HF SAM3 TRT engine.

If this still shows 78% recall / -0.08 score drift vs HF-PT, the bug is
in the ONNX export, not in FP16 precision. If it gets clean numbers,
the bug IS FP16 precision (and we need a different pinning strategy).
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
    "./sam3_hf_onnx_full/sam3_hf_fp32.engine"
)


def main() -> int:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(ONNX_PATH)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return 2
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    # NO FP16 flag — pure FP32

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0
    if serialized is None:
        print("Build failed")
        return 3
    blob = bytes(serialized)
    print(f"Built in {dt:.1f}s ({len(blob) / 1e6:.1f} MB)")
    with open(ENGINE_PATH, "wb") as f:
        f.write(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
