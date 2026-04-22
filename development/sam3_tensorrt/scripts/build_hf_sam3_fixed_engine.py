#!/usr/bin/env python3
"""Build TRT FP16 engine from the dummy-box-fixed HF ONNX."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import tensorrt as trt

ONNX_PATH = Path(
    "./sam3_hf_onnx_fixed/sam3_full.onnx"
)
ENGINE_PATH = Path(
    "./sam3_hf_onnx_fixed/sam3_hf_fp16.engine"
)


def main() -> int:
    if not ONNX_PATH.exists():
        print(f"ERROR: {ONNX_PATH} missing. Run export_hf_sam3_fixed.py first.")
        return 1

    logger = trt.Logger(trt.Logger.INFO)
    # Initialize TRT's plugin registry so ROIAlign (and others) are available
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing {ONNX_PATH} ...")
    t0 = time.perf_counter()
    if not parser.parse_from_file(str(ONNX_PATH)):
        for i in range(parser.num_errors):
            print(f"  {parser.get_error(i)}")
        return 2
    print(f"  parsed in {time.perf_counter() - t0:.1f}s")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    print("\nBuilding engine ...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0
    if serialized is None:
        print("ERROR: build returned None")
        return 3
    blob = bytes(serialized)
    print(f"  built in {dt:.1f}s ({len(blob) / 1e6:.1f} MB)")
    with open(ENGINE_PATH, "wb") as f:
        f.write(blob)
    print(f"Engine: {ENGINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
