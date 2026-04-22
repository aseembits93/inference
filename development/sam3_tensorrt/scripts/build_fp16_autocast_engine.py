#!/usr/bin/env python3
"""Build strongly-typed engine from fp16_autocast ONNX."""

import sys, time
from pathlib import Path
import tensorrt as trt

ONNX = Path("./sam3_onnx_exports/sam3_vision_backbone_fp16_autocast.onnx")
ENG = Path("./sam3_onnx_exports/sam3_vision_backbone_fp16_autocast.engine")


def main() -> int:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    print(f"Parsing {ONNX.name}...")
    with open(ONNX, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return 2

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)

    print("\nBuilding ...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR")
        return 3
    blob = bytes(serialized)
    print(f"  built in {time.perf_counter()-t0:.1f}s ({len(blob)/1e6:.1f} MB)")
    with open(ENG, "wb") as f:
        f.write(blob)
    print(f"Engine: {ENG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
