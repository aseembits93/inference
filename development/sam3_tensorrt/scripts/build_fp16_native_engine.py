#!/usr/bin/env python3
"""Build strongly-typed engine from FP16-native ONNX export."""

import sys, time
from pathlib import Path
import tensorrt as trt

ONNX = Path("./sam3_onnx_exports/sam3_vision_backbone_fp16_native.onnx")
ENG = Path("./sam3_onnx_exports/sam3_vision_backbone_fp16_native.engine")


def main() -> int:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    print(f"Parsing {ONNX.name}...")
    t0 = time.perf_counter()
    with open(ONNX, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return 2
    print(f"  parsed in {time.perf_counter()-t0:.1f}s")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    import os
    # Let user pick which tactic sources to disable via env var
    disable = os.environ.get("DISABLE_TACTIC_SOURCES", "").split(",")
    if disable and disable != [""]:
        src = 0
        for s in ["CUBLAS", "CUBLAS_LT", "CUDNN", "EDGE_MASK_CONVOLUTIONS", "JIT_CONVOLUTIONS"]:
            if s not in disable:
                src |= 1 << int(getattr(trt.TacticSource, s))
        config.set_tactic_sources(src)
        print(f"Tactic sources = {src} (disabled: {disable})")
    # Optionally restrict builder optimization level
    opt_level = int(os.environ.get("BUILDER_OPT_LEVEL", "-1"))
    if opt_level >= 0:
        config.builder_optimization_level = opt_level
        print(f"Builder optimization level: {opt_level}")

    print("Inputs :", [(network.get_input(i).name, network.get_input(i).dtype, network.get_input(i).shape)
                        for i in range(network.num_inputs)])
    print("Outputs:", [(network.get_output(i).name, network.get_output(i).dtype, network.get_output(i).shape)
                        for i in range(network.num_outputs)])

    print("\nBuilding engine ...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: build returned None")
        return 3
    blob = bytes(serialized)
    print(f"  built in {time.perf_counter()-t0:.1f}s ({len(blob)/1e6:.1f} MB)")
    with open(ENG, "wb") as f:
        f.write(blob)
    print(f"Engine: {ENG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
