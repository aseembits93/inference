#!/usr/bin/env python3
"""Inspect input/output dtypes of each built TRT engine."""

import sys
from pathlib import Path
import tensorrt as trt

ENGINE_DIR = Path("./sam3_onnx_exports")

def main() -> int:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    for p in sorted(ENGINE_DIR.glob("*.engine")):
        with open(p, "rb") as f:
            eng = runtime.deserialize_cuda_engine(f.read())
        if eng is None:
            print(f"{p.name}: FAILED to deserialize (likely OOM)")
            continue
        print(f"\n{p.name} ({p.stat().st_size/1e6:.0f} MB):")
        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            mode = eng.get_tensor_mode(name)
            dtype = eng.get_tensor_dtype(name)
            shape = eng.get_tensor_shape(name)
            tag = "IN" if mode == trt.TensorIOMode.INPUT else "OUT"
            print(f"  {tag} {name}: {dtype} {shape}")
        del eng
    return 0

if __name__ == "__main__":
    sys.exit(main())
