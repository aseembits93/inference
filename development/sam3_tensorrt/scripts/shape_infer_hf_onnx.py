#!/usr/bin/env python3
"""Apply ONNX shape inference (no full onnxsim) to the HF SAM3 ONNX.

onnxsim trips over the >2GB model size. Shape inference alone is what
we actually need: it's what reconciles the source {4} / target {5}
shape annotation mismatch that ORT warned about (and that probably
causes TRT to mis-select a kernel for the DETR decoder).

Uses onnx.shape_inference.infer_shapes_path(), which operates on the
FILE and its external-data siblings, avoiding the in-memory 2GB limit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import onnx
import onnx.shape_inference

SRC = Path(
    "./sam3_hf_onnx_full/sam3_full.onnx"
)
DST_DIR = Path("./sam3_hf_onnx_inferred")
DST_DIR.mkdir(parents=True, exist_ok=True)
DST = DST_DIR / "sam3_full_inferred.onnx"

# Copy external weight files alongside DST first (shape inference doesn't
# touch weights but the ONNX parser needs them adjacent).
# We'll symlink to avoid 3.2GB disk duplication.
SRC_DIR = SRC.parent


def main() -> int:
    # Clean any previous attempt
    for f in DST_DIR.iterdir():
        f.unlink()

    # Symlink every external weight file from SRC_DIR to DST_DIR so
    # the inferred ONNX can find them
    linked = 0
    for f in SRC_DIR.iterdir():
        if f.name == SRC.name:
            continue  # skip the main onnx proto
        target = DST_DIR / f.name
        target.symlink_to(f)
        linked += 1
    print(f"Symlinked {linked} external weight files to {DST_DIR}")

    # Run shape inference on the path so external weights resolve correctly
    print(f"\nRunning onnx.shape_inference.infer_shapes_path ...")
    t0 = time.perf_counter()
    onnx.shape_inference.infer_shapes_path(
        str(SRC),
        str(DST),
        check_type=False,
        strict_mode=False,
        data_prop=True,
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print(f"  output: {DST} ({DST.stat().st_size / 1e6:.1f} MB)")

    # Sanity: load header-only and verify
    model = onnx.load(str(DST), load_external_data=False)
    print(f"  ir_version={model.ir_version}, {len(model.graph.node)} nodes, "
          f"{len(model.graph.initializer)} initializers")

    # Check whether the previously-warned Concat node now has resolvable shapes
    target_name = "/sam3/vision_encoder/backbone/embeddings/Concat_output_0"
    for vi in list(model.graph.value_info) + list(model.graph.output):
        if vi.name == target_name:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            print(f"\nShape annotation for {target_name}: ndim={len(dims)} dims={dims}")
            break
    else:
        print(f"\n(no value_info for {target_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
