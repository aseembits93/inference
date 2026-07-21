#!/usr/bin/env python3
"""Insert Identity nodes between Softmax and downstream MatMul in HF SAM3
ONNX to prevent TRT's MHA fusion from matching the attention pattern.

Hypothesis: TRT's _gemm_mha_v2 fuses `MatMul -> [Mul] -> Softmax -> MatMul`
into a single kernel whose semantics differ subtly from the explicit op
sequence for HF's `Sam3Attention` (specifically around the attention
mask / relative position bias handling). By inserting an Identity between
the softmax output and its downstream MatMul, we force TRT to treat the
two matmuls as separate kernels, bypassing the buggy fused path.

Produces a new ONNX under sam3_hf_onnx_nofuse/ that TRT can still build
from but won't recognize as an MHA pattern.

Controlled by env var:
  BREAK_WHERE=decoder  (default) -- only detr_decoder
  BREAK_WHERE=all      -- every Softmax->MatMul anywhere
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs

SRC = Path(
    "./sam3_hf_onnx_full/sam3_full.onnx"
)
DST_DIR_TMPL = "./sam3_hf_onnx_nofuse_{tag}"


def main() -> int:
    where = os.environ.get("BREAK_WHERE", "decoder")
    if where == "decoder":
        pattern_in_name = ["detr_decoder"]
        dst_dir = Path(DST_DIR_TMPL.format(tag="decoder"))
    elif where == "all":
        pattern_in_name = [""]  # match all
        dst_dir = Path(DST_DIR_TMPL.format(tag="all"))
    elif where == "decoder_and_encoder":
        pattern_in_name = ["detr_decoder", "detr_encoder"]
        dst_dir = Path(DST_DIR_TMPL.format(tag="decoder_encoder"))
    else:
        print(f"unknown BREAK_WHERE={where}")
        return 1

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Clean any previous run's output
    for f in dst_dir.iterdir():
        f.unlink()

    # Symlink external weight files from SRC dir into DST dir so the
    # resulting ONNX can resolve them without copying 3.2 GB
    src_dir = SRC.parent
    for f in src_dir.iterdir():
        if f.name == SRC.name:
            continue
        (dst_dir / f.name).symlink_to(f)

    print(f"[1/3] Loading {SRC} via onnx-graphsurgeon ...")
    t0 = time.perf_counter()
    # gs.import_onnx doesn't need external data loaded; just proto structure
    proto = onnx.load(str(SRC), load_external_data=False)
    graph = gs.import_onnx(proto)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s: "
          f"{len(graph.nodes)} nodes, {len(graph.tensors())} tensors")

    # Walk nodes, find softmax->matmul pairs in the targeted region
    print(f"\n[2/3] Inserting Identity nodes (scope: {where}) ...")
    softmax_nodes = [n for n in graph.nodes if n.op == "Softmax"]
    print(f"  {len(softmax_nodes)} softmax nodes total")

    # Build consumer map: tensor -> list[node]
    consumers = {}
    for n in graph.nodes:
        for t in n.inputs:
            consumers.setdefault(t.name, []).append(n)

    n_inserted = 0
    for sm in softmax_nodes:
        # Is this softmax in a targeted region?
        if not any(p in sm.name for p in pattern_in_name) and pattern_in_name != [""]:
            continue
        # Softmax output tensor
        if not sm.outputs:
            continue
        sm_out = sm.outputs[0]
        # Find downstream MatMul consumers
        downstream_matmuls = [d for d in consumers.get(sm_out.name, []) if d.op == "MatMul"]
        if not downstream_matmuls:
            continue

        # Create a new Identity node that consumes sm_out and produces a new tensor
        # Replace the MatMul's input with the new tensor
        new_out = gs.Variable(
            name=f"{sm.name}_identity_out", dtype=sm_out.dtype, shape=sm_out.shape,
        )
        identity_node = gs.Node(
            op="Identity",
            name=f"{sm.name}_identity",
            inputs=[sm_out],
            outputs=[new_out],
        )
        graph.nodes.append(identity_node)

        # Rewire the downstream MatMul(s) to read from new_out instead of sm_out
        for mm in downstream_matmuls:
            for i, inp in enumerate(mm.inputs):
                if inp is sm_out:
                    mm.inputs[i] = new_out
        n_inserted += 1

    print(f"  inserted {n_inserted} Identity nodes")

    print(f"\n[3/3] Serializing modified graph to {dst_dir} ...")
    graph.cleanup().toposort()
    new_proto = gs.export_onnx(graph)

    # Preserve ir_version / opset info
    new_proto.ir_version = proto.ir_version
    if not any(o.domain == "" for o in new_proto.opset_import):
        op = new_proto.opset_import.add()
        op.domain = ""
        op.version = 17

    out_path = dst_dir / "sam3_full_nofuse.onnx"
    onnx.save(
        new_proto, str(out_path),
        save_as_external_data=False,  # we already symlinked external data
    )
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Verify the modified graph
    n_sm = sum(1 for n in new_proto.graph.node if n.op_type == "Softmax")
    n_id = sum(1 for n in new_proto.graph.node if n.op_type == "Identity"
               and "_identity" in (n.name or ""))
    print(f"  {n_sm} Softmax, {n_id} inserted Identity nodes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
