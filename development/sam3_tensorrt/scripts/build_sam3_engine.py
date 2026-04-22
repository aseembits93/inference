#!/usr/bin/env python3
"""Build a TRT engine from the SAM3 vision backbone ONNX.

Usage:
  build_sam3_engine.py [fp16|fp32]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import tensorrt as trt

import os as _os_m
ONNX_PATH = Path(_os_m.environ.get(
    "SAM3_ONNX_PATH",
    "./sam3_onnx_exports/sam3_vision_backbone_fp16.onnx",
))

PRECISION = sys.argv[1] if len(sys.argv) > 1 else "fp16"
_suffix = _os_m.environ.get("SAM3_ENGINE_SUFFIX", "")
ENGINE_PATH = Path(
    f"./sam3_onnx_exports/sam3_vision_backbone_{PRECISION}{_suffix}.engine"
)


def main() -> int:
    if not ONNX_PATH.exists():
        print(f"ERROR: {ONNX_PATH} does not exist. Run export first.")
        return 1

    print(f"Building TRT engine from {ONNX_PATH}")
    print(f"Precision: {PRECISION}")
    print(f"TRT version: {trt.__version__}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print("Parsing ONNX ...")
    t0 = time.perf_counter()
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return 2
    print(f"  parsed in {time.perf_counter() - t0:.1f}s")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    if PRECISION == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif PRECISION == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif PRECISION == "bf16_io":
        # BF16 precision AND cast the network I/O to BF16 so TRT is forced
        # to run BF16 kernels end-to-end (otherwise it may still pick FP32).
        config.set_flag(trt.BuilderFlag.BF16)
        for i in range(network.num_inputs):
            t = network.get_input(i)
            t.dtype = trt.DataType.BF16
        for i in range(network.num_outputs):
            t = network.get_output(i)
            t.dtype = trt.DataType.BF16
    elif PRECISION == "bf16_in":
        # BF16 compute, BF16 input (to force BF16 kernels), FP32 output
        # (to preserve precision for downstream PyTorch consumers).
        config.set_flag(trt.BuilderFlag.BF16)
        for i in range(network.num_inputs):
            t = network.get_input(i)
            t.dtype = trt.DataType.BF16
    elif PRECISION == "fp16_fp32weights":
        # Use original FP32 ONNX with FP16 flag — TRT will keep weights in
        # FP32 internally and only cast activations to FP16 at runtime.
        # This matches PyTorch's autocast semantics more closely.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    elif PRECISION == "fp16_all_add_fp32":
        # FP16 globally; pin every ElementWise-ADD to FP32.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type != trt.LayerType.ELEMENTWISE:
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
            except Exception:
                pass
        print(f"  forced {forced} ElementWise layers to FP32")
    elif PRECISION == "fp16_residual_fp32":
        # FP16 globally, but the residual-stream Add layers (ElementWise-ADD
        # whose output goes into the next block's input) stay in FP32. This
        # prevents FP16 rounding errors from accumulating in the residual.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        forced = 0
        # Heuristic: every Add whose name ends in "/Add" that is NOT inside
        # attn or qkv.bias. These are the residual adds after attn and mlp.
        import re
        pat = re.compile(r"^/trunk/blocks\.\d+/Add(_\d+)?$")
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name or ""
            if not pat.match(name):
                continue
            if layer.type not in (trt.LayerType.ELEMENTWISE,):
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
            except Exception:
                pass
        print(f"  forced {forced} residual Add layers to FP32")
    elif PRECISION == "fp16_no_mha_fusion":
        # FP16 globally; disable the TRT fused MHA kernel selection by setting
        # allowed tactic sources to exclude cuBLAS MHA kernels. Keep FP16
        # otherwise for max throughput.
        config.set_flag(trt.BuilderFlag.FP16)
        # Disable all optional fusion tactics that might pick up MHA
        config.set_tactic_sources(0)
    elif PRECISION == "fp16_all_matmul":
        # FP16 globally, pin ALL MatMul layers to FP32.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type != trt.LayerType.MATRIX_MULTIPLY:
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
            except Exception:
                pass
        print(f"  forced {forced} MatMul layers to FP32")
    elif PRECISION == "fp16_attn_matmul_only":
        # FP16 globally, pin ONLY the attention MatMuls to FP32 (the bmm's
        # and the qkv/out projection linears).
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = (layer.name or "").lower()
            if "attn" not in name:
                continue
            if layer.type != trt.LayerType.MATRIX_MULTIPLY:
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
            except Exception:
                pass
        print(f"  forced {forced} attn MatMul layers to FP32")
    elif PRECISION == "fp16_attn_no_matmul":
        # FP16 globally; attn layers in FP32 EXCEPT for MatMuls (qkv proj,
        # attn output proj, QK^T, and attn*V). MatMuls dominate attention
        # compute, so keeping those in FP16 matters for speed. The RoPE
        # math + softmax stay in FP32.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = (layer.name or "").lower()
            if "attn" not in name:
                continue
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.SLICE, trt.LayerType.GATHER,
                              trt.LayerType.CONCATENATION, trt.LayerType.CONSTANT,
                              trt.LayerType.SHUFFLE, trt.LayerType.IDENTITY,
                              trt.LayerType.MATRIX_MULTIPLY):
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
            except Exception:
                pass
        print(f"  forced {forced} attn non-MatMul layers to FP32")
    elif PRECISION == "fp16_attn_hard":
        # FP16 for the MLP/conv paths; attention path HARD-pinned to FP32.
        # Attention layers identified as: any layer whose name contains
        # "attn/" but NOT "attn/proj" (the output projection). We also keep
        # the attn.qkv linear in FP32 since it feeds the RoPE inputs.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = (layer.name or "").lower()
            if "attn" not in name:
                continue
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.SLICE, trt.LayerType.GATHER,
                              trt.LayerType.CONCATENATION, trt.LayerType.CONSTANT,
                              trt.LayerType.SHUFFLE, trt.LayerType.IDENTITY):
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
            except Exception:
                pass
        print(f"  forced {forced} attn layers to FP32")
    elif PRECISION == "fp16_attn_fp32":
        # FP16 globally, but force every layer whose name contains "attn" to
        # FP32. The vitdet attention uses real-arithmetic RoPE that TRT's
        # optimizer miscomputes in FP16 (magnitude ~2.5× off). Keep the
        # entire attention path in FP32 and let the ConvNeXt blocks run FP16.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = (layer.name or "").lower()
            if "attn" not in name:
                continue
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.SLICE, trt.LayerType.GATHER,
                              trt.LayerType.CONCATENATION, trt.LayerType.CONSTANT):
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
            except Exception:
                pass
        print(f"  forced {forced} attn layers to FP32")
    elif PRECISION == "fp16_rope_windowed_tf32":
        # Same as fp16_rope_windowed but also enable TF32 so the FP32
        # MatMuls in the windowed attention use TF32 tensor cores (if the
        # GPU supports them). T4 does not but A100/L4/Ada/Hopper do.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        GLOBAL_BLOCKS = {7, 15, 23, 31}
        rope_depth = int(_os_m.environ.get("ROPE_DEPTH", "8"))

        import re
        blk_re = re.compile(r"blocks\.(\d+)")
        tensor_consumers = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                tensor_consumers.setdefault(inp.name, []).append(i)
        seeds = set()
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name or ""
            m = blk_re.search(name)
            if not m:
                continue
            if int(m.group(1)) in GLOBAL_BLOCKS:
                continue
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                if "freqs_" in (inp.name or "").lower():
                    seeds.add(i)
                    break
        visited = set()
        frontier = list(seeds)
        hop = {i: 0 for i in seeds}
        while frontier:
            cur = frontier.pop(0)
            if cur in visited: continue
            visited.add(cur)
            layer = network.get_layer(cur)
            if hop[cur] >= rope_depth: continue
            if layer.type == trt.LayerType.CONVOLUTION: continue
            for j in range(layer.num_outputs):
                for consumer_idx in tensor_consumers.get(layer.get_output(j).name, []):
                    if consumer_idx not in visited:
                        hop[consumer_idx] = hop[cur] + 1
                        frontier.append(consumer_idx)
        forced = 0
        for i in visited:
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.CONSTANT, trt.LayerType.SLICE,
                              trt.LayerType.GATHER, trt.LayerType.SHUFFLE,
                              trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY):
                continue
            if layer.num_outputs == 0: continue
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
            except Exception:
                pass
        print(f"  forced {forced} RoPE layers to FP32 (windowed + TF32 flag, depth {rope_depth})")
    elif PRECISION == "fp16_rope_inputs_only":
        # FP16 globally; pin to FP32 only the RoPE inputs/math — crucially
        # NOT the MatMul/Softmax that follow. After the rotate_half ops
        # produce q/k in FP32, we cast them back to FP16 for attention.
        # Idea: precise RoPE inputs + FP16 attention BMM/softmax.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        GLOBAL_BLOCKS = {7, 15, 23, 31}
        rope_depth = int(_os_m.environ.get("ROPE_DEPTH", "4"))

        import re
        blk_re = re.compile(r"blocks\.(\d+)")

        tensor_consumers = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                tensor_consumers.setdefault(inp.name, []).append(i)

        seeds = set()
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name or ""
            m = blk_re.search(name)
            if not m:
                continue
            blk_idx = int(m.group(1))
            if blk_idx in GLOBAL_BLOCKS:
                continue
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                if "freqs_" in (inp.name or "").lower():
                    seeds.add(i)
                    break

        # Very shallow BFS - 4 hops is enough to cover the rotate_half
        # (Slice x 2, Neg, Concat, Mul, Mul, Add).
        visited = set()
        frontier = list(seeds)
        hop = {i: 0 for i in seeds}
        # Stop at MatMul (so BMM stays in FP16)
        STOP_TYPES = {
            trt.LayerType.CONVOLUTION,
            trt.LayerType.MATRIX_MULTIPLY,
            trt.LayerType.SOFTMAX,
        }
        while frontier:
            cur = frontier.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            layer = network.get_layer(cur)
            if hop[cur] >= rope_depth:
                continue
            if layer.type in STOP_TYPES:
                continue
            for j in range(layer.num_outputs):
                for consumer_idx in tensor_consumers.get(layer.get_output(j).name, []):
                    if consumer_idx not in visited:
                        hop[consumer_idx] = hop[cur] + 1
                        frontier.append(consumer_idx)

        forced = 0
        for i in visited:
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.CONSTANT, trt.LayerType.SLICE,
                              trt.LayerType.GATHER, trt.LayerType.SHUFFLE,
                              trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY):
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
            except Exception:
                pass
        print(f"  forced {forced} RoPE-INPUT layers to FP32 "
              f"(windowed blocks, depth {rope_depth}, stops at MatMul/Softmax)")
    elif PRECISION == "fp16_rope_custom":
        # FP16 globally; pin RoPE to FP32 only in windowed blocks within a
        # range [WIN_FROM, WIN_TO). Global blocks (7/15/23/31) always FP16.
        # This lets us tune which block range is FP32-pinned.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        GLOBAL_BLOCKS = {7, 15, 23, 31}
        rope_depth = int(_os_m.environ.get("ROPE_DEPTH", "8"))
        win_from = int(_os_m.environ.get("WIN_FROM", "0"))
        win_to = int(_os_m.environ.get("WIN_TO", "32"))

        import re
        blk_re = re.compile(r"blocks\.(\d+)")

        tensor_consumers = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                tensor_consumers.setdefault(inp.name, []).append(i)

        seeds = set()
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name or ""
            m = blk_re.search(name)
            if not m:
                continue
            blk_idx = int(m.group(1))
            if blk_idx in GLOBAL_BLOCKS:
                continue
            if not (win_from <= blk_idx < win_to):
                continue
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                if "freqs_" in (inp.name or "").lower():
                    seeds.add(i)
                    break

        visited = set()
        frontier = list(seeds)
        hop = {i: 0 for i in seeds}
        while frontier:
            cur = frontier.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            layer = network.get_layer(cur)
            if hop[cur] >= rope_depth:
                continue
            if layer.type == trt.LayerType.CONVOLUTION:
                continue
            for j in range(layer.num_outputs):
                for consumer_idx in tensor_consumers.get(layer.get_output(j).name, []):
                    if consumer_idx not in visited:
                        hop[consumer_idx] = hop[cur] + 1
                        frontier.append(consumer_idx)

        forced = 0
        for i in visited:
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.CONSTANT, trt.LayerType.SLICE,
                              trt.LayerType.GATHER, trt.LayerType.SHUFFLE,
                              trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY):
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
            except Exception:
                pass
        print(f"  forced {forced} RoPE layers to FP32 "
              f"(windowed blocks [{win_from},{win_to}), depth {rope_depth}, "
              f"{len(seeds)} seeds)")
    elif PRECISION == "fp16_rope_math_only":
        # FP16 globally; pin to FP32 ONLY the immediate RoPE math (Mul / Add
        # / Sub / Neg / Slice-Concat-rotate_half pattern) for windowed blocks.
        # Softmax and MatMuls stay in FP16 so attention kernels fuse fast.
        # This is a tighter version of `fp16_rope_windowed`.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        GLOBAL_BLOCKS = {7, 15, 23, 31}
        rope_depth = int(_os_m.environ.get("ROPE_DEPTH", "8"))

        import re
        blk_re = re.compile(r"blocks\.(\d+)")

        tensor_consumers = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                tensor_consumers.setdefault(inp.name, []).append(i)

        seeds = set()
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name or ""
            m = blk_re.search(name)
            if not m:
                continue
            blk_idx = int(m.group(1))
            if blk_idx in GLOBAL_BLOCKS:
                continue
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                if "freqs_" in (inp.name or "").lower():
                    seeds.add(i)
                    break

        visited = set()
        frontier = list(seeds)
        hop = {i: 0 for i in seeds}
        # Tighter stop set: also stop at MatMul (the QK^T and attn*V bmm's)
        # and at Softmax. This pins only the rotate-half / multiply math.
        STOP_TYPES = {
            trt.LayerType.CONVOLUTION,
            trt.LayerType.MATRIX_MULTIPLY,
            trt.LayerType.SOFTMAX,
            trt.LayerType.NORMALIZATION,
        }
        while frontier:
            cur = frontier.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            layer = network.get_layer(cur)
            if hop[cur] >= rope_depth:
                continue
            if layer.type in STOP_TYPES:
                continue
            for j in range(layer.num_outputs):
                for consumer_idx in tensor_consumers.get(layer.get_output(j).name, []):
                    if consumer_idx not in visited:
                        hop[consumer_idx] = hop[cur] + 1
                        frontier.append(consumer_idx)

        forced = 0
        for i in visited:
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.CONSTANT, trt.LayerType.SLICE,
                              trt.LayerType.GATHER, trt.LayerType.SHUFFLE,
                              trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY):
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
            except Exception:
                pass
        print(f"  forced {forced} RoPE-math layers to FP32 "
              f"(windowed blocks, depth {rope_depth}, {len(seeds)} seeds, "
              f"no MatMul/Softmax)")
    elif PRECISION == "fp16_rope_windowed":
        # FP16 globally; pin RoPE math to FP32 ONLY in windowed-attention
        # blocks (not blocks 7/15/23/31 which are global attention).
        # Global attention blocks get fused to `_gemm_mha_v2` in FP16 for
        # speed. The hypothesis: the numerical drift accumulates in the
        # many shorter-sequence windowed blocks and the 4 global blocks can
        # absorb FP16 losslessly because they only run 4 times.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        GLOBAL_BLOCKS = {7, 15, 23, 31}
        rope_depth = int(_os_m.environ.get("ROPE_DEPTH", "10"))

        import re
        blk_re = re.compile(r"blocks\.(\d+)")

        tensor_consumers = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                tensor_consumers.setdefault(inp.name, []).append(i)

        seeds = set()
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name or ""
            m = blk_re.search(name)
            if not m:
                continue
            blk_idx = int(m.group(1))
            if blk_idx in GLOBAL_BLOCKS:
                continue
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                if "freqs_" in (inp.name or "").lower():
                    seeds.add(i)
                    break

        visited = set()
        frontier = list(seeds)
        hop = {i: 0 for i in seeds}
        while frontier:
            cur = frontier.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            layer = network.get_layer(cur)
            if hop[cur] >= rope_depth:
                continue
            if layer.type == trt.LayerType.CONVOLUTION:
                continue
            for j in range(layer.num_outputs):
                for consumer_idx in tensor_consumers.get(layer.get_output(j).name, []):
                    if consumer_idx not in visited:
                        hop[consumer_idx] = hop[cur] + 1
                        frontier.append(consumer_idx)

        forced = 0
        for i in visited:
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.CONSTANT, trt.LayerType.SLICE,
                              trt.LayerType.GATHER, trt.LayerType.SHUFFLE,
                              trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY):
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
            except Exception:
                pass
        print(f"  forced {forced} RoPE-region layers to FP32 "
              f"(windowed blocks only, depth {rope_depth}, {len(seeds)} seeds)")
    elif PRECISION == "fp16_rope_late":
        # FP16 globally; pin RoPE math to FP32 only for blocks >= FP32_FROM.
        # Diagnostic showed TRT FP16 matches PT closely in blocks 0-10 and
        # diverges sharply from block 11 onward, so pinning only late blocks
        # should preserve correctness while freeing up speed in early blocks.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        fp32_from = int(_os_m.environ.get("FP32_FROM", "11"))
        rope_depth = int(_os_m.environ.get("ROPE_DEPTH", "10"))

        import re
        blk_re = re.compile(r"blocks\.(\d+)")

        # Build producer/consumer maps
        tensor_consumers = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                tensor_consumers.setdefault(inp.name, []).append(i)

        # Seeds: layers that consume a freqs_cos/freqs_sin tensor AND are in
        # a late block.
        seeds = set()
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            name = layer.name or ""
            m = blk_re.search(name)
            if not m:
                continue
            blk_idx = int(m.group(1))
            if blk_idx < fp32_from:
                continue
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                if "freqs_" in (inp.name or "").lower():
                    seeds.add(i)
                    break

        # BFS forward from seeds up to ROPE_DEPTH hops, stopping at conv.
        visited = set()
        frontier = list(seeds)
        hop = {i: 0 for i in seeds}
        while frontier:
            cur = frontier.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            layer = network.get_layer(cur)
            if hop[cur] >= rope_depth:
                continue
            if layer.type == trt.LayerType.CONVOLUTION:
                continue
            for j in range(layer.num_outputs):
                for consumer_idx in tensor_consumers.get(layer.get_output(j).name, []):
                    if consumer_idx not in visited:
                        hop[consumer_idx] = hop[cur] + 1
                        frontier.append(consumer_idx)

        forced = 0
        for i in visited:
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.CONSTANT, trt.LayerType.SLICE,
                              trt.LayerType.GATHER, trt.LayerType.SHUFFLE,
                              trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY):
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
            except Exception:
                pass
        print(f"  forced {forced} RoPE-region layers to FP32 "
              f"(blocks >= {fp32_from}, depth {rope_depth}, {len(seeds)} seeds)")
    elif PRECISION == "fp16_softmax_fp32":
        # FP16 globally, but Softmax in FP32.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type != trt.LayerType.SOFTMAX:
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
            except Exception:
                pass
        print(f"  forced {forced} Softmax layers to FP32")
    elif PRECISION == "fp16_half_fp32":
        # FP16 globally, but every layer whose name contains blocks.0..15 in FP32
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        import re
        pat = re.compile(r"blocks\.(\d+)")
        forced = 0
        first_half = _os_m.environ.get("FP32_HALF", "early")  # early | late
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.CONSTANT, trt.LayerType.SLICE,
                              trt.LayerType.GATHER, trt.LayerType.SHUFFLE,
                              trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY):
                continue
            m = pat.search(layer.name or "")
            if m is None:
                continue
            n = int(m.group(1))
            fp32_lo = int(_os_m.environ.get("FP32_LO", "0"))
            fp32_hi = int(_os_m.environ.get("FP32_HI", "16"))
            in_range = fp32_lo <= n < fp32_hi
            if (first_half == "early" and n < 16) or (first_half == "late" and n >= 16) or (first_half == "range" and in_range):
                try:
                    layer.precision = trt.DataType.FLOAT
                    for j in range(layer.num_outputs):
                        out = layer.get_output(j)
                        if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                            layer.set_output_type(j, trt.DataType.FLOAT)
                    forced += 1
                except Exception:
                    pass
        print(f"  forced {forced} layers ({first_half}-half blocks) to FP32")
    elif PRECISION == "fp16_norm_fp32":
        # FP16 globally, but every Normalization/Softmax/Reduce in FP32.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        target = {trt.LayerType.SOFTMAX, trt.LayerType.NORMALIZATION,
                  trt.LayerType.REDUCE}
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type not in target:
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
            except Exception:
                pass
        print(f"  forced {forced} norm/softmax/reduce layers to FP32")
    elif PRECISION == "fp16_rope_fp32":
        # FP16 globally, pin ONLY the RoPE-math layers to FP32. Identifies
        # layers by walking the consumers of every freqs_cos / freqs_sin
        # buffer down to the first Reshape/Flatten (the rotation region).
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        # Build producer->consumers map across the network
        tensor_producer = {}  # name -> layer_index
        tensor_consumers = {}  # name -> list[layer_index]
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_outputs):
                tensor_producer[layer.get_output(j).name] = i
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                tensor_consumers.setdefault(inp.name, []).append(i)

        # Seeds: any layer that has a freqs_cos/freqs_sin tensor as an input
        seeds = set()
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_inputs):
                inp = layer.get_input(j)
                if inp is None:
                    continue
                if "freqs_" in (inp.name or "").lower():
                    seeds.add(i)
                    break

        # BFS forward: collect all layers downstream up to depth D that are
        # NOT of type MatMul/Convolution (stop propagation at projection heads)
        visited = set()
        frontier = list(seeds)
        import os as _os
        DEPTH = int(_os.environ.get("ROPE_DEPTH", "10"))
        hop = {i: 0 for i in seeds}
        while frontier:
            cur = frontier.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            layer = network.get_layer(cur)
            d = hop[cur]
            if d >= DEPTH:
                continue
            # Stop only at Convolution (projection to FPN); allow MatMul
            # (attention scores & attn_out projection) since the scaling
            # after RoPE must also be in FP32.
            if layer.type in (trt.LayerType.CONVOLUTION,):
                continue
            for j in range(layer.num_outputs):
                for consumer_idx in tensor_consumers.get(layer.get_output(j).name, []):
                    if consumer_idx not in visited:
                        hop[consumer_idx] = d + 1
                        frontier.append(consumer_idx)

        forced = 0
        for i in visited:
            layer = network.get_layer(i)
            # Skip layers that compute indices or metadata
            if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CAST,
                              trt.LayerType.CONSTANT, trt.LayerType.SLICE,
                              trt.LayerType.GATHER, trt.LayerType.SHUFFLE,
                              trt.LayerType.CONCATENATION, trt.LayerType.IDENTITY):
                continue
            # Skip layers whose output isn't FP
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
            except Exception:
                pass
        print(f"  forced {forced} RoPE-region layers to FP32 (from {len(seeds)} seeds)")
    elif PRECISION == "fp16_precise":
        # FP16 globally, but force Cast and LayerNorm to FP32 — these are
        # the RoPE-precision-sensitive ops. OBEY_PRECISION_CONSTRAINTS forces
        # TRT to honor the per-layer overrides.
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.NORMALIZATION, trt.LayerType.CAST,
                              trt.LayerType.REDUCE, trt.LayerType.SOFTMAX):
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out.dtype in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                forced += 1
        print(f"  forced {forced} layers to FP32")
    elif PRECISION == "bf16_strict":
        # BF16 + OBEY_PRECISION_CONSTRAINTS: force TRT to actually run BF16
        # kernels instead of falling back to FP32 for "better accuracy".
        config.set_flag(trt.BuilderFlag.BF16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        SKIP_TYPES = {
            trt.LayerType.SOFTMAX, trt.LayerType.REDUCE, trt.LayerType.CAST,
            trt.LayerType.SHAPE, trt.LayerType.CONSTANT, trt.LayerType.IDENTITY,
            trt.LayerType.GATHER, trt.LayerType.SLICE, trt.LayerType.CONCATENATION,
            trt.LayerType.SHUFFLE,
        }
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in SKIP_TYPES:
                continue
            # Don't touch int-dtyped layers
            out0 = layer.get_output(0) if layer.num_outputs > 0 else None
            if out0 is None or out0.dtype not in (trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16):
                continue
            layer.precision = trt.DataType.BF16
            for j in range(layer.num_outputs):
                out = layer.get_output(j)
                if out.dtype == trt.DataType.FLOAT:
                    layer.set_output_type(j, trt.DataType.BF16)
    elif PRECISION == "fp16_bf16":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.BF16)
    elif PRECISION == "tf32":
        config.set_flag(trt.BuilderFlag.TF32)
    elif PRECISION == "best":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.BF16)
        config.set_flag(trt.BuilderFlag.TF32)

    print(
        "Network input(s):",
        [(network.get_input(i).name, network.get_input(i).shape)
         for i in range(network.num_inputs)],
    )
    print(
        "Network output(s):",
        [(network.get_output(i).name, network.get_output(i).shape)
         for i in range(network.num_outputs)],
    )

    print("\nBuilding engine (this takes several minutes) ...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    dt = time.perf_counter() - t0
    if serialized is None:
        print("ERROR: build_serialized_network returned None")
        return 3
    blob = bytes(serialized)
    print(f"  built in {dt:.1f}s  ({len(blob) / 1e6:.1f} MB)")

    with open(ENGINE_PATH, "wb") as f:
        f.write(blob)
    print(f"\nEngine written to {ENGINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
