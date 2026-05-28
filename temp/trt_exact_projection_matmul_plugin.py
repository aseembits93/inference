from typing import Tuple

import torch
import tensorrt.plugin as trtp
import triton
import triton.language as tl


PLUGIN_ID = "rfprobe::exact_projection_matmul"
TRITON_PLUGIN_ID = "rfprobe::triton_projection_matmul"
AOT_TRITON_PLUGIN_ID = "rfprobe::aot_triton_projection_matmul"
AOT_TRITON_ADD_PLUGIN_ID = "rfprobe::aot_triton_add"
_LOG_RUNTIME_ONCE = {"done": False}


@triton.jit
def _projection_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_size,
    n_size,
    k_size,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k_size, BLOCK_K)):
        a_mask = (offs_m[:, None] < m_size) & (offs_k[None, :] < k_size)
        b_mask = (offs_k[:, None] < k_size) & (offs_n[None, :] < n_size)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _projection_matmul_aot_kernel(
    a_ptr,
    b_ptr,
    m_size,
    n_size,
    k_size,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(n_size, BLOCK_N)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k_size, BLOCK_K)):
        a_mask = (offs_m[:, None] < m_size) & (offs_k[None, :] < k_size)
        b_mask = (offs_k[:, None] < k_size) & (offs_n[None, :] < n_size)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _add_aot_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@trtp.register(PLUGIN_ID)
def exact_projection_matmul_desc(
    inp0: trtp.TensorDesc, inp1: trtp.TensorDesc
) -> trtp.TensorDesc:
    # This plugin is only meant to replace projection-style matmuls where the
    # output shape matches the activations tensor shape.
    del inp1
    return inp0.like()


@trtp.impl(PLUGIN_ID)
def exact_projection_matmul_impl(
    inp0: trtp.Tensor,
    inp1: trtp.Tensor,
    outputs: Tuple[trtp.Tensor],
    stream: int,
) -> None:
    if not _LOG_RUNTIME_ONCE["done"]:
        print(
            "[plugin-runtime] "
            f"inp0.shape={tuple(inp0.shape)} inp0.strides={tuple(inp0.strides)} "
            f"inp1.shape={tuple(inp1.shape)} inp1.strides={tuple(inp1.strides)} "
            f"out.shape={tuple(outputs[0].shape)} out.strides={tuple(outputs[0].strides)}",
            flush=True,
        )
        _LOG_RUNTIME_ONCE["done"] = True
    lhs = torch.as_tensor(inp0, device="cuda")
    rhs = torch.as_tensor(inp1, device="cuda")
    out = torch.as_tensor(outputs[0], device="cuda")
    ext_stream = torch.cuda.ExternalStream(stream)
    with torch.cuda.stream(ext_stream):
        torch.matmul(lhs, rhs, out=out)


@trtp.register(TRITON_PLUGIN_ID)
def triton_projection_matmul_desc(
    inp0: trtp.TensorDesc, inp1: trtp.TensorDesc
) -> trtp.TensorDesc:
    del inp1
    return inp0.like()


@trtp.impl(TRITON_PLUGIN_ID)
def triton_projection_matmul_impl(
    inp0: trtp.Tensor,
    inp1: trtp.Tensor,
    outputs: Tuple[trtp.Tensor],
    stream: int,
) -> None:
    lhs = torch.as_tensor(inp0, device="cuda").reshape(-1, inp0.shape[-1])
    rhs = torch.as_tensor(inp1, device="cuda").reshape(inp1.shape[-2], inp1.shape[-1])
    out = torch.as_tensor(outputs[0], device="cuda").reshape(-1, outputs[0].shape[-1])

    m_size, k_size = lhs.shape
    _, n_size = rhs.shape
    ext_stream = torch.cuda.ExternalStream(stream)
    grid = (triton.cdiv(m_size, 64), triton.cdiv(n_size, 64))
    with torch.cuda.stream(ext_stream):
        _projection_matmul_kernel[grid](
            lhs,
            rhs,
            out,
            m_size,
            n_size,
            k_size,
            lhs.stride(0),
            lhs.stride(1),
            rhs.stride(0),
            rhs.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )


@trtp.register(AOT_TRITON_PLUGIN_ID)
def aot_triton_projection_matmul_desc(
    inp0: trtp.TensorDesc, inp1: trtp.TensorDesc
) -> trtp.TensorDesc:
    del inp1
    return inp0.like()


@trtp.aot_impl(AOT_TRITON_PLUGIN_ID)
def aot_triton_projection_matmul_impl(
    inp0: trtp.TensorDesc,
    inp1: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[str | bytes, str | bytes, trtp.KernelLaunchParams, trtp.SymIntExprs]:
    del outputs, tactic
    block_m = 64
    block_n = 64
    block_k = 32

    source = triton.compiler.ASTSource(
        fn=_projection_matmul_aot_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "m_size": "i32",
            "n_size": "i32",
            "k_size": "i32",
            "stride_am": "i32",
            "stride_ak": "i32",
            "stride_bk": "i32",
            "stride_bn": "i32",
            "stride_cm": "i32",
            "stride_cn": "i32",
            "c_ptr": "*fp16",
        },
        constexprs={
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
        },
    )
    compiled = triton.compile(source)

    m_size = inp0.shape_expr[0] * inp0.shape_expr[1]
    k_size = inp0.shape_expr[2]
    n_size = inp1.shape_expr[1]

    launch = trtp.KernelLaunchParams()
    launch.grid_x = trtp.cdiv(m_size, block_m) * trtp.cdiv(n_size, block_n)
    launch.block_x = compiled.metadata.num_warps * 32
    launch.shared_mem = compiled.metadata.shared

    extra_args = trtp.SymIntExprs(9)
    extra_args[0] = trtp.SymInt32(m_size)
    extra_args[1] = trtp.SymInt32(n_size)
    extra_args[2] = trtp.SymInt32(k_size)
    extra_args[3] = trtp.SymInt32(k_size)
    extra_args[4] = trtp.SymInt32(1)
    extra_args[5] = trtp.SymInt32(n_size)
    extra_args[6] = trtp.SymInt32(1)
    extra_args[7] = trtp.SymInt32(n_size)
    extra_args[8] = trtp.SymInt32(1)

    return compiled.metadata.name, compiled.asm["ptx"], launch, extra_args


@trtp.register(AOT_TRITON_ADD_PLUGIN_ID)
def aot_triton_add_desc(
    inp0: trtp.TensorDesc, inp1: trtp.TensorDesc
) -> trtp.TensorDesc:
    del inp1
    return inp0.like()


@trtp.aot_impl(AOT_TRITON_ADD_PLUGIN_ID)
def aot_triton_add_impl(
    inp0: trtp.TensorDesc,
    inp1: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[str | bytes, str | bytes, trtp.KernelLaunchParams, trtp.SymIntExprs]:
    del inp1, outputs, tactic
    block_size = 256

    source = triton.compiler.ASTSource(
        fn=_add_aot_kernel,
        signature={
            "x_ptr": "*fp16",
            "y_ptr": "*fp16",
            "n_elements": "i32",
            "out_ptr": "*fp16",
        },
        constexprs={"BLOCK_SIZE": block_size},
    )
    compiled = triton.compile(source)

    n_elements = inp0.shape_expr.numel()
    launch = trtp.KernelLaunchParams()
    launch.grid_x = trtp.cdiv(n_elements, block_size)
    launch.block_x = compiled.metadata.num_warps * 32
    launch.shared_mem = compiled.metadata.shared

    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(n_elements)

    return compiled.metadata.name, compiled.asm["ptx"], launch, extra_args
