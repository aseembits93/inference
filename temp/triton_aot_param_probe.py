import triton
import triton.language as tl


@triton.jit
def add_kernel_aot(
    x_ptr,
    n_elements,
    y_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + 1, mask=mask)


def main() -> None:
    src = triton.compiler.ASTSource(
        fn=add_kernel_aot,
        signature={
            "x_ptr": "*fp16",
            "n_elements": "i32",
            "y_ptr": "*fp16",
        },
        constexprs={"BLOCK_SIZE": 64},
    )
    compiled = triton.compile(src)
    print(compiled.metadata)
    for line in compiled.asm["ptx"].splitlines()[:40]:
        print(line)


if __name__ == "__main__":
    main()
