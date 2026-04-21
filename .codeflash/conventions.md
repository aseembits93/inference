# Codeflash Session Conventions

## Session Configuration (Autonomous Mode)
- **Run tag:** 2026-04-21
- **Target:** Heaviest model's `.infer()` method (to be identified from profiling)
- **Focus:** TensorRT GPU performance optimization
- **Scope:** End-to-end optimization of the inference pipeline

## Guard Command
Not specified. Tests will be run after each optimization to verify correctness.

## Autonomous Decisions
- Domain: GPU optimization (user explicitly mentioned "trt gpu performance")
- Target identification: Will profile available models to identify the "heaviest" one
- Benchmark tier: Will use available benchmarks in `tests/benchmarks/core/`

## Project Context
- Python 3.12.3
- PyTorch 2.10.0+cu128
- TensorRT 10.12.0.36
- Test framework: pytest with codeflash-benchmark plugin
