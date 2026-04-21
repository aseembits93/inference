# Setup Report

## Package Manager
**Detected:** uv (at `/home/ubuntu/.local/bin/uv`)

## Python Environment
- **Version:** Python 3.12.3
- **Virtual environment:** `/home/ubuntu/inference/.venv/`
- **Status:** Active and functional

## Dependencies
Project uses `pyproject.toml` for dependency management.

### Key Performance Libraries Installed
- **torch:** 2.10.0+cu128 (PyTorch with CUDA 12.8)
- **tensorrt:** 10.12.0.36 (NVIDIA TensorRT)
- **pytest:** 8.4.2
- **pytest-codeflash-benchmark:** 0.3.0 (benchmark plugin)

## Test Framework
- **Framework:** pytest
- **Test root:** `tests/inference/unit_tests`
- **Benchmarks root:** `tests/benchmarks`
- **Test command:** `/home/ubuntu/inference/.venv/bin/pytest`

### Available Benchmarks
Found 3 benchmark tests in `tests/benchmarks/core/`:
- `test_benchmark_equivalent_rfdetr`
- `test_benchmark_equivalent_yolov8n_seg`
- `test_benchmark_equivalent_yolov8n`

## GPU/Profiling Tools
- **CUDA:** 12.8 (via torch+cu128)
- **TensorRT:** 10.12.0.36 (installed)
- **torch.profiler:** Available (built into PyTorch 2.10)
- **nsys:** Not checked (NVIDIA Nsight Systems - to be verified if needed)

## Project Structure
- **Module root:** `inference/`
- **Ignore paths:** `inference/models/yolo26`
- **Formatters:** black

## Status
✓ Environment is ready for GPU optimization session
✓ All core dependencies installed
✓ PyTorch with CUDA 12.8 support
✓ TensorRT 10.12 available
✓ Benchmark tests discovered
