#!/usr/bin/env python3
"""
Build TRT engines for YOLO26 models from ONNX files.
This creates compatible engines for the current GPU (L4, compute 8.9).
"""

import os
import sys
import shutil
import json
from pathlib import Path

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    fp16_mode: bool = True,
    max_batch_size: int = 8,
) -> bool:
    """Build a TRT engine from ONNX model."""
    print(f"\nBuilding TRT engine:")
    print(f"  Input:  {onnx_path}")
    print(f"  Output: {engine_path}")
    print(f"  FP16:   {fp16_mode}")
    print(f"  Max batch: {max_batch_size}")

    # Create builder and network
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    print("  Parsing ONNX...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print(f"  ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(f"    {parser.get_error(error)}")
            return False

    # Configure builder
    config = builder.create_builder_config()

    # Set memory pool limit (8GB for L4)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode enabled")

    # Build engine
    print("  Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("  ERROR: Failed to build engine")
        return False

    # Save engine
    print(f"  Saving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print("  Engine built successfully!")
    return True


def create_model_package(
    onnx_base_dir: str,
    output_dir: str,
    model_type: str,
) -> bool:
    """Create a complete model package with TRT engine."""
    print(f"\n{'='*60}")
    print(f"Creating {model_type} model package")
    print(f"{'='*60}")

    # Source files
    onnx_path = Path(onnx_base_dir) / "model.onnx"
    class_names_path = Path(onnx_base_dir) / "class_names.txt"
    env_path = Path(onnx_base_dir) / "environment.json"

    # Validate source files exist
    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}")
        return False
    if not class_names_path.exists():
        print(f"ERROR: class_names.txt not found: {class_names_path}")
        return False
    if not env_path.exists():
        print(f"ERROR: environment.json not found: {env_path}")
        return False

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build TRT engine
    engine_path = output_path / "engine.plan"
    if not build_trt_engine(str(onnx_path), str(engine_path), fp16_mode=True):
        return False

    # Copy supporting files
    print("\n  Copying supporting files...")
    shutil.copy(class_names_path, output_path / "class_names.txt")

    # Load and update environment.json
    with open(env_path, 'r') as f:
        env_data = json.load(f)

    # Create inference_config.json from environment.json
    inference_config = {
        "image_pre_processing": {
            "resize_mode": env_data.get("PREPROCESSING_STRATEGY", "letterbox"),
            "input_color_format": "bgr",
        },
        "network_input": {
            "width": env_data.get("IMG_SIZE_WIDTH", 640),
            "height": env_data.get("IMG_SIZE_HEIGHT", 640),
        }
    }

    with open(output_path / "inference_config.json", 'w') as f:
        json.dump(inference_config, f, indent=2)

    # Create trt_config.json
    trt_config = {
        "fp16": True,
        "max_batch_size": 8,
    }

    with open(output_path / "trt_config.json", 'w') as f:
        json.dump(trt_config, f, indent=2)

    print(f"\n  Model package created at: {output_path}")
    print(f"  Files:")
    for file in output_path.iterdir():
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"    - {file.name} ({size_mb:.1f} MB)")

    return True


def main():
    print("YOLO26 TRT Engine Builder")
    print("="*60)
    print("Building TRT engines for current GPU (L4, compute 8.9)")
    print()

    # Model configurations
    models = [
        {
            "name": "yolo26-det",
            "source_dir": "/home/ubuntu/.cache/roboflow/tests/yolo26_det/1",
            "output_dir": "/home/ubuntu/.cache/roboflow/yolo26_trt_engines/yolo26-det",
        },
        {
            "name": "yolo26-seg",
            "source_dir": "/home/ubuntu/.cache/roboflow/tests/yolo26_seg/1",
            "output_dir": "/home/ubuntu/.cache/roboflow/yolo26_trt_engines/yolo26-seg",
        },
        {
            "name": "yolo26-pose",
            "source_dir": "/home/ubuntu/.cache/roboflow/tests/yolo26_pose/1",
            "output_dir": "/home/ubuntu/.cache/roboflow/yolo26_trt_engines/yolo26-pose",
        },
    ]

    success_count = 0
    for model_config in models:
        if create_model_package(
            onnx_base_dir=model_config["source_dir"],
            output_dir=model_config["output_dir"],
            model_type=model_config["name"],
        ):
            success_count += 1
        else:
            print(f"\nFAILED to build {model_config['name']}")

    print(f"\n{'='*60}")
    print(f"Build Summary: {success_count}/{len(models)} models built successfully")
    print(f"{'='*60}")

    if success_count == len(models):
        print("\nAll engines built successfully!")
        print("You can now benchmark with:")
        for model_config in models:
            print(f"  {model_config['output_dir']}")
        return 0
    else:
        print("\nSome builds failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
