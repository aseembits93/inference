#!/usr/bin/env python3
"""Check the dtype distribution of tensors in the parsed ONNX network."""

import sys
from collections import Counter
import tensorrt as trt


def main(onnx_path: str) -> int:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return 1

    tensor_dtypes = Counter()
    for i in range(net.num_layers):
        layer = net.get_layer(i)
        for j in range(layer.num_outputs):
            out = layer.get_output(j)
            if out is not None:
                tensor_dtypes[str(out.dtype)] += 1

    print(f"Network: {onnx_path}")
    print(f"Layers: {net.num_layers}")
    print(f"Output tensor dtypes:")
    for k, v in tensor_dtypes.most_common():
        print(f"  {k}: {v}")

    # Also check input dtypes
    print("\nInputs:")
    for i in range(net.num_inputs):
        t = net.get_input(i)
        print(f"  {t.name}: {t.dtype} {t.shape}")

    print("\nOutputs:")
    for i in range(net.num_outputs):
        t = net.get_output(i)
        print(f"  {t.name}: {t.dtype} {t.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
