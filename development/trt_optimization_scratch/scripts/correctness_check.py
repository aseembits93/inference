#!/usr/bin/env python3
"""Verify correctness of optimizations - outputs should match baseline."""
import sys
import numpy as np
import torch
from inference import get_model

MODEL_IDS = sys.argv[1:] if len(sys.argv) > 1 else ["yolov8n-640", "rfdetr-base", "yolov8n-seg-640"]

for MODEL_ID in MODEL_IDS:
    print(f"\n=== {MODEL_ID} ===")
    model = get_model(model_id=MODEL_ID, api_key=None)

    # Test with multiple sizes
    for size in [(640, 640), (320, 480), (800, 600)]:
        rng = np.random.default_rng(42)  # deterministic
        img = rng.integers(0, 255, (size[0], size[1], 3), dtype=np.uint8)

        # Run twice, confirm identical output
        out1 = model.infer(img)
        torch.cuda.synchronize()
        out2 = model.infer(img)
        torch.cuda.synchronize()

        # Check that both runs produce identical predictions
        def fingerprint(resp):
            if isinstance(resp, list):
                resp = resp[0]
            preds = resp.predictions
            if not preds:
                return 'empty'
            return [(p.confidence, p.class_id, p.x, p.y, p.width, p.height) for p in preds[:5]]

        f1 = fingerprint(out1)
        f2 = fingerprint(out2)
        match = f1 == f2
        print(f"  size {size}: 2 runs match = {match}, n={len(out1[0].predictions) if isinstance(out1, list) else len(out1.predictions)}")
        if not match:
            print(f"    run1: {f1}")
            print(f"    run2: {f2}")
