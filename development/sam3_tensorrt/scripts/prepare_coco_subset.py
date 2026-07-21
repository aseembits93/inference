#!/usr/bin/env python3
"""Download COCO val2017 annotations and 100 images spanning diverse classes.

Strategy: pick one prompt class per image (the dominant instance class
in that image, by mask area) so we have a natural ground-truth prompt.
Sample images so the 100 cover as many of the 80 COCO classes as possible.
Output: a manifest JSON with image filename + prompt text + local path.
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(os.environ.get("COCO_SUBSET", "/tmp/coco_val2017_subset"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL_TMPL = "http://images.cocodataset.org/val2017/{fname}"
ANN_CACHE = Path("/tmp/coco_annotations")
ANN_CACHE.mkdir(parents=True, exist_ok=True)
ANN_FILE = ANN_CACHE / "annotations" / "instances_val2017.json"

N_IMAGES = 100
SEED = 17


def fetch_annotations():
    if ANN_FILE.exists():
        print(f"[ann] already cached at {ANN_FILE}")
        return
    print(f"[ann] downloading {ANN_URL} ...")
    data = urllib.request.urlopen(ANN_URL).read()
    print(f"[ann] extracting to {ANN_CACHE} ({len(data)/1e6:.0f} MB) ...")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(ANN_CACHE)
    assert ANN_FILE.exists(), ANN_FILE


def main() -> int:
    random.seed(SEED)
    fetch_annotations()

    print(f"[ann] loading {ANN_FILE} ...")
    with open(ANN_FILE) as f:
        ann = json.load(f)

    # image_id -> record, cat_id -> name
    images_by_id = {im["id"]: im for im in ann["images"]}
    cat_name = {c["id"]: c["name"] for c in ann["categories"]}

    # For each image, find the dominant category by total mask/bbox area
    # (using bbox area as a proxy; no segmentation loading needed).
    area_per_image_cat = defaultdict(lambda: defaultdict(float))
    for a in ann["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        area_per_image_cat[a["image_id"]][a["category_id"]] += w * h

    # Dominant category per image
    image_prompt = {}
    for img_id, by_cat in area_per_image_cat.items():
        if not by_cat:
            continue
        dom_cat_id = max(by_cat, key=by_cat.get)
        image_prompt[img_id] = {
            "category_id": dom_cat_id,
            "prompt": cat_name[dom_cat_id],
            "area_share": by_cat[dom_cat_id] / sum(by_cat.values()),
        }

    # Group image_ids by their prompt class
    by_class = defaultdict(list)
    for img_id, info in image_prompt.items():
        by_class[info["prompt"]].append(img_id)

    print(f"[sample] {len(by_class)} distinct prompt classes available")

    # Round-robin sample: take up to ceil(N/80) from each class, shuffled
    per_class_cap = max(1, (N_IMAGES + len(by_class) - 1) // len(by_class))
    picked = []
    class_list = list(by_class.keys())
    random.shuffle(class_list)
    for cls in class_list:
        pool = by_class[cls][:]
        random.shuffle(pool)
        # Prefer images where the prompt class is clearly dominant (>40% area)
        # so the model has an unambiguous target.
        pool_sorted = sorted(
            pool,
            key=lambda iid: -image_prompt[iid]["area_share"],
        )
        picked.extend(pool_sorted[:per_class_cap])
        if len(picked) >= N_IMAGES:
            break
    picked = picked[:N_IMAGES]
    print(f"[sample] picked {len(picked)} images across {len({image_prompt[i]['prompt'] for i in picked})} classes")

    # Download images
    manifest = []
    for i, img_id in enumerate(picked):
        rec = images_by_id[img_id]
        fname = rec["file_name"]
        out_path = OUT_DIR / fname
        if not out_path.exists():
            try:
                url = IMG_URL_TMPL.format(fname=fname)
                data = urllib.request.urlopen(url, timeout=30).read()
                out_path.write_bytes(data)
            except Exception as e:
                print(f"[dl {i}/{len(picked)}] FAIL {fname}: {e}")
                continue
        info = image_prompt[img_id]
        manifest.append({
            "image_id": img_id,
            "file_name": fname,
            "local_path": str(out_path),
            "width": rec["width"],
            "height": rec["height"],
            "prompt": info["prompt"],
            "category_id": info["category_id"],
            "dominant_area_share": info["area_share"],
        })
        if (i + 1) % 10 == 0:
            print(f"[dl] {i+1}/{len(picked)} complete")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] manifest: {manifest_path} ({len(manifest)} images)")

    # Class distribution
    from collections import Counter
    c = Counter(m["prompt"] for m in manifest)
    print(f"\nClass distribution ({len(c)} classes):")
    for cls, n in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {cls}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
