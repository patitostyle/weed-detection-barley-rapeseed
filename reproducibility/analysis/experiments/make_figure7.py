"""Reproduce qualitative Barley 2 examples with the retained revision checkpoint.

Keep the two previously illustrated patch identities. Predict every retained
patch that overlaps either display window, preserving the original evaluation
batch membership. Average probabilities geographically, exactly as in the
numerical protocol, before deciding a class. No fitting or model selection.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

import run_revision as r

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
DATASET = "cebada_v2_C2_rgb_nir_re_indices"
EXAMPLES = ["patch_00020.tif", "patch_00490.tif"]
COLORS = ["#009E73", "#AA3377", "#B58C42", "#E4E7EB"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    provenance_path = HERE / "staged_v2/source_checkpoint_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    checkpoint = HERE / "experiments_v1/source_patch_valid_seed42/best.pth"
    if not checkpoint.exists():
        checkpoint = HERE / "experiments_v1/source_patch_valid_seed42/best.pth"
    assert sha256(checkpoint) == provenance["sha256"]
    complete = json.loads(checkpoint.with_name("complete.json").read_text())
    assert complete["seed"] == 42 and complete["selected_epoch"] == 12
    assert all("cebada_v2_C2" not in name for name in complete["train_ids"])
    tests = json.loads((HERE / "staged_v2/deeplab_benchmark/test_results.json").read_text())
    reference = next(x for x in tests if x["model"] == "UNet" and x["seed"] == 42)
    assert reference["checkpoint_sha256"] == provenance["sha256"]

    records = [dict(rec, dataset=DATASET) for rec in r.AUDIT[DATASET]["records"]]
    lookup = {rec["id"]: i for i, rec in enumerate(records)}
    gsd = records[0]["w"] / 256
    xmin = min(rec["x"] for rec in records)
    ymax = max(rec["y"] for rec in records)
    offsets = [(round((ymax - rec["y"]) / gsd),
                round((rec["x"] - xmin) / gsd)) for rec in records]

    def intersection(a, b):
        ay, ax = a
        by, bx = b
        y0, x0 = max(ay, by), max(ax, bx)
        y1, x1 = min(ay + 256, by + 256), min(ax + 256, bx + 256)
        if y0 >= y1 or x0 >= x1:
            return None
        return (slice(y0 - ay, y1 - ay), slice(x0 - ax, x1 - ax),
                slice(y0 - by, y1 - by), slice(x0 - bx, x1 - bx))

    neighbours = {name: [i for i, offset in enumerate(offsets)
                         if intersection(offsets[lookup[name]], offset) is not None]
                  for name in EXAMPLES}
    needed = sorted(set(i for indices in neighbours.values() for i in indices))
    # Preserve batches used by the original unique-pixel evaluation, including
    # otherwise unnecessary members of a selected batch, for numerical parity.
    batch_numbers = sorted({i // 4 for i in needed})
    inference_indices = sorted({i for b in batch_numbers
                                for i in range(4 * b, min(4 * b + 4, len(records)))})
    stats = json.loads((HERE / "staged_v2/source_scaling.json").read_text())["bounds"]
    data = r.Data([records[i] for i in inference_indices], "patch_valid", stats,
                  augment=False)
    index_to_local = {index: local for local, index in enumerate(inference_indices)}
    local_by_name = {name: j for j, name in enumerate(data.names)}

    r.legacy.seed_all(42)
    model = r.UNet(8, 3)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(r.DEVICE).eval()
    probs = {}
    with torch.inference_mode():
        for b in batch_numbers:
            ids = list(range(4 * b, min(4 * b + 4, len(records))))
            images = torch.stack([data.images[index_to_local[i]] for i in ids])
            with torch.autocast(device_type=r.DEVICE, enabled=r.DEVICE == "cuda"):
                result = model(images.to(r.DEVICE)).float().softmax(1).cpu().numpy()
            for index, probability in zip(ids, result):
                probs[index] = probability
    del model
    if r.DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Independently compare displayed labelled decisions with the evaluator's
    # own unique-pixel mappings. All contributors to each display are included.
    evaluator = r.metric(data)
    assert len(evaluator.parts) == 1
    _, unique = evaluator.parts[0]
    totals = np.zeros((3, unique.n), np.float64)
    for local, index in enumerate(inference_indices):
        rows, cols = unique.valid[local]
        for cls in range(3):
            totals[cls] += np.bincount(unique.maps[local],
                                      weights=probs[index][cls, rows, cols],
                                      minlength=unique.n)
    unique_prediction = (totals / unique.counts).argmax(0)

    arrays = {}
    descriptions = []
    for row, name in enumerate(EXAMPLES):
        index = lookup[name]
        total = np.zeros((3, 256, 256), np.float64)
        count = np.zeros((256, 256), np.int32)
        for source in neighbours[name]:
            dy, dx, sy, sx = intersection(offsets[index], offsets[source])
            total[:, dy, dx] += probs[source][:, sy, sx]
            count[dy, dx] += 1
        assert count.min() >= 1
        averaged = total / count[None]
        predicted = averaged.argmax(0).astype(np.uint8)
        local = local_by_name[DATASET + "/" + name]
        target = data.masks[local].numpy()
        valid = target != 255
        rr, cc = unique.valid[local]
        expected = unique_prediction[unique.maps[local]]
        assert np.array_equal(predicted[rr, cc], expected)
        assert np.array_equal(target[rr, cc], unique.labels[unique.maps[local]])
        _, input_validity, _ = r.raw(DATASET, name)
        display_valid = input_validity.all(0)
        assert display_valid[valid].all()

        image_path = r.ROOT / DATASET / "images" / name
        mask_path = r.ROOT / DATASET / "masks" / name
        with tifffile.TiffFile(image_path) as tif:
            scale_tag = tif.pages[0].tags.get("ModelPixelScaleTag")
            assert scale_tag is not None
            assert np.isclose(float(scale_tag.value[0]), gsd)

        cm = np.bincount(target[valid] * 3 + predicted[valid], minlength=9).reshape(3, 3)
        display_reference = np.where(valid, target, 3).astype(np.uint8)
        full_display = np.where(display_valid, predicted, 3).astype(np.uint8)
        masked_prediction = np.where(valid, predicted, 3).astype(np.uint8)
        arrays.update({f"rgb{row}": data.images[local][:3].numpy().transpose(1, 2, 0),
                       f"reference{row}": display_reference,
                       f"prediction{row}": full_display,
                       f"labelled_prediction{row}": masked_prediction,
                       f"probabilities{row}": averaged.astype(np.float32),
                       f"overlap_count{row}": count})
        descriptions.append({
            "patch": name,
            "image_sha256": sha256(image_path),
            "mask_sha256": sha256(mask_path),
            "contributing_retained_patches": [records[i]["id"] for i in neighbours[name]],
            "overlap_count_range": [int(count.min()), int(count.max())],
            "reference_counts_crop_weed_soil": cm.sum(1).tolist(),
            "confusion_at_labelled_pixels": cm.tolist(),
            "predicted_counts_all_valid_input_pixels":
                np.bincount(predicted[display_valid], minlength=3).tolist(),
            "unlabelled_pixels": int((~valid).sum()),
            "invalid_input_pixels": int((~display_valid).sum()),
            "unique_evaluator_decisions_match": True,
            "unique_evaluator_labels_match": True,
        })

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "Figure7_arrays.npz", names=np.array(EXAMPLES), **arrays)
    output = {
        "dataset": DATASET,
        "selection": "same two patch identities as the previous figure; no selection using revised predictions",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": provenance["sha256"],
        "checkpoint_seed": 42,
        "checkpoint_selected_epoch": 12,
        "checkpoint_selection_data": "spatially separated Barley 1 validation only",
        "class_order": ["crop", "weed", "soil"],
        "preprocessing": "Data(norm='patch_valid') from the revised run_revision.py; no augmentation",
        "probability_aggregation": "all retained overlapping Barley 2 patches before argmax",
        "inference_batch_size": 4,
        "inference_batch_membership": "same as complete retained-dataset evaluation",
        "inference_patch_count": len(inference_indices),
        "neighbour_patch_count": len(needed),
        "geographic_pixel_size_m": gsd,
        "scale_bar_m": 5,
        "device": r.DEVICE,
        "source_code_sha256": sha256(Path(__file__)),
        "normalization_code_sha256": sha256(HERE / "run_revision.py"),
        "renderer_code_sha256": sha256(HERE / "render_figure7.py"),
        "training_performed": False,
        "unlabelled_predictions_validated": False,
        "examples": descriptions,
    }
    (OUT / "Figure7_data.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    # Render in a fresh process to keep the plotting and PyTorch OpenMP runtime
    # dependencies separate on this Windows scientific environment.
    subprocess.run([sys.executable, str(HERE / "render_figure7.py")], check=True)
    print(json.dumps(output, indent=2), flush=True)
    print("FIGURE 7 COMPLETE: revised checkpoint, overlap averages and reference support verified.", flush=True)


if __name__ == "__main__":
    main()
