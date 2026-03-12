"""
Simple evaluation for CLIP zero-shot scene labels.
CSV format:
image_path,day_night,indoor_outdoor
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scene_classifier import classify_scene


def load_labels(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row.get("image_path", "").strip()
            day_night = row.get("day_night", "").strip().lower()
            indoor_outdoor = row.get("indoor_outdoor", "").strip().lower()
            if not image_path or not day_night or not indoor_outdoor:
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "day_night": day_night,
                    "indoor_outdoor": indoor_outdoor,
                    "scene": f"{day_night}_{indoor_outdoor}",
                }
            )
    return rows


def evaluate(rows):
    total = len(rows)
    if total == 0:
        raise ValueError("No valid rows found in labels file.")

    correct_scene = 0
    correct_day_night = 0
    correct_indoor_outdoor = 0

    details = []
    for row in rows:
        pred = classify_scene(row["image_path"])
        is_day_night = pred["day_night"] == row["day_night"]
        is_indoor_outdoor = pred["indoor_outdoor"] == row["indoor_outdoor"]
        is_scene = pred["scene"] == row["scene"]

        correct_day_night += 1 if is_day_night else 0
        correct_indoor_outdoor += 1 if is_indoor_outdoor else 0
        correct_scene += 1 if is_scene else 0

        details.append(
            {
                "image": row["image_path"],
                "label": row["scene"],
                "pred": pred["scene"],
                "confidence": pred["confidence"],
                "day_night_ok": is_day_night,
                "indoor_outdoor_ok": is_indoor_outdoor,
            }
        )

    return {
        "total": total,
        "scene_acc": correct_scene / total,
        "day_night_acc": correct_day_night / total,
        "indoor_outdoor_acc": correct_indoor_outdoor / total,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate scene classifier")
    parser.add_argument(
        "--labels",
        type=str,
        default="scene_labels.csv",
        help="Path to CSV labels file",
    )
    parser.add_argument("--show", action="store_true", help="Show per-image results")
    args = parser.parse_args()

    csv_path = Path(args.labels)
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels file not found: {csv_path}")

    rows = load_labels(csv_path)
    result = evaluate(rows)

    print(f"Total: {result['total']}")
    print(f"Scene accuracy: {result['scene_acc']:.2%}")
    print(f"Day/Night accuracy: {result['day_night_acc']:.2%}")
    print(f"Indoor/Outdoor accuracy: {result['indoor_outdoor_acc']:.2%}")

    if args.show:
        for d in result["details"]:
            print(
                f"{d['image']} | label={d['label']} | pred={d['pred']} | "
                f"conf={d['confidence']}"
            )


if __name__ == "__main__":
    main()

