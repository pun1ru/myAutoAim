"""Convert a captured manifest and its images from BMP to JPG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = source_root / "manifests" / "capture_4s_outpost_4s_range.csv"
    output_manifest = output_root / "manifests" / manifest_path.name
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8", newline="") as source_file:
        rows = list(csv.DictReader(source_file))

    with output_manifest.open("w", encoding="utf-8", newline="") as target_file:
        fieldnames = ["file", "scene", "source_sequence", "elapsed_seconds"]
        writer = csv.DictWriter(target_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            source = source_root / row["file"]
            relative = Path(row["file"]).with_suffix(".jpg")
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {source}")
            if not cv2.imwrite(
                str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Could not write {destination}")
            writer.writerow({**row, "file": relative.as_posix()})

    print(f"Converted {len(rows)} images to {output_root}")


if __name__ == "__main__":
    main()
