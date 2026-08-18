"""Convert a CVAT COCO Keypoints export into an Ultralytics pose dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


POINT_COLORS = [
    (0, 255, 255),   # bottom_left
    (0, 255, 0),     # top_left
    (255, 255, 0),   # top_right
    (255, 0, 255),   # bottom_right
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path, help="CVAT COCO Keypoints JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "armor_pose_dataset",
        help="Generated YOLO dataset directory",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate(data: dict, image_dir: Path) -> tuple[list[dict], dict[int, list[dict]], list[str]]:
    categories = data.get("categories", [])
    if len(categories) != 1:
        raise ValueError(f"Expected exactly one category, got {len(categories)}")

    point_names = categories[0].get("keypoints", [])
    if len(point_names) != 4:
        raise ValueError(f"Expected 4 keypoints, got {len(point_names)}")

    images = data.get("images", [])
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    known_ids = {image["id"] for image in images}

    for annotation in data.get("annotations", []):
        image_id = annotation["image_id"]
        if image_id not in known_ids:
            raise ValueError(f"Annotation {annotation['id']} references unknown image {image_id}")
        keypoints = annotation.get("keypoints", [])
        if len(keypoints) != 12 or not all(math.isfinite(value) for value in keypoints):
            raise ValueError(f"Annotation {annotation['id']} has invalid keypoints")
        if len(annotation.get("bbox", [])) != 4:
            raise ValueError(f"Annotation {annotation['id']} has an invalid bounding box")
        annotations_by_image[image_id].append(annotation)

    missing = [image["file_name"] for image in images if not (image_dir / image["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image(s), first: {missing[0]}")

    return images, annotations_by_image, point_names


def normalized_label(annotation: dict, width: int, height: int) -> str:
    x, y, box_width, box_height = annotation["bbox"]
    values = [
        0,
        (x + box_width / 2) / width,
        (y + box_height / 2) / height,
        box_width / width,
        box_height / height,
    ]
    keypoints = annotation["keypoints"]
    for index in range(0, len(keypoints), 3):
        point_x, point_y, visibility = keypoints[index : index + 3]
        values.extend((point_x / width, point_y / height, visibility))
    return " ".join(str(value) if isinstance(value, int) else f"{value:.8f}" for value in values)


def draw_preview(source: Path, destination: Path, annotations: list[dict], point_names: list[str]) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"OpenCV could not read {source}")

    for annotation in annotations:
        points = []
        keypoints = annotation["keypoints"]
        for index in range(4):
            x, y, visibility = keypoints[index * 3 : index * 3 + 3]
            point = (round(x), round(y))
            points.append(point)
            if visibility > 0:
                cv2.circle(image, point, 5, POINT_COLORS[index], -1, cv2.LINE_AA)
                cv2.putText(
                    image,
                    str(index + 1),
                    (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    POINT_COLORS[index],
                    2,
                    cv2.LINE_AA,
                )
        cv2.polylines(image, [np.array(points)], True, (0, 165, 255), 2, cv2.LINE_AA)

    legend = "  ".join(f"{index + 1}:{name}" for index, name in enumerate(point_names))
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 760), 34), (0, 0, 0), -1)
    cv2.putText(image, legend, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write {destination}")


def main() -> None:
    args = parse_args()
    json_path = args.json_path.resolve()
    image_dir = json_path.parent
    output = args.output.resolve()

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    images, annotations_by_image, point_names = validate(data, image_dir)

    shuffled = images.copy()
    random.Random(args.seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * args.val_ratio))
    validation_ids = {image["id"] for image in shuffled[:validation_count]}

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    preview_dir = output / "previews"
    preview_dir.mkdir(parents=True)

    split_counts = {"train": 0, "val": 0}
    previewed = 0
    for image in images:
        split = "val" if image["id"] in validation_ids else "train"
        split_counts[split] += 1
        source = image_dir / image["file_name"]
        destination = output / "images" / split / image["file_name"]
        shutil.copy2(source, destination)

        label_path = output / "labels" / split / f"{Path(image['file_name']).stem}.txt"
        labels = [
            normalized_label(annotation, image["width"], image["height"])
            for annotation in annotations_by_image.get(image["id"], [])
        ]
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="ascii")

        if labels and previewed < args.preview_count:
            draw_preview(source, preview_dir / image["file_name"], annotations_by_image[image["id"]], point_names)
            previewed += 1

    yaml_path = output / "armor_pose.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [4, 3]",
                "flip_idx: [3, 2, 1, 0]",
                "names:",
                "  0: armor",
                "",
            ]
        ),
        encoding="utf-8",
    )

    annotation_count = sum(len(items) for items in annotations_by_image.values())
    negative_count = sum(not annotations_by_image.get(image["id"]) for image in images)
    print(f"Created dataset: {output}")
    print(f"Images: {len(images)} (train={split_counts['train']}, val={split_counts['val']})")
    print(f"Instances: {annotation_count}; negative images: {negative_count}")
    print(f"Keypoint order: {', '.join(point_names)}")
    print(f"Dataset config: {yaml_path}")
    print(f"Annotation previews: {preview_dir}")


if __name__ == "__main__":
    main()
