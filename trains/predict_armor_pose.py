"""Run an armor pose model on an image, directory, or video."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Image, image directory, video, or camera index")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "runs" / "armor_pose" / "weights" / "best.pt",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="armor_pose_predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights.resolve()))
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str((ROOT / "runs").resolve()),
        name=args.name,
        exist_ok=True,
        save=True,
    )


if __name__ == "__main__":
    main()
