"""Make raw RGB-only time strips for input QA, without model or GT numeric data."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(0, 56, 5)))
    args = parser.parse_args()
    w, h = 300, 190
    canvas = np.full((2 * (h+24), 6*w, 3), 255, np.uint8)
    for n, idx in enumerate(args.indices[:12]):
        image = cv2.imread(str(args.scene / "rgb" / f"{idx:06d}.png"))
        if image is None:
            continue
        area = canvas[(n//6)*(h+24):(n//6+1)*(h+24), (n%6)*w:(n%6+1)*w]
        cv2.putText(area, f"{args.scene.name} frame {idx}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1)
        area[24:24+h] = cv2.resize(image, (w,h))
    if args.output.exists():
        raise FileExistsError(args.output)
    cv2.imwrite(str(args.output), canvas)


if __name__ == "__main__":
    main()
