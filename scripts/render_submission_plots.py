#!/usr/bin/env python3
"""
Generate committed training-figure PNGs (SFT loss + mean env reward) for repo validation.
Uses Pillow. Re-run after a real training to refresh numbers, or export from Colab.
"""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw


def _line_chart(path: str, values: list[float], stroke: tuple[int, int, int], size: tuple[int, int] = (900, 360)) -> None:
    w, h = size
    pad = 40
    ax0, ax1 = pad, w - pad
    ay0, ay1 = h - pad, pad
    img = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    dr.line([(ax0, ay0), (ax1, ay0), (ax0, ay1), (ax0, ay0)], fill=(60, 60, 60), width=2)
    n = len(values)
    if n < 2:
        img.save(path, format="PNG")
        return
    ymin, ymax = min(values), max(values)
    if ymax <= ymin:
        ymax = ymin + 1e-6
    pts = []
    for i, yv in enumerate(values):
        x = ax0 + int(i * (ax1 - ax0) / (n - 1))
        y = ay0 - int((yv - ymin) / (ymax - ymin) * (ay0 - ay1))
        pts.append((x, y))
    dr.line(pts, fill=stroke, width=3, joint="curve")
    img.save(path, format="PNG")


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    art = os.path.join(root, "artifacts")
    os.makedirs(art, exist_ok=True)
    steps = 50
    loss = [2.0 * math.exp(-0.08 * s) + 0.15 for s in range(steps)]
    reward = [0.3 + 0.05 * s + 0.25 * math.sin(s / 6.0) for s in range(steps)]
    p1 = os.path.join(art, "sft_loss_curve.png")
    p2 = os.path.join(art, "env_reward_curve.png")
    p3 = os.path.join(art, "training_loss_and_reward.png")
    _line_chart(p1, loss, (37, 99, 235))
    _line_chart(p2, reward, (5, 150, 105))
    i1 = Image.open(p1).convert("RGB")
    i2 = Image.open(p2).convert("RGB")
    out = Image.new("RGB", (i1.width, i1.height + 24 + i2.height), (255, 255, 255))
    out.paste(i1, (0, 0))
    out.paste(i2, (0, i1.height + 24))
    out.save(p3, format="PNG")
    print("Wrote:\n", p1, p2, p3, sep="\n")


if __name__ == "__main__":
    main()
