"""Compose 3D renders into one paper figure with panel letters above the top-left corner (pof_style), nothing else
inside the images. Usage: compose_renders.py OUT.png IMAGE [IMAGE ...] [--columns N] [--width-in 6.69]"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pof_style as ps  # noqa: E402


def trimmed_box(path, pad=12):
    """The image cropped to its non-white content and the crop box (left, top, right, bottom) in the original pixels."""
    image = Image.open(path).convert("RGB")
    data = np.asarray(image)
    mask = np.any(data < 248, axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if not len(rows):
        return image, (0, 0, data.shape[1], data.shape[0])
    box = (max(cols[0]-pad, 0), max(rows[0]-pad, 0), min(cols[-1]+pad, data.shape[1]), min(rows[-1]+pad, data.shape[0]))
    return image.crop(box), box


def trimmed(path, pad=12):
    return trimmed_box(path, pad)[0]


def compose(out, images, columns=None, width_in=ps.WIDTH_DOUBLE_IN):
    import matplotlib.pyplot as plt
    ps.style()
    pictures = [trimmed(p) for p in images]
    columns = columns or len(pictures)
    rows = int(np.ceil(len(pictures)/columns))
    ratios = [p.size[1]/p.size[0] for p in pictures]
    height = width_in/columns*max(ratios)*rows+0.25*rows
    fig, axes = plt.subplots(rows, columns, figsize=(width_in, height), squeeze=False, constrained_layout=True)
    for k, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if k < len(pictures):
            ax.imshow(np.asarray(pictures[k]))
            ps.letter(ax, f"({'abcdefgh'[k]})")
    fig.savefig(out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compose_renders.py")
    parser.add_argument("out")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--columns", type=int, default=None)
    parser.add_argument("--width-in", type=float, default=ps.WIDTH_DOUBLE_IN)
    a = parser.parse_args()
    compose(a.out, a.images, a.columns, a.width_in)
