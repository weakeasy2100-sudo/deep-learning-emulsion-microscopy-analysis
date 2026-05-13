"""Shared utility functions used across V1–V7 pipeline scripts."""

import numpy as np


def crop_patch(img, cy, cx, size):
    # Square crop centred at (cy, cx); reflect-pads if the crop window exceeds the image boundary.
    half = size // 2
    r0, r1 = cy - half, cy + half
    c0, c1 = cx - half, cx + half

    pad_top    = max(0, -r0)
    pad_bottom = max(0, r1 - img.shape[0])
    pad_left   = max(0, -c0)
    pad_right  = max(0, c1 - img.shape[1])

    if pad_top or pad_bottom or pad_left or pad_right:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
        cy += pad_top
        cx += pad_left
        r0, r1 = cy - half, cy + half
        c0, c1 = cx - half, cx + half

    return img[r0:r1, c0:c1]
