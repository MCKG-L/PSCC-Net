from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb_image(path):
    with Image.open(path) as image:
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        return np.array(image.convert('RGB'))


def read_mask(path):
    """Read a single-channel mask and accept either 0/1 or 0/255 encoding."""
    with Image.open(path) as image:
        if image.mode == '1':
            image = image.convert('L')
        if image.mode in ('RGB', 'RGBA', 'LA'):
            image = image.convert('L')
        mask = np.array(image, dtype=np.float32)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError(f'Expected a finite single-channel mask: {path}')
    if mask.min() < 0 or mask.max() > 255:
        raise ValueError(f'Mask values must be in 0..1 or 0..255: {path}')
    if mask.max() <= 1:
        mask *= 255
    return mask.astype(np.uint8)


def mask_path(image_path, cls):
    path = Path(image_path)
    # Replace a directory component, never arbitrary occurrences in the root/name.
    parts = list(path.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == 'fake':
            parts[index] = 'mask'
            break
    else:
        raise ValueError(f'Expected a fake/ image directory: {image_path}')
    path = Path(*parts)
    if cls in (1, 3):
        path = path.with_suffix('.png')
    return path
