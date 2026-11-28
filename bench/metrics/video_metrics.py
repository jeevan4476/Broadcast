"""Deterministic synthetic portrait frames and video quality/parity metrics.

Provides:
- Deterministic 1280x720 (or scaled) synthetic RGB portrait frame generator
  with textured room background, centered human head/shoulders silhouette,
  facial landmarks (eyes/mouth), and optional low-light sensor noise.
- Ground-truth foreground mask for objective segmentation IoU/MAE scoring.
- Frame-to-frame PSNR (dB), SSIM, alpha matte IoU/MAE, background blur
  variance reduction, and side-by-side PNG strip renderer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

try:
    from PIL import Image, ImageDraw  # type: ignore

    HAS_PIL = True
except ImportError:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    HAS_PIL = False


def generate_synthetic_portrait(
    width: int = 1280,
    height: int = 720,
    frame_index: int = 0,
    add_sensor_noise: bool = False,
) -> tuple[Any, Any]:
    """Generate a deterministic RGB portrait frame + ground-truth binary foreground mask.

    Returns `(frame_rgb, gt_mask_f32)` as NumPy arrays when NumPy is installed,
    or `(PIL.Image RGB, PIL.Image L)` when running in NumPy-free CI.
    """
    # Slight deterministic horizontal sway so temporal models & Auto Frame see natural motion
    sway = math.sin(frame_index * 0.25) * 0.025
    cx = 0.50 + sway
    cy = 0.42

    if HAS_NUMPY:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xn = xx / max(1.0, float(width))
        yn = yy / max(1.0, float(height))

        # High-frequency textured office background (bookshelf stripes + window + checkerboard details)
        bg = np.zeros((height, width, 3), dtype=np.float32)
        bg[:, :, 0] = 62.0 + 38.0 * np.sin(xn * 28.0) + 18.0 * np.cos(yn * 22.0)
        bg[:, :, 1] = 68.0 + 32.0 * np.cos(xn * 19.0 + yn * 14.0)
        bg[:, :, 2] = 82.0 + 40.0 * np.sin(yn * 26.0)
        # Bright window rectangle on right side
        win = (xn > 0.72) & (xn < 0.94) & (yn > 0.10) & (yn < 0.65)
        bg[win] = np.array([210.0, 222.0, 236.0], dtype=np.float32)
        # Dark shelf vertical bars on left side
        shelf = (xn > 0.06) & (xn < 0.26) & (yn > 0.15) & (yn < 0.85) & (((xx.astype(np.int32) // 18) % 2) == 0)
        bg[shelf] = np.array([34.0, 38.0, 44.0], dtype=np.float32)

        # Foreground silhouette: head ellipse + neck + torso/shoulders
        head_rx, head_ry = 0.11, 0.19
        head_d2 = ((xn - cx) / head_rx) ** 2 + ((yn - cy) / head_ry) ** 2
        head_mask = head_d2 <= 1.0

        neck_mask = (np.abs(xn - cx) < 0.042) & (yn >= cy + 0.15) & (yn <= cy + 0.27)

        sh_rx, sh_ry = 0.27, 0.34
        sh_cy = cy + 0.50
        sh_d2 = ((xn - cx) / sh_rx) ** 2 + ((yn - sh_cy) / sh_ry) ** 2
        torso_mask = (sh_d2 <= 1.0) & (yn >= cy + 0.22)

        fg_mask = (head_mask | neck_mask | torso_mask).astype(np.float32)

        img = bg.copy()
        # Torso clothing (dark navy jacket with collar texture)
        img[torso_mask] = np.array([36.0, 48.0, 74.0], dtype=np.float32)
        # Skin tone on head + neck with soft directional shading
        shade = np.clip(1.0 - 0.28 * head_d2, 0.65, 1.05)
        skin_mask = head_mask | neck_mask
        img[:, :, 0] = np.where(skin_mask, 212.0 * shade, img[:, :, 0])
        img[:, :, 1] = np.where(skin_mask, 168.0 * shade, img[:, :, 1])
        img[:, :, 2] = np.where(skin_mask, 140.0 * shade, img[:, :, 2])

        # Hair crown
        hair_mask = head_mask & (yn < (cy - 0.06 + 0.03 * np.sin((xn - cx) * 35.0)))
        img[hair_mask] = np.array([38.0, 30.0, 26.0], dtype=np.float32)

        # Eyes
        for ex_off in (-0.04, 0.04):
            eye_d2 = ((xn - (cx + ex_off)) / 0.016) ** 2 + ((yn - (cy - 0.02)) / 0.014) ** 2
            img[eye_d2 <= 1.0] = np.array([240.0, 240.0, 240.0], dtype=np.float32)
            pupil_d2 = ((xn - (cx + ex_off)) / 0.007) ** 2 + ((yn - (cy - 0.02)) / 0.009) ** 2
            img[pupil_d2 <= 1.0] = np.array([32.0, 28.0, 26.0], dtype=np.float32)

        if add_sensor_noise:
            rng = np.random.default_rng(1000 + frame_index)
            img = img + rng.normal(0.0, 14.0, img.shape).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8), fg_mask

    if not HAS_PIL:
        raise RuntimeError("Either numpy or Pillow is required to generate synthetic video frames")

    img = Image.new("RGB", (width, height), (64, 70, 84))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    mdraw = ImageDraw.Draw(mask)
    for x in range(0, width, 32):
        draw.line([(x, 0), (x, height)], fill=(96, 102, 118), width=4)
    draw.rectangle([int(width * 0.72), int(height * 0.1), int(width * 0.94), int(height * 0.65)], fill=(210, 222, 236))
    hx, hy = int(cx * width), int(cy * height)
    hrx, hry = int(0.11 * width), int(0.19 * height)
    srx, sry = int(0.27 * width), int(0.34 * height)
    sy = int((cy + 0.50) * height)
    draw.ellipse([hx - srx, sy - sry, hx + srx, sy + sry], fill=(36, 48, 74))
    mdraw.ellipse([hx - srx, sy - sry, hx + srx, sy + sry], fill=255)
    draw.ellipse([hx - hrx, hy - hry, hx + hrx, hy + hry], fill=(212, 168, 140))
    mdraw.ellipse([hx - hrx, hy - hry, hx + hrx, hy + hry], fill=255)
    return img, mask


def compute_psnr_db(img_a: Any, img_b: Any) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two RGB images."""
    if HAS_NUMPY:
        a = np.asarray(img_a, dtype=np.float64)
        b = np.asarray(img_b, dtype=np.float64)
        mse = float(np.mean((a - b) ** 2))
    else:
        ba = img_a.tobytes()
        bb = img_b.tobytes()
        n = min(len(ba), len(bb))
        if n == 0:
            return 99.0
        mse = sum((ba[i] - bb[i]) ** 2 for i in range(n)) / n
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def compute_ssim(img_a: Any, img_b: Any) -> float:
    """Compute global/block-averaged luminance SSIM in [-1, 1] (1.0 = identical)."""
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    if HAS_NUMPY:
        a = np.asarray(img_a, dtype=np.float64)
        b = np.asarray(img_b, dtype=np.float64)
        if a.ndim == 3:
            a = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
            b = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
        # Compute over 8x8 non-overlapping blocks for fast, spatially aware SSIM
        h = (a.shape[0] // 8) * 8
        w = (a.shape[1] // 8) * 8
        if h == 0 or w == 0:
            return 1.0
        ab = a[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
        bb = b[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
        mu_a = np.mean(ab, axis=1)
        mu_b = np.mean(bb, axis=1)
        var_a = np.var(ab, axis=1)
        var_b = np.var(bb, axis=1)
        cov = np.mean((ab - mu_a[:, None]) * (bb - mu_b[:, None]), axis=1)
        num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
        den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
        return float(np.mean(num / np.maximum(den, 1e-12)))

    # Pure-Python global luminance SSIM fallback
    ba = img_a.convert("L").tobytes()
    bb = img_b.convert("L").tobytes()
    n = min(len(ba), len(bb))
    if n == 0:
        return 1.0
    mu_a = sum(ba[:n]) / n
    mu_b = sum(bb[:n]) / n
    var_a = sum((ba[i] - mu_a) ** 2 for i in range(n)) / n
    var_b = sum((bb[i] - mu_b) ** 2 for i in range(n)) / n
    cov = sum((ba[i] - mu_a) * (bb[i] - mu_b) for i in range(n)) / n
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float(num / max(den, 1e-12))


def compute_mask_iou_mae(mask_a: Any, mask_b: Any, threshold: float = 0.5) -> dict[str, float]:
    """Compute binary IoU and continuous MAE between two [0..1] alpha mattes."""
    if HAS_NUMPY:
        a = np.asarray(mask_a, dtype=np.float32)
        b = np.asarray(mask_b, dtype=np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        if b.max() > 1.5:
            b = b / 255.0
        mae = float(np.mean(np.abs(a - b)))
        ab = a >= threshold
        bb = b >= threshold
        inter = int(np.sum(ab & bb))
        union = int(np.sum(ab | bb))
        iou = (inter / union) if union > 0 else 1.0
        return {"iou": round(iou, 4), "mae": round(mae, 4)}

    ba = mask_a.tobytes()
    bb = mask_b.tobytes()
    n = min(len(ba), len(bb))
    if n == 0:
        return {"iou": 1.0, "mae": 0.0}
    thresh_u8 = int(threshold * 255)
    inter = union = 0
    abs_sum = 0.0
    for i in range(n):
        abs_sum += abs(ba[i] - bb[i]) / 255.0
        a1 = ba[i] >= thresh_u8
        b1 = bb[i] >= thresh_u8
        if a1 and b1:
            inter += 1
        if a1 or b1:
            union += 1
    return {"iou": round((inter / union) if union > 0 else 1.0, 4), "mae": round(abs_sum / n, 4)}


def compute_bg_smoothness_delta(raw_rgb: Any, proc_rgb: Any, fg_mask: Any) -> float:
    """Measure horizontal gradient energy reduction on the background region (%)."""
    if not HAS_NUMPY:
        return 0.0
    r = np.asarray(raw_rgb, dtype=np.float32)
    p = np.asarray(proc_rgb, dtype=np.float32)
    bg = np.asarray(fg_mask, dtype=np.float32) < 0.2
    bg_slice = bg[:, 1:]
    if not np.any(bg_slice):
        return 0.0
    raw_grad = float(np.mean(np.abs(np.diff(r, axis=1))[bg_slice]))
    proc_grad = float(np.mean(np.abs(np.diff(p, axis=1))[bg_slice]))
    if raw_grad <= 1e-6:
        return 0.0
    return round((1.0 - proc_grad / raw_grad) * 100.0, 2)


def save_comparison_strip(panels: list[tuple[str, Any]], out_path: Path) -> str | None:
    """Save a horizontal comparison strip PNG for visual inspection."""
    if not HAS_PIL or not panels:
        return None
    pil_panels: list[tuple[str, Any]] = []
    for label, img in panels:
        if HAS_NUMPY and isinstance(img, np.ndarray):
            pil_panels.append((label, Image.fromarray(img.astype(np.uint8), mode="RGB")))
        elif isinstance(img, Image.Image):
            pil_panels.append((label, img.convert("RGB")))
    if not pil_panels:
        return None
    pw, ph = 426, 240
    strip = Image.new("RGB", (pw * len(pil_panels), ph + 28), (18, 20, 24))
    draw = ImageDraw.Draw(strip)
    for idx, (label, im) in enumerate(pil_panels):
        resized = im.resize((pw, ph), Image.Resampling.BILINEAR)
        strip.paste(resized, (idx * pw, 28))
        draw.text((idx * pw + 10, 8), label, fill=(220, 230, 240))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(str(out_path))
    return str(out_path)
