import math
from typing import Tuple

import numpy as np
from PIL import Image
import torch


def _tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    return arr


def _numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(arr)


def _border_based_mask(rgb: np.ndarray, diff_threshold: float = 0.08) -> np.ndarray:
    h, w, _ = rgb.shape
    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    return diff > diff_threshold


def _build_mask(img: np.ndarray, border_diff_threshold: float) -> np.ndarray:
    if img.shape[2] >= 4:
        alpha = img[:, :, 3]
        mask = alpha > 0.05
        if mask.mean() > 0.005:
            return mask

    rgb = img[:, :, :3]
    mask = _border_based_mask(rgb, border_diff_threshold)

    # Fallback for very dark/very bright cutouts
    if mask.mean() < 0.005:
        gray = rgb.mean(axis=2)
        mask = (gray < 0.97) & (gray > 0.02)
    return mask


def _estimate_bottom_profile_angle(mask: np.ndarray, trim_side_fraction: float) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) < 100:
        return 0.0, 0.0

    x_min, x_max = xs.min(), xs.max()
    width = max(1, x_max - x_min + 1)
    trim = int(width * trim_side_fraction)

    left = x_min + trim
    right = x_max - trim
    if right <= left:
        left, right = x_min, x_max

    profile_x = []
    profile_y = []

    for x in range(left, right + 1):
        col = np.where(mask[:, x])[0]
        if len(col) == 0:
            continue
        # bottom-most object pixel in this column
        y = int(col.max())
        profile_x.append(x)
        profile_y.append(y)

    if len(profile_x) < 20:
        return 0.0, 0.0

    px = np.asarray(profile_x, dtype=np.float32)
    py = np.asarray(profile_y, dtype=np.float32)

    # robust fit via median-based residual filtering
    slope, intercept = np.polyfit(px, py, 1)
    pred = slope * px + intercept
    resid = py - pred
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    keep = np.abs(resid - np.median(resid)) < 2.5 * mad

    if keep.sum() >= 20:
        px2 = px[keep]
        py2 = py[keep]
        slope, intercept = np.polyfit(px2, py2, 1)
        pred = slope * px2 + intercept
        resid = py2 - pred
        used_n = len(px2)
    else:
        used_n = len(px)

    angle_deg = math.degrees(math.atan(float(slope)))

    # confidence: many samples + low residual variance
    residual_scale = float(np.std(resid)) if len(resid) > 1 else 999.0
    width_span = max(1.0, float(px.max() - px.min()))
    straightness = max(0.0, 1.0 - (residual_scale / max(8.0, width_span * 0.03)))
    density = min(1.0, used_n / max(30.0, width_span * 0.75))
    confidence = max(0.0, min(1.0, 0.55 * straightness + 0.45 * density))
    return angle_deg, confidence


def _crop_to_content(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return arr
    pad = 4
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(arr.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(arr.shape[1], int(xs.max()) + pad + 1)
    return arr[y0:y1, x0:x1, :]


def _rotate_single_image(
    img: np.ndarray,
    angle_deg: float,
    auto_crop_to_content: bool,
    border_diff_threshold: float,
) -> np.ndarray:
    is_rgba = img.shape[2] >= 4
    if is_rgba:
        rgba = (np.clip(img[:, :, :4], 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(rgba, mode="RGBA")
        rotated = pil.rotate(
            angle_deg,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0, 0),
        )
        arr = np.asarray(rotated).astype(np.float32) / 255.0
        if auto_crop_to_content:
            mask = arr[:, :, 3] > 0.02
            arr = _crop_to_content(arr, mask)
        return arr

    rgb = (np.clip(img[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    border = pil.getpixel((0, 0))
    rotated = pil.rotate(
        angle_deg,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=border,
    )
    arr = np.asarray(rotated).astype(np.float32) / 255.0

    if auto_crop_to_content:
        mask = _build_mask(arr, border_diff_threshold)
        arr = _crop_to_content(arr, mask)
    return arr


class VehicleAutoLevel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_correction_degrees": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "trim_side_fraction": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 0.4, "step": 0.01}),
                "border_diff_threshold": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.5, "step": 0.01}),
                "min_mask_coverage": ("FLOAT", {"default": 0.15, "min": 0.001, "max": 0.9, "step": 0.01}),
                "min_confidence": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "manual_angle_offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "auto_crop_to_content": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("image", "angle_deg", "confidence")
    FUNCTION = "run"
    CATEGORY = "Autobook/vehicle"

    def run(
        self,
        image,
        max_correction_degrees=7.0,
        trim_side_fraction=0.18,
        border_diff_threshold=0.08,
        min_mask_coverage=0.15,
        min_confidence=0.35,
        manual_angle_offset=0.0,
        auto_crop_to_content=True,
    ):
        batch = _tensor_to_numpy(image)
        out_images = []
        angles = []
        confidences = []

        for i in range(batch.shape[0]):
            img = batch[i]
            mask = _build_mask(img, border_diff_threshold)
            coverage = float(mask.mean())

            if coverage < min_mask_coverage:
                correction = float(manual_angle_offset)
                confidence = 0.0
                out = _rotate_single_image(img, correction, auto_crop_to_content, border_diff_threshold) if abs(correction) > 1e-4 else img
            else:
                est_angle, confidence = _estimate_bottom_profile_angle(mask, trim_side_fraction)
                est_angle = max(-max_correction_degrees, min(max_correction_degrees, est_angle))
                correction = est_angle + float(manual_angle_offset)

                if confidence < min_confidence and abs(manual_angle_offset) < 1e-4:
                    correction = 0.0

                if abs(correction) > 1e-4:
                    out = _rotate_single_image(img, correction, auto_crop_to_content, border_diff_threshold)
                else:
                    out = img

            out_images.append(out)
            angles.append(np.array([correction], dtype=np.float32))
            confidences.append(np.array([confidence], dtype=np.float32))

        # pad images in batch to same size if rotate expanded them differently
        max_h = max(im.shape[0] for im in out_images)
        max_w = max(im.shape[1] for im in out_images)
        max_c = max(im.shape[2] for im in out_images)

        padded = []
        for im in out_images:
            canvas = np.zeros((max_h, max_w, max_c), dtype=np.float32)
            if im.shape[2] < max_c:
                extra = np.ones((im.shape[0], im.shape[1], max_c - im.shape[2]), dtype=np.float32)
                im = np.concatenate([im, extra], axis=2)
            y = (max_h - im.shape[0]) // 2
            x = (max_w - im.shape[1]) // 2
            canvas[y:y + im.shape[0], x:x + im.shape[1], :] = im
            padded.append(canvas)

        image_out = _numpy_to_tensor(np.stack(padded, axis=0))
        angle_out = torch.from_numpy(np.concatenate(angles, axis=0))
        confidence_out = torch.from_numpy(np.concatenate(confidences, axis=0))
        return (image_out, angle_out, confidence_out)


NODE_CLASS_MAPPINGS = {
    "VehicleAutoLevel": VehicleAutoLevel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VehicleAutoLevel": "Vehicle Auto Level",
}
