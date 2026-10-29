"""
SAM 3 (Segment Anything Model 3) utility: prompt-based instance segmentation.
Returns extracted crops and their positions on the original image for reconstruction.
Load/unload is controlled from outside: load_sam3() / unload_sam3(); pass the result into extract_by_prompt.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import weakref

from scripts.runtime.gpu import release_after_call, release_cuda_memory
from scripts.runtime.ollama import prepare_torch_model

import numpy as np
from PIL import Image

from .image_utils import ensure_rgb_hwc, load_rgb_flat, numpy_to_pil


@dataclass
class Sam3Extract:
    """One extracted instance: crop image, bbox on original, score, and full-size mask."""

    image: Image.Image  # RGBA crop
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) on original image
    score: float
    mask_original: np.ndarray  # (H, W) float 0–1, same size as original (for reconstruction)
    duplicate_detections: list[dict] = field(default_factory=list)


@dataclass
class Sam3Loaded:
    """Loaded SAM 3 model and processor; pass to extract_by_prompt or unload with unload_sam3."""

    model: Any  # Sam3Model
    processor: Any  # Sam3Processor
    device: str


# Defaults; callers do not need to change these unless tuning.
DEFAULT_MODEL_ID = "facebook/sam3"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MASK_THRESHOLD = 0.5
# When passing a point (x, y) to SAM3, we use a small box around it (HF API uses boxes, not points).
POINT_TO_BOX_RADIUS = 5
_LIVE_SAM3: "weakref.WeakValueDictionary[int, Sam3Loaded]" = weakref.WeakValueDictionary()


def _cleanup_sam_resources(
    model: Any | None,
    processor: Any | None,
    device: Optional[str],
) -> None:
    """
    Best-effort SAM cleanup for error paths and normal unloads.
    Ensures CUDA cache is released even if init/inference failed midway.
    """
    if model is not None:
        try:
            model.to("cpu")
        except Exception as exc:
            print(f"SAM offload warning: {exc}")
    release_cuda_memory()


def is_valid_box(
    box: Optional[Tuple[int, int, int, int]],
    min_size: int = 3,
) -> bool:
    """Check box is not None and has width/height >= min_size."""
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def _tensor_to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _resize_mask_to(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize mask (float 0–1) to (target_w, target_h)."""
    u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(u8).resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.array(pil, dtype=np.float64) / 255.0


@release_after_call
def load_sam3(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
) -> Sam3Loaded:
    """
    Load SAM 3 model and processor. Call from outside when you want to control load time.
    Pass the return value to extract_by_prompt(..., model_and_processor=...); when done, call unload_sam3.
    """
    try:
        from transformers import Sam3Model, Sam3Processor
        import torch
    except ImportError as err:
        raise ImportError("Install: pip install transformers torch Pillow") from err

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prepare_torch_model()
    model = None
    processor = None
    try:
        model = Sam3Model.from_pretrained(model_id).to(device)
        processor = Sam3Processor.from_pretrained(model_id)
        loaded = Sam3Loaded(model=model, processor=processor, device=device)
        _LIVE_SAM3[id(loaded)] = loaded
        return loaded
    except Exception:
        # If init fails after allocating VRAM, force cleanup before propagating.
        _cleanup_sam_resources(model, processor, device)
        raise


@release_after_call
def unload_sam3(loaded: Sam3Loaded) -> None:
    """
    Release the loaded model and processor so memory can be freed.
    Call from outside when you are done with this instance.
    """
    model = getattr(loaded, "model", None)
    processor = getattr(loaded, "processor", None)
    device = getattr(loaded, "device", None)
    _LIVE_SAM3.pop(id(loaded), None)
    # Drop references from the container first, then free local refs.
    try:
        loaded.model = None
    except Exception:
        pass
    try:
        loaded.processor = None
    except Exception:
        pass
    _cleanup_sam_resources(model, processor, device)


def unload_all_sam3() -> int:
    """
    Unload all currently tracked SAM3 instances created via load_sam3().
    Returns number of unload attempts performed.
    """
    count = 0
    for key, loaded in list(_LIVE_SAM3.items()):
        try:
            unload_sam3(loaded)
            count += 1
        except Exception:
            pass
        finally:
            _LIVE_SAM3.pop(key, None)
    return count


@release_after_call
def extract_by_prompt(
    prompt: str,
    image: Path | str | Image.Image,
    *,
    box: Optional[Tuple[int, int, int, int]] = None,
    point: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
    model_and_processor: Optional[Sam3Loaded] = None,
    model_id: str = DEFAULT_MODEL_ID,
    threshold: float = DEFAULT_THRESHOLD,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    device: Optional[str] = None,
) -> List[Sam3Extract]:
    """
    Run SAM 3 with text prompt, optionally with box (x1,y1,x2,y2) or point(s). Uses input_boxes.

    Args:
        prompt: Text prompt (e.g. "person", "flag").
        image: Image path or PIL Image (RGB).
        box: Optional (x1, y1, x2, y2) from vision. Passed directly as input_boxes.
        point: Optional (x, y) or list of (x, y). Converted to small boxes if box is None.
        model_and_processor: Pre-loaded from load_sam3(); if None, loads internally.

    Returns:
        List of Sam3Extract (image, bbox, score, mask_original).
    """
    if isinstance(image, (Path, str)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        # Use load_rgb_flat so WebP/PNG transparency is composited on white (avoids SAM artifacts)
        pil_image = load_rgb_flat(path)
    else:
        pil_image = image.convert("RGB")

    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
    except ImportError as err:
        raise ImportError("Install: pip install transformers torch Pillow") from err

    owned_model = None  # only set when we load internally, so we can free in finally
    if model_and_processor is not None:
        model = model_and_processor.model
        processor = model_and_processor.processor
        device = model_and_processor.device
    else:
        prepare_torch_model()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = None
        processor = None
        try:
            model = Sam3Model.from_pretrained(model_id).to(device)
            processor = Sam3Processor.from_pretrained(model_id)
            owned_model = (model, processor, device)
        except Exception:
            # Handles partial init failures (e.g., OOM during model/processor load).
            _cleanup_sam_resources(model, processor, device)
            raise

    try:
        img_w, img_h = pil_image.size  # PIL: (width, height)
        use_boxes = False
        if box is not None:
            # Clamp to image bounds and ensure valid x1<x2, y1<y2 (SAM expects pixel xyxy)
            x1 = max(0, min(int(box[0]), img_w - 1))
            y1 = max(0, min(int(box[1]), img_h - 1))
            x2 = max(0, min(int(box[2]), img_w))
            y2 = max(0, min(int(box[3]), img_h))
            if x1 >= x2:
                x2 = x1 + 1
            if y1 >= y2:
                y2 = y1 + 1
            input_boxes = [[[float(x1), float(y1), float(x2), float(y2)]]]
            input_boxes_labels = [[1]]
            use_boxes = True
        else:
            points_list: List[Tuple[int, int]] = []
            if point is not None:
                points_list = [point] if isinstance(point, tuple) else point
            if points_list:
                r = POINT_TO_BOX_RADIUS
                input_boxes = [[
                    [max(0, x - r), max(0, y - r), min(img_w, x + r + 1), min(img_h, y + r + 1)]
                    for x, y in points_list
                ]]
                input_boxes_labels = [[1] * len(points_list)]
                use_boxes = True

        if use_boxes:
            # SAM3 processor expects absolute XYXY boxes and handles normalization internally.
            inputs = processor(
                images=[pil_image],
                text=prompt.strip(),
                input_boxes=input_boxes,
                input_boxes_labels=input_boxes_labels,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = inputs.get("original_sizes")
            if isinstance(target_sizes, torch.Tensor):
                target_sizes = target_sizes.tolist()
            if target_sizes is None:
                target_sizes = [[img_h, img_w]]
        else:
            img_inputs = processor(images=pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)
            text_inputs = processor(text=prompt.strip(), return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(vision_embeds=vision_embeds, **text_inputs)
            target_sizes = img_inputs.get("original_sizes").tolist()

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )[0]

        masks = results.get("masks")
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])

        if masks is None or len(masks) == 0:
            return []

        # When we prompted with a single box, use it for output so label and location stay matched.
        prompt_box: Optional[Tuple[int, int, int, int]] = None
        if box is not None:
            px1 = max(0, min(int(box[0]), img_w - 1))
            py1 = max(0, min(int(box[1]), img_h - 1))
            px2 = max(0, min(int(box[2]), img_w))
            py2 = max(0, min(int(box[3]), img_h))
            if px1 >= px2:
                px2 = px1 + 1
            if py1 >= py2:
                py2 = py1 + 1
            prompt_box = (px1, py1, px2, py2)

        masks_np = _tensor_to_numpy(masks)
        boxes_np = np.array([_tensor_to_numpy(b) for b in boxes])
        scores_np = np.array([_tensor_to_numpy(s) for s in scores])

        img = ensure_rgb_hwc(np.array(pil_image))
        h, w = img.shape[:2]

        out: List[Sam3Extract] = []
        for i in range(len(masks_np)):
            mask = np.asarray(masks_np[i]).squeeze()
            if mask.shape[:2] != (h, w):
                mask = _resize_mask_to(mask, w, h)
            score = float(scores_np[i].item() if scores_np[i].size == 1 else scores_np[i])

            if prompt_box is not None:
                x1, y1, x2, y2 = prompt_box
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
            elif i < len(boxes_np):
                sam_box = boxes_np[i]
                x1, y1, x2, y2 = (int(round(x)) for x in sam_box)
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
            else:
                ys, xs = np.where(mask > float(mask_threshold))
                if xs.size == 0 or ys.size == 0:
                    continue
                x1, x2 = int(xs.min()), int(xs.max()) + 1
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            if prompt_box is not None:
                # The saved crop and reconstruction mask must describe the same
                # region. SAM may predict a containing object outside a small box.
                bounded = np.zeros_like(mask)
                bounded[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                mask = bounded
            if not np.any(mask[y1:y2, x1:x2] > float(mask_threshold)):
                continue

            crop_img = img[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]
            if crop_mask.ndim > 2:
                crop_mask = crop_mask.squeeze()
            if crop_mask.shape[:2] != crop_img.shape[:2]:
                crop_mask = _resize_mask_to(crop_mask, crop_img.shape[1], crop_img.shape[0])

            alpha = (np.clip(crop_mask, 0, 1) * 255).astype(np.uint8)
            if alpha.ndim == 2:
                alpha = alpha[..., np.newaxis]
            rgba = np.concatenate([crop_img, alpha], axis=-1)
            crop_pil = numpy_to_pil(rgba)

            out.append(
                Sam3Extract(
                    image=crop_pil,
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    mask_original=mask.astype(np.float64),
                )
            )

        return out
    finally:
        if owned_model is not None:
            model, processor, dev = owned_model
            _cleanup_sam_resources(model, processor, dev)


def deduplicate_results(
    results: List[tuple[str, str, Sam3Extract]],
    tolerance: int = 2,
    *,
    mask_threshold: float = 0.5,
) -> List[tuple[str, str, Sam3Extract]]:
    """Keep the strongest mask per occupied region, independent of label and padding.

    Pixel tolerance applies only to the box fallback for mask-less extracts. A zero
    tolerance must not turn off mask-based duplicate protection.
    """
    from .segmentation_identity import RegionIndex, extract_region
    from .detection_views import box_iou

    known = RegionIndex()
    kept: list[int] = []
    for index in sorted(range(len(results)), key=lambda i: (-float(results[i][2].score), i)):
        group, label, extract = results[index]
        region = extract_region(extract, threshold=mask_threshold)
        duplicate = known.match(region)
        if duplicate is None and extract.mask_original is None and tolerance > 0:
            duplicate = next((prior for prior in kept if results[prior][2].mask_original is None
                              and all(abs(a - b) <= tolerance for a, b in zip(extract.bbox, results[prior][2].bbox))
                              and box_iou(extract.bbox, results[prior][2].bbox) >= 0.9), None)
        if duplicate is None:
            known.add(index, region)
            kept.append(index)
        else:
            canonical = results[duplicate][2]
            evidence = {"label": label, "category": group, "bbox": list(extract.bbox), "sam_score": float(extract.score)}
            for record in [evidence, *extract.duplicate_detections]:
                if record not in canonical.duplicate_detections:
                    canonical.duplicate_detections.append(record)
    return [results[index] for index in sorted(kept)]


def _named_granular_results(
    results: List[tuple[str, str, Sam3Extract]],
) -> List[tuple[str, str, str, Sam3Extract]]:
    """Assign stable names (label_0, label_1, ...) per label; return (name, group, label, ex)."""
    label_count: dict[str, int] = {}
    out: List[tuple[str, str, str, Sam3Extract]] = []
    for group, label, ex in results:
        i = label_count.get(label, 0)
        label_count[label] = i + 1
        name = f"{label}_{i}"
        out.append((name, group, label, ex))
    return out


def _bbox_position_size(bbox: tuple[int, int, int, int]) -> tuple[list[float], list[int]]:
    """From (x1, y1, x2, y2) return (center [cx, cy], size [width, height])."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    return ([round(cx, 2), round(cy, 2)], [width, height])


def build_extracts_metadata(results: List[tuple[str, str, Sam3Extract]]) -> dict:
    """Describe each canonical (category, label, instance) extraction."""
    extracts = []
    for name, group, label, extract in _named_granular_results(results):
        position, size = _bbox_position_size(extract.bbox)
        extracts.append({
            "name": name,
            "label": label,
            "category": group,
            "sam_score": float(extract.score),
            "bbox": list(extract.bbox),
            "bbox_position": position,
            "bbox_size": size,
            "group": group if group in {"people", "texts"} else label,
            **({"duplicate_detections": extract.duplicate_detections} if extract.duplicate_detections else {}),
        })
    return {"extracts": extracts}


def named_extracts_for_images(
    results: List[tuple[str, str, Sam3Extract]],
) -> List[tuple[str, Image.Image]]:
    """
    From granular results (group, label, single extract), return [(name, image), ...]
    with same naming as build_extracts_metadata, for saving segment images.
    """
    named = _named_granular_results(results)
    return [(name, ex.image) for name, _g, _l, ex in named]
