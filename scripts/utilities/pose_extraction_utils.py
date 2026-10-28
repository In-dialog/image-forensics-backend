"""
Pose extraction via OpenPose (controlnet_aux). Resize-for-detect, run detector, return/save pose image and JSON (points + lines).
Reusable by tests and by the run_pose_extraction executor.
"""
import json
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image

# Optional at import; required when running
try:
    import numpy as np
    import torch
    from controlnet_aux.open_pose import OpenposeDetector, draw_poses
    from controlnet_aux.util import HWC3, resize_image
except ImportError:
    np = None
    torch = None
    OpenposeDetector = None
    draw_poses = None
    HWC3 = None
    resize_image = None

# Body skeleton connections (1-based keypoint indices), same as controlnet_aux open_pose.util
BODY_LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],
    [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
    [2, 1], [1, 15], [15, 17], [1, 16], [16, 18],
]


def _resize_for_detect(image: Image.Image, max_longer_side: Optional[int]) -> Image.Image:
    """Resize so longer side is at most max_longer_side (if set), dimensions multiple of 64."""
    w, h = image.size
    if max_longer_side is None:
        new_w = (w + 63) // 64 * 64
        new_h = (h + 63) // 64 * 64
        if (new_w, new_h) != (w, h):
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image
    if max(w, h) <= max_longer_side:
        new_w = (w + 63) // 64 * 64
        new_h = (h + 63) // 64 * 64
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    scale = max_longer_side / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = (new_w + 63) // 64 * 64
    new_h = (new_h + 63) // 64 * 64
    return image.resize((max(64, new_w), max(64, new_h)), Image.Resampling.LANCZOS)


def load_detector(
    model_id: str = "lllyasviel/Annotators",
    device: Optional[str] = None,
) -> tuple[Any, str]:
    """Load OpenPose detector. Returns (detector, device_str)."""
    if OpenposeDetector is None or torch is None:
        raise ImportError("torch and controlnet_aux are required for pose extraction")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = OpenposeDetector.from_pretrained(model_id)
    detector = detector.to(device)
    return detector, device


def pose_result_to_json(
    poses: Any,
    image_width: int,
    image_height: int,
    *,
    decimals: int = 2,
) -> dict:
    """
    Convert controlnet_aux List[PoseResult] to JSON-serializable dict with points and lines.
    Keypoints are in pixel coordinates (same frame as the pose image).
    """
    def round_xy(x: float, y: float) -> List[float]:
        return [round(x, decimals), round(y, decimals)]

    people = []
    for pose in poses:
        body = pose.body
        if body is None or not body.keypoints:
            continue
        kps = body.keypoints
        # Points: id 1..18, pixel coords (keypoints are normalized 0-1)
        points = []
        for i, kp in enumerate(kps):
            if kp is None or (getattr(kp, "x", None) is None):
                continue
            x_px = kp.x * image_width
            y_px = kp.y * image_height
            points.append({"id": i + 1, "x": round(x_px, decimals), "y": round(y_px, decimals)})
        # Lines: body limbs
        lines = []
        for (i1, i2) in BODY_LIMB_SEQ:
            idx1, idx2 = i1 - 1, i2 - 1
            if idx1 >= len(kps) or idx2 >= len(kps):
                continue
            k1, k2 = kps[idx1], kps[idx2]
            if k1 is None or k2 is None:
                continue
            x1 = k1.x * image_width
            y1 = k1.y * image_height
            x2 = k2.x * image_width
            y2 = k2.y * image_height
            lines.append({"start": round_xy(x1, y1), "end": round_xy(x2, y2)})
        people.append({"points": points, "lines": lines})
    return {"image_width": image_width, "image_height": image_height, "people": people}


def extract_pose_and_poses(
    image_path: Path,
    detector: Any,
    *,
    hand_and_face: bool = True,
    detect_resolution: int = 512,
) -> tuple[Image.Image, Any, int, int]:
    """
    Run OpenPose using controlnet_aux pipeline; return (pose_image_pil, poses, width, height).
    Pose image and keypoints use the same (detect_resolution) coordinate frame.
    """
    from .image_utils import load_rgb

    if HWC3 is None or resize_image is None or draw_poses is None:
        raise ImportError("controlnet_aux is required for extract_pose_and_poses")
    img = load_rgb(image_path)
    img_np = np.array(img, dtype=np.uint8)
    img_np = HWC3(img_np)
    img_np = resize_image(img_np, detect_resolution)
    H, W, _ = img_np.shape
    poses = detector.detect_poses(img_np, include_hand=hand_and_face, include_face=hand_and_face)
    canvas = draw_poses(
        poses, H, W,
        draw_body=True,
        draw_hand=hand_and_face,
        draw_face=hand_and_face,
    )
    canvas = HWC3(canvas)
    pose_pil = Image.fromarray(canvas)
    return pose_pil, poses, W, H


def extract_pose(
    image_path: Path,
    detector: Any,
    *,
    hand_and_face: bool = True,
    max_longer_side: Optional[int] = 512,
) -> Image.Image:
    """Run OpenPose on image and return pose/skeleton image (PIL)."""
    from .image_utils import load_rgb

    image = load_rgb(image_path)
    image = _resize_for_detect(image, max_longer_side)
    pose_image = detector(image, hand_and_face=hand_and_face)
    if isinstance(pose_image, Image.Image):
        return pose_image
    import numpy as np

    if hasattr(pose_image, "cpu"):
        pose_image = pose_image.cpu().numpy()
    if isinstance(pose_image, np.ndarray):
        if pose_image.ndim == 3 and pose_image.shape[0] == 3:
            pose_image = np.transpose(pose_image, (1, 2, 0))
        return Image.fromarray(
            (pose_image * 255).astype(np.uint8)
            if pose_image.max() <= 1
            else pose_image.astype(np.uint8)
        )
    return Image.fromarray(pose_image)


def run_pose_extraction_and_save(
    image_path: Path,
    output_dir: Path,
    *,
    detector: Optional[Any] = None,
    model_id: str = "lllyasviel/Annotators",
    hand_and_face: bool = True,
    max_longer_side: Optional[int] = 512,
    pose_filename: Optional[str] = None,
) -> Path:
    """
    Run OpenPose on the image and save pose map to output_dir/{stem}_pose.webp (transparent bg).
    If detector is None, loads and uses model_id. Returns path to the saved pose image.
    """
    from .image_utils import ensure_dir

    if detector is None:
        detector, _ = load_detector(model_id)
    pose_image = extract_pose(
        image_path,
        detector,
        hand_and_face=hand_and_face,
        max_longer_side=max_longer_side,
    )
    ensure_dir(output_dir)
    stem = image_path.stem
    out_name = pose_filename or f"{stem}_pose.webp"
    out_path = output_dir / out_name
    arr = np.array(pose_image.convert("RGBA"))
    black_mask = (arr[:, :, 0] <= 5) & (arr[:, :, 1] <= 5) & (arr[:, :, 2] <= 5)
    arr[black_mask, 3] = 0
    Image.fromarray(arr).save(out_path, "WEBP", quality=90, method=6)
    return out_path
