"""
Composition / leading lines utility. Returns line segments as (start, end) points only;
no drawing. Uses gradient-based clustering on a downscaled image.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage import exposure
from skimage.transform import resize
from skimage.draw import line as draw_line

# ---------------------------------------------------------------------------
# Base lines: indices and masks (no PIL; we only care about begin/end, masks for algo)
# ---------------------------------------------------------------------------


def get_base_line_indices(ws: int) -> np.ndarray:
    """
    All potential line (start, end) pairs for a ws×ws square.
    Edges: 1=top, 2=right, 3=bottom, 4=left. Skips 5 px from borders.
    Returns array of shape (N, 2, 2): each row is [[x0,y0], [x1,y1]].
    """
    out: List[np.ndarray] = []
    for i in range(0, ws - 5):
        for j in range(5, ws):
            # Adjacent: edge 1→2, then rotations 4→1, 3→4, 2→3
            out.append(np.array([[i, 0], [ws - 1, j]]))
            out.append(np.array([[j, 0], [0, ws - 1 - i]]))
            out.append(np.array([[0, ws - 1 - j], [ws - 1 - i, ws - 1]]))
            out.append(np.array([[ws - 1, i], [ws - 1 - j, ws - 1]]))
            # Opposite: 1→3 and 4→2 (transpose)
            out.append(np.array([[i, 0], [j, ws - 1]]))
            out.append(np.array([[0, i], [ws - 1, j]]))
    return np.asarray(out)


def _line_mask(ws: int, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Rasterize one line into a ws×ws binary mask. Points are (x,y) = (col, row)."""
    mask = np.zeros((ws, ws), dtype=np.float64)
    r0, c0 = int(start[1]), int(start[0])
    r1, c1 = int(end[1]), int(end[0])
    rr, cc = draw_line(r0, c0, r1, c1)
    valid = (rr >= 0) & (rr < ws) & (cc >= 0) & (cc < ws)
    mask[rr[valid], cc[valid]] = 1.0
    return mask


def get_base_lines(ws: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Base lines as (masks, indices). Masks used internally for gradient weighting;
    indices are (start, end) per line. Returns (all_lines_masks, all_lines_index).
    """
    indices = get_base_line_indices(ws)
    masks = np.array([_line_mask(ws, idx[0], idx[1]) for idx in indices])
    return masks, indices


# ---------------------------------------------------------------------------
# Gradient and clustering (from run_composition_lines)
# ---------------------------------------------------------------------------


def _normalise(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def gradient_4d(img: np.ndarray) -> np.ndarray:
    """Absolute gradient in 4 directions per pixel; shape (H,W)."""
    row, col = img.shape
    g4d = np.zeros((row, col))
    for i in range(row - 1):
        for j in range(col - 1):
            g4d[i, j] = (
                abs(img[i + 1, j] - img[i, j])
                + abs(img[i, j + 1] - img[i, j])
                + abs(img[i + 1, j - 1] - img[i, j])
                + abs(img[i + 1, j + 1] - img[i, j])
            )
    return _normalise(g4d)


def _slope(v1: np.ndarray, v2: np.ndarray) -> float:
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    if dx == 0:
        return 5.0 * dy
    return dy / dx


def _cluster_region(
    center_line: np.ndarray, scale: int = 4, window_size: int = 64
) -> np.ndarray:
    H = W = window_size
    s_mask = np.zeros((H, W))
    ix, iy = int(center_line[0][0]), int(center_line[0][1])
    pixel_range = int(min(H, W) / scale)
    k = _slope(center_line[0], center_line[1])

    if abs(k) > 1:
        while ix > 0:
            iy = int(round(((ix - center_line[0][1]) / k) + center_line[0][0]))
            front_y = max(0, iy - pixel_range)
            back_y = min(W, iy + pixel_range + 1)
            s_mask[ix, front_y:back_y] = 1
            ix -= 1
        ix = int(center_line[0][0])
        while ix < H:
            iy = int(round(((ix - center_line[0][1]) / k) + center_line[0][0]))
            front_y = max(0, iy - pixel_range)
            back_y = min(W, iy + pixel_range + 1)
            s_mask[ix, front_y:back_y] = 1
            ix += 1
    else:
        while iy > 0:
            ix = int(round(((iy - center_line[0][0]) * k) + center_line[0][1]))
            front_x = max(0, ix - pixel_range)
            back_x = min(H, ix + pixel_range + 1)
            s_mask[front_x:back_x, iy] = 1
            iy -= 1
        iy = int(center_line[0][1])
        while iy < W:
            ix = int(round(((iy - center_line[0][0]) * k) + center_line[0][1]))
            front_x = max(0, ix - pixel_range)
            back_x = min(H, ix + pixel_range + 1)
            s_mask[front_x:back_x, iy] = 1
            iy += 1
    return s_mask


def _judge_vertex_advanced(
    line_mask: np.ndarray,
    v1: np.ndarray,
    v1w: float,
    force_list: list,
    w_seuil: float = 4,
    intersect_percent: float = 0.7,
) -> Tuple[bool, list]:
    for cl in force_list:
        v_possible = cl[0] * line_mask
        if np.sum(v_possible) > (np.sum(line_mask) * intersect_percent):
            if abs(cl[1][1] - v1w) < w_seuil:
                cl.append([v1, v1w, line_mask])
            return True, force_list
    return False, force_list


def _force_lines_cluster_integration(cluster: list) -> list:
    return [line_set[1] for line_set in cluster]


def _refine(
    line_list: list, fg: int, wg: float, iP: float, ws: int
) -> list:
    wlist = [l[1] for l in line_list]
    npw_list = np.array(wlist)
    sort_weight = npw_list.argsort()[::-1]
    force_list: list = []
    for n, w_id in enumerate(sort_weight):
        if n == 0:
            g_mask = _cluster_region(line_list[w_id][0], fg, ws)
            force_list.append([g_mask, line_list[w_id]])
        else:
            judge, force_list = _judge_vertex_advanced(
                line_list[w_id][2],
                line_list[w_id][0],
                npw_list[w_id],
                force_list,
                wg,
                iP,
            )
            if not judge:
                g_mask = _cluster_region(line_list[w_id][0], fg, ws)
                force_list.append([g_mask, line_list[w_id]])
    return _force_lines_cluster_integration(force_list)


def _find_salient_line_cluster(
    gradient_4d: np.ndarray,
    all_lines: np.ndarray,
    all_lines_index: np.ndarray,
    ws: int,
    org_w: int,
    org_h: int,
) -> np.ndarray:
    """Returns force lines as (N, 2, 2) in original image coordinates."""
    weight_list = [np.sum(gradient_4d * l) for l in all_lines]
    np_weight_list = np.array(weight_list)
    sort_weight_list = np_weight_list.argsort()[::-1]
    fine_grained0 = 8
    inte_prec0 = 0.8
    force_lines_cluster: list = []

    for n, w_id in enumerate(sort_weight_list[:300]):
        if n == 0:
            group_mask = _cluster_region(all_lines_index[w_id], fine_grained0, ws)
            force_lines_cluster.append(
                [group_mask, [all_lines_index[w_id], np_weight_list[w_id], all_lines[w_id]]]
            )
        else:
            if (np_weight_list[sort_weight_list[n - 1]] - np_weight_list[w_id]) > 10:
                break
            judge, force_lines_cluster = _judge_vertex_advanced(
                all_lines[w_id],
                all_lines_index[w_id],
                np_weight_list[w_id],
                force_lines_cluster,
                2,
                inte_prec0,
            )
            if not judge:
                group_mask = _cluster_region(all_lines_index[w_id], fine_grained0, ws)
                force_lines_cluster.append(
                    [group_mask, [all_lines_index[w_id], np_weight_list[w_id], all_lines[w_id]]]
                )

    force_lines_rough = _force_lines_cluster_integration(force_lines_cluster)
    force_lines_rough_new = force_lines_rough
    force_lines_rough_org: list = []
    fine_grained = 7
    w_grained = 3
    inte_prec = 0.7

    for _ in range(10000):
        if len(force_lines_rough_new) == len(force_lines_rough_org):
            if fine_grained <= 4 and w_grained >= 10:
                break
        force_lines_rough_org = force_lines_rough_new
        force_lines_rough_new = _refine(
            force_lines_rough_new, fine_grained, w_grained, inte_prec, ws
        )
        if fine_grained > 4:
            fine_grained -= 1
        if inte_prec > 0.6:
            inte_prec -= 0.05
        if w_grained < 10:
            w_grained += 1

    force_lines = np.array([l[0] for l in force_lines_rough_new])
    scale = 1.0 / ws
    hw_scale = np.array(
        [[np.ceil(org_w * scale), np.ceil(org_h * scale)],
         [np.ceil(org_w * scale), np.ceil(org_h * scale)]]
    )
    hws = np.expand_dims(hw_scale, 0).repeat(force_lines.shape[0], axis=0)
    return force_lines * hws


# ---------------------------------------------------------------------------
# Public API: return only (start, end) for each line
# ---------------------------------------------------------------------------


def get_composition_lines(
    image: Union[str, Path, np.ndarray],
    window_size: int = 64,
) -> np.ndarray:
    """
    Compute leading/composition lines for the image. No drawing.

    Args:
        image: Path to image file or (H, W, 3) RGB array.
        window_size: Internal grid size for line candidates (default 64).

    Returns:
        Lines as ndarray of shape (N, 2, 2), dtype float. Each row is
        [[x0, y0], [x1, y1]] in original image pixel coordinates (start, end).
    """
    if isinstance(image, (str, Path)):
        img = imread(str(image))
    else:
        img = np.asarray(image)

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("Image must be 3- or 4-channel (RGB/RGBA)")
    if img.shape[2] == 4:
        img = rgba2rgb(img)
    org_h, org_w = img.shape[0], img.shape[1]

    resized = resize(img, (window_size, window_size))
    log_img = exposure.adjust_log(resized, 1)
    gray = rgb2gray(log_img)
    grad = gradient_4d(gray)

    all_lines, all_lines_index = get_base_lines(window_size)
    lines = _find_salient_line_cluster(
        grad, all_lines, all_lines_index, window_size, org_w, org_h
    )
    return lines
