#!/usr/bin/env python3
"""
Post-analysis utility for raw matrices saved by the YOLO + FoundationStereo pipeline.

Workflow:
1) Load raw matrices from an output/raw folder: depth.npy (meters), edges.npy (uint8).
    Optional: flatten very-far depths (e.g., sky): if depth > T, set to T
    to avoid triggering false edges in depth gradients.
2) Buffer (dilate) the detected block edge map to create a wider boundary band.
3) Compute a tight ROI bounding box around the buffered edges (with optional margin).
4) Clip the depth map to this ROI.
5) Detect "sudden depth change" edges inside the ROI using gradient magnitude (Sobel) or Laplacian.
6) Save results (NPY + PNG) for downstream numerical analysis.

Outputs (written next to the raw folder by default under raw/post):
- buffered_edges.png: Dilated edge mask at full image resolution
- roi_meta.json: ROI coordinates and parameters used
- depth_roi.npy / depth_roi.png: Clipped depth region (meters) + visualization
- depth_grad.npy: Gradient magnitude inside ROI (meters per pixel)
- depth_edge.npy / depth_edge.png: Binary depth-edge map inside ROI
- overlay_roi.png: Depth visualization with depth-edges overlaid

Example:
  python analyze_depth_edges.py --raw_dir C:/path/to/output/raw --buffer_px 6 --roi_margin_px 10 --method sobel --edge_percentile 95

"""

import os
import json
import argparse
import numpy as np
import cv2


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_raw(raw_dir: str):
    depth_path = os.path.join(raw_dir, 'depth.npy')
    edges_path = os.path.join(raw_dir, 'edges.npy')
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Missing depth.npy at {depth_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing edges.npy at {edges_path}")
    depth = np.load(depth_path)  # float32 meters
    edges = np.load(edges_path)  # uint8 (0 or 255)
    # Normalize edges to binary mask {0,1}
    if edges.dtype != np.uint8:
        edges = edges.astype(np.uint8)
    edges_bin = (edges > 0).astype(np.uint8)
    return depth, edges_bin


def buffer_edges(edges_bin: np.ndarray, buffer_px: int) -> np.ndarray:
    if buffer_px <= 0:
        return edges_bin.copy()
    k = 2 * buffer_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dilated = cv2.dilate(edges_bin, kernel, iterations=1)
    return dilated


def compute_roi_from_mask(mask: np.ndarray, margin: int = 0):
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None  # no ROI
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    if margin > 0:
        y1 -= margin
        y2 += margin
        x1 -= margin
        x2 += margin
    # clip to image bounds
    h, w = mask.shape[:2]
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(h - 1, y2)
    x2 = min(w - 1, x2)
    return x1, y1, x2, y2


def visualize_depth(depth: np.ndarray, invalid_val: float = np.nan) -> np.ndarray:
    d = depth.copy()
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    dmin = float(np.percentile(d[valid], 5))
    dmax = float(np.percentile(d[valid], 95))
    if dmax <= dmin:
        dmax = dmin + 1e-6
    d_clipped = np.clip(d, dmin, dmax)
    d_norm = (d_clipped - dmin) / (dmax - dmin)
    d_norm[~valid] = 0
    img_u8 = (np.clip(d_norm, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def fill_invalid_with_median(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(d[valid]))
    d[~valid] = med
    return d


def detect_depth_edges(depth_roi: np.ndarray, method: str, edge_thresh: float | None, edge_percentile: float | None):
    # Prepare: fill invalid for gradient computation
    d_filled = fill_invalid_with_median(depth_roi.astype(np.float32))

    if method == 'sobel':
        dx = cv2.Sobel(d_filled, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(d_filled, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(dx * dx + dy * dy)
    elif method == 'laplacian':
        grad = cv2.Laplacian(d_filled, cv2.CV_32F, ksize=3)
        grad = np.abs(grad)
    elif method == 'canny':
        # Normalize to 8-bit for canny
        valid = np.isfinite(depth_roi) & (depth_roi > 0)
        if valid.any():
            dmin = float(np.percentile(depth_roi[valid], 5))
            dmax = float(np.percentile(depth_roi[valid], 95))
            if dmax <= dmin:
                dmax = dmin + 1e-6
            d_norm = (np.clip(depth_roi, dmin, dmax) - dmin) / (dmax - dmin)
        else:
            d_norm = np.zeros_like(depth_roi, dtype=np.float32)
        img_u8 = (np.clip(d_norm, 0, 1) * 255).astype(np.uint8)
        # Canny thresholds: auto from percentile if not provided
        if edge_thresh is None and edge_percentile is not None:
            low = int(np.percentile(img_u8, edge_percentile * 0.5))
            high = int(np.percentile(img_u8, edge_percentile))
        else:
            # Default thresholds
            low, high = 50, 150
        edges = cv2.Canny(img_u8, low, high)
        grad = edges.astype(np.float32)  # use as binary magnitude
    else:
        raise ValueError("Unknown method. Choose from: sobel, laplacian, canny")

    # Threshold to binary edges
    if method in ('sobel', 'laplacian'):
        if edge_thresh is not None:
            th = float(edge_thresh)
        elif edge_percentile is not None:
            th = float(np.percentile(grad, edge_percentile))
        else:
            th = float(np.percentile(grad, 95))  # default
        edges_bin = (grad >= th).astype(np.uint8) * 255
    else:
        # canny already binary
        edges_bin = (grad > 0).astype(np.uint8) * 255

    return grad, edges_bin


def main():
    parser = argparse.ArgumentParser(description='Post-analysis of raw depth/edge matrices')
    parser.add_argument('--raw_dir', type=str, required=True, help='Path to output/raw directory')
    parser.add_argument('--buffer_px', type=int, default=6, help='Buffer (dilation) radius in pixels')
    parser.add_argument('--roi_margin_px', type=int, default=8, help='Extra margin around buffered ROI in pixels')
    parser.add_argument('--method', type=str, default='sobel', choices=['sobel', 'laplacian', 'canny'], help='Depth edge detection method')
    parser.add_argument('--edge_thresh', type=float, default=None, help='Absolute threshold for gradient magnitude (meters/pixel for sobel/laplacian)')
    parser.add_argument('--edge_percentile', type=float, default=95.0, help='Percentile for automatic thresholding if edge_thresh is not provided')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory (defaults to <raw_dir>/post)')
    parser.add_argument('--flatten_far_m', type=float, default=5.0, help='Flatten far depths: if depth > T (meters), set to T')

    args = parser.parse_args()

    depth, edges_bin = load_raw(args.raw_dir)

    # Flatten very-far depths to avoid sky/background artifacts: if depth > T, set to T
    if args.flatten_far_m is not None:
        T = float(args.flatten_far_m)
        far_mask = np.isfinite(depth) & (depth > T)
        if far_mask.any():
            depth = depth.copy()
            depth[far_mask] = T

    # Buffer/dilate edges and save full-res buffered mask
    buf = buffer_edges(edges_bin, args.buffer_px)

    post_dir = args.out_dir or os.path.join(args.raw_dir, 'post')
    ensure_dir(post_dir)

    cv2.imwrite(os.path.join(post_dir, 'buffered_edges.png'), (buf * 255).astype(np.uint8))

    # ROI from buffered mask
    roi = compute_roi_from_mask(buf, args.roi_margin_px)
    if roi is None:
        print('No edges found to build ROI. Exiting.')
        return
    x1, y1, x2, y2 = roi

    # Clip depth to ROI
    depth_roi = depth[y1:y2 + 1, x1:x2 + 1]

    # Detect depth edges inside ROI
    grad, depth_edges = detect_depth_edges(
        depth_roi, method=args.method,
        edge_thresh=args.edge_thresh,
        edge_percentile=args.edge_percentile
    )

    # Visualize depth ROI and overlay
    depth_roi_vis = visualize_depth(depth_roi)
    overlay = depth_roi_vis.copy()
    # draw depth edges in red
    cnts, _ = cv2.findContours((depth_edges > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 1)

    # Save artifacts
    np.save(os.path.join(post_dir, 'depth_roi.npy'), depth_roi.astype(np.float32))
    np.save(os.path.join(post_dir, 'depth_grad.npy'), grad.astype(np.float32))
    np.save(os.path.join(post_dir, 'depth_edge.npy'), (depth_edges > 0).astype(np.uint8))
    cv2.imwrite(os.path.join(post_dir, 'depth_roi.png'), depth_roi_vis)
    cv2.imwrite(os.path.join(post_dir, 'depth_edge.png'), depth_edges)
    cv2.imwrite(os.path.join(post_dir, 'overlay_roi.png'), overlay)

    # Save ROI metadata
    meta = {
        'raw_dir': os.path.abspath(args.raw_dir),
        'post_dir': os.path.abspath(post_dir),
        'roi': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
        'buffer_px': int(args.buffer_px),
        'roi_margin_px': int(args.roi_margin_px),
        'method': args.method,
        'edge_thresh': None if args.edge_thresh is None else float(args.edge_thresh),
        'edge_percentile': float(args.edge_percentile),
        'flatten_far_m': None if args.flatten_far_m is None else float(args.flatten_far_m),
    }
    with open(os.path.join(post_dir, 'roi_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('Post-analysis complete:')
    print(f" - Buffered edges: {os.path.join(post_dir, 'buffered_edges.png')}")
    print(f" - Depth ROI (npy/png): {os.path.join(post_dir, 'depth_roi.npy')} | {os.path.join(post_dir, 'depth_roi.png')}")
    print(f" - Depth gradient (npy): {os.path.join(post_dir, 'depth_grad.npy')}")
    print(f" - Depth edges (npy/png): {os.path.join(post_dir, 'depth_edge.npy')} | {os.path.join(post_dir, 'depth_edge.png')}")
    print(f" - Overlay: {os.path.join(post_dir, 'overlay_roi.png')}")
    print(f" - ROI meta: {os.path.join(post_dir, 'roi_meta.json')}")


if __name__ == '__main__':
    main()
