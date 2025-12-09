#!/usr/bin/env python3
"""
Corner Detection Library
Contains core functions and classes for the corner detection pipeline.
"""

import os
import sys
import glob
import json
import time
import numpy as np
import cv2
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# Optional imports
try:
    import imageio
except ImportError:
    imageio = None
    print("Warning: imageio not available. EXR loading will fail.")

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("Warning: Open3D not available. 3D processing will fail.")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Pipeline configuration parameters"""
    # Depth edge analysis parameters
    raw_dir = None  # Auto-detect if None
    buffer_px = 2
    roi_margin_px = 2
    method = 'canny'  # 'sobel' | 'laplacian' | 'canny'
    edge_thresh = None
    edge_sigma = 0.6
    max_edge_dist_px = 15
    smooth = True
    smooth_ksize = 5
    smooth_sigma = 1.0
    filter_top_instance = True
    
    # 3D edge extraction parameters
    voxel_size = None
    downsample_voxel = 0.01
    remove_outliers = True
    nb_neighbors = 20
    std_ratio = 1.0
    bpa_radius_factor = 1.5
    bpa_radii = None
    normal_radius = None
    angle_thresh_deg = 20.0
    min_edge_length = 0.01
    plane_distance = None  # Computed from spacing
    num_planes = 3
    ransac_n = 3
    num_iterations = 2000
    ortho_thresh_deg = 20.0
    min_inlier_ratio = 0.1
    
    # Output
    output_dir = r'../run_files/macro/3d_edge_output'

    # Input
    input_dir = r'./scripts/run_files/input_3d_blocks' #r'../run_files/macro/input_3d_blocks'


# ============================================================================
# HELPER FUNCTIONS - DEPTH ANALYSIS
# ============================================================================

def load_instance_metadata(raw_dir: str):
    """Load YOLO instance metadata (masks and confidences) if available."""
    masks_npz = os.path.join(raw_dir, 'instance_masks.npz')
    if os.path.exists(masks_npz):
        try:
            data = np.load(masks_npz, allow_pickle=True)
            masks = data.get('masks', [])
            confidences = data.get('confidences', [])
            if len(masks) > 0 and len(confidences) > 0:
                return {'masks': list(masks), 'confidences': list(confidences)}
        except Exception as e:
            print(f'Warning: Failed to load {masks_npz}: {e}')
    return None


def filter_edges_by_top_instance(edges_bin: np.ndarray, raw_dir: str):
    """Filter edges to only include the instance with highest confidence."""
    metadata = load_instance_metadata(raw_dir)
    
    if metadata is None:
        return edges_bin
    
    masks = metadata['masks']
    confidences = metadata['confidences']
    
    if len(masks) == 0:
        return edges_bin
    
    max_idx = int(np.argmax(confidences))
    top_mask = masks[max_idx]
    
    if top_mask.shape != edges_bin.shape:
        top_mask = cv2.resize(top_mask, (edges_bin.shape[1], edges_bin.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
    
    filtered_edges = np.zeros_like(edges_bin, dtype=np.uint8)
    filtered_edges[top_mask > 0] = edges_bin[top_mask > 0]
    
    return filtered_edges


def load_raw(raw_dir: str, filter_top_instance: bool = True):
    """
    Step 1: Load raw depth and edges from raw_dir.
    """
    depth_npy = os.path.join(raw_dir, 'depth.npy')
    edges_npy = os.path.join(raw_dir, 'edges.npy')
    depth_exr = os.path.join(raw_dir, 'depth.exr')
    edges_png = os.path.join(raw_dir, 'edges.png')

    if os.path.exists(depth_npy):
        depth = np.load(depth_npy).astype(np.float32)
    elif os.path.exists(depth_exr):
        d = imageio.imread(depth_exr)
        if d.ndim == 3 and d.shape[2] >= 1:
            depth = d[..., 0].astype(np.float32)
        else:
            depth = d.astype(np.float32)
    else:
        raise FileNotFoundError('Could not find depth.npy or depth.exr in raw_dir')

    zero_mask = (depth == 0)
    if zero_mask.any():
        depth[zero_mask] = 1.0

    if os.path.exists(edges_npy):
        edges = np.load(edges_npy)
        if edges.dtype != np.uint8:
            edges = edges.astype(np.uint8)
        edges_bin = (edges > 0).astype(np.uint8)
    elif os.path.exists(edges_png):
        e = cv2.imread(edges_png, cv2.IMREAD_GRAYSCALE)
        edges_bin = (e > 0).astype(np.uint8)
    else:
        raise FileNotFoundError('Could not find edges.npy or edges.png in raw_dir')
    
    if filter_top_instance:
        edges_bin = filter_edges_by_top_instance(edges_bin, raw_dir)
    
    return depth, edges_bin


def buffer_edges(edges_bin: np.ndarray, buffer_px: int) -> np.ndarray:
    """
    Step 3: Apply morphological dilation to buffer edges.
    """
    if buffer_px <= 0:
        return edges_bin.copy()
    k = 2 * buffer_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(edges_bin, kernel, iterations=1)


def compute_roi_from_mask(mask: np.ndarray, margin: int = 0):
    """
    Step 4: Compute bounding box ROI from mask.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    if margin > 0:
        y1 -= margin
        y2 += margin
        x1 -= margin
        x2 += margin
    h, w = mask.shape[:2]
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(h - 1, y2)
    x2 = min(w - 1, x2)
    return x1, y1, x2, y2


def fill_invalid_with_median(   depth: np.ndarray) -> np.ndarray:
    """Fill invalid depth values with median."""
    d = depth.copy()
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    # Use extreme large value to ensure gradient at edges
    d[~valid] = np.max(d[valid])
    return d


def detect_depth_edges(depth_roi: np.ndarray, method: str, edge_thresh: float | None,
                      edge_sigma: float | None, smooth: bool = True,
                      smooth_ksize: int = 5, smooth_sigma: float = 1.0):
    """
    Step 6: Detect depth edges using specified method.
    """
    d_filled = fill_invalid_with_median(depth_roi.astype(np.float32))

    if smooth_ksize is None or smooth_ksize <= 0:
        smooth_ksize = 1
    if smooth_ksize % 2 == 0:
        smooth_ksize += 1

    if method == 'sobel':
        d_proc = d_filled
        if smooth and smooth_ksize > 1:
            d_proc = cv2.GaussianBlur(d_filled, (smooth_ksize, smooth_ksize), smooth_sigma)
        dx = cv2.Sobel(d_proc, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(d_proc, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(dx * dx + dy * dy)
        if edge_thresh is not None:
            th = float(edge_thresh)
        elif edge_sigma is not None:
            th = float(np.percentile(grad, edge_sigma))
        else:
            th = float(np.percentile(grad, 95))
        edges_bin = (grad >= th).astype(np.uint8) * 255
        return grad, edges_bin

    elif method == 'laplacian':
        d_proc = d_filled
        if smooth and smooth_ksize > 1:
            d_proc = cv2.GaussianBlur(d_filled, (smooth_ksize, smooth_ksize), smooth_sigma)
        grad = cv2.Laplacian(d_proc, cv2.CV_32F, ksize=3)
        grad = np.abs(grad)
        if edge_thresh is not None:
            th = float(edge_thresh)
        elif edge_sigma is not None:
            th = float(np.percentile(grad, edge_sigma))
        else:
            th = float(np.percentile(grad, 95))
        edges_bin = (grad >= th).astype(np.uint8) * 255
        return grad, edges_bin

    elif method == 'canny':
        d_proc = d_filled
        if smooth and smooth_ksize > 1:
            d_proc = cv2.GaussianBlur(d_proc, (smooth_ksize, smooth_ksize), smooth_sigma)
        
        dx = cv2.Sobel(d_proc, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(d_proc, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(dx * dx + dy * dy)

        valid = np.isfinite(d_proc) & (d_proc > 0)
        if valid.any():
            dmin = float(np.percentile(d_proc[valid], 5))
            dmax = float(np.percentile(d_proc[valid], 95))
            if dmax <= dmin:
                dmax = dmin + 1e-6
            d_norm = (np.clip(d_proc, dmin, dmax) - dmin) / (dmax - dmin)
        else:
            d_norm = np.zeros_like(d_proc, dtype=np.float32)
        img_u8 = (np.clip(d_norm, 0, 1) * 255).astype(np.uint8)

        if edge_thresh is None and edge_sigma is not None:
            nz = img_u8 > 0
            v = float(img_u8[nz].mean()) if nz.any() else float(img_u8.mean())
            low = int(max(0, (1.0 - edge_sigma) * v))
            high = int(min(255, (1.0 + edge_sigma) * v))
        else:
            low, high = 50, 150

        edges = cv2.Canny(img_u8, low, high)
        edges_bin = (edges > 0).astype(np.uint8) * 255
        return grad, edges_bin

    raise ValueError("Unknown method. Choose from: sobel, laplacian, canny")


def create_mosaic(images, max_cols=None):
    """Helper to create a mosaic from a list of images."""
    if not images:
        return None
    n = len(images)
    if max_cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    else:
        cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    h, w = images[0].shape[:2]
    mosaic = np.zeros((rows * h, cols * w), dtype=images[0].dtype)
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        mosaic[r*h:(r+1)*h, c*w:(c+1)*w] = img
    return mosaic


def find_closed_loop(depth_edges_filtered: np.ndarray, depth_roi: np.ndarray, 
                    buffer_px: int):
    """
    Step 7: Find closed loop using iterative morphological closing.
    Returns: best_loop, iteration_count, mosaic_image
    """
    mask = (depth_edges_filtered > 0).astype(np.uint8) * 255
    
    roi_h, roi_w = depth_roi.shape[:2]
    min_dimension_threshold = roi_h * roi_w * 0.3 
    
    epsilon_frac = 0.005
    min_vertices = 4
    max_iterations = 50
    
    best_loop = None
    working_mask = mask.copy()
    iteration = 0
    
    debug_frames = []
    
    while best_loop is None and iteration <= max_iterations:
        debug_frames.append(working_mask.copy())
        
        # Find contours with hierarchy to detect holes
        cnts, hierarchy = cv2.findContours(working_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        large_closed_contours = []
        if hierarchy is not None:
            for i, c in enumerate(cnts):
                
                # Get bounding box dimensions
                x, y, w, h = cv2.boundingRect(c)
                max_dim = cv2.contourArea(c)
                
                # Check if contour is closed (has child or parent, i.e., part of a ring structure)
                # hierarchy: [Next, Previous, First_Child, Parent]
                # opened means isolated: no child (<0) AND no parent (<0)
                opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0

                # Check if dimension exceeds threshold and is not opened
                if max_dim >= min_dimension_threshold and not opened:
                    large_closed_contours.append(c)
        
        if large_closed_contours:
            largest_contour = min(large_closed_contours, key=cv2.contourArea)
            peri = float(cv2.arcLength(largest_contour, True))
            if peri > 0:
                eps = float(epsilon_frac * peri)
                approx = cv2.approxPolyDP(largest_contour, eps, True)
                if approx is not None and len(approx) >= min_vertices:
                    best_loop = approx
                    break
        
        if iteration < max_iterations:
            ks = 4 * iteration + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
            working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            iteration += 1
        else:
            break
    
    mosaic = create_mosaic(debug_frames)
    return best_loop, iteration, mosaic


def extract_boundary_points(depth_roi: np.ndarray, best_loop, roi):
    """
    Step 8: Extract 3D points within boundary polygon.
    """
    if best_loop is None or roi is None:
        return None
    
    x1, y1, x2, y2 = roi
    poly = best_loop.reshape(-1, 2).astype(np.int32)
    
    mask = np.zeros(depth_roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    
    depth_outside_zero = depth_roi.copy()
    depth_outside_zero[mask == 0] = 0
    
    bx, by, bw, bh = cv2.boundingRect(poly)
    if bw <= 0 or bh <= 0:
        return None
    
    cropped_depth = depth_outside_zero[by:by+bh, bx:bx+bw]
    cropped_mask = mask[by:by+bh, bx:bx+bw]
    
    pts = []
    rows, cols = np.where(cropped_mask > 0)
    for r, c in zip(rows, cols):
        z = float(cropped_depth[r, c])
        if not np.isfinite(z) or z <= 0:
            continue
        gx = float(x1 + bx + c)
        gy = float(y1 + by + r)
        pts.append((gx, gy, z))
    
    return np.array(pts, dtype=float)


def find_camera_intrinsics(search_dirs):
    """
    Step 0.5 / 8.6: Find and load camera intrinsics from step0.frame_data.json.
    """
    names = ['step0.frame_data.json', 'frame_data.json']
    for root in search_dirs:
        try:
            root_abs = os.path.abspath(root)
            if not os.path.isdir(root_abs):
                continue
            for dirpath, dirnames, filenames in os.walk(root_abs):
                for n in filenames:
                    if n.lower() in names:
                        json_path = os.path.join(dirpath, n)
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        
                        captures = data.get('captures', [])
                        cam = None
                        for c in captures:
                            if c.get('@type', '').endswith('RGBCamera'):
                                cam = c
                                break
                        
                        if cam is None:
                            continue
                        
                        dim = cam.get('dimension', None)
                        proj = cam.get('matrix', None)
                        if (not dim) or (not proj) or len(dim) < 2 or len(proj) < 9:
                            continue
                        
                        width = float(dim[0])
                        height = float(dim[1])
                        P00 = float(proj[0])
                        P11 = float(proj[4])
                        fx = P00 * width / 2.0
                        fy = P11 * height / 2.0
                        cx = width / 2.0
                        cy = height / 2.0
                        
                        return fx, fy, cx, cy, json_path
        except Exception:
            pass
    return None, None, None, None, None


def convert_to_camera_space(pts_pixel: np.ndarray, fx: float, fy: float, 
                           cx: float, cy: float):
    """
    Step 8.6: Convert pixel coordinates to camera-space metric coordinates.
    """
    u = pts_pixel[:, 0]
    v = pts_pixel[:, 1]
    Z = pts_pixel[:, 2]
    
    valid = np.isfinite(Z) & (Z > 0)
    
    # Filter Z outliers
    if valid.any():
        q1 = np.percentile(Z[valid], 25)
        q3 = np.percentile(Z[valid], 75)
        iqr = q3 - q1
        upper_fence = q3 + 3.0 * iqr
        if upper_fence > 0:
            valid_outlier = (Z <= upper_fence)
            valid = valid & valid_outlier
    
    u = u[valid]
    v = v[valid]
    Z = Z[valid]
    
    Xc = (u - cx) / fx * Z
    Yc = (v - cy) / fy * Z
    Zc = Z
    
    return np.stack([Xc, Yc, Zc], axis=1)


# ============================================================================
# HELPER FUNCTIONS - 3D PROCESSING
# ============================================================================

def build_mesh_from_pointcloud(pcd, bpa_radii):
    """Build mesh using Ball Pivoting Algorithm."""
    if not pcd.has_normals():
        raise RuntimeError("Point cloud must have normals")
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(bpa_radii)
    )
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    return mesh


def segment_planes(pts_all: np.ndarray, plane_distance: float, num_planes: int = 3,
                  ransac_n: int = 3, num_iterations: int = 2000):
    """Segment multiple planes using RANSAC."""
    if o3d is None:
        raise RuntimeError("Open3D required for plane segmentation")
    
    pc_mesh = o3d.geometry.PointCloud()
    pc_mesh.points = o3d.utility.Vector3dVector(pts_all)
    
    remaining_idx = np.arange(pts_all.shape[0])
    planes = []
    
    for i in range(num_planes):
        if len(remaining_idx) < ransac_n:
            break
        
        sub_pc = o3d.geometry.PointCloud()
        sub_pc.points = o3d.utility.Vector3dVector(pts_all[remaining_idx])
        
        try:
            model, inliers_local = sub_pc.segment_plane(
                distance_threshold=plane_distance,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            
            # Refinement iteration
            if len(inliers_local) > ransac_n:
                current_inliers = np.array(inliers_local, dtype=int)
                pts_sub = np.asarray(sub_pc.points)
                
                for _ in range(50):
                    if len(current_inliers) < ransac_n:
                        break
                    
                    pts_in = pts_sub[current_inliers]
                    centroid = pts_in.mean(axis=0)
                    centered = pts_in - centroid
                    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                    normal = Vt[-1]
                    normal = normal / (np.linalg.norm(normal) + 1e-12)
                    d_val = -np.dot(normal, centroid)
                    
                    dists = np.abs(np.dot(pts_in, normal) + d_val)
                    mask = dists <= plane_distance
                    
                    if np.all(mask):
                        model = (normal[0], normal[1], normal[2], d_val)
                        break
                    
                    current_inliers = current_inliers[mask]
                    model = (normal[0], normal[1], normal[2], d_val)
                
                inliers_local = current_inliers.tolist()
        except Exception as e:
            print(f'Plane {i+1} segmentation failed:', e)
            break
        
        if len(inliers_local) == 0:
            break
        
        inliers_global = remaining_idx[np.array(inliers_local, dtype=int)]
        a, b, c, d = model
        n = np.array([a, b, c], dtype=float)
        n_norm = np.linalg.norm(n) + 1e-12
        n_unit = n / n_norm
        d_unit = float(d) / n_norm
        
        planes.append({
            'normal': n_unit,
            'd': d_unit,
            'model_raw': (a, b, c, d),
            'inliers': inliers_global
        })
        
        mask_keep = np.ones(len(remaining_idx), dtype=bool)
        mask_keep[np.array(inliers_local, dtype=int)] = False
        remaining_idx = remaining_idx[mask_keep]
    
    return planes


def compute_box_corners(plane_models: list, pts_all: np.ndarray):
    """Compute the single intersection corner of 3 planes."""
    if len(plane_models) < 3:
        return np.array([], dtype=float)
    
    # Use the first 3 planes
    Ns = np.stack([p['normal'] for p in plane_models[:3]], axis=0)
    d_vals = np.array([p['d'] for p in plane_models[:3]], dtype=float)
    
    # Solve Ns * x = -d_vals
    try:
        corner = np.linalg.solve(Ns, -d_vals)
        return np.array([corner], dtype=float)
    except np.linalg.LinAlgError:
        try:
            corner, _, _, _ = np.linalg.lstsq(Ns, -d_vals, rcond=None)
            return np.array([corner], dtype=float)
        except Exception:

            return np.array([], dtype=float)

def process_planes_and_corners(planes, pts_all, ortho_thresh_deg=20.0, min_inlier_ratio=0.1):
    """
    Filter planes based on orthogonality to the prime plane (most points)
    and compute corners based on the number of valid planes.
    """
    if not planes:
        return [], np.array([], dtype=float)

    # Filter planes by inlier count
    total_points = len(pts_all)
    min_inliers = int(min_inlier_ratio * total_points)
    print(f"  Filtering planes with < {min_inliers} inliers ({min_inlier_ratio*100:.1f}% of {total_points})")
    
    filtered_planes = []
    for p in planes:
        if len(p['inliers']) >= min_inliers:
            filtered_planes.append(p)
        else:
            print(f"  Dropping plane with {len(p['inliers'])} inliers (Threshold: {min_inliers})")
            
    if not filtered_planes:
        print("  No planes left after inlier filtering.")
        return [], np.array([], dtype=float)
        
    planes = filtered_planes

    # 1. Identify Prime Plane (most inliers)
    planes_sorted = sorted(planes, key=lambda p: len(p['inliers']), reverse=True)
    prime_plane = planes_sorted[0]
    valid_planes = [prime_plane]

    # 2. Check Orthogonality
    # Threshold: |dot| < sin(thresh)
    threshold_val = np.sin(np.deg2rad(ortho_thresh_deg))
    
    print(f"  Prime plane has {len(prime_plane['inliers'])} inliers. Normal: {prime_plane['normal']}")

    for i in range(1, len(planes_sorted)):
        p = planes_sorted[i]
        dot_val = np.abs(np.dot(prime_plane['normal'], p['normal']))
        if dot_val < threshold_val:
            valid_planes.append(p)
            print(f"  Keeping plane (inliers: {len(p['inliers'])}) - Orthogonal (dot={dot_val:.3f})")
        else:
            print(f"  Discarding plane (inliers: {len(p['inliers'])}) - Not orthogonal (dot={dot_val:.3f})")

    # 3. Compute Corners
    corners = []
    num_valid = len(valid_planes)
    print(f"  Using {num_valid} valid planes for corner calculation.")

    if num_valid >= 3:
        # 3 Planes -> 1 Intersection
        Ns = np.stack([p['normal'] for p in valid_planes[:3]], axis=0)
        d_vals = np.array([p['d'] for p in valid_planes[:3]], dtype=float)
        try:
            corner = np.linalg.solve(Ns, -d_vals)
            corners = [corner]
        except np.linalg.LinAlgError:
            corner, _, _, _ = np.linalg.lstsq(Ns, -d_vals, rcond=None)
            corners = [corner]

    elif num_valid == 2:
        # 2 Planes -> 2 Points (Line segment endpoints)
        p1 = valid_planes[0]
        p2 = valid_planes[1]
        n1 = p1['normal']
        n2 = p2['normal']
        d1 = p1['d']
        d2 = p2['d']
        
        line_dir = np.cross(n1, n2)
        norm_dir = np.linalg.norm(line_dir)
        
        if norm_dir > 1e-6:
            line_dir /= norm_dir
            
            # Point on line
            A = np.vstack([n1, n2, line_dir])
            b = np.array([-d1, -d2, 0])
            try:
                x0 = np.linalg.solve(A, b)
            except:
                x0, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Project inliers
            idx1 = p1['inliers']
            idx2 = p2['inliers']
            pts_in = np.concatenate([pts_all[idx1], pts_all[idx2]], axis=0)
            
            t_vals = np.dot(pts_in - x0, line_dir)
            t_min = np.min(t_vals)
            t_max = np.max(t_vals)
            
            corners = [x0 + t_min * line_dir, x0 + t_max * line_dir]

    elif num_valid == 1:
        # 1 Plane -> 4 Points (Rectangle)
        p = valid_planes[0]
        n = p['normal']
        pts = pts_all[p['inliers']]
        
        # Local coordinates
        helper = np.array([1, 0, 0])
        if np.abs(np.dot(helper, n)) > 0.9:
            helper = np.array([0, 1, 0])
        x_prime = np.cross(n, helper)
        x_prime /= np.linalg.norm(x_prime)
        y_prime = np.cross(n, x_prime)
        
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        u = np.dot(pts_centered, x_prime)
        v = np.dot(pts_centered, y_prime)
        uv = np.stack([u, v], axis=1).astype(np.float32)
        
        rect = cv2.minAreaRect(uv)
        box_2d = cv2.boxPoints(rect)
        
        for pt_2d in box_2d:
            pt_3d = centroid + pt_2d[0] * x_prime + pt_2d[1] * y_prime
            corners.append(pt_3d)

    return valid_planes, np.array(corners, dtype=float)

def find_ground_truth_corners(json_path: str):
    """Extract ground truth corners from frame_data.json."""
    with open(json_path, 'r') as f:
        frame_data = json.load(f)
    
    corner_positions = []
    capture = frame_data.get('metrics', [])[2].get('values', [])[0].get('instances', [])
    for meta in capture:
        transform = meta.get('transformRecord', {})
        position = transform.get('position', None)
        if position and isinstance(position, list) and len(position) == 3:
            corner_positions.append([float(position[0]), float(position[1]), float(position[2])])
    
    if len(corner_positions) >= 8:
        return np.array(corner_positions[:8], dtype=float)
    return None


def get_camera_transform(json_path: str):
    """Extract camera position and rotation from frame_data.json."""
    with open(json_path, 'r') as f:
        frame_data = json.load(f)
    
    ego = frame_data.get('captures', [])[0]
    if ego:
        camera_position = ego.get('position', None)
        camera_rotation_quat = ego.get('rotation', None)
        return camera_position, camera_rotation_quat
    
    return None, None


def transform_to_world_space(corners_camera: np.ndarray, camera_position, camera_rotation_quat):
    """Transform camera-space corners to world space."""
    if camera_position is None or camera_rotation_quat is None:
        return corners_camera
    
    # Convert from CV camera space (Y-down) to Unity camera space (Y-up)
    # Unity Camera: X-Right, Y-Up, Z-Forward
    # CV Camera: X-Right, Y-Down, Z-Forward
    corners_unity_local = corners_camera.copy()
    corners_unity_local[:, 1] *= -1
    
    qx, qy, qz, qw = camera_rotation_quat
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    t = np.array(camera_position, dtype=float)
    
    return (R @ corners_unity_local.T).T + t


def compare_corners(gt_corners: np.ndarray, detected_corners_world: np.ndarray):
    """Compare ground truth and detected corners."""
    distances = cdist(gt_corners, detected_corners_world, metric='euclidean')
    
    matches = []
    for i, gt_pt in enumerate(gt_corners):
        min_idx = int(np.argmin(distances[i]))
        min_dist = float(distances[i, min_idx])
        detected_pt = detected_corners_world[min_idx]
        matches.append({
            'gt_idx': i,
            'gt_coord': gt_pt,
            'detected_idx': min_idx,
            'detected_coord': detected_pt,
            'distance': min_dist
        })
    
    best_match = min(matches, key=lambda m: m['distance'])
    return best_match, matches


def visualize_depth(depth: np.ndarray) -> np.ndarray:
    """Visualize depth map using Turbo colormap."""
    d = depth.copy()
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    
    d_valid = d[valid]
    dmin = np.percentile(d_valid, 5)
    dmax = np.percentile(d_valid, 95)
    if dmax <= dmin: dmax = dmin + 1e-6
    
    d_norm = np.clip((d - dmin) / (dmax - dmin), 0, 1)
    d_u8 = (d_norm * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    d_color[~valid] = 0
    return d_color


def project_points(points_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Project 3D points to 2D pixel coordinates."""
    if points_3d.shape[0] == 0:
        return np.zeros((0, 2))
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]
    
    # Avoid division by zero
    Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    
    u = (X * fx / Z) + cx
    v = (Y * fy / Z) + cy
    return np.stack([u, v], axis=1)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config: Config):
    """Execute the complete corner detection pipeline."""
    import time
    start_time = time.time()
    
    print("=" * 80)
    print("CORNER DETECTION PIPELINE")
    print("=" * 80)
    
    if config.output_dir and not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Stage 1: Find and load raw data
    print("\n[Stage 1] Loading raw depth and edge data...")
    if config.raw_dir is None:
        raise ValueError("raw_dir must be specified in config")
    
    depth, edges_bin = load_raw(config.raw_dir, config.filter_top_instance)
    print(f"  Depth: {depth.shape}, Edges: {int((edges_bin>0).sum())} pixels")
    
    if config.output_dir:
        d_vis = visualize_depth(depth)
        e_vis = cv2.cvtColor(edges_bin, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([d_vis, e_vis])
        cv2.imwrite(os.path.join(config.output_dir, '01_raw_input.png'), combined)

    # Stage 2: Buffer edges and compute ROI
    print("\n[Stage 2] Computing ROI from edges...")
    buf = buffer_edges(edges_bin, config.buffer_px)
    roi = compute_roi_from_mask(buf, config.roi_margin_px)
    if roi is None:
        raise RuntimeError("No ROI found")
    print(f"  ROI: {roi}")
    
    # Stage 3: Detect depth edges
    print("\n[Stage 3] Detecting depth edges...")
    x1, y1, x2, y2 = roi
    depth_roi_raw = depth[y1:y2+1, x1:x2+1]
        # Filter outliers: clip to 5-95 percentile of valid pixels
    depth_roi = depth_roi_raw.copy()
    valid_mask = (depth_roi > 0) & np.isfinite(depth_roi)
    
    dmin, dmax = 0.0, 1.0 # Default
    if valid_mask.any():
        valid_pixels = depth_roi[valid_mask]
        dmin = float(np.percentile(valid_pixels, 5))
        dmax = float(np.percentile(valid_pixels, 95))
        depth_roi[valid_mask] = np.clip(depth_roi[valid_mask], dmin, dmax)

    # Apply morphological smoothing
    k = max(1, int(round(config.buffer_px / 4)))
    ks = 5 * k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    d_filled = fill_invalid_with_median(depth_roi.astype(np.float32))
    
    mean = np.nanmedian(depth_roi)  # Diagonal depth of module
    std = np.nanstd(depth_roi)
    valid_mask = ((depth_roi < mean + 2 * std ) & (depth_roi > mean - 2 * std )).astype(np.uint8) * 255
    mask_eroded = cv2.erode(valid_mask, kernel, iterations=1)
    mask_smooth = cv2.dilate(mask_eroded, kernel, iterations=1)
    depth_roi_smooth = d_filled.copy()
    depth_roi_smooth[mask_smooth == 0] = 0
        # Remove outlier with standard deviation clipping inside the smoothed mask
    valid_smooth = (mask_smooth > 0)
    if valid_smooth.any():
        d_vals = depth_roi_smooth[valid_smooth]
        d_mean = float(np.mean(d_vals))
        d_std = float(np.std(d_vals))
        lower_bound = d_mean - 3.0 * d_std
        upper_bound = d_mean + 3.0 * d_std
        low_outliers = (depth_roi_smooth < lower_bound)
        high_outliers = (depth_roi_smooth > upper_bound)
        outlier_count = int((low_outliers).sum()) + int((high_outliers).sum())
        if outlier_count > 0:
            depth_roi_smooth[low_outliers] = d_mean + 3.0 * d_std
            depth_roi_smooth[high_outliers] = d_mean - 3.0 * d_std
            print(f'Removed {outlier_count} outlier pixels from smoothed depth using 2-sigma clipping.')

    grad, depth_edges = detect_depth_edges(depth_roi_smooth, config.method, 
                                          config.edge_thresh, config.edge_sigma,
                                          smooth=False)
    print(f"  Detected {int((depth_edges>0).sum())} edge pixels")
    
    # Stage 4: Filter edges by proximity
    loaded_edges_roi = (edges_bin[y1:y2+1, x1:x2+1] > 0).astype(np.uint8)
    depth_edges_filtered, dt = filter_depth_edges(depth_edges, loaded_edges_roi, config.max_edge_dist_px, config.buffer_px)
    
    print(f"  Kept {int((depth_edges_filtered>0).sum())} edge pixels")
    
    # Stage 5: Find closed loop
    print("\n[Stage 5] Finding closed loop...")
    best_loop, iterations, mosaic = find_closed_loop(depth_edges_filtered, depth_roi, config.buffer_px)
    if best_loop is None:
        raise RuntimeError("No closed loop found")
    print(f"  Found loop with {len(best_loop)} vertices after {iterations} iterations")
    
    if config.output_dir:
        if mosaic is not None:
            cv2.imwrite(os.path.join(config.output_dir, '05_closing_process.png'), mosaic)
        
        vis_loop = visualize_depth(depth_roi)
        # best_loop is in ROI coordinates
        cv2.drawContours(vis_loop, [best_loop], -1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(config.output_dir, '05_closed_loop.png'), vis_loop)

    # Stage 6: Extract boundary points
    print("\n[Stage 6] Extracting 3D points within boundary...")
    pts_pixel = extract_boundary_points(depth_roi, best_loop, roi)
    if pts_pixel is None:
        raise RuntimeError("Failed to extract boundary points")
    print(f"  Extracted {len(pts_pixel)} points")
    
    # Stage 7: Find camera intrinsics and convert to camera space
    print("\n[Stage 7] Converting to camera-space coordinates...")
    search_dirs = [config.raw_dir, config.input_dir, os.path.dirname(config.input_dir)]
    fx, fy, cx, cy, json_path = find_camera_intrinsics(search_dirs)
    if fx is None:
        raise RuntimeError("Could not find camera intrinsics")
    print(f"  Intrinsics: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    
    pts_camera = convert_to_camera_space(pts_pixel, fx, fy, cx, cy)
    print(f"  Converted to {len(pts_camera)} camera-space points")
    
    # Stage 8: Build point cloud and estimate parameters
    print("\n[Stage 8] Building point cloud and mesh...")
    if o3d is None:
        raise RuntimeError("Open3D required for 3D processing")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_camera)
    
    # Estimate spacing
    kdt = cKDTree(pts_camera)
    dists, _ = kdt.query(pts_camera, k=6)
    spacing = float(np.mean(dists[:, -1]))
    print(f"  Estimated spacing: {spacing:.6f} m")
    
    # Downsample
    if config.downsample_voxel > 0:
        pcd = pcd.voxel_down_sample(config.downsample_voxel)
        print(f"  Downsampled to {len(pcd.points)} points")
    
    # Remove outliers
    if config.remove_outliers:
        pcd, ind = pcd.remove_statistical_outlier(config.nb_neighbors, config.std_ratio)
        print(f"  After outlier removal: {len(pcd.points)} points")
    
    # Compute normals
    normal_radius = spacing * 2.0 if config.normal_radius is None else config.normal_radius
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(normal_radius))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # Build mesh
    bpa_radii = config.bpa_radii
    if bpa_radii is None:
        r = config.bpa_radius_factor * spacing
        bpa_radii = [r, 2*r, 4*r]
    
    mesh = build_mesh_from_pointcloud(pcd, bpa_radii)
    print(f"  Mesh: {len(mesh.triangles)} triangles, {len(mesh.vertices)} vertices")
    
    # Stage 9: Segment planes
    print("\n[Stage 9] Segmenting planes...")
    pts_all = np.asarray(mesh.vertices)
    plane_distance = max(spacing * 1.5, 1e-3)
    planes = segment_planes(pts_all, plane_distance, config.num_planes, 
                           config.ransac_n, config.num_iterations)
    print(f"  Found {len(planes)} planes")
    
    if config.output_dir and planes:
        # Project mesh vertices back to 2D for visualization (Step 9)
        pts_2d = project_points(pts_all, fx, fy, cx, cy)
        
        x1, y1, x2, y2 = roi
        # Create blank black image for planes
        vis_planes = np.zeros((depth[y1:y2+1, x1:x2+1].shape[0], depth[y1:y2+1, x1:x2+1].shape[1], 3), dtype=np.uint8)
        
        # Offset points by ROI top-left
        pts_2d_roi = pts_2d - np.array([x1, y1])
        
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        
        for i, plane in enumerate(planes):
            inliers = plane['inliers']
            plane_pts = pts_2d_roi[inliers].astype(np.int32)
            
            for pt in plane_pts:
                if 0 <= pt[0] < vis_planes.shape[1] and 0 <= pt[1] < vis_planes.shape[0]:
                    vis_planes[pt[1], pt[0]] = colors[i % len(colors)]
        
        cv2.imwrite(os.path.join(config.output_dir, '09_planes.png'), vis_planes)

    # Stage 10: Compute box corners
    print("\n[Stage 10] Computing box corners...")
    planes, box_corners = process_planes_and_corners(planes, pts_all, config.ortho_thresh_deg, config.min_inlier_ratio)
    print(f"  Computed {len(box_corners)} box corners")
    
    # Stage 11: Compare with ground truth
    print("\n[Stage 11] Comparing with ground truth...")
    gt_corners = find_ground_truth_corners(json_path)
    if gt_corners is None:
        print("  WARNING: No ground truth corners found")
        return None
    print(f"  Found {len(gt_corners)} ground truth corners")
    
    camera_position, camera_rotation_quat = get_camera_transform(json_path)
    box_corners_world = transform_to_world_space(box_corners, camera_position, 
                                                 camera_rotation_quat)
    
    best_match, all_matches = compare_corners(gt_corners, box_corners_world)
    
    # Calculate runtime
    end_time = time.time()
    runtime = end_time - start_time
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nRUNTIME: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    print("\nBEST MATCH:")
    print(f"  Ground Truth Corner {best_match['gt_idx']}: {best_match['gt_coord']}")
    print(f"  Detected Corner {best_match['detected_idx']}: {best_match['detected_coord']}")
    print(f"  Distance: {best_match['distance']:.6f} m")
    
    if camera_position is not None:
        cam_dist = np.linalg.norm(np.array(camera_position) - best_match['gt_coord'])
        print(f"  Camera to GT Corner Distance: {cam_dist:.6f} m")
        best_match['camera_to_gt_distance'] = cam_dist
    else:
        best_match['camera_to_gt_distance'] = None

    if config.output_dir and 'vis_planes' in locals():
        # Step 10 Visualization: Use plane image from Step 9 and add corners
        vis_corners = vis_planes.copy()
        x1, y1, x2, y2 = roi
        
        # Draw Detected Corners (Blue)
        if len(box_corners) > 0:
            corners_2d = project_points(box_corners, fx, fy, cx, cy)
            corners_2d_roi = corners_2d - np.array([x1, y1])
            for pt in corners_2d_roi.astype(np.int32):
                 cv2.circle(vis_corners, tuple(pt), 5, (255, 0, 0), -1) # Blue
                 cv2.circle(vis_corners, tuple(pt), 3, (0, 0, 0), -1)

        # Draw Ground Truth Corners (Green)
        if gt_corners is not None and camera_position is not None and camera_rotation_quat is not None:
            # Transform GT corners (World) -> Camera Space
            # P_world = R * P_cam + t  =>  P_cam = R.T * (P_world - t)
            qx, qy, qz, qw = camera_rotation_quat
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
            ])
            t = np.array(camera_position, dtype=float)
            
            gt_corners_cam = (R.T @ (gt_corners - t).T).T
            
            # Convert to CV Camera Space (Y-down) for projection
            gt_corners_cam[:, 1] *= -1
            
            gt_2d = project_points(gt_corners_cam, fx, fy, cx, cy)
            gt_2d_roi = gt_2d - np.array([x1, y1])
            
            for pt in gt_2d_roi.astype(np.int32):
                cv2.circle(vis_corners, tuple(pt), 5, (0, 255, 0), -1) # Green
                cv2.circle(vis_corners, tuple(pt), 3, (0, 0, 0), -1)
        
        cv2.imwrite(os.path.join(config.output_dir, '10_planes_corners.png'), vis_corners)

    print("\nSTATISTICS:")
    all_dists = [m['distance'] for m in all_matches]
    print(f"  Mean distance: {np.mean(all_dists):.6f} m")
    print(f"  Std distance: {np.std(all_dists):.6f} m")
    print(f"  Min distance: {np.min(all_dists):.6f} m")
    print(f"  Max distance: {np.max(all_dists):.6f} m")
    print("=" * 80)
    
    return best_match


def auto_detect_raw_dir(search_roots):
    """
    Step 0: Auto-detect the most recent raw directory.
    """
    def has_required_files(run_dir: str) -> bool:
        d_npy = os.path.join(run_dir, 'depth.npy')
        d_exr = os.path.join(run_dir, 'depth.exr')
        e_npy = os.path.join(run_dir, 'edges.npy')
        e_png = os.path.join(run_dir, 'edges.png')
        return (os.path.isfile(d_npy) or os.path.isfile(d_exr)) and \
               (os.path.isfile(e_npy) or os.path.isfile(e_png))
    
    candidates = []
    for root in search_roots:
        root_abs = os.path.abspath(root)
        if not os.path.isdir(root_abs):
            continue
        for dirpath, dirnames, filenames in os.walk(root_abs):
            if os.path.basename(dirpath).lower() == 'raw' and has_required_files(dirpath):
                try:
                    mtimes = [os.path.getmtime(os.path.join(dirpath, f)) for f in filenames]
                    score = max(mtimes) if mtimes else 0
                except Exception:
                    score = 0
                candidates.append((score, dirpath))
    
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None

def filter_depth_edges(depth_edges: np.ndarray, loaded_edges_roi: np.ndarray, max_edge_dist_px: float, buffer_px: int) -> np.ndarray:
    """
    Step 6.5: Filter depth edges by proximity to loaded edges and smooth.
    """
    inv = (1 - loaded_edges_roi).astype(np.uint8) * 255
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    valid = dt <= max_edge_dist_px
    depth_edges_filtered = ((depth_edges > 0) & valid).astype(np.uint8) * 255
    
    # Morphological closing to smooth filtered edges
    k = max(1, int(round(buffer_px / 4)))
    ks = 3 * k
    kernel = np.ones((ks, ks), np.uint8)
    depth_edges_filtered = cv2.morphologyEx(depth_edges_filtered, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return depth_edges_filtered, dt

def smooth_depth_roi(depth_roi: np.ndarray, buffer_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 5.5: Apply morphological smoothing to depth ROI.
    
    Returns:
        depth_roi_smooth: Smoothed depth map
        valid_mask: Initial valid mask
        mask_eroded: Eroded mask
        mask_smooth: Dilated (smoothed) mask
        roi_eroded: Eroded visualization (for debug)
        roi_dilated: Dilated visualization (for debug)
    """
    # Use a small elliptical kernel; scale lightly with buffer_px if available
    k = max(1, int(round(buffer_px / 4)))
    ks = 5 * k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

    d_filled = fill_invalid_with_median(depth_roi.astype(np.float32))
    
    # Create binary mask of valid/non-zero depth region
    mean = np.nanmedian(depth_roi)  # Diagonal depth of module
    std = np.nanstd(depth_roi)
    valid_mask = ((depth_roi < mean + 2 * std ) & (depth_roi > mean - 2 * std )).astype(np.uint8) * 255
    
    # Apply morphological closing to the mask (dilate then erode) to smooth boundary
    mask_eroded = cv2.erode(valid_mask, kernel, iterations=1)
    mask_smooth = cv2.dilate(mask_eroded, kernel, iterations=1)
    
    # Apply smoothed mask to filled depth
    depth_roi_smooth = d_filled.copy()
    depth_roi_smooth[mask_smooth == 0] = 0  # zero out regions outside smoothed mask

    # Remove outlier with standard deviation clipping inside the smoothed mask
    valid_smooth = (mask_smooth > 0)
    if valid_smooth.any():
        d_vals = depth_roi_smooth[valid_smooth]
        d_mean = float(np.mean(d_vals))
        d_std = float(np.std(d_vals))
        lower_bound = d_mean - 3.0 * d_std
        upper_bound = d_mean + 3.0 * d_std
        low_outliers = (depth_roi_smooth < lower_bound)
        high_outliers = (depth_roi_smooth > upper_bound)
        outlier_count = int((low_outliers).sum()) + int((high_outliers).sum())
        if outlier_count > 0:
            depth_roi_smooth[low_outliers] = d_mean + 3.0 * d_std
            depth_roi_smooth[high_outliers] = d_mean - 3.0 * d_std
            print(f'Removed {outlier_count} outlier pixels from smoothed depth using 2-sigma clipping.')
            
    return depth_roi_smooth, valid_mask, mask_eroded, mask_smooth
