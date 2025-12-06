#!/usr/bin/env python3
"""
Corner Detection Pipeline
Combines depth edge analysis and 3D edge extraction to detect box corners.
Outputs only the final best match corner and distance comparison with ground truth.
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
    max_edge_dist_px = 10
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
    """Load depth and edges from raw_dir."""
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
    """Apply morphological dilation to buffer edges."""
    if buffer_px <= 0:
        return edges_bin.copy()
    k = 2 * buffer_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(edges_bin, kernel, iterations=1)


def compute_roi_from_mask(mask: np.ndarray, margin: int = 0):
    """Compute bounding box ROI from mask."""
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
    """Detect depth edges using specified method."""
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


def find_closed_loop(depth_edges_filtered: np.ndarray, depth_roi: np.ndarray, 
                    buffer_px: int):
    """Find closed loop using iterative morphological closing."""
    mask = (depth_edges_filtered > 0).astype(np.uint8) * 255
    
    roi_h, roi_w = depth_roi.shape[:2]
    min_dimension_threshold = roi_h * roi_w * 0.3 
    
    epsilon_frac = 0.005
    min_vertices = 4
    max_iterations = 20
    
    best_loop = None
    working_mask = mask.copy()
    iteration = 0
    
    while best_loop is None and iteration <= max_iterations:
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
            largest_contour = max(large_closed_contours, key=cv2.contourArea)
            peri = float(cv2.arcLength(largest_contour, True))
            if peri > 0:
                eps = float(epsilon_frac * peri)
                approx = cv2.approxPolyDP(largest_contour, eps, True)
                if approx is not None and len(approx) >= min_vertices:
                    best_loop = approx
                    break
        
        if iteration < max_iterations:
            ks = 2*iteration + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
            working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            iteration += 1
        else:
            break
    
    return best_loop


def extract_boundary_points(depth_roi: np.ndarray, best_loop, roi):
    """Extract 3D points within boundary polygon."""
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
    """Find and load camera intrinsics from step0.frame_data.json."""
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
    """Convert pixel coordinates to camera-space metric coordinates."""
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
    """Compute 8 box corners from 3 planes using percentile-based far planes."""
    if len(plane_models) < 3:
        raise ValueError("Need at least 3 planes")
    
    Ns = np.stack([p['normal'] for p in plane_models], axis=0)
    s0 = np.array([-p['d'] for p in plane_models], dtype=float)
    
    # Compute far parallel plane offsets
    s_pairs = []
    for i in range(3):
        n = Ns[i]
        s_all = pts_all.dot(n)
        q05, q95 = np.quantile(s_all, [0.05, 0.95])
        s_far = q05 if abs(q05 - s0[i]) > abs(q95 - s0[i]) else q95
        if abs(s_far - s0[i]) < 1e-3:
            s_far = float(s_all.min()) if abs(s_all.min() - s0[i]) > abs(s_all.max() - s0[i]) else float(s_all.max())
        s_pairs.append([float(s0[i]), float(s_far)])
    
    try:
        Ns_inv = np.linalg.inv(Ns)
    except np.linalg.LinAlgError:
        Ns_inv = np.linalg.pinv(Ns)
    
    import itertools
    corner_keys = list(itertools.product([0, 1], repeat=3))
    corners = []
    for key in corner_keys:
        s_vec = np.array([s_pairs[i][key[i]] for i in range(3)], dtype=float)
        x = Ns_inv.dot(s_vec)
        corners.append(x)
    
    return np.array(corners, dtype=float)


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
        if len(corner_positions) >= 4:
            break
    
    if len(corner_positions) >= 4:
        return np.array(corner_positions[:4], dtype=float)
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
    
    qx, qy, qz, qw = camera_rotation_quat
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    t = np.array(camera_position, dtype=float)
    
    return (R @ corners_camera.T).T + t


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
    
    # Stage 1: Find and load raw data
    print("\n[Stage 1] Loading raw depth and edge data...")
    if config.raw_dir is None:
        raise ValueError("raw_dir must be specified in config")
    
    depth, edges_bin = load_raw(config.raw_dir, config.filter_top_instance)
    print(f"  Depth: {depth.shape}, Edges: {int((edges_bin>0).sum())} pixels")
    
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
    print("\n[Stage 4] Filtering edges by proximity to loaded edges...")
    loaded_edges_roi = (edges_bin[y1:y2+1, x1:x2+1] > 0).astype(np.uint8)
    inv = (1 - loaded_edges_roi).astype(np.uint8) * 255
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    valid = dt <= config.max_edge_dist_px
    depth_edges_filtered = ((depth_edges > 0) & valid).astype(np.uint8) * 255
    
    # Morphological closing to smooth filtered edges
    k = max(1, int(round(config.buffer_px / 4)))
    ks = 3 * k
    kernel = np.ones((ks, ks), np.uint8)
    depth_edges_filtered = cv2.morphologyEx(depth_edges_filtered, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    print(f"  Kept {int((depth_edges_filtered>0).sum())} edge pixels")
    
    # Stage 5: Find closed loop
    print("\n[Stage 5] Finding closed loop...")
    best_loop = find_closed_loop(depth_edges_filtered, depth_roi, config.buffer_px)
    if best_loop is None:
        raise RuntimeError("No closed loop found")
    print(f"  Found loop with {len(best_loop)} vertices")
    
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
    
    # Stage 10: Compute box corners
    print("\n[Stage 10] Computing box corners...")
    box_corners = compute_box_corners(planes, pts_all)
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
    
    print("\nSTATISTICS:")
    all_dists = [m['distance'] for m in all_matches]
    print(f"  Mean distance: {np.mean(all_dists):.6f} m")
    print(f"  Std distance: {np.std(all_dists):.6f} m")
    print(f"  Min distance: {np.min(all_dists):.6f} m")
    print(f"  Max distance: {np.max(all_dists):.6f} m")
    print("=" * 80)
    
    return best_match


def auto_detect_raw_dir(search_roots):
    """Auto-detect the most recent raw directory."""
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


def main():
    """Main entry point."""
    config = Config()
    
    # Auto-detect raw_dir if not specified
    if config.raw_dir is None:
        search_roots = [
            './scripts/run_files/batch_outputs',
        ]
        config.raw_dir = auto_detect_raw_dir(search_roots)
        if config.raw_dir is None:
            print("ERROR: Could not auto-detect raw_dir. Please specify it in the Config class.")
            return 1
        print(f"Auto-detected raw_dir: {config.raw_dir}\n")
    
    try:
        result = run_pipeline(config)
        return 0 if result is not None else 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
