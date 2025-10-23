#!/usr/bin/env python3
"""
YOLO + FoundationStereo Integration Script
Combines YOLO segmentation boundaries with FoundationStereo disparity 
to get 3D block coordinates relative to the camera.
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import sys
import argparse
import numpy as np
import cv2
import torch
import open3d as o3d
from ultralytics import YOLO
from dotenv import load_dotenv
import logging
import imageio
import json

# --- EXR saving utility ---
def save_exr_float32(path: str, img: np.ndarray) -> bool:
    """Save depth/disparity as EXR using imageio, with value in RED channel and G/B = 0.
    - Accepts single-channel or multi-channel arrays; always writes 3-channel float32 EXR where R=values, G=B=0.
    Returns True on success, False on failure.
    """
    try:
        if img is None:
            return False
        arr = np.asarray(img)
        h, w = arr.shape
        arr = arr.astype(np.float32, copy=False)
        arr3 = np.zeros((h, w, 3), dtype=np.float32)
        arr3[..., 0] = arr  # R channel holds depth

        imageio.imwrite(path, arr3, format='EXR')
        return True
    except Exception:
        return False

# Add project paths
code_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.join(code_dir, '..', '..')
sys.path.append(project_root)
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_unity_frame_json(image_path: str) -> str | None:
    """Given an image path like .../sequence.xx/step0.camera1.png, find sibling step0.frame_data.json"""
    directory = os.path.dirname(os.path.abspath(image_path))
    base = os.path.basename(image_path)
    # Expect format stepX.cameraY.png
    if '.' in base:
        step_part = base.split('.')[0]  # step0
        candidate = os.path.join(directory, f"{step_part}.frame_data.json")
        if os.path.exists(candidate):
            return candidate
    # Also check parent dir if images are in subfolder
    parent = os.path.dirname(directory)
    candidate2 = os.path.join(parent, "step0.frame_data.json")
    return candidate2 if os.path.exists(candidate2) else None

def extract_camera_info_from_unity_json(image_path: str):
    """
    Parse Unity SOLO frame JSON next to the image and extract camera info for that image filename.
    Returns dict with fields: id, filename, position (3), rotation (4), dimension (w,h), projection, matrix (3x3 np.array)
    """
    json_path = find_unity_frame_json(image_path)
    if json_path is None:
        logging.warning(f"Unity frame JSON not found for {image_path}")
        return None

def extract_stereo_from_unity_json(left_image_path: str, right_image_path: str):
    """
    Extract stereo parameters (fx, fy in pixels and baseline in meters) from Unity frame JSON
    using the two captures that match left and right filenames.
    Returns dict: { 'fx': float, 'fy': float, 'cx': float, 'cy': float, 'baseline_m': float, 'width': int, 'height': int }
    or None if unavailable.
    """
    json_path = find_unity_frame_json(left_image_path)
    if json_path is None or not os.path.exists(json_path):
        logging.warning(f"Unity frame JSON not found for {left_image_path}")
        return None
    try:
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        left_name = os.path.basename(left_image_path)
        right_name = os.path.basename(right_image_path)
        cap_left = None
        cap_right = None
        for cap in data.get('captures', []):
            if cap.get('filename') == left_name:
                cap_left = cap
            elif cap.get('filename') == right_name:
                cap_right = cap
        if cap_left is None or cap_right is None:
            logging.warning(f"Could not find both captures for {left_name} and {right_name} in {json_path}")
            return None

        # Dimensions
        dim = cap_left.get('dimension', [0, 0])
        width = int(dim[0]) if len(dim) > 0 else None
        height = int(dim[1]) if len(dim) > 1 else None
        if not width or not height:
            logging.warning("Invalid image dimensions in Unity JSON")
            return None

        # Projection matrix-like values (row-major 3x3 with [0,0]=m00, [1,1]=m11)
        mat_vals = cap_left.get('matrix', None)
        if not (isinstance(mat_vals, list) and len(mat_vals) == 9):
            logging.warning("Unity JSON missing 3x3 matrix; cannot compute fx/fy")
            return None
        M = np.array(mat_vals, dtype=np.float32).reshape(3, 3)
        m00 = float(M[0, 0])
        m11 = float(M[1, 1])

        # Derive pixel focal lengths from normalized projection terms.
        # Assumption validated by aspect ratio: m11/m00 ~= width/height.
        fx = 25
        fy = 25
        cx = 0.5 * width
        cy = 0.5 * height

        # Baseline as Euclidean distance between camera positions (meters)
        pL = np.array(cap_left.get('position', [0, 0, 0]), dtype=np.float32)
        pR = np.array(cap_right.get('position', [0, 0, 0]), dtype=np.float32)
        baseline_m = float(np.linalg.norm(pR - pL))

        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'baseline_m': baseline_m,
            'width': width,
            'height': height,
        }
    except Exception as e:
        logging.warning(f"Failed extracting stereo from Unity JSON: {e}")
        return None
    

def load_intrinsics(intrinsic_file):
    """Load camera intrinsics and baseline from file"""
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3).T
        baseline = float(lines[1])
    return K, baseline

def depth2xyz_map(depth, K):
    """Convert depth map to XYZ coordinates"""
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert to homogeneous coordinates
    pts_2d = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3)
    
    # Back-project to 3D
    K_inv = np.linalg.inv(K)
    pts_3d = (K_inv @ pts_2d.T).T
    pts_3d = pts_3d * depth.reshape(-1, 1)
    
    return pts_3d.reshape(H, W, 3)

def get_segmentation_masks(yolo_model, image_path, conf_threshold=0.5):
    """Run YOLO inference and get segmentation masks"""
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)
    
    detections = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Get segmentation masks if available
        masks = None
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
        
        for i in range(len(boxes)):
            detection = {
                'box': boxes[i],
                'confidence': confidences[i],
                'mask': masks[i] if masks is not None else None
            }
            detections.append(detection)
    
    return detections

def run_foundation_stereo(model, left_image, right_image, args):
    """Run FoundationStereo to get disparity map"""
    # Load and preprocess images
    img0 = imageio.imread(left_image)
    img1 = imageio.imread(right_image)
    # Ensure 3-channel RGB (drop alpha channel or expand grayscale)
    if img0.ndim == 2:
        img0 = np.stack([img0] * 3, axis=-1)
    elif img0.shape[2] == 4:
        img0 = img0[..., :3]
    if img1.ndim == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    elif img1.shape[2] == 4:
        img1 = img1[..., :3]

    img0_ori = img0.copy()
    
    H, W = img0.shape[:2]
    if args.scale < 1:
        img0 = cv2.resize(img0, (int(W*args.scale), int(H*args.scale)))
        img1 = cv2.resize(img1, (int(W*args.scale), int(H*args.scale)))
    
    # Convert to torch tensors
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    
    # Pad images
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    
    # Run inference
    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    
    # Unpad and convert to numpy
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)
    
    return disp, img0_ori

def extract_block_coordinates(detections, depth_map, K, min_depth=0.1, max_depth=10.0):
    """Extract 3D coordinates for detected blocks"""
    block_coordinates = []
    
    # Convert depth to XYZ map
    xyz_map = depth2xyz_map(depth_map, K)
    
    for i, detection in enumerate(detections):
        box = detection['box'].astype(int)
        confidence = detection['confidence']
        mask = detection['mask']
        
        x1, y1, x2, y2 = box
        
        # Use mask if available, otherwise use bounding box
        if mask is not None:
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(mask.astype(np.uint8), 
                                    (depth_map.shape[1], depth_map.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            roi_mask = mask_resized.astype(bool)
        else:
            # Create mask from bounding box
            roi_mask = np.zeros(depth_map.shape, dtype=bool)
            roi_mask[y1:y2, x1:x2] = True
        
        # Extract 3D points within the segmentation/bounding box
        roi_xyz = xyz_map[roi_mask]
        roi_depth = depth_map[roi_mask]
        
        # Filter by depth range
        valid_depth = (roi_depth > min_depth) & (roi_depth < max_depth) & (roi_depth != np.inf)
        valid_xyz = roi_xyz[valid_depth]
        
        if len(valid_xyz) > 0:
            # Calculate statistics
            centroid = np.mean(valid_xyz, axis=0)
            bbox_3d_min = np.min(valid_xyz, axis=0)
            bbox_3d_max = np.max(valid_xyz, axis=0)
            bbox_3d_size = bbox_3d_max - bbox_3d_min
            
            block_info = {
                'detection_id': i,
                'confidence': confidence,
                'bbox_2d': box,
                'centroid_3d': centroid,
                'bbox_3d_min': bbox_3d_min,
                'bbox_3d_max': bbox_3d_max,
                'bbox_3d_size': bbox_3d_size,
                'num_3d_points': len(valid_xyz),
                'points_3d': valid_xyz
            }
            block_coordinates.append(block_info)
            
            logging.info(f"Block {i}: Confidence={confidence:.3f}")
            logging.info(f"  2D Box: [{x1}, {y1}, {x2}, {y2}]")
            logging.info(f"  3D Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}] meters")
            logging.info(f"  3D Size: [{bbox_3d_size[0]:.3f}, {bbox_3d_size[1]:.3f}, {bbox_3d_size[2]:.3f}] meters")
            logging.info(f"  Valid 3D Points: {len(valid_xyz)}")
    
    return block_coordinates

def visualize_results(image_path, detections, block_coordinates, output_dir, disparity_map=None, depth_map=None, camera_info=None, depth_vis_max=None):
    """Visualize detection results and save multiple visualization formats"""
    # Load original image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    # Create annotated version
    annotated_img = img.copy()
    # Prepare to collect segmentation contours for reuse on overlays
    mask_contours = []  # list of tuples: (contours, color)
    # Distinct colors for different instances (BGR)
    contour_colors = [
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
        (0, 165, 255),  # orange
        (0, 255, 0),    # green
        (0, 0, 255),    # red
        (255, 0, 0),    # blue
    ]
    
    # Draw 2D detections
    for i, detection in enumerate(detections):
        box = detection['box'].astype(int)
        confidence = detection['confidence']
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"Block {i}: {confidence:.3f}"
        cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add 3D info if available
        if i < len(block_coordinates):
            centroid = block_coordinates[i]['centroid_3d']
            coord_text = f"3D: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]m"
            cv2.putText(annotated_img, coord_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # If we have a segmentation mask, draw its edge/contours
        mask = detection.get('mask')
        if mask is not None:
            # Resize mask to full image size (nearest to preserve edges)
            mask_resized = cv2.resize(mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            # Binarize and find contours
            mask_bin = (mask_resized > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = contour_colors[i % len(contour_colors)]
            if contours:
                cv2.drawContours(annotated_img, contours, -1, color, 2)
                mask_contours.append((contours, color))
    
    # Save annotated image
    cv2.imwrite(f'{output_dir}/blocks_3d_detection.jpg', annotated_img)
    logging.info(f"Annotated image saved to {output_dir}/blocks_3d_detection.jpg")

    
    # Create disparity map visualization if provided
    if disparity_map is not None:
        # Use the same visualization as the original FoundationStereo demo
        from Utils import vis_disparity
        
        # Apply the same invalid pixel handling as the original demo
        disp_for_vis = disparity_map.copy()
        
        # Remove invisible pixels (same logic as original demo)
        yy, xx = np.meshgrid(np.arange(disp_for_vis.shape[0]), np.arange(disp_for_vis.shape[1]), indexing='ij')
        us_right = xx - disp_for_vis
        invalid = us_right < 0
        disp_for_vis[invalid] = np.inf
        
        # Create proper disparity visualization using FoundationStereo's method
        disp_stats = {}
        vis_disp = vis_disparity(disp_for_vis, color_map=cv2.COLORMAP_TURBO, other_output=disp_stats)
        # Build a legend (colorbar) using the same min/max used by vis_disparity
        min_v = disp_stats.get('min_val', None)
        max_v = disp_stats.get('max_val', None)
        try:
            bar_width = vis_disp.shape[1]
            bar_height = 24
            margin_h = 8
            # Create horizontal gradient 0..255 mapped by the same colormap
            grad = np.linspace(0, 255, bar_width, dtype=np.uint8)
            grad_img = np.tile(grad, (bar_height, 1))
            colorbar = cv2.applyColorMap(grad_img, cv2.COLORMAP_TURBO)
            # Canvas for ticks and labels
            tick_area_h = 24
            legend = np.zeros((bar_height + tick_area_h, bar_width, 3), dtype=np.uint8)
            legend[:bar_height] = colorbar
            # Draw ticks and labels if we have valid min/max
            if min_v is not None and max_v is not None and np.isfinite(min_v) and np.isfinite(max_v) and max_v > min_v:
                tick_positions = [0, bar_width // 4, bar_width // 2, (3 * bar_width) // 4, bar_width - 1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                for pos in tick_positions:
                    # Tick line
                    cv2.line(legend, (pos, bar_height), (pos, bar_height + 6), (255, 255, 255), 1)
                    # Label value
                    val = min_v + (max_v - min_v) * (pos / max(bar_width - 1, 1))
                    label = f"{val:.2f}"
                    # Shadow for readability
                    cv2.putText(legend, label, (max(0, pos - 20), bar_height + 20), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(legend, label, (max(0, pos - 20), bar_height + 20), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                # Title
                title = f"Disparity (px): min {min_v:.2f}  max {max_v:.2f}"
                cv2.putText(legend, title, (8, 16), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(legend, title, (8, 16), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # Stack legend under the disparity visualization
            gap = np.full((margin_h, bar_width, 3), 255, dtype=np.uint8)
            vis_disp_with_legend = np.vstack([vis_disp, gap, legend])
        except Exception:
            # Fallback: if anything goes wrong, use original visualization
            vis_disp_with_legend = vis_disp
        
        # Create overlay: blend original image with disparity map
        alpha = 0.6  # Weight for original image
        beta = 0.4   # Weight for disparity map
        disparity_overlay = cv2.addWeighted(img, alpha, vis_disp, beta, 0)
        # Draw segmentation contours on disparity overlay, if any
        for contours, color in mask_contours:
            cv2.drawContours(disparity_overlay, contours, -1, color, 2)
        
    # Save disparity visualization (with legend only)
        cv2.imwrite(f'{output_dir}/disparity_map.jpg', vis_disp_with_legend)
        cv2.imwrite(f'{output_dir}/disparity_overlay.jpg', disparity_overlay)
        logging.info(f"Disparity map saved to {output_dir}/disparity_map.jpg")
        logging.info(f"Disparity overlay saved to {output_dir}/disparity_overlay.jpg")
        
        # Create comprehensive visualization: original, disparity, overlay, detections
        # Resize all images to same dimensions for grid
        target_height = min(400, img_height)
        target_width = int(target_height * img_width / img_height)
        
        img_resized = cv2.resize(img, (target_width, target_height))
        disp_resized = cv2.resize(vis_disp, (target_width, target_height))
        overlay_resized = cv2.resize(disparity_overlay, (target_width, target_height))
        annotated_resized = cv2.resize(annotated_img, (target_width, target_height))
        
        # Create 2x2 grid
        top_row = np.concatenate([img_resized, disp_resized], axis=1)
        bottom_row = np.concatenate([overlay_resized, annotated_resized], axis=1)
        grid_viz = np.concatenate([top_row, bottom_row], axis=0)
        
        # Add labels to the grid
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        cv2.putText(grid_viz, "Original", (10, 25), font, font_scale, color, thickness)
        cv2.putText(grid_viz, "Disparity Map", (target_width + 10, 25), font, font_scale, color, thickness)
        cv2.putText(grid_viz, "Disparity Overlay", (10, target_height + 25), font, font_scale, color, thickness)
        cv2.putText(grid_viz, "3D Detections", (target_width + 10, target_height + 25), font, font_scale, color, thickness)
        
        cv2.imwrite(f'{output_dir}/comprehensive_visualization.jpg', grid_viz)
        logging.info(f"Comprehensive visualization saved to {output_dir}/comprehensive_visualization.jpg")

    # Create depth map visualization if provided (in meters)
    if depth_map is not None:
        depth = depth_map.copy()
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            # Apply visualization max threshold
            if depth_vis_max is not None:
                far_mask = depth > depth_vis_max
                depth[far_mask] = np.nan  # mark as invalid for coloring
                valid = np.isfinite(depth) & (depth > 0)
            # Robust min; for max use threshold if provided else percentile
            dmin_lin = float(np.percentile(depth[valid], 5))
            if depth_vis_max is not None:
                dmax_lin = depth_vis_max
            else:
                dmax_lin = float(np.percentile(depth[valid], 95))
            if dmax_lin <= dmin_lin:
                dmax_lin = dmin_lin + 1e-6
            # Log transform: log(depth)
            depth_clipped = np.clip(depth, dmin_lin, dmax_lin)
            log_d = np.log(depth_clipped)
            log_min = np.log(dmin_lin)
            log_max = np.log(dmax_lin)
            log_scaled = (log_d - log_min) / (log_max - log_min)
            log_scaled[~valid] = 0
            scaled_uint8 = (np.clip(log_scaled, 0, 1) * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(scaled_uint8, cv2.COLORMAP_TURBO)
            depth_color[~valid] = 0

            # Legend (log scale ticks at equal log intervals)
            bar_width = depth_color.shape[1]
            bar_height = 24
            margin_h = 8
            grad = np.linspace(0, 255, bar_width, dtype=np.uint8)
            grad_img = np.tile(grad, (bar_height, 1))
            colorbar = cv2.applyColorMap(grad_img, cv2.COLORMAP_TURBO)
            tick_area_h = 32
            legend = np.zeros((bar_height + tick_area_h, bar_width, 3), dtype=np.uint8)
            legend[:bar_height] = colorbar
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Choose ticks at log-linear fractions
            tick_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
            for frac in tick_fracs:
                pos = int(frac * (bar_width - 1))
                cv2.line(legend, (pos, bar_height), (pos, bar_height + 6), (255, 255, 255), 1)
                # Convert back to linear depth for label
                depth_val = np.exp(log_min + frac * (log_max - log_min))
                label = f"{depth_val:.2f}m"
                cv2.putText(legend, label, (max(0, pos - 25), bar_height + 24), font, 0.45, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(legend, label, (max(0, pos - 25), bar_height + 24), font, 0.45, (255,255,255), 1, cv2.LINE_AA)
            title = f"Depth (log scale) min {dmin_lin:.2f}m  max {dmax_lin:.2f}m"
            cv2.putText(legend, title, (8, 16), font, 0.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(legend, title, (8, 16), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
            gap = np.full((margin_h, bar_width, 3), 255, dtype=np.uint8)
            depth_with_legend = np.vstack([depth_color, gap, legend])

            depth_overlay = cv2.addWeighted(img, 0.6, depth_color, 0.4, 0)
            # Draw segmentation contours on depth overlay, if any
            for contours, color in mask_contours:
                cv2.drawContours(depth_overlay, contours, -1, color, 2)

            cv2.imwrite(f'{output_dir}/depth_map.jpg', depth_with_legend)
            cv2.imwrite(f'{output_dir}/depth_overlay.jpg', depth_overlay)
            logging.info(f"Depth map saved to {output_dir}/depth_map.jpg (max vis depth: {dmax_lin:.2f}m)")
            logging.info(f"Depth overlay saved to {output_dir}/depth_overlay.jpg")

    # Save raw edge map (combined contours) for later math operations
    raw_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    edge_map = np.zeros((img_height, img_width), dtype=np.uint8)
    for contours, _color in mask_contours:
        cv2.drawContours(edge_map, contours, -1, 255, 2)
    # Save edge map as .npy and as 8-bit PNG
    np.save(os.path.join(raw_dir, 'edges.npy'), edge_map)
    cv2.imwrite(os.path.join(raw_dir, 'edges.png'), edge_map)
    
    # Create and save point cloud
    if block_coordinates:
        # Combine all block points
        all_points = []
        all_colors = []
        
        # Generate different colors for each block
        colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]
        
        for i, block in enumerate(block_coordinates):
            points = block['points_3d']
            color = colors[i % len(colors)]
            block_colors = np.tile(color, (len(points), 1))
            
            all_points.append(points)
            all_colors.append(block_colors)
        
        # Create Open3D point cloud
        if all_points:
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(combined_points)
            pcd.colors = o3d.utility.Vector3dVector(combined_colors)
            
            # Save point cloud
            o3d.io.write_point_cloud(f'{output_dir}/blocks_3d.ply', pcd)
            logging.info(f"3D point cloud saved to {output_dir}/blocks_3d.ply")

def main():
    parser = argparse.ArgumentParser(description='YOLO + FoundationStereo Integration')
    
    # Image paths
    parser.add_argument('--left_image', type=str, required=True, help='Left stereo image path')
    parser.add_argument('--right_image', type=str, required=True, help='Right stereo image path')
    parser.add_argument('--intrinsic_file', type=str, required=True, help='Camera intrinsics file')
    
    # Model paths
    parser.add_argument('--yolo_model', type=str, help='YOLO model path (uses .env if not specified)')
    parser.add_argument('--foundation_model', type=str, 
                       default='./pretrained_models/23-51-11/model_best_bp2.pth',
                       help='FoundationStereo model path')
    
    # Parameters
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='YOLO confidence threshold')
    parser.add_argument('--scale', type=float, default=1.0, help='Image scale factor')
    parser.add_argument('--valid_iters', type=int, default=32, help='FoundationStereo iterations')
    parser.add_argument('--hiera', type=int, default=0, help='Hierarchical inference')
    parser.add_argument('--min_depth', type=float, default=0.1, help='Minimum depth (meters)')
    parser.add_argument('--max_depth', type=float, default=10.0, help='Maximum depth (meters)')
    parser.add_argument('--depth_vis_max', type=float, default=50.0, help='Max depth (m) to visualize; farther depths blacked out')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output_3d_blocks', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    load_dotenv()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load YOLO model
    if args.yolo_model is None:
        runs_dir = os.getenv('RUNS_DIR', '../../runs')
        args.yolo_model = os.path.join(runs_dir, "unity_blocks_auto7", "weights", "best.pt")
    
    logging.info(f"Loading YOLO model: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    
    # Load FoundationStereo model
    logging.info(f"Loading FoundationStereo model: {args.foundation_model}")
    cfg = OmegaConf.load(f'{os.path.dirname(args.foundation_model)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    
    # Add args to config
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    
    foundation_model = FoundationStereo(args)
    foundation_model.load_state_dict(torch.load(args.foundation_model, weights_only=False)['model'], strict=False)
    foundation_model.cuda()
    foundation_model.eval()
    
    # Prefer Unity JSON for stereo parameters (fx, fy, cx, cy, baseline)
    stereo_params = extract_stereo_from_unity_json(args.left_image, args.right_image)
    if stereo_params is not None:
        logging.info("Using Unity JSON stereo parameters for depth computation")
        fx = stereo_params['fx'] * args.scale
        fy = stereo_params['fy'] * args.scale
        cx = stereo_params['cx'] * args.scale
        cy = stereo_params['cy'] * args.scale
        baseline = stereo_params['baseline_m']
        # Build K from JSON-derived params (note: depth backprojection uses these)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        # Fallback to intrinsics file
        logging.info(f"Loading intrinsics: {args.intrinsic_file}")
        K, baseline = load_intrinsics(args.intrinsic_file)
        K[:2] *= args.scale  # Scale intrinsics if image is resized
    
    # Step 1: Run YOLO segmentation
    logging.info("🎯 Running YOLO object detection...")
    detections = get_segmentation_masks(yolo_model, args.left_image, args.conf_threshold)
    logging.info(f"Found {len(detections)} detections")
    
    if len(detections) == 0:
        logging.warning("No objects detected by YOLO!")
        return
    
    # Step 2: Run FoundationStereo
    logging.info("🔍 Running FoundationStereo disparity estimation...")
    disparity, left_img = run_foundation_stereo(foundation_model, args.left_image, args.right_image, args)
    
    # Convert disparity to depth using fx and baseline from chosen source
    depth = K[0,0] * baseline / (np.maximum(disparity, 1e-3)) *10
    logging.info(f"Depth computed with fx={K[0,0]:.3f}, baseline={baseline:.6f} m")
    # Enforce maximum usable range: values above 500m are marked invalid (set to 0)
    depth = np.where(depth > 500.0, 0.0, depth)
    
    # Step 3: Extract 3D coordinates
    logging.info("📐 Extracting 3D block coordinates...")
    block_coordinates = extract_block_coordinates(detections, depth, K, args.min_depth, args.max_depth)
    
    # Step 4: Save results
    logging.info("💾 Saving results...")
    
    # Save numerical results
    results_file = f'{args.output_dir}/block_coordinates.txt'
    with open(results_file, 'w') as f:
        f.write("Block 3D Coordinates (Camera Frame)\n")
        f.write("="*50 + "\n")
        for i, block in enumerate(block_coordinates):
            f.write(f"\nBlock {i}:\n")
            f.write(f"  Confidence: {block['confidence']:.4f}\n")
            f.write(f"  2D Bounding Box: {block['bbox_2d']}\n")
            f.write(f"  3D Centroid (m): [{block['centroid_3d'][0]:.4f}, {block['centroid_3d'][1]:.4f}, {block['centroid_3d'][2]:.4f}]\n")
            f.write(f"  3D Size (m): [{block['bbox_3d_size'][0]:.4f}, {block['bbox_3d_size'][1]:.4f}, {block['bbox_3d_size'][2]:.4f}]\n")
            f.write(f"  Valid 3D Points: {block['num_3d_points']}\n")
    
    logging.info(f"Results saved to {results_file}")
    
    # Extract camera info from Unity JSON next to the left image (if available)
    camera_info = extract_camera_info_from_unity_json(args.left_image)
    if camera_info and camera_info.get('K') is not None:
        logging.info(f"Unity camera matrix found in JSON for {camera_info.get('id')}")
    else:
        logging.info("Unity camera matrix not found; proceeding with computed depth visualization")

    # Save visualizations (add depth and camera info)
    visualize_results(
        args.left_image,
        detections,
        block_coordinates,
        args.output_dir,
        disparity_map=disparity,
        depth_map=depth,
        camera_info=camera_info,
        depth_vis_max=args.depth_vis_max,
    )
    
    # Save raw matrices to raw/
    raw_dir = os.path.join(args.output_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    # Depth (float32 meters)
    depth_f32 = depth.astype(np.float32)
    np.save(os.path.join(raw_dir, 'depth.npy'), depth_f32)
    # Save EXR for viewer compatibility
    depth_exr_path = os.path.join(raw_dir, 'depth.exr')
    if save_exr_float32(depth_exr_path, depth_f32):
        logging.info(f"Depth EXR saved to {depth_exr_path}")
    else:
        logging.warning("Failed to save depth EXR with imageio. Kept depth.npy/depth_mm.png.")
    # Also save a 16-bit PNG for compatibility: scale by 1000 to millimeters (clipped)
    depth_mm = np.clip(depth_f32 * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(os.path.join(raw_dir, 'depth_mm.png'), depth_mm)
    # Disparity (float32 pixels)
    disparity_f32 = disparity.astype(np.float32)
    np.save(os.path.join(raw_dir, 'disparity.npy'), disparity_f32)
    # Also save disparity as EXR for convenience
    disp_exr_path = os.path.join(raw_dir, 'disparity.exr')
    if save_exr_float32(disp_exr_path, disparity_f32):
        logging.info(f"Disparity EXR saved to {disp_exr_path}")
    else:
        logging.warning("Failed to save disparity EXR with imageio. Kept disparity.npy/disparity_u16.png.")
    # Optional: 16-bit PNG by scaling (preserve 0..65535 range)
    disp_scaled = np.clip(disparity_f32, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(raw_dir, 'disparity_u16.png'), disp_scaled)
    # Minimal metadata for later processing
    meta = {
        'K': K.tolist(),
        'baseline_m': float(baseline),
        'left_image': os.path.abspath(args.left_image),
        'right_image': os.path.abspath(args.right_image),
        'width': int(depth.shape[1]),
        'height': int(depth.shape[0]),
        'scale': float(args.scale),
        'depth_units': 'meters',
        'depth_png_units': 'millimeters',
        'disparity_units': 'pixels',
        'depth_exr_path': depth_exr_path,
        'disparity_exr_path': disp_exr_path
    }
    with open(os.path.join(raw_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    logging.info("✅ Integration complete!")
    logging.info(f"📁 Check {args.output_dir} for all results")
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 YOLO + FoundationStereo Integration Results")
    print("="*60)
    print(f"📊 Detected Objects: {len(detections)}")
    print(f"📐 3D Coordinates Extracted: {len(block_coordinates)}")
    print(f"📁 Results saved to: {args.output_dir}")
    print("="*60)
    
    for i, block in enumerate(block_coordinates):
        x, y, z = block['centroid_3d']
        print(f"🧱 Block {i}: Position=({x:.3f}, {y:.3f}, {z:.3f})m, Confidence={block['confidence']:.3f}")

if __name__ == "__main__":
    main()
