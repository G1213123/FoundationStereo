#!/usr/bin/env python3
"""
YOLO + FoundationStereo Integration Script
Combines YOLO segmentation boundaries with FoundationStereo disparity 
to get 3D block coordinates relative to the camera.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import open3d as o3d
from ultralytics import YOLO
from dotenv import load_dotenv
import logging

# Add project paths
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
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

def load_intrinsics(intrinsic_file):
    """Load camera intrinsics and baseline from file"""
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
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
    img0 = cv2.imread(left_image)
    img1 = cv2.imread(right_image)
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

def visualize_results(image_path, detections, block_coordinates, output_dir):
    """Visualize detection results and save point cloud"""
    # Load original image
    img = cv2.imread(image_path)
    
    # Draw 2D detections
    for i, detection in enumerate(detections):
        box = detection['box'].astype(int)
        confidence = detection['confidence']
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"Block {i}: {confidence:.3f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add 3D info if available
        if i < len(block_coordinates):
            centroid = block_coordinates[i]['centroid_3d']
            coord_text = f"3D: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]m"
            cv2.putText(img, coord_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Save annotated image
    cv2.imwrite(f'{output_dir}/blocks_3d_detection.jpg', img)
    logging.info(f"Annotated image saved to {output_dir}/blocks_3d_detection.jpg")
    
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
                       default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth',
                       help='FoundationStereo model path')
    
    # Parameters
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='YOLO confidence threshold')
    parser.add_argument('--scale', type=float, default=1.0, help='Image scale factor')
    parser.add_argument('--valid_iters', type=int, default=32, help='FoundationStereo iterations')
    parser.add_argument('--hiera', type=int, default=0, help='Hierarchical inference')
    parser.add_argument('--min_depth', type=float, default=0.1, help='Minimum depth (meters)')
    parser.add_argument('--max_depth', type=float, default=10.0, help='Maximum depth (meters)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output_3d_blocks', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    load_dotenv()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load YOLO model
    if args.yolo_model is None:
        runs_dir = os.getenv('RUNS_DIR', './runs')
        args.yolo_model = os.path.join(runs_dir, "unity_blocks_auto6", "weights", "best.pt")
    
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
    
    foundation_model = torch.nn.DataParallel(FoundationStereo(args), device_ids=[0])
    foundation_model.load_state_dict(torch.load(args.foundation_model)['model'], strict=False)
    foundation_model = foundation_model.module
    foundation_model.cuda()
    foundation_model.eval()
    
    # Load camera intrinsics
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
    
    # Convert disparity to depth
    depth = K[0,0] * baseline / (disparity + 1e-8)  # Add small epsilon to avoid division by zero
    
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
    
    # Save visualizations
    visualize_results(args.left_image, detections, block_coordinates, args.output_dir)
    
    # Save depth map
    np.save(f'{args.output_dir}/depth_map.npy', depth)
    
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
