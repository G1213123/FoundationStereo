#!/usr/bin/env python3
"""
Depth Map Viewer for YOLO + FoundationStereo Integration Results
Visualizes the depth maps generated from disparity estimation
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import glob

def visualize_depth_map(depth_file, save_visualization=True):
    """Load and visualize a depth map file"""
    print(f"📂 Loading depth map: {depth_file}")
    
    if not os.path.exists(depth_file):
        print(f"❌ File not found: {depth_file}")
        return False
    
    try:
        # Load depth map
        depth = np.load(depth_file)
        print(f"✅ Loaded depth map: {depth.shape}")
        
        # Remove infinite values for statistics
        finite_depth = depth[np.isfinite(depth)]
        if len(finite_depth) == 0:
            print("❌ No finite depth values found!")
            return False
        
        # Calculate depth range
        depth_min, depth_max = finite_depth.min(), finite_depth.max()
        depth_range = depth_max - depth_min
        
        print(f"📊 Depth statistics:")
        print(f"   Shape: {depth.shape}")
        print(f"   Min depth: {depth_min:.6f}m")
        print(f"   Max depth: {depth_max:.6f}m")
        print(f"   Mean depth: {finite_depth.mean():.6f}m")
        print(f"   Depth range: {depth_range:.6f}m")
        print(f"   Finite pixels: {len(finite_depth)}/{depth.size} ({100*len(finite_depth)/depth.size:.1f}%)")
        
        # Check for very narrow depth range (likely indicates scale issues)
        if depth_range < 0.1:
            print(f"⚠️  WARNING: Very narrow depth range ({depth_range:.6f}m) - possible scale issue!")
            print(f"   Expected depth range for real scenes: 1-500m")
            print(f"   Current range suggests depth values may need scaling")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Depth Map Analysis - {os.path.basename(depth_file)}', fontsize=16)
        
        # 1. Raw depth map
        ax1 = axes[0, 0]
        depth_vis = depth.copy()
        depth_vis[~np.isfinite(depth_vis)] = 0  # Set invalid to 0 for display
        im1 = ax1.imshow(depth_vis, cmap='plasma', vmin=0, vmax=np.percentile(finite_depth, 95))
        ax1.set_title('Raw Depth Map')
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Height (pixels)')
        plt.colorbar(im1, ax=ax1, label='Depth (meters)')
        
        # 2. Depth map with better contrast
        ax2 = axes[0, 1]
        # Use percentile-based clipping for better visualization
        p5, p95 = np.percentile(finite_depth, [5, 95])
        depth_clipped = np.clip(depth_vis, p5, p95)
        im2 = ax2.imshow(depth_clipped, cmap='viridis')
        ax2.set_title(f'Depth Map (Clipped: {p5:.2f}-{p95:.2f}m)')
        ax2.set_xlabel('Width (pixels)')
        ax2.set_ylabel('Height (pixels)')
        plt.colorbar(im2, ax=ax2, label='Depth (meters)')
        
        # 3. Depth histogram
        ax3 = axes[1, 0]
        ax3.hist(finite_depth, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlabel('Depth (meters)')
        ax3.set_ylabel('Number of pixels')
        ax3.set_title('Depth Distribution')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(finite_depth.mean(), color='red', linestyle='--', label=f'Mean: {finite_depth.mean():.3f}m')
        ax3.legend()
        
        # 4. Valid depth mask
        ax4 = axes[1, 1]
        valid_mask = np.isfinite(depth).astype(np.uint8) * 255
        ax4.imshow(valid_mask, cmap='gray')
        ax4.set_title(f'Valid Depth Mask ({100*len(finite_depth)/depth.size:.1f}% valid)')
        ax4.set_xlabel('Width (pixels)')
        ax4.set_ylabel('Height (pixels)')
        
        plt.tight_layout()
        
        if save_visualization:
            # Save visualization
            output_dir = os.path.dirname(depth_file)
            vis_file = os.path.join(output_dir, 'depth_map_visualization.png')
            plt.savefig(vis_file, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to: {vis_file}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading depth map: {e}")
        return False

def list_available_depth_maps():
    """List all available depth map files"""
    depth_pattern = "./output_3d_blocks/**/depth_map.npy"
    depth_files = glob.glob(depth_pattern, recursive=True)
    
    if not depth_files:
        print("❌ No depth map files found!")
        print("💡 Run the integration first: python scripts/unity_yolo/run_3d_integration.py")
        return []
    
    print("📁 Available depth map files:")
    for i, depth_file in enumerate(depth_files):
        # Extract sequence info from path
        path_parts = depth_file.split(os.sep)
        sequence_info = path_parts[-2] if len(path_parts) > 1 else "unknown"
        print(f"   {i+1}: {sequence_info} -> {depth_file}")
    
    return depth_files

def main():
    parser = argparse.ArgumentParser(description='Depth Map Viewer')
    parser.add_argument('--file', type=str, help='Specific depth map file to view')
    parser.add_argument('--sequence', type=str, help='View specific sequence (e.g., solo_9_sequence.0_step0)')
    parser.add_argument('--list', action='store_true', help='List all available depth maps')
    parser.add_argument('--no_save', action='store_true', help='Do not save visualization to file')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_depth_maps()
        return
    
    save_vis = not args.no_save
    
    if args.file:
        # View specific file
        visualize_depth_map(args.file, save_vis)
    elif args.sequence:
        # View specific sequence
        depth_file = f"./output_3d_blocks/{args.sequence}/depth_map.npy"
        visualize_depth_map(depth_file, save_vis)
    else:
        # Interactive mode - show all available files
        depth_files = list_available_depth_maps()
        
        if not depth_files:
            return
        
        if len(depth_files) == 1:
            print(f"\n🎯 Opening the only available file...")
            visualize_depth_map(depth_files[0], save_vis)
        else:
            print(f"\n🔢 Select a file to view (1-{len(depth_files)}):")
            try:
                choice = int(input("Enter number: ")) - 1
                if 0 <= choice < len(depth_files):
                    visualize_depth_map(depth_files[choice], save_vis)
                else:
                    print("❌ Invalid choice!")
            except (ValueError, KeyboardInterrupt):
                print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
