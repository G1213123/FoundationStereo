#!/usr/bin/env python3
"""
3D Point Cloud Viewer for YOLO + FoundationStereo Integration Results
Opens and visualizes the generated block point clouds
"""

import open3d as o3d
import os
import argparse
import glob

def view_point_cloud(ply_file):
    """Load and visualize a point cloud file"""
    print(f"📂 Loading point cloud: {ply_file}")
    
    # Check if file exists
    if not os.path.exists(ply_file):
        print(f"❌ File not found: {ply_file}")
        return False
    
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_file)
        
        # Check if point cloud has points
        if len(pcd.points) == 0:
            print(f"❌ Point cloud is empty: {ply_file}")
            return False
        
        print(f"✅ Loaded {len(pcd.points)} points")
        
        # Print point cloud info
        print(f"📊 Point cloud bounds:")
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        print(f"   Min: [{min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}]")
        print(f"   Max: [{max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f}]")
        
        # Visualize
        print("🎨 Opening 3D viewer...")
        print("💡 Controls:")
        print("   - Mouse: Rotate/Pan/Zoom")
        print("   - ESC: Exit")
        print("   - R: Reset view")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"3D Blocks - {os.path.basename(ply_file)}")
        vis.add_geometry(pcd)
        
        # Set point size for better visibility
        opt = vis.get_render_option()
        opt.point_size = 2.0
        
        # Run visualizer
        vis.run()
        vis.destroy_window()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading point cloud: {e}")
        return False

def list_available_clouds():
    """List all available point cloud files"""
    output_pattern = "./output_3d_blocks/**/blocks_3d.ply"
    ply_files = glob.glob(output_pattern, recursive=True)
    
    if not ply_files:
        print("❌ No point cloud files found!")
        print("💡 Run the integration first: python scripts/unity_yolo/run_3d_integration.py")
        return []
    
    print("📁 Available point cloud files:")
    for i, ply_file in enumerate(ply_files):
        # Extract sequence info from path
        path_parts = ply_file.split(os.sep)
        sequence_info = path_parts[-2] if len(path_parts) > 1 else "unknown"
        print(f"   {i+1}: {sequence_info} -> {ply_file}")
    
    return ply_files

def main():
    parser = argparse.ArgumentParser(description='3D Point Cloud Viewer')
    parser.add_argument('--file', type=str, help='Specific PLY file to view')
    parser.add_argument('--sequence', type=str, help='View specific sequence (e.g., solo_9_sequence.0_step0)')
    parser.add_argument('--list', action='store_true', help='List all available point clouds')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_clouds()
        return
    
    if args.file:
        # View specific file
        view_point_cloud(args.file)
    elif args.sequence:
        # View specific sequence
        ply_file = f"./output_3d_blocks/{args.sequence}/blocks_3d.ply"
        view_point_cloud(ply_file)
    else:
        # Interactive mode - show all available files
        ply_files = list_available_clouds()
        
        if not ply_files:
            return
        
        if len(ply_files) == 1:
            print(f"\n🎯 Opening the only available file...")
            view_point_cloud(ply_files[0])
        else:
            print(f"\n🔢 Select a file to view (1-{len(ply_files)}):")
            try:
                choice = int(input("Enter number: ")) - 1
                if 0 <= choice < len(ply_files):
                    view_point_cloud(ply_files[choice])
                else:
                    print("❌ Invalid choice!")
            except (ValueError, KeyboardInterrupt):
                print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
