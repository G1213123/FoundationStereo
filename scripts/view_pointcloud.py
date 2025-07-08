import os
import sys
import argparse
import open3d as o3d
import numpy as np

def view_pointcloud(ply_file_path):
    """
    Load and visualize a PLY point cloud file using Open3D
    
    Args:
        ply_file_path (str): Path to the PLY file
    """
    if not os.path.exists(ply_file_path):
        print(f"Error: PLY file not found at: {ply_file_path}")
        return False
    
    print(f"Loading point cloud from: {ply_file_path}")
    
    try:
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(ply_file_path)
        
        if len(pcd.points) == 0:
            print("Error: Point cloud is empty or failed to load")
            return False
        
        print(f"Point cloud loaded successfully!")
        print(f"Number of points: {len(pcd.points)}")
        
        # Check if point cloud has colors
        if len(pcd.colors) > 0:
            print(f"Point cloud has colors: {len(pcd.colors)} color values")
        else:
            print("Point cloud has no color information")
        
        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Point Cloud Viewer - {os.path.basename(ply_file_path)}", 
            width=1200, 
            height=800
        )
        vis.add_geometry(pcd)
        
        # Set visualization options
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = [0.1, 0.1, 0.1]  # Dark background
        render_option.show_coordinate_frame = True
        
        # Print controls
        print("\n" + "="*50)
        print("POINT CLOUD VIEWER CONTROLS:")
        print("="*50)
        print("Mouse Controls:")
        print("  - Left mouse + drag: Rotate view")
        print("  - Right mouse + drag: Pan view")
        print("  - Mouse wheel: Zoom in/out")
        print("  - Middle mouse + drag: Pan view")
        print("\nKeyboard Controls:")
        print("  - R: Reset view to default")
        print("  - H: Print help")
        print("  - ESC or Q: Exit viewer")
        print("  - F: Toggle fullscreen")
        print("  - S: Save screenshot")
        print("="*50)
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        
        print("Point cloud viewer closed.")
        return True
        
    except Exception as e:
        print(f"Error loading or displaying point cloud: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View PLY point cloud files using Open3D")
    parser.add_argument(
        "ply_file", 
        type=str, 
        help="Path to the PLY file to visualize"
    )
    parser.add_argument(
        "--point_size", 
        type=float, 
        default=2.0, 
        help="Size of points in the visualization (default: 2.0)"
    )
    parser.add_argument(
        "--background", 
        type=str, 
        default="dark", 
        choices=["dark", "light", "white", "black"],
        help="Background color theme (default: dark)"
    )
    
    args = parser.parse_args()
    
    # Set background color based on argument
    bg_colors = {
        "dark": [0.1, 0.1, 0.1],
        "light": [0.9, 0.9, 0.9], 
        "white": [1.0, 1.0, 1.0],
        "black": [0.0, 0.0, 0.0]
    }
    
    # Load and view the point cloud
    success = view_pointcloud(args.ply_file)
    
    if not success:
        sys.exit(1)
