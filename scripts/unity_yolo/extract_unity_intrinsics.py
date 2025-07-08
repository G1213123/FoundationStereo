import json
import numpy as np
import argparse
import os
import math

def unity_matrix_to_opencv_intrinsics(unity_matrix, image_width, image_height):
    """
    Convert Unity's projection matrix to OpenCV camera intrinsics matrix
    
    Unity's projection matrix format (column-major):
    [0] = 2*near / (right-left) = 2*near / width  -> relates to fx
    [1] = 0
    [2] = 0  
    [3] = 0
    [4] = 2*near / (top-bottom) = 2*near / height -> relates to fy
    [5] = 0
    [6] = 0
    [7] = 0  
    [8] = -(far+near)/(far-near) -> depth related
    
    For Unity cameras, we need to convert to OpenCV format:
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    
    # Unity uses a different coordinate system and projection matrix format
    # We need to extract focal length information from the projection matrix
    
    # Unity's projection matrix elements
    m00 = unity_matrix[0]  # Related to horizontal FOV
    m11 = unity_matrix[4]  # Related to vertical FOV
    
    # Convert Unity projection matrix to focal lengths in pixels
    # Unity's projection matrix: m00 = 2*near/width_in_world_units
    # We need to convert this to pixels
    
    # For Unity, the projection matrix elements relate to FOV:
    # m00 = 2*near/width, m11 = 2*near/height (in world coordinates)
    # To get focal length in pixels: f = (image_dimension/2) / tan(fov/2)
    
    # From Unity's projection matrix, we can derive:
    # fov_x = 2 * atan(1/m00), fov_y = 2 * atan(1/m11)
    fov_x = 2 * math.atan(1.0 / m00)
    fov_y = 2 * math.atan(1.0 / m11)
    
    # Convert FOV to focal length in pixels
    fx = (image_width / 2.0) / math.tan(fov_x / 2.0)
    fy = (image_height / 2.0) / math.tan(fov_y / 2.0)
    
    # Principal point (usually center of image)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Create OpenCV intrinsics matrix
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    return K

def calculate_baseline(pos1, pos2):
    """
    Calculate the baseline (distance) between two camera positions
    """
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    baseline = np.linalg.norm(pos2 - pos1)
    return baseline

def extract_camera_params(json_file_path):
    """
    Extract camera parameters from Unity frame data JSON
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Find camera1 and camera2 data
    camera1_data = None
    camera2_data = None
    
    for capture in data['captures']:
        if capture['id'] == 'camera1':
            camera1_data = capture
        elif capture['id'] == 'camera2':
            camera2_data = capture
    
    if camera1_data is None or camera2_data is None:
        raise ValueError("Could not find both camera1 and camera2 in the JSON file")
    
    # Extract image dimensions (should be same for both cameras)
    width, height = camera1_data['dimension']
    width, height = int(width), int(height)
    
    # Extract projection matrices
    matrix1 = camera1_data['matrix']
    matrix2 = camera2_data['matrix']
    
    # Convert to OpenCV intrinsics (assuming both cameras have same intrinsics)
    K1 = unity_matrix_to_opencv_intrinsics(matrix1, width, height)
    K2 = unity_matrix_to_opencv_intrinsics(matrix2, width, height)
    
    # Calculate baseline from positions
    pos1 = camera1_data['position']
    pos2 = camera2_data['position']
    baseline = calculate_baseline(pos1, pos2)
    
    # Unity units are typically in meters, but let's verify the scale
    print(f"Camera 1 position: {pos1}")
    print(f"Camera 2 position: {pos2}")
    print(f"Calculated baseline: {baseline:.6f} units")
    print(f"Camera 1 intrinsics matrix:")
    print(K1)
    print(f"Camera 2 intrinsics matrix:")
    print(K2)
    
    # Use camera1 intrinsics (they should be identical for stereo setup)
    return K1, baseline, width, height

def save_intrinsics_file(K, baseline, output_path):
    """
    Save intrinsics in the format expected by FoundationStereo
    """
    # Flatten the 3x3 matrix to 1x9 format
    K_flat = K.flatten()
    
    with open(output_path, 'w') as f:
        # First line: flattened intrinsics matrix (9 values)
        f.write(' '.join([f"{val:.6f}" for val in K_flat]) + '\n')
        # Second line: baseline in meters
        f.write(f"{baseline:.6f}\n")
    
    print(f"Intrinsics saved to: {output_path}")
    print(f"Content:")
    print(f"Line 1 (K matrix flattened): {' '.join([f'{val:.6f}' for val in K_flat])}")
    print(f"Line 2 (baseline): {baseline:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract camera intrinsics from Unity JSON frame data")
    parser.add_argument(
        "json_file", 
        type=str, 
        help="Path to Unity frame_data.json file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="unity_intrinsics.txt",
        help="Output file for camera intrinsics (default: unity_intrinsics.txt)"
    )
    parser.add_argument(
        "--baseline_scale", 
        type=float, 
        default=1.0,
        help="Scale factor for baseline if Unity units are not in meters (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        exit(1)
    
    try:
        # Extract camera parameters
        K, baseline, width, height = extract_camera_params(args.json_file)
        
        # Apply baseline scale if needed
        baseline_scaled = baseline * args.baseline_scale
        
        print(f"\nExtracted Parameters:")
        print(f"Image dimensions: {width} x {height}")
        print(f"Focal length (fx, fy): ({K[0,0]:.2f}, {K[1,1]:.2f})")
        print(f"Principal point (cx, cy): ({K[0,2]:.2f}, {K[1,2]:.2f})")
        print(f"Original baseline: {baseline:.6f}")
        print(f"Scaled baseline: {baseline_scaled:.6f}")
        
        # Save intrinsics file
        save_intrinsics_file(K, baseline_scaled, args.output)
        
        print(f"\nYou can now use this intrinsics file with FoundationStereo:")
        print(f"python scripts/run_demo.py --intrinsic_file {args.output} ...")
        
    except Exception as e:
        print(f"Error processing JSON file: {str(e)}")
        exit(1)
