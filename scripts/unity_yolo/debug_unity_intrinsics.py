import json
import numpy as np
import math
import os

def debug_unity_intrinsics():
    """
    Debug Unity camera intrinsics calculation and identify issues
    """
    
    print("="*60)
    print("UNITY CAMERA INTRINSICS DEBUGGING")
    print("="*60)
    
    # Load the Unity JSON file
    json_file = "C:/Users/1213123/AppData/LocalLow/DefaultCompany/My project (1)/solo_7/sequence.49/step0.frame_data.json"
    
    if not os.path.exists(json_file):
        print(f"ERROR: JSON file not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract camera data
    cameras = {}
    for capture in data['captures']:
        cameras[capture['id']] = capture
    
    print("\n1. UNITY CAMERA DATA ANALYSIS")
    print("-" * 40)
    
    for cam_id in ['camera1', 'camera2']:
        if cam_id not in cameras:
            print(f"ERROR: {cam_id} not found in JSON")
            continue
            
        cam = cameras[cam_id]
        print(f"\n{cam_id.upper()}:")
        print(f"  Position: {cam['position']}")
        print(f"  Rotation (quaternion): {cam['rotation']}")
        print(f"  Dimensions: {cam['dimension']} (width x height)")
        print(f"  Projection: {cam['projection']}")
        print(f"  Matrix: {cam['matrix']}")
    
    # Check if both cameras have identical projection matrices
    matrix1 = np.array(cameras['camera1']['matrix'])
    matrix2 = np.array(cameras['camera2']['matrix'])
    print(f"\nProjection matrices identical: {np.allclose(matrix1, matrix2)}")
    
    # Extract basic parameters
    pos1 = np.array(cameras['camera1']['position'])
    pos2 = np.array(cameras['camera2']['position'])
    baseline_3d = np.linalg.norm(pos2 - pos1)
    
    width, height = cameras['camera1']['dimension']
    width, height = int(width), int(height)
    
    print(f"\nBaseline distance (3D): {baseline_3d:.6f} Unity units")
    print(f"Image dimensions: {width} x {height}")
    
    # Analyze the projection matrix
    print("\n2. PROJECTION MATRIX ANALYSIS")
    print("-" * 40)
    
    proj_matrix_flat = matrix1
    print(f"Unity projection matrix (9 elements): {proj_matrix_flat}")
    
    # Try different interpretations of Unity's projection matrix
    print(f"\nMatrix elements:")
    print(f"  [0] = {proj_matrix_flat[0]:.6f} (m00 - horizontal scaling)")
    print(f"  [1] = {proj_matrix_flat[1]:.6f} (m01)")
    print(f"  [2] = {proj_matrix_flat[2]:.6f} (m02)")
    print(f"  [3] = {proj_matrix_flat[3]:.6f} (m10)")
    print(f"  [4] = {proj_matrix_flat[4]:.6f} (m11 - vertical scaling)")
    print(f"  [5] = {proj_matrix_flat[5]:.6f} (m12)")
    print(f"  [6] = {proj_matrix_flat[6]:.6f} (m20)")
    print(f"  [7] = {proj_matrix_flat[7]:.6f} (m21)")
    print(f"  [8] = {proj_matrix_flat[8]:.6f} (m22 - depth related)")
    
    # Calculate FOV from projection matrix
    m00 = proj_matrix_flat[0]  # Horizontal scaling
    m11 = proj_matrix_flat[4]  # Vertical scaling
    
    print(f"\nProjection matrix interpretation:")
    print(f"  m00 (horizontal): {m00:.6f}")
    print(f"  m11 (vertical): {m11:.6f}")
    
    # Calculate FOV
    fov_x_rad = 2 * math.atan(1.0 / m00)
    fov_y_rad = 2 * math.atan(1.0 / m11)
    fov_x_deg = math.degrees(fov_x_rad)
    fov_y_deg = math.degrees(fov_y_rad)
    
    print(f"  Calculated FOV X: {fov_x_deg:.2f} degrees ({fov_x_rad:.4f} radians)")
    print(f"  Calculated FOV Y: {fov_y_deg:.2f} degrees ({fov_y_rad:.4f} radians)")
    
    # Calculate focal lengths
    fx = (width / 2.0) / math.tan(fov_x_rad / 2.0)
    fy = (height / 2.0) / math.tan(fov_y_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    
    print(f"  Calculated focal lengths: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
    
    # Create intrinsics matrix
    K_calculated = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    
    print("\n3. INTRINSICS MATRIX COMPARISON")
    print("-" * 40)
    
    print("Calculated K matrix:")
    print(K_calculated)
    
    # Load current intrinsics file
    intrinsics_file = "./unity_seq49_intrinsics.txt"
    if os.path.exists(intrinsics_file):
        print(f"\nCurrent intrinsics file: {intrinsics_file}")
        with open(intrinsics_file, 'r') as f:
            lines = f.readlines()
            K_flat_current = list(map(float, lines[0].strip().split()))
            baseline_current = float(lines[1].strip())
        
        K_current = np.array(K_flat_current).reshape(3, 3)
        print("Current K matrix from file:")
        print(K_current)
        print(f"Current baseline: {baseline_current:.6f}")
        
        print(f"\nDifference between calculated and current:")
        print(f"  fx diff: {abs(fx - K_current[0,0]):.6f}")
        print(f"  fy diff: {abs(fy - K_current[1,1]):.6f}")
        print(f"  cx diff: {abs(cx - K_current[0,2]):.6f}")
        print(f"  cy diff: {abs(cy - K_current[1,2]):.6f}")
    
    # Compare with asset intrinsics
    asset_intrinsics = "./assets/K.txt"
    if os.path.exists(asset_intrinsics):
        print(f"\nAsset intrinsics file: {asset_intrinsics}")
        with open(asset_intrinsics, 'r') as f:
            lines = f.readlines()
            K_asset_flat = list(map(float, lines[0].strip().split()))
            baseline_asset = float(lines[1].strip())
        
        K_asset = np.array(K_asset_flat).reshape(3, 3)
        print("Asset K matrix:")
        print(K_asset)
        print(f"Asset baseline: {baseline_asset:.6f}")
        
        print(f"\nComparison with asset:")
        print(f"  Focal length ratio (Unity/Asset): fx={fx/K_asset[0,0]:.3f}, fy={fy/K_asset[1,1]:.3f}")
        print(f"  Baseline ratio (Unity/Asset): {baseline_3d/baseline_asset:.3f}")
    
    print("\n4. BASELINE ANALYSIS")
    print("-" * 40)
    
    # Calculate different baseline interpretations
    print(f"Camera positions:")
    print(f"  Camera 1: {pos1}")
    print(f"  Camera 2: {pos2}")
    print(f"  Difference: {pos2 - pos1}")
    
    # Calculate baseline in different ways
    baseline_euclidean = np.linalg.norm(pos2 - pos1)
    baseline_x_only = abs(pos2[0] - pos1[0])
    baseline_horizontal = np.linalg.norm([pos2[0] - pos1[0], pos2[2] - pos1[2]])  # X-Z plane
    
    print(f"\nBaseline calculations:")
    print(f"  3D Euclidean distance: {baseline_euclidean:.6f}")
    print(f"  X-axis difference only: {baseline_x_only:.6f}")
    print(f"  Horizontal plane (X-Z): {baseline_horizontal:.6f}")
    
    print("\n5. SCALE ANALYSIS")
    print("-" * 40)
    
    # Unity units might not be meters - let's analyze the scale
    print("Unity scale analysis:")
    print(f"  Unity baseline: {baseline_3d:.6f} units")
    print(f"  If Unity units are centimeters: {baseline_3d/100:.6f} meters")
    print(f"  If Unity units are millimeters: {baseline_3d/1000:.6f} meters")
    print(f"  Current file baseline: {baseline_current:.6f}")
    
    # Suggest potential fixes
    print("\n6. POTENTIAL ISSUES AND FIXES")
    print("-" * 40)
    
    print("Potential issues:")
    if baseline_3d > 1.0:
        print(f"  - Large baseline ({baseline_3d:.3f}) suggests Unity units might not be meters")
        print(f"    Try scaling: --baseline_scale 0.01 (if centimeters) or 0.001 (if millimeters)")
    
    if abs(fx - fy) > 1:
        print(f"  - fx ({fx:.2f}) and fy ({fy:.2f}) differ significantly")
        print("    This might indicate non-square pixels or calibration issues")
    
    print(f"\nRecommended intrinsics file content:")
    K_flat = K_calculated.flatten()
    print(f"Line 1: {' '.join([f'{val:.6f}' for val in K_flat])}")
    print(f"Line 2: {baseline_3d:.6f}  # or try with scaling: {baseline_3d*0.01:.6f} or {baseline_3d*0.001:.6f}")
    
    return K_calculated, baseline_3d, fov_x_deg, fov_y_deg

if __name__ == "__main__":
    debug_unity_intrinsics()
