import os
import cv2
import glob
import argparse

def process_sequence(seq_dir):
    depth_map_path = os.path.join(seq_dir, 'depth_map.jpg')
    depth_overlay_path = os.path.join(seq_dir, 'depth_overlay.jpg')
    output_path = os.path.join(seq_dir, 'depth_overlay_merged.jpg')

    if not os.path.exists(depth_map_path):
        print(f"Skipping {seq_dir}: depth_map.jpg not found")
        return
    
    if not os.path.exists(depth_overlay_path):
        print(f"Skipping {seq_dir}: depth_overlay.jpg not found")
        return

    # Read images
    depth_map = cv2.imread(depth_map_path)
    depth_overlay = cv2.imread(depth_overlay_path)

    if depth_map is None or depth_overlay is None:
        print(f"Error reading images in {seq_dir}")
        return

    # Check dimensions
    if depth_map.shape != depth_overlay.shape:
        print(f"Warning: Dimensions mismatch in {seq_dir}. Resizing depth_overlay to match depth_map.")
        depth_overlay = cv2.resize(depth_overlay, (depth_map.shape[1], depth_map.shape[0]))

    height, width = depth_overlay.shape[:2]
    split_row = 401

    if height <= split_row:
        print(f"Warning: Image height ({height}) is smaller than split row ({split_row}) in {seq_dir}. Skipping.")
        return

    # Merge: Top part from overlay, Bottom part from depth map
    # The user said: "merge the depth map image bottom (from px 401 to the bottom) to the image depth overlay"
    # So we keep overlay top (0 to 401) and replace bottom with depth map (401 to end)
    
    merged_image = depth_overlay.copy()
    merged_image[split_row:, :] = depth_map[split_row:, :]

    # Save result
    cv2.imwrite(output_path, merged_image)
    print(f"Saved merged image to {output_path}")

def main():
    base_dir = r'C:\Users\1213123\Documents\Scripts\FoundationStereo\scripts\run_files\batch_outputs'
    
    # Find all seq directories
    seq_dirs = glob.glob(os.path.join(base_dir, 'seq*'))
    
    print(f"Found {len(seq_dirs)} sequences in {base_dir}")
    
    for seq_dir in seq_dirs:
        if os.path.isdir(seq_dir):
            process_sequence(seq_dir)

if __name__ == "__main__":
    main()
