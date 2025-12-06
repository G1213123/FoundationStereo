import numpy as np
from pathlib import Path
import argparse
import os
import glob
import warnings

# Prefer OpenCV for EXR; fallback to imageio
try:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2
except Exception as e:
    cv2 = None

try:
    import imageio.v3 as iio
except Exception as e:
    iio = None

def _as_single_channel_float(arr):
    if arr is None:
        return None
    arr = np.array(arr)
    # Resize shape
    if arr.shape[-1] == 4:
        arr = arr[..., 2]
    elif arr.ndim == 3:
        arr = arr[..., 2]
    # Clean non-finite
    mask = ~np.isfinite(arr)
    if mask.any():
        arr = arr.copy()
        arr[mask] = np.nan
    return arr

def read_exr_cv2(path):
    if cv2 is not None:
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return _as_single_channel_float(img)
    elif iio is not None:
        img = iio.imread(str(path))
        return _as_single_channel_float(img)
    else:
        raise RuntimeError("No available EXR reader (cv2 or imageio).")

def compare_depth(input_path, output_path):
    if not input_path.exists():
        print(f"Missing input: {input_path}")
        return None
    if not output_path.exists():
        print(f"Missing output: {output_path}")
        return None

    try:
        depth_in = read_exr_cv2(input_path)
        depth_out = read_exr_cv2(output_path)
    except Exception as e:
        print(f"Error reading files {input_path} or {output_path}: {e}")
        return None

    if depth_in is None or depth_out is None:
        return None

    # If shapes differ, resize output to match input
    if depth_in.shape != depth_out.shape:
        # warnings.warn(f'Shape mismatch {depth_in.shape} vs {depth_out.shape}; resizing output to input')
        if cv2 is None:
            raise RuntimeError('OpenCV required to resize depth; please install opencv-python')
        h, w = depth_in.shape[:2]
        depth_out = cv2.resize(depth_out, (w, h), interpolation=cv2.INTER_NEAREST)

    # Calculate difference
    valid = np.isfinite(depth_in) & np.isfinite(depth_out)
    if not valid.any():
        print(f"No overlapping finite pixels for {input_path.name}")
        return None

    diff = np.zeros_like(depth_in, dtype=np.float32)
    diff[valid] = depth_out[valid] - depth_in[valid]
    
    ssd = np.sum(diff[valid]**2)
    mse = np.mean(diff[valid]**2)
    
    return ssd, mse

def main():
    parser = argparse.ArgumentParser(description="Compare depth maps and calculate SSD.")
    parser.add_argument("--input", type=str, required=True, help="Input directory or file path.")
    parser.add_argument("--output", type=str, required=True, help="Output directory or file path.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of images to process.")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file() and output_path.is_file():
        # Single file comparison
        result = compare_depth(input_path, output_path)
        if result:
            ssd, mse = result
            print(f"Comparison Result:")
            print(f"SSD: {ssd:.4f}")
            print(f"MSE: {mse:.4f}")
        return

    if not input_path.is_dir() or not output_path.is_dir():
        print("Error: Both inputs must be directories or both must be files.")
        return

    # Directory comparison
    input_files = sorted(list(input_path.glob("*.exr")))
    
    if not input_files:
        print(f"No EXR files found in {input_path}")
        return

    total_ssd = 0.0
    total_mse = 0.0
    count = 0
    
    print(f"Found {len(input_files)} input files. Processing up to {args.limit}...")

    for in_file in input_files:
        if count >= args.limit:
            break
            
        # Construct corresponding output path
        # Assuming same filename
        out_file = output_path / in_file.name
        
        if not out_file.exists():
            print(f"Output file not found for {in_file.name}, skipping.")
            continue

        result = compare_depth(in_file, out_file)
        if result:
            ssd, mse = result
            total_ssd += ssd
            total_mse += mse
            count += 1
            print(f"Processed {in_file.name}: SSD={ssd:.4f}, MSE={mse:.4f}")

    if count > 0:
        mean_ssd = total_ssd / count
        mean_mse = total_mse / count
        print(f"\nSummary over {count} images:")
        print(f"Mean Sum of Squared Differences (SSD): {mean_ssd:.4f}")
        print(f"Mean Mean Squared Error (MSE): {mean_mse:.4f}")
    else:
        print("No images processed.")

if __name__ == "__main__":
    main()
