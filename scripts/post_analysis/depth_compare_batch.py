import numpy as np
from pathlib import Path
import argparse
import os
import glob
import warnings
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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

def read_depth(path):
    path = Path(path)
    if path.suffix.lower() == '.exr':
        return read_exr_cv2(path)
    elif path.suffix.lower() == '.npy':
        try:
            arr = np.load(path)
            return _as_single_channel_float(arr)
        except Exception as e:
            print(f"Error reading NPY {path}: {e}")
            return None
    else:
        print(f"Unsupported depth format: {path.suffix}")
        return None

def compare_depth(input_path, output_path, max_eval_depth=50.0):
    if not input_path.exists():
        print(f"Missing input: {input_path}")
        return None
    if not output_path.exists():
        print(f"Missing output: {output_path}")
        return None

    try:
        depth_in = read_depth(input_path)
        depth_out = read_depth(output_path)
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
    
    if max_eval_depth is not None:
        valid &= (depth_in <= max_eval_depth)

    if not valid.any():
        print(f"No overlapping finite pixels (after masking > {max_eval_depth}) for {input_path.name}")
        return None

    diff = np.zeros_like(depth_in, dtype=np.float32)
    diff[valid] = depth_out[valid] - depth_in[valid]
    
    ssd = np.sum(diff[valid]**2)
    mse = np.mean(diff[valid]**2)
    mae = np.mean(np.abs(diff[valid]))
    med_ae = np.median(np.abs(diff[valid]))
    
    return ssd, mse, mae, med_ae

def main():
    parser = argparse.ArgumentParser(description="Batch Compare depth maps and calculate SSD.")
    parser.add_argument('--batch_outputs', type=str, default='../run_files/batch_outputs', help='Folder containing batch outputs (seqX/raw/depth.exr)')
    parser.add_argument('--batch_inputs', type=str, default='../run_files/batch_inputs', help='Folder containing batch inputs (seqX/step0.camera1.Depth.exr)')
    parser.add_argument('--output_dir', type=str, default='batch_depth_compare', help='Folder to save comparison results')
    parser.add_argument('--specific_seq', type=str, default="*", help="Glob pattern for specific sequences")
    parser.add_argument('--max_eval_depth', type=float, default=50.0, help='Maximum valid depth for ground truth (mask background)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all sequence folders in batch_inputs
    # Structure: batch_inputs/seqX
    batch_outputs_path = Path(args.batch_outputs)
    batch_inputs_path = Path(args.batch_inputs)
    
    seq_folders = sorted(list(batch_inputs_path.glob(f'seq{args.specific_seq}')))
    
    print(f"Found {len(seq_folders)} sequences in {batch_inputs_path}")
    print(f"Outputs expected in {batch_outputs_path}")
    print(f"Results will be saved to {output_dir}")
    
    results_data = []
    total_ssd = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_med_ae = 0.0
    count = 0
    
    for seq_folder in seq_folders:
        seq_id = seq_folder.name
        print(f"Processing {seq_id}...")
        
        # Ground Truth Depth: batch_inputs/seqX/step0.camera1.Depth.exr
        gt_depth_path = seq_folder / 'step0.camera1.Depth.exr'
        
        # Predicted Depth: batch_outputs/seqX/raw/depth.exr OR depth.npy
        pred_depth_path_exr = batch_outputs_path / seq_id / 'raw' / 'depth.exr'
        pred_depth_path_npy = batch_outputs_path / seq_id / 'raw' / 'depth.npy'
        
        pred_depth_path = None
        if pred_depth_path_exr.exists():
            pred_depth_path = pred_depth_path_exr
        elif pred_depth_path_npy.exists():
            pred_depth_path = pred_depth_path_npy
        
        if not gt_depth_path.exists():
             # Try alternative name if needed, or just report missing
             print(f"  GT depth not found at {gt_depth_path}")
             continue
             
        if pred_depth_path is None:
            print(f"  Predicted depth not found at {pred_depth_path_exr} or {pred_depth_path_npy}")
            continue
            
        result = compare_depth(gt_depth_path, pred_depth_path, max_eval_depth=args.max_eval_depth)
        
        if result:
            ssd, mse, mae, med_ae = result
            total_ssd += ssd
            total_mse += mse
            total_mae += mae
            total_med_ae += med_ae
            count += 1
            results_data.append((seq_id, ssd, mse, mae, med_ae))
            print(f"  SSD: {ssd:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, MedAE: {med_ae:.4f}")
        else:
            print(f"  Comparison failed for {seq_id}")

    if count > 0:
        mean_ssd = total_ssd / count
        mean_mse = total_mse / count
        mean_mae = total_mae / count
        mean_med_ae = total_med_ae / count
        
        print("\n" + "="*40)
        print(f"Batch Depth Comparison Complete")
        print(f"Processed {count} sequences successfully")
        print(f"Mean SSD: {mean_ssd:.4f}")
        print(f"Mean MSE: {mean_mse:.4f}")
        print(f"Mean MAE: {mean_mae:.4f}")
        print(f"Mean MedAE: {mean_med_ae:.4f}")
        print("="*40)
        
        # Save summary
        summary_path = output_dir / "depth_compare_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Batch Depth Comparison Summary\n")
            f.write(f"==============================\n")
            f.write(f"Total sequences found: {len(seq_folders)}\n")
            f.write(f"Successful comparisons: {count}\n")
            f.write(f"Mean SSD: {mean_ssd:.4f}\n")
            f.write(f"Mean MSE: {mean_mse:.4f}\n")
            f.write(f"Mean MAE: {mean_mae:.4f}\n")
            f.write(f"Mean MedAE: {mean_med_ae:.4f}\n")
            
            f.write(f"\nPer-Sequence Results:\n")
            f.write(f"---------------------\n")
            f.write(f"{'Sequence ID':<20} | {'SSD':<15} | {'MSE':<15} | {'MAE':<15} | {'MedAE':<15}\n")
            for sid, ssd, mse, mae, med_ae in results_data:
                f.write(f"{sid:<20} | {ssd:<15.4f} | {mse:<15.4f} | {mae:<15.4f} | {med_ae:<15.4f}\n")
        print(f"Summary saved to {summary_path}")

        # Generate Separate MAE and MedAE Bar Charts
        if plt is not None:
            try:
                seq_ids = [r[0] for r in results_data]
                maes = [r[3] for r in results_data]
                med_aes = [r[4] for r in results_data]
                
                x_pos = np.arange(len(seq_ids))

                # 1. MAE Chart
                plt.figure(figsize=(12, 6))
                if len(seq_ids) > 50:
                    plt.bar(x_pos, maes, color='skyblue', edgecolor='black', alpha=0.7)
                    plt.xlabel('Sequence Index')
                else:
                    plt.bar(x_pos, maes, color='skyblue', edgecolor='black', alpha=0.7)
                    plt.xticks(x_pos, seq_ids, rotation=90, fontsize=8)
                    plt.xlabel('Sequence ID')

                plt.axhline(y=mean_mae, color='r', linestyle='--', linewidth=2, label=f'Mean MAE: {mean_mae:.4f} m')
                plt.ylabel('Mean Absolute Error (m)')
                plt.title('Mean Absolute Error per Sequence')
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                mae_plot_path = output_dir / "mae_bar_chart.png"
                plt.savefig(mae_plot_path)
                plt.close()
                print(f"MAE bar chart saved to {mae_plot_path}")

                # 2. MedAE Chart
                plt.figure(figsize=(12, 6))
                if len(seq_ids) > 50:
                    plt.bar(x_pos, med_aes, color='orange', edgecolor='black', alpha=0.7)
                    plt.xlabel('Sequence Index')
                else:
                    plt.bar(x_pos, med_aes, color='orange', edgecolor='black', alpha=0.7)
                    plt.xticks(x_pos, seq_ids, rotation=90, fontsize=8)
                    plt.xlabel('Sequence ID')

                plt.axhline(y=mean_med_ae, color='r', linestyle='--', linewidth=2, label=f'Mean MedAE: {mean_med_ae:.4f} m')
                plt.ylabel('Median Absolute Error (m)')
                plt.title('Median Absolute Error per Sequence')
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                med_plot_path = output_dir / "medae_bar_chart.png"
                plt.savefig(med_plot_path)
                plt.close()
                print(f"MedAE bar chart saved to {med_plot_path}")

            except Exception as e:
                print(f"Error generating plots: {e}")
        else:
            print("matplotlib not installed; skipping chart generation.")

    else:
        print("No successful comparisons.")

if __name__ == "__main__":
    main()
