#!/usr/bin/env python3
"""
Batch Corner Detection Script
Runs the corner detection pipeline on a batch of sequences and computes statistics.
"""

import sys
import os
import argparse
import glob
import numpy as np
import time
from contextlib import redirect_stdout

# Add current directory to path to import pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from corner_detection_lib import Config, run_pipeline

def main():
    parser = argparse.ArgumentParser(description='Batch Corner Detection')
    parser.add_argument('--batch_outputs', type=str, default='../run_files/batch_outputs', help='Folder containing batch outputs (seqX/raw)')
    parser.add_argument('--batch_inputs', type=str, default='../run_files/batch_inputs', help='Folder containing batch inputs (seqX/step0.frame_data.json)')
    parser.add_argument('--output_dir', type=str, default='batch_macro_detection', help='Folder to save detection results')
    parser.add_argument('--specific_seq', type=str, default="*")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Find all sequence folders in batch_outputs
    # Structure: batch_outputs/seqX/raw
    seq_folders = glob.glob(os.path.join(args.batch_outputs, f'seq{args.specific_seq}'))
    # Sort for consistent order
    seq_folders.sort()
    
    distances = []
    results_data = []
    processing_stats = []
    
    print(f"Found {len(seq_folders)} sequences in {args.batch_outputs}")
    print(f"Inputs expected in {args.batch_inputs}")
    print(f"Outputs will be saved to {args.output_dir}")
    
    for seq_folder in seq_folders:
        seq_id = os.path.basename(seq_folder)
        print(f"Processing {seq_id}...")
        
        raw_dir = os.path.join(seq_folder, 'raw')
        if not os.path.exists(raw_dir):
            print(f"  Skipping {seq_id}: raw directory not found at {raw_dir}")
            continue
            
        input_dir = os.path.join(args.batch_inputs, seq_id)
        if not os.path.exists(input_dir):
            print(f"  Skipping {seq_id}: input directory not found at {input_dir}")
            continue
            
        # Configure pipeline
        config = Config()
        config.raw_dir = raw_dir
        config.input_dir = input_dir
        config.output_dir = os.path.join(args.output_dir, seq_id) 
        
        # Run pipeline and save output
        # Ensure output directory exists (it might be created by run_pipeline, but we need it for the text file)
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
            
        output_file = os.path.join(config.output_dir, "detection.txt")
        
        start_time = time.time()
        try:
            # Capture stdout to file
            with open(output_file, 'w') as f:
                with redirect_stdout(f):
                    result = run_pipeline(config)
            
            if result:
                distances.append(result['distance'])
                cam_dist = result.get('camera_to_gt_distance')
                results_data.append((seq_id, result['distance'], cam_dist))
                
                msg = f"  Success! Distance: {result['distance']:.6f} m"
                if cam_dist is not None:
                    msg += f", Cam-GT Dist: {cam_dist:.6f} m"
                print(msg)
            else:
                print(f"  Failed to detect corners or match ground truth.")
                
        except Exception as e:
            print(f"  Error processing {seq_id}: {e}")
            # Print traceback to console for debugging
            import traceback
            traceback.print_exc()
            
        end_time = time.time()
        duration = end_time - start_time
        processing_stats.append({'seq_id': seq_id, 'time': duration})
            
    if distances:
        avg_dist = np.mean(distances)
        print("\n" + "="*40)
        print(f"Batch Processing Complete")
        print(f"Processed {len(distances)} sequences successfully")
        print(f"Average Distance: {avg_dist:.6f} m")
        print("="*40)
        
        # Save summary
        summary_path = os.path.join(args.output_dir, "batch_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Batch Processing Summary\n")
            f.write(f"========================\n")
            f.write(f"Total sequences found: {len(seq_folders)}\n")
            f.write(f"Successful detections: {len(distances)}\n")
            f.write(f"Average Distance: {avg_dist:.6f} m\n")
            f.write(f"Std Dev: {np.std(distances):.6f} m\n")
            f.write(f"Min: {np.min(distances):.6f} m\n")
            f.write(f"Max: {np.max(distances):.6f} m\n")
            
            f.write(f"\nPer-Sequence Results:\n")
            f.write(f"---------------------\n")
            f.write(f"{'Sequence ID':<20} | {'Error (m)':<12} | {'Cam-GT Dist (m)':<15}\n")
            for sid, dist, cam_dist in results_data:
                cam_str = f"{cam_dist:.6f}" if cam_dist is not None else "N/A"
                f.write(f"{sid:<20} | {dist:<12.6f} | {cam_str:<15}\n")
        print(f"Summary saved to {summary_path}")
    else:
        print("No successful detections.")

    # --- New Timing Stats & Plotting ---
    if processing_stats:
        raw_times = [s['time'] for s in processing_stats]
        
        # Calculate initialization overhead
        init_overhead = 0.0
        adjusted_times = list(raw_times)
        
        if len(raw_times) > 1:
            # Calculate mean of subsequent runs (steady state)
            steady_times = raw_times[1:]
            mean_steady_time = sum(steady_times) / len(steady_times)
            
            # If first run is significantly slower, treat difference as init overhead
            if raw_times[0] > mean_steady_time:
                init_overhead = raw_times[0] - mean_steady_time
                # Discount the init time from the first sequence for stats/plotting
                adjusted_times[0] = mean_steady_time
                print(f"Discounted {init_overhead:.4f}s initialization overhead from first sequence.")

        mean_time = sum(adjusted_times) / len(adjusted_times)
        print(f"Mean processing time: {mean_time:.4f} seconds")
        
        stats_file = os.path.join(args.output_dir, "processing_stats.txt")
        with open(stats_file, "w") as f:
            f.write(f"Batch Processing Stats\n")
            f.write(f"======================\n")
            f.write(f"Total sequences processed: {len(processing_stats)}\n")
            f.write(f"Mean processing time: {mean_time:.4f} seconds\n")
            if init_overhead > 0:
                f.write(f"Initialization overhead (seq 0): {init_overhead:.4f} seconds\n")
            f.write(f"\nIndividual times (adjusted):\n")
            for i, stat in enumerate(processing_stats):
                f.write(f"Sequence {stat['seq_id']}: {adjusted_times[i]:.4f} seconds\n")
        print(f"Stats saved to {stats_file}")

        # Generate Plot
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            seq_indices = range(len(adjusted_times))
            
            # Plot runtimes
            plt.plot(seq_indices, adjusted_times, marker='o', linestyle='-', linewidth=2, label='Sequence Runtime')
            
            # Plot mean line
            plt.axhline(y=mean_time, color='r', linestyle='--', label=f'Mean: {mean_time:.2f}s')
            
            # Add init time info
            if init_overhead > 0:
                # Add a dummy line for the legend to show init time
                plt.plot([], [], ' ', label=f'Init Overhead: {init_overhead:.2f}s')
            
            plt.title('Processing Runtime vs Sequence')
            plt.xlabel('Sequence Index')
            plt.ylabel('Time (seconds)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plot_file = os.path.join(args.output_dir, "runtime_plot.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Runtime plot saved to {plot_file}")
            
        except ImportError:
            print("matplotlib not found. Skipping plot generation.")
        except Exception as e:
            print(f"Error generating plot: {e}")

if __name__ == "__main__":
    main()
