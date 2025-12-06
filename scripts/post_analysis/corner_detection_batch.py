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
from contextlib import redirect_stdout

# Add current directory to path to import pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from corner_detection_pipeline import Config, run_pipeline

def main():
    parser = argparse.ArgumentParser(description='Batch Corner Detection')
    parser.add_argument('--batch_outputs', type=str, default='../run_files/batch_outputs', help='Folder containing batch outputs (seqX/raw)')
    parser.add_argument('--batch_inputs', type=str, default='../run_files/batch_inputs', help='Folder containing batch inputs (seqX/step0.frame_data.json)')
    parser.add_argument('--output_dir', type=str, default='batch_macro_detection', help='Folder to save detection results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Find all sequence folders in batch_outputs
    # Structure: batch_outputs/seqX/raw
    seq_folders = glob.glob(os.path.join(args.batch_outputs, 'seq*'))
    # Sort for consistent order
    seq_folders.sort()
    
    distances = []
    results_data = []
    
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
        output_file = os.path.join(args.output_dir, f"{seq_id}_detection.txt")
        
        try:
            # Capture stdout to file
            with open(output_file, 'w') as f:
                with redirect_stdout(f):
                    result = run_pipeline(config)
            
            if result:
                distances.append(result['distance'])
                results_data.append((seq_id, result['distance']))
                print(f"  Success! Distance: {result['distance']:.6f} m")
            else:
                print(f"  Failed to detect corners or match ground truth.")
                
        except Exception as e:
            print(f"  Error processing {seq_id}: {e}")
            # Print traceback to console for debugging
            import traceback
            traceback.print_exc()
            
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
            for sid, dist in results_data:
                f.write(f"{sid}: {dist:.6f} m\n")
        print(f"Summary saved to {summary_path}")
    else:
        print("No successful detections.")

if __name__ == "__main__":
    main()
