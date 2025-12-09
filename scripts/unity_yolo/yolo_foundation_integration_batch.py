#!/usr/bin/env python3
"""
Batch processing script for YOLO + FoundationStereo Integration.
Randomly selects a batch of sequences from a folder and processes them.
"""

import os
import sys
import argparse
import random
import glob
import logging
import time
import shutil
from dotenv import load_dotenv
import torch
from ultralytics import YOLO
from omegaconf import OmegaConf

# Add project paths
code_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.join(code_dir, '..', '..')
sys.path.append(project_root)
sys.path.append(code_dir) # Allow importing from current directory

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from Utils import *

# Import from the main integration script
from yolo_foundation_integration import process_stereo_pair, setup_logging, find_unity_frame_json

def find_sequences(input_folder):
    """
    Find all valid stereo sequences in the input folder.
    Assumes naming convention: sequence.{id}_step{step}_camera{cam}.png
    Returns a list of tuples: (sequence_id, left_image_path, right_image_path)
    """
    sequences = []
    
    # Find all camera1 images
    # Pattern: sequence.*_camera1.png
    # We can use glob
    search_pattern = os.path.join(input_folder, "sequence.*", "*camera1.png")
    left_images = glob.glob(search_pattern)
    
    for left_img in left_images:
        # Construct expected right image path
        # Replace camera1 with camera2
        right_img = left_img.replace("camera1", "camera2")
        
        if os.path.exists(right_img):
            # Extract sequence ID for logging/reference
            basename = os.path.basename(left_img)
            # Example: sequence.0_step0_camera1.png
            # We can just use the basename as ID or parse it
            seq_id = basename.split("_camera1")[0]
            sequences.append((seq_id, left_img, right_img))
        else:
            # logging.warning(f"Missing right image for {left_img}")
            pass
            
    return sequences

def main():
    parser = argparse.ArgumentParser(description='Batch YOLO + FoundationStereo Integration')
    
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing image sequences')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of sequences to process randomly')
    parser.add_argument('--output_folder', type=str, default='../run_files/batch_outputs', help='Root output folder')
    
    # Model paths (same as original script)
    parser.add_argument('--yolo_model', type=str, help='YOLO model path (uses .env if not specified)')
    parser.add_argument('--foundation_model', type=str, 
                       default='./pretrained_models/23-51-11/model_best_bp2.pth',
                       help='FoundationStereo model path')
    
    # Parameters (same as original script)
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='YOLO confidence threshold')
    parser.add_argument('--scale', type=float, default=1.0, help='Image scale factor')
    parser.add_argument('--valid_iters', type=int, default=32, help='FoundationStereo iterations')
    parser.add_argument('--hiera', type=int, default=0, help='Hierarchical inference')
    parser.add_argument('--min_depth', type=float, default=0.1, help='Minimum depth (meters)')
    parser.add_argument('--max_depth', type=float, default=10.0, help='Maximum depth (meters)')
    parser.add_argument('--depth_vis_max', type=float, default=500.0, help='Max depth (m) to visualize')
    parser.add_argument('--minimal_output', action='store_true', help='Generate minimal output (only depth.npy and edges.npy)')
    
    args = parser.parse_args()
    
    setup_logging()
    load_dotenv()
    
    # Find sequences
    logging.info(f"Scanning {args.input_folder} for sequences...")
    all_sequences = find_sequences(args.input_folder)
    logging.info(f"Found {len(all_sequences)} valid stereo sequences.")
    
    if len(all_sequences) == 0:
        logging.error("No valid sequences found. Exiting.")
        return

    # Select batch
    batch_size = min(args.batch_size, len(all_sequences))
    selected_sequences = random.sample(all_sequences, batch_size)
    logging.info(f"Randomly selected {batch_size} sequences for processing.")
    
    # Load models
    # Load YOLO model
    if args.yolo_model is None:
        runs_dir = os.getenv('RUNS_DIR', '../../runs')
        # Try to find a default model if not specified
        default_model = os.path.join(runs_dir, "unity_blocks_auto7", "weights", "best.pt")
        if os.path.exists(default_model):
            args.yolo_model = default_model
        else:
             # Fallback or error
             logging.warning(f"Default YOLO model not found at {default_model}. Please specify --yolo_model.")
    
    if args.yolo_model:
        logging.info(f"Loading YOLO model: {args.yolo_model}")
        yolo_model = YOLO(args.yolo_model)
    else:
        logging.error("No YOLO model specified.")
        return

    # Load FoundationStereo model
    logging.info(f"Loading FoundationStereo model: {args.foundation_model}")
    if not os.path.exists(args.foundation_model):
        logging.error(f"FoundationStereo model not found at {args.foundation_model}")
        return

    cfg = OmegaConf.load(f'{os.path.dirname(args.foundation_model)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    
    # Add args to config
    # We need to create a config object that FoundationStereo expects
    # It seems it expects args to have certain attributes.
    # In the original script:
    # for k in args.__dict__: cfg[k] = args.__dict__[k]
    # args = OmegaConf.create(cfg)
    
    # We need to be careful not to overwrite args that we need for the loop
    # So let's create a separate model_args object
    model_cfg = cfg.copy()
    for k in args.__dict__:
        model_cfg[k] = args.__dict__[k]
    model_args = OmegaConf.create(model_cfg)
    
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    
    foundation_model = FoundationStereo(model_args)
    foundation_model.load_state_dict(torch.load(args.foundation_model, weights_only=False)['model'], strict=False)
    foundation_model.cuda()
    foundation_model.eval()
    
    # Process sequences
    processing_stats = []
    for i, (seq_id, left_img, right_img) in enumerate(selected_sequences):
        logging.info(f"Processing sequence {i+1}/{batch_size}: {seq_id}")
        
        # Create output directory for this sequence
        seq_output_dir = os.path.join(args.output_folder, f'seq{i}')
        os.makedirs(seq_output_dir, exist_ok=True)
        
        # Save input files
        root_input_save_dir = os.path.join(os.path.dirname(args.output_folder), 'batch_inputs')
        seq_input_save_dir = os.path.join(root_input_save_dir, f'seq{i}')
        os.makedirs(seq_input_save_dir, exist_ok=True)
        
        try:
            # Copy images
            shutil.copy2(left_img, os.path.join(seq_input_save_dir, os.path.basename(left_img)))
            shutil.copy2(right_img, os.path.join(seq_input_save_dir, os.path.basename(right_img)))
            
            # Copy JSON if found
            json_path = find_unity_frame_json(left_img)
            if json_path and os.path.exists(json_path):
                shutil.copy2(json_path, os.path.join(seq_input_save_dir, os.path.basename(json_path)))
                
            # Copy Depth EXR if found
            depth_candidates = [
                left_img.replace("camera1.png", "camera1.Depth.exr"),
                os.path.splitext(left_img)[0] + ".Depth.exr"
            ]
            for depth_cand in depth_candidates:
                if os.path.exists(depth_cand):
                    shutil.copy2(depth_cand, os.path.join(seq_input_save_dir, os.path.basename(depth_cand)))
                    break

            start_time = time.time()
            process_stereo_pair(
                yolo_model, 
                foundation_model, 
                left_img, 
                right_img, 
                seq_output_dir, 
                model_args # Pass the OmegaConf args expected by the function
            )
            end_time = time.time()
            duration = end_time - start_time
            processing_stats.append({'seq_id': seq_id, 'time': duration})
            logging.info(f"Sequence {seq_id} processed in {duration:.4f} seconds")

        except Exception as e:
            logging.error(f"Error processing sequence {seq_id}: {e}")
            import traceback
            traceback.print_exc()

    logging.info("Batch processing complete!")

    # Save stats
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
                logging.info(f"Discounted {init_overhead:.4f}s initialization overhead from first sequence.")

        mean_time = sum(adjusted_times) / len(adjusted_times)
        logging.info(f"Mean processing time: {mean_time:.4f} seconds")
        
        stats_file = os.path.join(args.output_folder, "processing_stats.txt")
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
        logging.info(f"Stats saved to {stats_file}")

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
            
            plot_file = os.path.join(args.output_folder, "runtime_plot.png")
            plt.savefig(plot_file)
            plt.close()
            logging.info(f"Runtime plot saved to {plot_file}")
            
        except ImportError:
            logging.warning("matplotlib not found. Skipping plot generation.")
        except Exception as e:
            logging.error(f"Error generating plot: {e}")


if __name__ == "__main__":
    main()

