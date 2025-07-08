#!/usr/bin/env python3
"""
Unity YOLO + FoundationStereo Integration
Automatically processes Unity stereo images to get 3D block coordinates
"""

import os
import argparse
from dotenv import load_dotenv
import subprocess
import sys

def find_unity_stereo_images(unity_data_dir, sequence="sequence.0", step="step0"):
    """Find left and right camera images from Unity data"""
    sequence_dir = os.path.join(unity_data_dir, sequence)
    
    if not os.path.exists(sequence_dir):
        available_sequences = [d for d in os.listdir(unity_data_dir) if d.startswith('sequence.')]
        raise FileNotFoundError(f"Sequence '{sequence}' not found. Available: {available_sequences}")
    
    # Look for camera images
    left_image = os.path.join(sequence_dir, f"{step}.camera1.png")
    right_image = os.path.join(sequence_dir, f"{step}.camera2.png")
    
    if not os.path.exists(left_image):
        raise FileNotFoundError(f"Left camera image not found: {left_image}")
    if not os.path.exists(right_image):
        raise FileNotFoundError(f"Right camera image not found: {right_image}")
    
    return left_image, right_image

def main():
    parser = argparse.ArgumentParser(description='Unity YOLO + FoundationStereo Integration')
    parser.add_argument('--sequence', type=str, default='sequence.0', help='Unity sequence to process')
    parser.add_argument('--step', type=str, default='step0', help='Unity step to process')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='YOLO confidence threshold')
    parser.add_argument('--min_depth', type=float, default=0.01, help='Minimum depth (meters)')
    parser.add_argument('--max_depth', type=float, default=50.0, help='Maximum depth (meters)')
    parser.add_argument('--scale', type=float, default=1.0, help='Image scale factor')
    parser.add_argument('--output_dir', type=str, help='Output directory (auto-generated if not specified)')
    parser.add_argument('--intrinsics', type=str, help='Intrinsics file (uses .env default if not specified)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get paths from environment
    unity_project_path = os.getenv('UNITY_PROJECT_PATH')
    unity_dataset_name = os.getenv('UNITY_DATASET_NAME', 'solo_9')
    
    if not unity_project_path:
        print("❌ UNITY_PROJECT_PATH not found in .env file")
        return 1
    
    # Construct Unity data directory
    unity_data_dir = os.path.join(unity_project_path, unity_dataset_name)
    
    print(f"🎯 Unity YOLO + FoundationStereo Integration")
    print(f"📁 Unity Data: {unity_data_dir}")
    print(f"🎬 Processing: {args.sequence}/{args.step}")
    
    try:
        # Find stereo images
        left_image, right_image = find_unity_stereo_images(unity_data_dir, args.sequence, args.step)
        print(f"📷 Left Image: {os.path.basename(left_image)}")
        print(f"📷 Right Image: {os.path.basename(right_image)}")
        
        # Set up output directory
        if args.output_dir is None:
            args.output_dir = f"./output_3d_blocks/{unity_dataset_name}_{args.sequence}_{args.step}"        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Set up intrinsics file
        if args.intrinsics is None:
            args.intrinsics = os.getenv('INTRINSICS_FILE', './unity_seq49_intrinsics_cm.txt')
        
        # Build command for main integration script
        script_path = os.path.join(os.path.dirname(__file__), 'yolo_foundation_integration.py')
        python_exe = os.getenv('PYTHON_ENV', sys.executable)
        cmd = [
            python_exe, script_path,
            '--left_image', left_image,
            '--right_image', right_image,
            '--intrinsic_file', args.intrinsics,
            '--conf_threshold', str(args.conf_threshold),
            '--min_depth', str(args.min_depth),
            '--max_depth', str(args.max_depth),
            '--scale', str(args.scale),
            '--output_dir', args.output_dir
        ]
        
        print(f"🚀 Running integration...")
        print(f"📋 Command: {' '.join(cmd)}")
        
        # Change to project root directory for relative imports
        original_cwd = os.getcwd()
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        project_root = os.path.abspath(project_root)
        os.chdir(project_root)
        
        try:
            # Run the integration script
            result = subprocess.run(cmd, capture_output=False)
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
        if result.returncode == 0:
            print(f"\n✅ Integration completed successfully!")
            print(f"📁 Results saved to: {args.output_dir}")
        else:
            print(f"\n❌ Integration failed with return code: {result.returncode}")
            return result.returncode
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
