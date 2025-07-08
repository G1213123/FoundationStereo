#!/usr/bin/env python3
"""
Test the YOLO + FoundationStereo 3D integration pipeline
"""

import os
import sys
from dotenv import load_dotenv

def test_integration():
    print("🧪 Testing YOLO + FoundationStereo 3D Integration")
    print("="*50)
    
    # Load environment
    load_dotenv()
    
    # Check environment variables
    unity_project_path = os.getenv('UNITY_PROJECT_PATH')
    unity_dataset_name = os.getenv('UNITY_DATASET_NAME', 'solo_9')
    intrinsics_file = os.getenv('INTRINSICS_FILE', './unity_seq49_intrinsics_cm.txt')
    runs_dir = os.getenv('RUNS_DIR', './runs')
    
    print(f"📁 Unity Project: {unity_project_path}")
    print(f"📂 Dataset: {unity_dataset_name}")
    print(f"📐 Intrinsics: {intrinsics_file}")
    print(f"🏃 Runs Dir: {runs_dir}")
    
    # Check if paths exist
    if not unity_project_path or not os.path.exists(unity_project_path):
        print("❌ Unity project path not found!")
        return False
    
    unity_data_dir = os.path.join(unity_project_path, unity_dataset_name)
    if not os.path.exists(unity_data_dir):
        print(f"❌ Unity data directory not found: {unity_data_dir}")
        return False
    
    if not os.path.exists(intrinsics_file):
        print(f"❌ Intrinsics file not found: {intrinsics_file}")
        return False
    
    # Check for YOLO model
    yolo_model_path = os.path.join(runs_dir, "unity_blocks_auto6", "weights", "best.pt")
    if not os.path.exists(yolo_model_path):
        print(f"❌ YOLO model not found: {yolo_model_path}")
        return False
    
    # Check for FoundationStereo model
    foundation_model_path = "./pretrained_models/23-51-11/model_best_bp2.pth"
    if not os.path.exists(foundation_model_path):
        print(f"❌ FoundationStereo model not found: {foundation_model_path}")
        return False
    
    # Check for test images
    test_sequence = "sequence.0"
    test_step = "step0"
    
    left_image = os.path.join(unity_data_dir, test_sequence, f"{test_step}.camera1.png")
    right_image = os.path.join(unity_data_dir, test_sequence, f"{test_step}.camera2.png")
    
    if not os.path.exists(left_image):
        print(f"❌ Left test image not found: {left_image}")
        # Try to find any available sequence
        sequences = [d for d in os.listdir(unity_data_dir) if d.startswith('sequence.')]
        if sequences:
            print(f"💡 Available sequences: {sequences}")
        return False
    
    if not os.path.exists(right_image):
        print(f"❌ Right test image not found: {right_image}")
        return False
    
    print("✅ All required files found!")
    print(f"📷 Test images: {test_sequence}/{test_step}")
    print(f"🤖 YOLO model: {os.path.basename(yolo_model_path)}")
    print(f"🔍 FoundationStereo model: {os.path.basename(foundation_model_path)}")
    
    # Show sample command
    print("\n🚀 Sample command to run 3D integration:")
    print(f"python scripts/unity_yolo/run_3d_integration.py --sequence {test_sequence} --step {test_step}")
    
    return True

if __name__ == "__main__":
    if test_integration():
        print("\n🎉 Ready to run 3D integration!")
        exit(0)
    else:
        print("\n❌ Setup incomplete. Please check the issues above.")
        exit(1)
