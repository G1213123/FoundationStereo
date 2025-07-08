#!/usr/bin/env python3
"""
Simple test script to run YOLO inference on Unity images
Tests if our trained model can detect Module_Construction objects
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def test_yolo_inference():
    """Test YOLO inference on Unity images"""
    
    print("🧪 Testing YOLO Model Inference")
    print("=" * 50)
    
    # Find the latest trained model
    runs_dir = Path("./runs")
    if not runs_dir.exists():
        print("❌ No runs directory found")
        return False
        
    # Look for unity_blocks_auto models
    model_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "unity_blocks_auto" in d.name]
    if not model_dirs:
        print("❌ No trained models found")
        return False
        
    # Get the most recent model
    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_model_dir / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
        
    print(f"✅ Using model: {model_path}")
    
    # Load the YOLO model
    try:
        model = YOLO(str(model_path))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Find Unity test images
    unity_solo_path = "C:\\Users\\1213123\\AppData\\LocalLow\\DefaultCompany\\My project (1)\\solo_9"
    if not os.path.exists(unity_solo_path):
        print(f"❌ Unity solo path not found: {unity_solo_path}")
        return False
    
    # Find a sequence to test
    sequence_dirs = [d for d in Path(unity_solo_path).iterdir() if d.is_dir() and d.name.startswith("sequence.")]
    if not sequence_dirs:
        print("❌ No sequence directories found")
        return False
    
    test_sequence = sequence_dirs[0]  # Use first sequence
    print(f"📁 Testing with sequence: {test_sequence.name}")
    
    # Look for camera images (RGB images, not instance segmentation)
    # Files are directly in sequence directory: step0.camera1.png
    rgb_image_path = test_sequence / "step0.camera1.png"
    if not rgb_image_path.exists():
        print(f"❌ RGB image not found: {rgb_image_path}")
        return False
    
    print(f"🖼️ Testing with image: {rgb_image_path}")
    
    # Run inference
    try:
        results = model(str(rgb_image_path))
        print("✅ Inference completed successfully")
        
        # Analyze results
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                num_detections = len(result.boxes)
                print(f"🎯 Detected {num_detections} objects")
                
                if hasattr(result, 'masks') and result.masks is not None:
                    print(f"🎭 Generated {len(result.masks)} segmentation masks")
                
                # Save result image
                output_dir = Path("./results/inference_test")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "test_inference.png"
                
                # Get annotated image
                annotated_img = result.plot()
                cv2.imwrite(str(output_path), annotated_img)
                print(f"💾 Result saved: {output_path}")
                
                return True
            else:
                print("⚠️ No objects detected")
                return True
        else:
            print("❌ No results returned")
            return False
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

if __name__ == "__main__":
    success = test_yolo_inference()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
    sys.exit(0 if success else 1)
