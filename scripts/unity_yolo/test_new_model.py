#!/usr/bin/env python3
"""Test the newly trained YOLO model with 50 sequences"""

import os
from ultralytics import YOLO
import cv2
import numpy as np
from dotenv import load_dotenv

def test_new_model():
    """Test the newly trained model with different confidence thresholds"""
    
    # Load environment variables
    load_dotenv()
    
    # Get paths from .env
    runs_dir = os.getenv('RUNS_DIR', './runs')
    unity_project_path = os.getenv('UNITY_PROJECT_PATH')
    unity_dataset_name = os.getenv('UNITY_DATASET_NAME', 'solo_9')
    
    if not unity_project_path:
        print("❌ UNITY_PROJECT_PATH not found in .env file")
        return
    
    # Construct Unity data directory path
    unity_data_dir = os.path.join(unity_project_path, unity_dataset_name)
    
    # Find the latest model (unity_blocks_auto6)
    model_dirs = []
    for d in os.listdir(runs_dir):
        if d.startswith("unity_blocks_auto"):
            model_dirs.append(d)
    
    # Get the latest (highest number)
    if model_dirs:
        latest_dir = sorted(model_dirs, key=lambda x: int(x.replace("unity_blocks_auto", "") or "0"))[-1]
        model_path = os.path.join(runs_dir, latest_dir, "weights", "best.pt")
    else:
        print("❌ No unity_blocks_auto models found")
        return
    
    print(f"🧪 Testing model: {model_path}")
    
    # Load model
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test image path
    test_image = os.path.join(unity_data_dir, "sequence.0", "step0.camera1.png")
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"🖼️ Testing image: {test_image}")
    
    # Test different confidence thresholds
    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    for conf in thresholds:
        try:
            results = model(test_image, conf=conf, verbose=False)
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            status = "✅" if num_detections > 0 else "❌"
            print(f"   conf={conf:.2f}: {num_detections} detections {status}")
            
            # Show confidence scores if any detections
            if num_detections > 0 and results[0].boxes.conf is not None:
                conf_scores = results[0].boxes.conf.cpu().numpy()
                print(f"      Confidence scores: {conf_scores}")
                
        except Exception as e:
            print(f"   conf={conf:.2f}: Error - {e}")
    
    # Test with very low confidence to see all possible detections
    print(f"\n🔍 Testing with conf=0.001 to see all possible detections:")
    try:
        results = model(test_image, conf=0.001, verbose=False)
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"   Found {num_detections} total detections")
        
        if num_detections > 0 and results[0].boxes.conf is not None:
            conf_scores = results[0].boxes.conf.cpu().numpy()
            print(f"   Confidence range: {conf_scores.min():.4f} - {conf_scores.max():.4f}")
            print(f"   All confidence scores: {conf_scores}")
            
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_new_model()
