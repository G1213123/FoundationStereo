#!/usr/bin/env python3
"""Updated YOLO inference script that returns only the best detection"""

import os
from ultralytics import YOLO
import cv2
import numpy as np
from dotenv import load_dotenv

def test_yolo_best_detection():
    """Test YOLO model and return only the best detection"""
    
    # Load environment variables
    load_dotenv()
    
    # Get paths from .env
    runs_dir = os.getenv('RUNS_DIR', './runs')
    unity_project_path = os.getenv('UNITY_PROJECT_PATH')
    unity_dataset_name = os.getenv('UNITY_DATASET_NAME', 'solo_9')
    results_dir = os.getenv('RESULTS_DIR', './results')
    
    if not unity_project_path:
        print("❌ UNITY_PROJECT_PATH not found in .env file")
        return
    
    # Construct Unity data directory path
    unity_data_dir = os.path.join(unity_project_path, unity_dataset_name)
    
    # Find the latest/best model
    if not os.path.exists(runs_dir):
        print(f"❌ Runs directory not found: {runs_dir}")
        return
        
    model_dirs = [d for d in os.listdir(runs_dir) if d.startswith("unity_blocks_auto")]
    
    if not model_dirs:
        print("❌ No trained models found")
        return
    
    # Get the latest model
    latest_dir = sorted(model_dirs, key=lambda x: int(x.replace("unity_blocks_auto", "") or "0"))[-1]
    model_path = os.path.join(runs_dir, latest_dir, "weights", "best.pt")
    
    print(f"🤖 Using model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Test images directory
    # Test multiple images
    test_sequences = ["sequence.0", "sequence.1"]
    
    for seq in test_sequences:
        image_path = os.path.join(unity_data_dir, seq, "step0.camera1.png")
        
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue
            
        print(f"\n🖼️ Testing: {seq}/step0.camera1.png")
        
        # Use optimal confidence threshold for best detection only
        results = model(image_path, conf=0.5, save=True, project=results_dir)
        
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        
        if num_detections > 0:
            confidence_scores = results[0].boxes.conf.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            print(f"✅ Found {num_detections} high-confidence detection(s)")
            for i, (conf, box) in enumerate(zip(confidence_scores, boxes)):
                print(f"   Detection {i+1}: confidence={conf:.3f}, box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
        else:
            print(f"❌ No high-confidence detections (conf >= 0.5)")
            
            # Show best available detection
            fallback_results = model(image_path, conf=0.1, verbose=False)
            if fallback_results[0].boxes is not None and len(fallback_results[0].boxes) > 0:
                best_conf = fallback_results[0].boxes.conf.cpu().numpy().max()
                print(f"💡 Best available detection: {best_conf:.3f} confidence")

if __name__ == "__main__":
    test_yolo_best_detection()
    # Load environment variables for display
    load_dotenv()
    results_path = os.getenv('RESULTS_DIR', './results')
    print(f"\n🎯 Results saved to: {results_path}")
    print(f"📁 Check the saved images to see detected objects with bounding boxes")
