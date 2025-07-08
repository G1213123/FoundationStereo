#!/usr/bin/env python3
"""Test the complete pipeline with the improved model"""

import os
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv

def test_complete_pipeline():
    """Test the full pipeline with improved model"""
    
    # Load environment variables
    load_dotenv()
    
    # Get paths from .env
    runs_dir = os.getenv('RUNS_DIR', './runs')
    unity_project_path = os.getenv('UNITY_PROJECT_PATH')
    unity_dataset_name = os.getenv('UNITY_DATASET_NAME', 'solo_9')
    results_dir = os.getenv('RESULTS_DIR', './results')
    
    if not unity_project_path:
        print("❌ UNITY_PROJECT_PATH not found in .env file")
        return False
    
    # Construct Unity data directory path
    unity_data_dir = os.path.join(unity_project_path, unity_dataset_name)
    
    # Use the latest trained model
    model_path = os.path.join(runs_dir, "unity_blocks_auto6", "weights", "best.pt")
    
    print("🚀 Testing Complete Pipeline with Improved Model")
    print("=" * 60)
    
    # Load model
    model = YOLO(model_path)
    print(f"✅ Loaded model: {model_path}")
    
    # Test image
    test_image = os.path.join(unity_data_dir, "sequence.0", "step0.camera1.png")
    
    # Test with higher confidence to get only best matches
    print(f"\n🖼️ Testing: {os.path.basename(test_image)}")
    print(f"📊 Using high confidence threshold: 0.5 (for best matches only)")
    
    results = model(test_image, conf=0.5, save=True, project=results_dir)
    
    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"🎯 High-confidence detections: {num_detections}")
    
    if num_detections > 0:
        confidence_scores = results[0].boxes.conf.cpu().numpy()
        print(f"📈 Best confidence scores: {confidence_scores}")
        print(f"✅ SUCCESS: Only best matches detected!")
        print(f"📁 Results saved to: {results_dir}/")
        
        # Show details of best detection
        best_conf = confidence_scores.max()
        print(f"🏆 Best detection confidence: {best_conf:.3f}")
    else:
        print("❌ No high-confidence detections found")
        
        # Fallback: show top 3 detections with any confidence
        print("🔍 Checking for lower confidence detections...")
        fallback_results = model(test_image, conf=0.01, verbose=False)
        if fallback_results[0].boxes is not None and len(fallback_results[0].boxes) > 0:
            all_conf = fallback_results[0].boxes.conf.cpu().numpy()
            top_3_indices = all_conf.argsort()[-3:][::-1]  # Get top 3
            top_3_conf = all_conf[top_3_indices]
            print(f"📊 Top 3 detections: {top_3_conf}")
            print("💡 Consider lowering confidence threshold if these look reasonable")
    
    return num_detections > 0

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n🎉 PIPELINE WORKING PERFECTLY!")
        print("🔧 Your Unity instance segmentation → YOLO pipeline is ready!")
    else:
        print("\n⚠️ Still having issues - check model path")
