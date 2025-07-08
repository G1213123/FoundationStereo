#!/usr/bin/env python3
"""Get only the best detection from YOLO model"""

import os
from ultralytics import YOLO
import cv2
import numpy as np
from dotenv import load_dotenv

def get_best_detection():
    """Get only the single best detection"""
    
    # Load environment variables
    load_dotenv()
    
    # Use paths from .env
    runs_dir = os.getenv('RUNS_DIR', './runs')
    unity_project_path = os.getenv('UNITY_PROJECT_PATH')
    unity_dataset_name = os.getenv('UNITY_DATASET_NAME', 'solo_9')
    results_dir = os.getenv('RESULTS_DIR', './results')
    
    if not unity_project_path:
        print("❌ UNITY_PROJECT_PATH not found in .env file")
        return
    
    # Construct Unity data directory path
    unity_data_dir = os.path.join(unity_project_path, unity_dataset_name)
    
    # Use the latest trained model
    model_path = os.path.join(runs_dir, "unity_blocks_auto6", "weights", "best.pt")
    
    print("🎯 Finding Best Detection Only")
    print("=" * 40)
    
    # Load model
    model = YOLO(model_path)
    print(f"✅ Loaded model: {model_path}")
    
    # Test image
    test_image = os.path.join(unity_data_dir, "sequence.27", "step0.camera1.png")
    
    print(f"🖼️ Analyzing: {os.path.basename(test_image)}")
    
    # Get all detections with very low threshold
    results = model(test_image, conf=0.001, verbose=False)
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        print("❌ No detections found at all")
        return
    
    # Get all confidence scores
    all_confidences = results[0].boxes.conf.cpu().numpy()
    all_boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # Find the best detection
    best_idx = np.argmax(all_confidences)
    best_confidence = all_confidences[best_idx]
    best_box = all_boxes[best_idx]
    
    print(f"📊 Total detections found: {len(all_confidences)}")
    print(f"🏆 BEST DETECTION:")
    print(f"   Confidence: {best_confidence:.4f}")
    print(f"   Bounding box: [{best_box[0]:.1f}, {best_box[1]:.1f}, {best_box[2]:.1f}, {best_box[3]:.1f}]")
    
    # Show comparison with reasonable thresholds
    reasonable_thresholds = [0.1, 0.25, 0.5, 0.7]
    print(f"\n📈 Detection counts at different thresholds:")
    for thresh in reasonable_thresholds:
        count = np.sum(all_confidences >= thresh)
        status = "✅" if count > 0 else "❌"
        print(f"   conf >= {thresh}: {count} detections {status}")
    
    # Recommend best threshold
    if best_confidence >= 0.5:
        recommended_conf = 0.5
    elif best_confidence >= 0.25:
        recommended_conf = 0.25
    elif best_confidence >= 0.1:
        recommended_conf = 0.1
    else:
        recommended_conf = best_confidence * 0.9  # Slightly below best
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Use confidence threshold: {recommended_conf:.3f}")
    print(f"   This will give you only the best detection(s)")
    
    # Test the recommended threshold
    final_results = model(test_image, conf=recommended_conf, save=True, project=results_dir)
    final_count = len(final_results[0].boxes) if final_results[0].boxes is not None else 0
    
    print(f"\n🎯 FINAL RESULT with conf={recommended_conf:.3f}:")
    print(f"   Detections: {final_count}")
    if final_count > 0:
        final_conf_scores = final_results[0].boxes.conf.cpu().numpy()
        print(f"   Confidence scores: {final_conf_scores}")
        print(f"✅ Results saved to: {results_dir}/")
    
    return recommended_conf, final_count

if __name__ == "__main__":
    try:
        recommended_conf, count = get_best_detection()
        print(f"\n🎉 SUCCESS! Use conf={recommended_conf:.3f} for best results")
    except Exception as e:
        print(f"❌ Error: {e}")
