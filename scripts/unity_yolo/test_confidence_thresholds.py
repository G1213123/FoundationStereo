#!/usr/bin/env python3
"""
Test YOLO with different confidence thresholds and analyze predictions
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def test_yolo_with_different_thresholds():
    """Test YOLO with different confidence thresholds"""
    
    print("🔧 Testing YOLO with Different Confidence Thresholds")
    print("=" * 60)
    
    # Load model
    model_path = "runs/unity_blocks_auto5/weights/best.pt"
    model = YOLO(model_path)
    print(f"✅ Using model: {model_path}")
    
    # Test image
    test_image = "C:\\Users\\1213123\\AppData\\LocalLow\\DefaultCompany\\My project (1)\\solo_9\\sequence.0\\step0.camera1.png"
    print(f"🖼️ Test image: {Path(test_image).name}")
    
    # Test with different confidence thresholds
    confidence_thresholds = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
    
    for conf in confidence_thresholds:
        print(f"\n🎯 Testing with confidence threshold: {conf}")
        
        try:
            # Run inference with specific confidence
            results = model(test_image, conf=conf, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                
                # Check boxes
                if hasattr(result, 'boxes') and result.boxes is not None:
                    num_boxes = len(result.boxes)
                    print(f"   📦 Boxes detected: {num_boxes}")
                    
                    if num_boxes > 0:
                        # Print box details
                        for i, box in enumerate(result.boxes):
                            conf_score = float(box.conf[0]) if hasattr(box, 'conf') else "N/A"
                            cls_id = int(box.cls[0]) if hasattr(box, 'cls') else "N/A"
                            print(f"      Box {i}: confidence={conf_score:.3f}, class={cls_id}")
                
                # Check masks
                if hasattr(result, 'masks') and result.masks is not None:
                    num_masks = len(result.masks)
                    print(f"   🎭 Masks detected: {num_masks}")
                
                # Save result if detections found
                if (hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0):
                    output_dir = Path("results/confidence_test")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    annotated_img = result.plot()
                    output_path = output_dir / f"detection_conf_{conf}.png"
                    cv2.imwrite(str(output_path), annotated_img)
                    print(f"   💾 Saved: {output_path}")
                else:
                    print(f"   ❌ No detections")
            else:
                print(f"   ❌ No results returned")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
    
    # Also try with different IoU thresholds
    print(f"\n🔧 Testing with low confidence (0.01) and different IoU thresholds:")
    iou_thresholds = [0.1, 0.3, 0.5, 0.7]
    
    for iou in iou_thresholds:
        try:
            results = model(test_image, conf=0.01, iou=iou, verbose=False)
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                num_detections = len(results[0].boxes)
                print(f"   IoU {iou}: {num_detections} detections")
            else:
                print(f"   IoU {iou}: 0 detections")
        except Exception as e:
            print(f"   IoU {iou}: Error - {e}")

if __name__ == "__main__":
    test_yolo_with_different_thresholds()
