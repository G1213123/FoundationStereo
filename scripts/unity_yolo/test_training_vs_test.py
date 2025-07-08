#!/usr/bin/env python3
"""
Test YOLO on a training image vs test image
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def test_on_training_vs_test():
    """Compare YOLO performance on training vs test images"""
    
    print("🔄 Comparing Training vs Test Image Performance")
    print("=" * 60)
    
    # Load model
    model_path = "runs/unity_blocks_auto5/weights/best.pt"
    model = YOLO(model_path)
    
    # Test on a different sequence that was likely in training
    training_image = "C:\\Users\\1213123\\AppData\\LocalLow\\DefaultCompany\\My project (1)\\solo_9\\sequence.1\\step0.camera1.png"
    test_image = "C:\\Users\\1213123\\AppData\\LocalLow\\DefaultCompany\\My project (1)\\solo_9\\sequence.0\\step0.camera1.png"
    
    images_to_test = [
        ("Training-like (sequence.1)", training_image),
        ("Test (sequence.0)", test_image)
    ]
    
    for name, image_path in images_to_test:
        print(f"\n🖼️ Testing {name}: {Path(image_path).name}")
        
        if not Path(image_path).exists():
            print(f"   ❌ Image not found: {image_path}")
            continue
        
        # Test with moderate confidence
        results = model(image_path, conf=0.05, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                num_boxes = len(result.boxes)
                print(f"   📦 Detections: {num_boxes}")
                
                if num_boxes > 0:
                    # Print top 3 detections
                    for i, box in enumerate(result.boxes[:3]):
                        conf_score = float(box.conf[0]) if hasattr(box, 'conf') else "N/A"
                        print(f"      Detection {i+1}: confidence={conf_score:.3f}")
                    
                    # Save result
                    output_dir = Path("results/training_vs_test")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
                    output_path = output_dir / f"{safe_name}_detection.png"
                    annotated_img = result.plot()
                    cv2.imwrite(str(output_path), annotated_img)
                    print(f"   💾 Saved: {output_path}")
                else:
                    print(f"   ❌ No detections with conf > 0.05")
            else:
                print(f"   ❌ No boxes found")
        else:
            print(f"   ❌ No results")

if __name__ == "__main__":
    test_on_training_vs_test()
