#!/usr/bin/env python3
"""
Analyze why YOLO detected no objects
Compare training data vs test image
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_detection_failure():
    """Analyze why no objects were detected"""
    
    print("🔍 Analyzing Detection Failure")
    print("=" * 50)
    
    # 1. Check the test image we used
    test_image_path = "C:\\Users\\1213123\\AppData\\LocalLow\\DefaultCompany\\My project (1)\\solo_9\\sequence.0\\step0.camera1.png"
    instance_seg_path = "C:\\Users\\1213123\\AppData\\LocalLow\\DefaultCompany\\My project (1)\\solo_9\\sequence.0\\step0.camera1.instance segmentation.png"
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    # Load the RGB test image
    test_img = cv2.imread(test_image_path)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    print(f"✅ Test image shape: {test_img.shape}")
    
    # Load the instance segmentation to see if there should be objects
    if Path(instance_seg_path).exists():
        instance_img = cv2.imread(instance_seg_path)
        unique_colors = np.unique(instance_img.reshape(-1, 3), axis=0)
        print(f"✅ Instance segmentation has {len(unique_colors)} unique colors:")
        for color in unique_colors:
            count = np.sum(np.all(instance_img == color, axis=2))
            print(f"   Color {color}: {count} pixels")
    else:
        print(f"❌ Instance segmentation not found: {instance_seg_path}")
    
    # 2. Check what our training images looked like
    vis_dir = Path("visualizations/training_data")
    if vis_dir.exists():
        training_samples = list(vis_dir.glob("*.png"))
        print(f"✅ Found {len(training_samples)} training visualizations")
    else:
        print("❌ No training visualizations found")
    
    # 3. Compare image sizes and characteristics
    print(f"\n📊 Test Image Analysis:")
    print(f"   Shape: {test_img.shape}")
    print(f"   Min/Max pixel values: {test_img.min()}/{test_img.max()}")
    print(f"   Mean pixel value: {test_img.mean():.2f}")
    
    # 4. Save comparison images
    output_dir = Path("results/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the test image
    cv2.imwrite(str(output_dir / "test_rgb_image.png"), test_img)
    print(f"💾 Test RGB image saved: {output_dir / 'test_rgb_image.png'}")
    
    if Path(instance_seg_path).exists():
        instance_img = cv2.imread(instance_seg_path)
        cv2.imwrite(str(output_dir / "test_instance_segmentation.png"), instance_img)
        print(f"💾 Instance segmentation saved: {output_dir / 'test_instance_segmentation.png'}")
    
    # 5. Check if this image was used in training
    print(f"\n🔍 Checking if test image was in training data...")
    
    # Load the dataset info
    dataset_dir = Path("datasets/unity_blocks_auto")
    if dataset_dir.exists():
        train_txt = dataset_dir / "train.txt"
        val_txt = dataset_dir / "val.txt"
        test_txt = dataset_dir / "test.txt"
        
        if train_txt.exists():
            with open(train_txt, 'r') as f:
                train_images = f.read().strip().split('\n')
            print(f"   Training set: {len(train_images)} images")
            
            # Check if our test image path appears in training
            test_found = any("sequence.0" in img for img in train_images)
            print(f"   sequence.0 in training: {test_found}")
    
    print(f"\n💡 Possible reasons for no detection:")
    print(f"   1. Object too small or occluded")
    print(f"   2. Different lighting/appearance than training")
    print(f"   3. Model confidence threshold too high")
    print(f"   4. Need more training data")
    print(f"   5. Image preprocessing differences")

if __name__ == "__main__":
    analyze_detection_failure()
