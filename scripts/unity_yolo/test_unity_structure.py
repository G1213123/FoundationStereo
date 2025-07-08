"""
Unity Instance Segmentation Quick Test
Quickly examines Unity sequence folders to understand the structure
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

def quick_unity_analysis():
    """Quick analysis of Unity sequence structure"""
    
    # Load config
    if os.path.exists(".env"):
        load_dotenv(".env")
    
    unity_solo_9 = os.getenv('UNITY_SOLO_9_PATH', '')
    
    if not unity_solo_9:
        print("❌ UNITY_SOLO_9_PATH not set in .env file")
        return
    
    solo_dir = Path(unity_solo_9)
    if not solo_dir.exists():
        print(f"❌ Unity solo_9 directory not found: {unity_solo_9}")
        return
    
    print(f"🔍 Analyzing Unity solo_9 directory: {unity_solo_9}")
    
    # Find sequence directories
    sequence_dirs = [d for d in solo_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
    print(f"Found {len(sequence_dirs)} sequence directories")
    
    if not sequence_dirs:
        print("❌ No sequence directories found!")
        return
    
    # Analyze first sequence
    seq_dir = sequence_dirs[0]
    print(f"\\nAnalyzing first sequence: {seq_dir.name}")
    
    # List all files
    all_files = list(seq_dir.iterdir())
    print(f"Total files in sequence: {len(all_files)}")
    
    # Categorize files
    png_files = [f for f in all_files if f.suffix == '.png']
    json_files = [f for f in all_files if f.suffix == '.json']
    
    print(f"PNG files: {len(png_files)}")
    print(f"JSON files: {len(json_files)}")
    
    # Find instance segmentation files
    instance_seg_files = [f for f in png_files if 'instance segmentation' in f.name]
    rgb_files = [f for f in png_files if 'instance segmentation' not in f.name]
    
    print(f"\\nInstance segmentation files: {len(instance_seg_files)}")
    print(f"RGB files: {len(rgb_files)}")
    
    if instance_seg_files:
        print("\\nInstance segmentation files found:")
        for f in instance_seg_files[:5]:  # Show first 5
            print(f"  {f.name}")
    
    if rgb_files:
        print("\\nRGB files found:")
        for f in rgb_files[:5]:  # Show first 5
            print(f"  {f.name}")
    
    # Analyze a sample instance segmentation file
    if instance_seg_files:
        print(f"\\n📊 Analyzing sample instance segmentation file...")
        sample_seg = instance_seg_files[0]
        
        # Load the image
        seg_img = cv2.imread(str(sample_seg), cv2.IMREAD_UNCHANGED)
        if seg_img is not None:
            print(f"Image shape: {seg_img.shape}")
            print(f"Image dtype: {seg_img.dtype}")
            
            # Analyze pixel values
            if len(seg_img.shape) == 3:
                print("Multi-channel image detected")
                for i in range(seg_img.shape[2]):
                    channel_data = seg_img[:, :, i]
                    unique_vals = np.unique(channel_data)
                    print(f"  Channel {i}: {len(unique_vals)} unique values, range: {np.min(unique_vals)} - {np.max(unique_vals)}")
                
                # Try R channel as instance IDs
                instance_ids = seg_img[:, :, 0]
            else:
                print("Single-channel image detected")
                instance_ids = seg_img
                unique_vals = np.unique(instance_ids)
                print(f"Unique values: {len(unique_vals)}, range: {np.min(unique_vals)} - {np.max(unique_vals)}")
            
            # Show unique instance IDs
            unique_instances = np.unique(instance_ids)
            print(f"\\nUnique instance IDs found: {len(unique_instances)}")
            print(f"Instance IDs: {unique_instances[:20]}")  # Show first 20
            
            # Count pixels per instance
            print(f"\\nPixel counts per instance:")
            for instance_id in unique_instances[:10]:  # Show first 10
                pixel_count = np.sum(instance_ids == instance_id)
                print(f"  Instance {instance_id}: {pixel_count} pixels")
        else:
            print("❌ Could not load instance segmentation image")
    
    # Check for JSON metadata
    if json_files:
        print(f"\\n📄 Analyzing JSON metadata...")
        sample_json = json_files[0]
        
        try:
            with open(sample_json, 'r') as f:
                data = json.load(f)
            
            print(f"JSON structure:")
            print(f"  Root keys: {list(data.keys())}")
            
            # Look for captures
            if 'captures' in data:
                captures = data['captures']
                print(f"  Number of captures: {len(captures)}")
                
                if captures:
                    capture = captures[0]
                    print(f"  Sample capture keys: {list(capture.keys())}")
                    
                    # Look for annotations
                    if 'annotations' in capture:
                        annotations = capture['annotations']
                        print(f"  Number of annotations: {len(annotations)}")
                        
                        for i, annotation in enumerate(annotations[:3]):  # Show first 3
                            print(f"    Annotation {i}: {annotation.get('@type', 'unknown')}")
                            
                            # Look for instance segmentation annotation
                            if 'InstanceSegmentation' in annotation.get('@type', ''):
                                values = annotation.get('values', [])
                                print(f"      Instance segmentation values: {len(values)}")
                                
                                for j, value in enumerate(values[:5]):  # Show first 5
                                    instance_id = value.get('instanceId', 'unknown')
                                    label_name = value.get('labelName', 'unknown')
                                    print(f"        Instance {j}: ID={instance_id}, Label={label_name}")
        
        except Exception as e:
            print(f"❌ Error reading JSON: {e}")
    
    print(f"\\n✅ Analysis complete!")
    print(f"\\nNext steps:")
    print(f"1. Use this information to configure the YOLO training script")
    print(f"2. Run: python scripts/unity_instance_seg_yolo.py --action visualize")
    print(f"3. Then: python scripts/unity_instance_seg_yolo.py --action train")

if __name__ == "__main__":
    quick_unity_analysis()
