"""
Unity Instance Segmentation JSON Channel Analyzer
Examines Unity JSON files to find which color channel contains instance data
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

def analyze_unity_segmentation_json():
    """Analyze Unity JSON to understand instance segmentation encoding"""
    
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
    
    print(f"🔍 Analyzing Unity instance segmentation JSON structure...")
    
    # Find sequence directories with objects
    sequence_dirs = [d for d in solo_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
    
    for seq_dir in sequence_dirs[:5]:  # Check first 5 sequences
        print(f"\\n📂 Sequence: {seq_dir.name}")
        
        # Find JSON files
        json_files = list(seq_dir.glob("step*.frame_data.json"))
        if not json_files:
            continue
        
        json_file = json_files[0]
        print(f"📄 JSON file: {json_file.name}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Examine captures
            for capture_idx, capture in enumerate(data.get('captures', [])):
                print(f"\\n📷 Capture {capture_idx}: {capture.get('id', 'unknown')}")
                
                # Look for annotations
                for ann_idx, annotation in enumerate(capture.get('annotations', [])):
                    ann_type = annotation.get('@type', '')
                    print(f"  📝 Annotation {ann_idx}: {ann_type}")
                    
                    if 'InstanceSegmentation' in ann_type:
                        print(f"    🎯 Found Instance Segmentation Annotation!")
                        
                        # Print all keys in the annotation
                        print(f"    Keys: {list(annotation.keys())}")
                        
                        # Look for color channel information
                        if 'values' in annotation:
                            values = annotation['values']
                            print(f"    Number of values: {len(values)}")
                            
                            for val_idx, value in enumerate(values[:3]):  # Show first 3
                                print(f"    Value {val_idx}:")
                                print(f"      Keys: {list(value.keys())}")
                                
                                # Print all value data
                                for key, val in value.items():
                                    if isinstance(val, (str, int, float, bool)):
                                        print(f"      {key}: {val}")
                                    elif isinstance(val, list) and len(val) < 10:
                                        print(f"      {key}: {val}")
                                    else:
                                        print(f"      {key}: {type(val).__name__}")
                        
                        # Look for definition or specification fields
                        for key, val in annotation.items():
                            if key not in ['@type', 'values']:
                                print(f"    {key}: {val}")
                
                # Also check if there's color channel info in capture metadata
                if 'imageFormat' in capture:
                    print(f"  📸 Image format: {capture['imageFormat']}")
                
                # Look for any segmentation-related metadata
                for key, val in capture.items():
                    if 'segmentation' in key.lower() or 'channel' in key.lower() or 'color' in key.lower():
                        print(f"  🔍 Relevant metadata - {key}: {val}")
        
        except Exception as e:
            print(f"❌ Error reading JSON: {e}")
            continue
        
        # Also examine the corresponding instance segmentation image
        seg_files = list(seq_dir.glob("step*.camera*.instance segmentation.png"))
        if seg_files:
            seg_file = seg_files[0]
            print(f"\\n🖼️  Instance segmentation image: {seg_file.name}")
            
            seg_img = cv2.imread(str(seg_file), cv2.IMREAD_UNCHANGED)
            if seg_img is not None:
                print(f"   Image shape: {seg_img.shape}")
                print(f"   Image dtype: {seg_img.dtype}")
                
                if len(seg_img.shape) == 3:
                    for channel in range(seg_img.shape[2]):
                        channel_data = seg_img[:, :, channel]
                        unique_vals = np.unique(channel_data)
                        print(f"   Channel {channel}: {len(unique_vals)} unique values")
                        if len(unique_vals) <= 10:
                            print(f"     Values: {unique_vals}")
                        else:
                            print(f"     Range: {np.min(unique_vals)} - {np.max(unique_vals)}")
        
        # Stop after first sequence with data
        break

if __name__ == "__main__":
    analyze_unity_segmentation_json()
