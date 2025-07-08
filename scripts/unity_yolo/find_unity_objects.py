"""
Find Unity sequences with actual objects
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

def find_sequences_with_objects():
    """Find Unity sequences that actually contain objects"""
    
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
    
    print(f"🔍 Searching for Unity sequences with objects...")
    
    # Find sequence directories
    sequence_dirs = [d for d in solo_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
    print(f"Total sequences: {len(sequence_dirs)}")
    
    sequences_with_objects = []
    
    # Check first 50 sequences for speed
    for i, seq_dir in enumerate(sequence_dirs[:50]):
        if i % 10 == 0:
            print(f"Checking sequence {i}...")
        
        # Look for JSON files
        json_files = list(seq_dir.glob("*.json"))
        if not json_files:
            continue
        
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            object_count = 0
            
            # Count objects in annotations
            for capture in data.get('captures', []):
                for annotation in capture.get('annotations', []):
                    if 'InstanceSegmentation' in annotation.get('@type', ''):
                        values = annotation.get('values', [])
                        object_count += len(values)
                    elif 'BoundingBox' in annotation.get('@type', ''):
                        values = annotation.get('values', [])
                        object_count += len(values)
            
            if object_count > 0:
                sequences_with_objects.append({
                    'sequence': seq_dir.name,
                    'path': str(seq_dir),
                    'object_count': object_count
                })
                
                print(f"✓ Found sequence with {object_count} objects: {seq_dir.name}")
                
                # Also check the instance segmentation image
                seg_files = list(seq_dir.glob("*.instance segmentation.png"))
                if seg_files:
                    seg_img = cv2.imread(str(seg_files[0]), cv2.IMREAD_UNCHANGED)
                    if seg_img is not None:
                        # Check different channels for instance data
                        for channel in range(min(4, seg_img.shape[2] if len(seg_img.shape) == 3 else 1)):
                            if len(seg_img.shape) == 3:
                                channel_data = seg_img[:, :, channel]
                            else:
                                channel_data = seg_img
                            
                            unique_vals = np.unique(channel_data)
                            if len(unique_vals) > 2:  # More than just 0 and background
                                print(f"  Channel {channel}: {len(unique_vals)} unique values: {unique_vals[:10]}")
                
                # Stop after finding 5 good sequences
                if len(sequences_with_objects) >= 5:
                    break
        
        except Exception as e:
            continue
    
    print(f"\\n✅ Found {len(sequences_with_objects)} sequences with objects")
    
    if sequences_with_objects:
        print("\\nSequences with objects:")
        for seq in sequences_with_objects:
            print(f"  {seq['sequence']}: {seq['object_count']} objects")
        
        # Test with the first good sequence
        test_seq = sequences_with_objects[0]
        print(f"\\n🔍 Testing with sequence: {test_seq['sequence']}")
        
        return test_seq['path']
    else:
        print("\\n❌ No sequences with objects found in first 50 sequences")
        print("You may need to check more sequences or verify Unity data generation")
        return None

if __name__ == "__main__":
    good_sequence = find_sequences_with_objects()
    
    if good_sequence:
        print(f"\\n🚀 You can test with this sequence:")
        print(f"python scripts/unity_instance_seg_yolo.py --action visualize --sequence '{good_sequence}'")
    else:
        print("\\n⚠ Try checking your Unity data generation settings")
        print("Make sure objects are being spawned and annotated correctly")
