"""
Unity Instance Segmentation Analysis
Analyzes Unity's instance segmentation PNG files and converts them for YOLO training
"""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import argparse

class UnityInstanceSegmentationAnalyzer:
    """Analyze and convert Unity instance segmentation data"""
    
    def __init__(self, config_file: str = ".env"):
        # Load environment configuration
        if os.path.exists(config_file):
            load_dotenv(config_file)
        
        self.unity_solo_9_path = os.getenv('UNITY_SOLO_9_PATH', '')
        self.unity_solo_7_path = os.getenv('UNITY_SOLO_7_PATH', '')
        
    def analyze_sequence_structure(self, solo_path: str) -> Dict:
        """Analyze the structure of Unity sequence folders"""
        
        solo_dir = Path(solo_path)
        if not solo_dir.exists():
            print(f"❌ Unity solo directory not found: {solo_path}")
            return {}
        
        print(f"🔍 Analyzing Unity sequence structure in: {solo_path}")
        
        analysis = {
            'sequences': [],
            'file_types': set(),
            'camera_types': set(),
            'annotation_types': set()
        }
        
        # Find all sequence directories
        sequence_dirs = [d for d in solo_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
        print(f"Found {len(sequence_dirs)} sequences")
        
        for seq_dir in sequence_dirs:
            seq_info = {
                'name': seq_dir.name,
                'path': str(seq_dir),
                'files': [],
                'json_data': None
            }
            
            # List all files in sequence
            files = list(seq_dir.iterdir())
            for file in files:
                file_info = {
                    'name': file.name,
                    'type': file.suffix,
                    'size_mb': file.stat().st_size / (1024*1024) if file.is_file() else 0
                }
                
                seq_info['files'].append(file_info)
                analysis['file_types'].add(file.suffix)
                
                # Parse filename for camera and annotation info
                if 'camera' in file.name:
                    parts = file.name.split('.')
                    if len(parts) >= 3:
                        analysis['camera_types'].add(parts[1])  # camera1, camera2, etc.
                        if len(parts) >= 4:
                            analysis['annotation_types'].add('.'.join(parts[2:]))
                
                # Load JSON metadata if available
                if file.suffix == '.json' and 'frame_data' in file.name:
                    try:
                        with open(file, 'r') as f:
                            seq_info['json_data'] = json.load(f)
                    except Exception as e:
                        print(f"⚠ Error loading JSON {file}: {e}")
            
            analysis['sequences'].append(seq_info)
            
            # Only analyze first few sequences for speed
            if len(analysis['sequences']) >= 3:
                break
        
        return analysis
    
    def examine_instance_segmentation(self, sequence_path: str, step: str = "step0") -> Dict:
        """Examine instance segmentation PNG files in detail"""
        
        seq_dir = Path(sequence_path)
        if not seq_dir.exists():
            print(f"❌ Sequence directory not found: {sequence_path}")
            return {}
        
        print(f"🔍 Examining instance segmentation in: {seq_dir.name}")
        
        # Find instance segmentation files
        instance_seg_files = list(seq_dir.glob(f"{step}.camera*.instance segmentation.png"))
        rgb_files = list(seq_dir.glob(f"{step}.camera*.png"))
        json_files = list(seq_dir.glob(f"{step}.frame_data.json"))
        
        print(f"Found files:")
        print(f"  Instance segmentation: {len(instance_seg_files)}")
        print(f"  RGB images: {len(rgb_files)}")
        print(f"  JSON metadata: {len(json_files)}")
        
        analysis = {
            'files': {
                'instance_seg': [str(f) for f in instance_seg_files],
                'rgb': [str(f) for f in rgb_files],
                'json': [str(f) for f in json_files]
            },
            'segmentation_analysis': []
        }
        
        # Load JSON metadata for instance ID mapping
        instance_mapping = {}
        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    json_data = json.load(f)
                
                # Extract instance mapping from annotations
                for capture in json_data.get('captures', []):
                    for annotation in capture.get('annotations', []):
                        if 'InstanceSegmentation' in annotation.get('@type', ''):
                            for value in annotation.get('values', []):
                                instance_id = value.get('instanceId', 0)
                                label_name = value.get('labelName', 'unknown')
                                instance_mapping[instance_id] = label_name
                
                print(f"Instance mapping: {instance_mapping}")
                
            except Exception as e:
                print(f"⚠ Error loading JSON metadata: {e}")
        
        # Analyze each instance segmentation image
        for seg_file in instance_seg_files:
            seg_analysis = self._analyze_segmentation_image(seg_file, instance_mapping)
            analysis['segmentation_analysis'].append(seg_analysis)
        
        return analysis
    
    def _analyze_segmentation_image(self, seg_file: Path, instance_mapping: Dict) -> Dict:
        """Analyze a single instance segmentation image"""
        
        print(f"\\n📊 Analyzing: {seg_file.name}")
        
        # Load instance segmentation image
        seg_img = cv2.imread(str(seg_file), cv2.IMREAD_UNCHANGED)
        
        if seg_img is None:
            return {'error': f'Could not load {seg_file}'}
        
        print(f"Image shape: {seg_img.shape}, dtype: {seg_img.dtype}")
        
        # Analyze unique instance IDs
        if len(seg_img.shape) == 3:
            # Convert RGB to single channel instance IDs
            # Unity typically encodes instance IDs in RGB channels
            if seg_img.shape[2] == 3:
                # Convert RGB to instance ID (assuming R channel contains instance IDs)
                instance_ids = seg_img[:, :, 0]  # or combine channels if needed
            else:
                instance_ids = seg_img[:, :, 0]
        else:
            instance_ids = seg_img
        
        unique_ids = np.unique(instance_ids)
        print(f"Unique instance IDs: {unique_ids}")
        
        # Analyze each instance
        instances = []
        for instance_id in unique_ids:
            if instance_id == 0:  # Skip background
                continue
            
            mask = (instance_ids == instance_id)
            pixel_count = np.sum(mask)
            
            if pixel_count < 10:  # Skip very small instances
                continue
            
            # Get bounding box
            y_coords, x_coords = np.where(mask)
            if len(x_coords) == 0:
                continue
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1
            
            # Get class name from mapping
            class_name = instance_mapping.get(int(instance_id), f'instance_{instance_id}')
            
            instance_info = {
                'instance_id': int(instance_id),
                'class_name': class_name,
                'pixel_count': int(pixel_count),
                'bbox': [int(x_min), int(y_min), int(bbox_width), int(bbox_height)],
                'mask_available': True
            }
            
            instances.append(instance_info)
            print(f"  Instance {instance_id} ({class_name}): {pixel_count} pixels, bbox: {instance_info['bbox']}")
        
        return {
            'file': str(seg_file),
            'image_shape': seg_img.shape,
            'num_instances': len(instances),
            'instances': instances,
            'unique_ids': unique_ids.tolist()
        }
    
    def convert_to_yolo_masks(self, 
                             sequence_path: str, 
                             output_dir: str,
                             target_classes: List[str] = None) -> bool:
        """Convert Unity instance segmentation to YOLO mask format"""
        
        if not target_classes:
            target_classes = ['Module_Construction', 'block', 'cube', 'object']
        
        seq_dir = Path(sequence_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Converting Unity masks to YOLO format...")
        print(f"Source: {seq_dir.name}")
        print(f"Output: {output_dir}")
        print(f"Target classes: {target_classes}")
        
        # Find all steps in sequence
        steps = set()
        for file in seq_dir.glob("step*.png"):
            step = file.name.split('.')[0]
            steps.add(step)
        
        steps = sorted(list(steps))
        print(f"Found steps: {steps}")
        
        converted_count = 0
        
        for step in steps:
            # Find RGB and instance segmentation for this step
            rgb_files = list(seq_dir.glob(f"{step}.camera*.png"))
            seg_files = list(seq_dir.glob(f"{step}.camera*.instance segmentation.png"))
            json_files = list(seq_dir.glob(f"{step}.frame_data.json"))
            
            # Filter RGB files to exclude segmentation files
            rgb_files = [f for f in rgb_files if 'segmentation' not in f.name]
            
            if not rgb_files or not seg_files:
                continue
            
            # Load instance mapping from JSON
            instance_mapping = {}
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        json_data = json.load(f)
                    
                    for capture in json_data.get('captures', []):
                        for annotation in capture.get('annotations', []):
                            if 'InstanceSegmentation' in annotation.get('@type', ''):
                                for value in annotation.get('values', []):
                                    instance_id = value.get('instanceId', 0)
                                    label_name = value.get('labelName', 'unknown')
                                    instance_mapping[instance_id] = label_name
                except:
                    pass
            
            # Process each camera
            for rgb_file, seg_file in zip(rgb_files, seg_files):
                if self._convert_single_image(rgb_file, seg_file, output_path, 
                                            instance_mapping, target_classes, step):
                    converted_count += 1
        
        print(f"✓ Converted {converted_count} images")
        return converted_count > 0
    
    def _convert_single_image(self, rgb_file: Path, seg_file: Path, output_dir: Path,
                             instance_mapping: Dict, target_classes: List[str], step: str) -> bool:
        """Convert a single image pair to YOLO format"""
        
        # Load images
        rgb_img = cv2.imread(str(rgb_file))
        seg_img = cv2.imread(str(seg_file), cv2.IMREAD_UNCHANGED)
        
        if rgb_img is None or seg_img is None:
            return False
        
        img_height, img_width = rgb_img.shape[:2]
        
        # Get camera ID from filename
        camera_id = rgb_file.name.split('.')[1]  # camera1, camera2, etc.
        
        # Output filenames
        output_name = f"{step}_{camera_id}"
        rgb_output = output_dir / "images" / f"{output_name}.png"
        label_output = output_dir / "labels" / f"{output_name}.txt"
        mask_output = output_dir / "masks" / f"{output_name}"
        
        # Create directories
        rgb_output.parent.mkdir(parents=True, exist_ok=True)
        label_output.parent.mkdir(parents=True, exist_ok=True)
        mask_output.mkdir(parents=True, exist_ok=True)
        
        # Copy RGB image
        cv2.imwrite(str(rgb_output), rgb_img)
        
        # Process instance segmentation
        if len(seg_img.shape) == 3:
            instance_ids = seg_img[:, :, 0]  # Assuming R channel contains instance IDs
        else:
            instance_ids = seg_img
        
        yolo_annotations = []
        mask_count = 0
        
        unique_ids = np.unique(instance_ids)
        for instance_id in unique_ids:
            if instance_id == 0:  # Skip background
                continue
            
            # Get class name
            class_name = instance_mapping.get(int(instance_id), 'unknown')
            if class_name not in target_classes:
                continue
            
            # Get class index
            class_idx = target_classes.index(class_name)
            
            # Create mask
            mask = (instance_ids == instance_id).astype(np.uint8)
            pixel_count = np.sum(mask)
            
            if pixel_count < 10:  # Skip tiny instances
                continue
            
            # Get bounding box
            y_coords, x_coords = np.where(mask)
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # Convert to YOLO format (normalized center + size)
            center_x = (x_min + x_max) / 2 / img_width
            center_y = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min + 1) / img_width
            height = (y_max - y_min + 1) / img_height
            
            # Save individual mask
            mask_file = mask_output / f"mask_{mask_count}.png"
            cv2.imwrite(str(mask_file), mask * 255)
            
            # Convert mask to polygon (simplified)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Convert to normalized coordinates
                polygon = []
                for point in simplified_contour:
                    x, y = point[0]
                    polygon.extend([x / img_width, y / img_height])
                
                # YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
                if len(polygon) >= 6:  # At least 3 points
                    poly_str = ' '.join([f"{coord:.6f}" for coord in polygon])
                    yolo_annotations.append(f"{class_idx} {poly_str}")
                    mask_count += 1
        
        # Save YOLO annotations
        if yolo_annotations:
            with open(label_output, 'w') as f:
                f.write('\\n'.join(yolo_annotations))
            return True
        
        return False
    
    def visualize_segmentation(self, sequence_path: str, step: str = "step0", 
                              output_dir: str = "./visualizations") -> None:
        """Visualize Unity instance segmentation for debugging"""
        
        seq_dir = Path(sequence_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find files
        rgb_files = [f for f in seq_dir.glob(f"{step}.camera*.png") if 'segmentation' not in f.name]
        seg_files = list(seq_dir.glob(f"{step}.camera*.instance segmentation.png"))
        
        for rgb_file, seg_file in zip(rgb_files, seg_files):
            # Load images
            rgb_img = cv2.imread(str(rgb_file))
            seg_img = cv2.imread(str(seg_file), cv2.IMREAD_UNCHANGED)
            
            if rgb_img is None or seg_img is None:
                continue
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original RGB
            axes[0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'RGB: {rgb_file.name}')
            axes[0].axis('off')
            
            # Instance segmentation
            if len(seg_img.shape) == 3:
                instance_ids = seg_img[:, :, 0]
            else:
                instance_ids = seg_img
            
            # Create colored visualization
            unique_ids = np.unique(instance_ids)
            colored_seg = np.zeros((*instance_ids.shape, 3), dtype=np.uint8)
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
            for i, instance_id in enumerate(unique_ids):
                if instance_id == 0:
                    continue
                mask = instance_ids == instance_id
                colored_seg[mask] = (colors[i][:3] * 255).astype(np.uint8)
            
            axes[1].imshow(colored_seg)
            axes[1].set_title(f'Instance Segmentation: {len(unique_ids)-1} instances')
            axes[1].axis('off')
            
            # Overlay
            overlay = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) * 0.7 + colored_seg * 0.3
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('RGB + Segmentation Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            camera_id = rgb_file.name.split('.')[1]
            save_path = output_path / f"unity_segmentation_{step}_{camera_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Unity Instance Segmentation Analysis")
    parser.add_argument("--action", choices=['analyze', 'examine', 'convert', 'visualize'], 
                       required=True, help="Action to perform")
    parser.add_argument("--solo_path", type=str, help="Path to Unity solo directory")
    parser.add_argument("--sequence", type=str, help="Specific sequence directory to examine")
    parser.add_argument("--step", type=str, default="step0", help="Step to examine")
    parser.add_argument("--output", type=str, default="./unity_yolo_dataset", 
                       help="Output directory for conversion")
    parser.add_argument("--target_classes", type=str, nargs='+', 
                       default=['Module_Construction', 'block', 'cube'], 
                       help="Target class names")
    
    args = parser.parse_args()
    
    analyzer = UnityInstanceSegmentationAnalyzer()
    
    if args.action == 'analyze':
        solo_path = args.solo_path or analyzer.unity_solo_9_path
        analysis = analyzer.analyze_sequence_structure(solo_path)
        
        print("\\n" + "="*60)
        print("UNITY SEQUENCE ANALYSIS")
        print("="*60)
        print(f"File types found: {sorted(analysis['file_types'])}")
        print(f"Camera types: {sorted(analysis['camera_types'])}")
        print(f"Annotation types: {sorted(analysis['annotation_types'])}")
        
        print(f"\\nSequences ({len(analysis['sequences'])}):")
        for seq in analysis['sequences']:
            print(f"  {seq['name']}: {len(seq['files'])} files")
            
            # Show sample files
            png_files = [f for f in seq['files'] if f['type'] == '.png']
            json_files = [f for f in seq['files'] if f['type'] == '.json']
            
            print(f"    PNG files: {len(png_files)}")
            print(f"    JSON files: {len(json_files)}")
            
            # Show instance segmentation files
            instance_seg_files = [f for f in png_files if 'instance segmentation' in f['name']]
            if instance_seg_files:
                print(f"    Instance segmentation files: {len(instance_seg_files)}")
                for f in instance_seg_files[:3]:  # Show first 3
                    print(f"      {f['name']}")
    
    elif args.action == 'examine':
        if args.sequence:
            analysis = analyzer.examine_instance_segmentation(args.sequence, args.step)
        else:
            solo_path = args.solo_path or analyzer.unity_solo_9_path
            solo_dir = Path(solo_path)
            sequences = [d for d in solo_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
            if sequences:
                analysis = analyzer.examine_instance_segmentation(str(sequences[0]), args.step)
            else:
                print("❌ No sequences found")
                return
        
        print("\\n" + "="*60)
        print("INSTANCE SEGMENTATION EXAMINATION")
        print("="*60)
        
        for seg_analysis in analysis.get('segmentation_analysis', []):
            print(f"\\nFile: {Path(seg_analysis['file']).name}")
            print(f"Instances found: {seg_analysis['num_instances']}")
            for instance in seg_analysis['instances']:
                print(f"  {instance['class_name']} (ID: {instance['instance_id']}): "
                      f"{instance['pixel_count']} pixels, bbox: {instance['bbox']}")
    
    elif args.action == 'convert':
        if args.sequence:
            success = analyzer.convert_to_yolo_masks(args.sequence, args.output, args.target_classes)
        else:
            solo_path = args.solo_path or analyzer.unity_solo_9_path
            solo_dir = Path(solo_path)
            sequences = [d for d in solo_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
            
            total_converted = 0
            for seq_dir in sequences[:5]:  # Convert first 5 sequences
                print(f"\\nConverting sequence: {seq_dir.name}")
                if analyzer.convert_to_yolo_masks(str(seq_dir), args.output, args.target_classes):
                    total_converted += 1
            
            print(f"\\n✓ Converted {total_converted} sequences to YOLO format")
            print(f"Output directory: {args.output}")
    
    elif args.action == 'visualize':
        if args.sequence:
            analyzer.visualize_segmentation(args.sequence, args.step)
        else:
            solo_path = args.solo_path or analyzer.unity_solo_9_path
            solo_dir = Path(solo_path)
            sequences = [d for d in solo_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
            if sequences:
                analyzer.visualize_segmentation(str(sequences[0]), args.step)
            else:
                print("❌ No sequences found")

if __name__ == "__main__":
    main()
