"""
Unity Instance Segmentation to YOLO Training Pipeline
Directly uses Unity's instance segmentation PNG files for YOLO training
"""

import os
import cv2
import json
import numpy as np
import yaml
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dotenv import load_dotenv

class UnityInstanceSegmentationYOLO:
    """YOLO training pipeline using Unity's instance segmentation PNG files"""
    
    def __init__(self, config_file: str = ".env"):
        # Load environment configuration
        if os.path.exists(config_file):
            load_dotenv(config_file)
        
        # Unity paths
        self.unity_project_path = os.getenv('UNITY_PROJECT_PATH', '')
        self.unity_dataset_name = os.getenv('UNITY_DATASET_NAME', 'solo_9')
        
        # Other paths
        self.foundation_stereo_root = os.getenv('FOUNDATION_STEREO_ROOT', '.')
        self.python_env = os.getenv('PYTHON_ENV', 'python')
        
        self.project_name = os.getenv('PROJECT_NAME', 'unity_instance_seg')
        self.dataset_root = os.path.join(os.getenv('DATASET_ROOT', './datasets'), self.project_name)
        self.intrinsics_file = os.getenv('INTRINSICS_FILE', './unity_seq49_intrinsics_cm.txt')
        
        # Training parameters
        self.model_size = os.getenv('MODEL_SIZE', 'n')
        self.epochs = int(os.getenv('EPOCHS', '50'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '8'))
        self.image_size = int(os.getenv('IMAGE_SIZE', '640'))
        self.device = os.getenv('DEVICE', '0')
        
        # Output directories
        self.runs_dir = os.getenv('RUNS_DIR', './runs')
        self.results_dir = os.getenv('RESULTS_DIR', './results')
        self.visualizations_dir = os.getenv('VISUALIZATIONS_DIR', './visualizations')
        
        # Create output directories
        for dir_path in [self.dataset_root, self.runs_dir, self.results_dir, self.visualizations_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def extract_unity_instance_data(self, 
                                   solo_path: str,
                                   target_classes: List[str] = None,
                                   max_sequences: int = 10) -> List[Dict]:
        """
        Extract instance segmentation data from Unity's PNG files
        
        Args:
            solo_path: Path to Unity solo directory
            target_classes: List of class names to extract
            max_sequences: Maximum number of sequences to process
        
        Returns:
            List of processed instance data
        """
        if not target_classes:
            target_classes = ['Module_Construction', 'block', 'cube', 'object']
        
        solo_dir = Path(solo_path)
        if not solo_dir.exists():
            print(f"❌ Unity solo directory not found: {solo_path}")
            return []
        
        print(f"🔍 Extracting Unity instance segmentation data from: {solo_path}")
        print(f"Target classes: {target_classes}")
        
        # Find all sequence directories
        sequence_dirs = [d for d in solo_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
        sequence_dirs = sequence_dirs[:max_sequences]  # Limit number of sequences
        print(f"Processing {len(sequence_dirs)} sequences")
        
        extracted_data = []
        
        for seq_dir in sequence_dirs:
            print(f"\\nProcessing sequence: {seq_dir.name}")
            
            # Find all step files
            steps = set()
            for file in seq_dir.glob("step*.png"):
                step = file.name.split('.')[0]
                steps.add(step)
            
            steps = sorted(list(steps))
            print(f"  Found {len(steps)} steps")
            
            # Load instance mapping from JSON
            mapping_data = self._load_instance_mapping(seq_dir)
            
            for step in steps:
                step_data = self._process_step(seq_dir, step, mapping_data, target_classes)
                if step_data:
                    extracted_data.extend(step_data)
        
        print(f"\\n✓ Extracted {len(extracted_data)} image-segmentation pairs")
        return extracted_data
    
    def _load_instance_mapping(self, seq_dir: Path) -> Dict:
        """Load instance ID to class name mapping from JSON metadata with color channel info"""
        
        json_files = list(seq_dir.glob("step*.frame_data.json"))
        instance_mapping = {}
        color_mapping = {}  # Maps RGB color to instance info
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract instance mapping from annotations
                for capture in data.get('captures', []):
                    for annotation in capture.get('annotations', []):
                        if 'InstanceSegmentation' in annotation.get('@type', ''):
                            # Unity instance segmentation format
                            instances = annotation.get('instances', [])
                            
                            for instance in instances:
                                instance_id = instance.get('instanceId', 0)
                                label_name = instance.get('labelName', 'unknown')
                                color = instance.get('color', [0, 0, 0, 255])  # RGBA
                                
                                instance_mapping[instance_id] = label_name
                                
                                # Store color mapping - Unity JSON uses RGBA, but OpenCV loads as BGR
                                if len(color) >= 3:
                                    # Unity JSON: [R, G, B, A] but OpenCV loads PNG as BGR
                                    # So we need to convert: JSON [R,G,B] -> OpenCV [B,G,R]
                                    bgr_key = (color[2], color[1], color[0])  # Convert RGB to BGR
                                    color_mapping[bgr_key] = {
                                        'instance_id': instance_id,
                                        'label_name': label_name,
                                        'color': color,
                                        'bgr_color': bgr_key
                                    }
                            
                            # Also check old format for compatibility
                            for value in annotation.get('values', []):
                                instance_id = value.get('instanceId', 0)
                                label_name = value.get('labelName', 'unknown')
                                instance_mapping[instance_id] = label_name
                
            except Exception as e:
                print(f"⚠ Error loading JSON {json_file}: {e}")
                continue
        
        # Return both mappings
        return {'instance_mapping': instance_mapping, 'color_mapping': color_mapping}
    
    def _process_step(self, seq_dir: Path, step: str, mapping_data: Dict, 
                     target_classes: List[str]) -> List[Dict]:
        """Process a single step to extract RGB and instance segmentation pairs"""
        
        # Find RGB and instance segmentation files for this step
        rgb_files = [f for f in seq_dir.glob(f"{step}.camera*.png") if 'segmentation' not in f.name]
        seg_files = list(seq_dir.glob(f"{step}.camera*.instance segmentation.png"))
        
        if not rgb_files or not seg_files:
            return []
        
        step_data = []
        
        # Match RGB and segmentation files by camera
        for rgb_file in rgb_files:
            # Extract camera ID from filename (e.g., "step0.camera1.png" -> "camera1")
            camera_id = '.'.join(rgb_file.name.split('.')[1:-1])  # camera1, camera2, etc.
            
            # Find corresponding segmentation file
            seg_file = None
            for sf in seg_files:
                if camera_id in sf.name:
                    seg_file = sf
                    break
            
            if seg_file is None:
                continue
            
            # Process this RGB-segmentation pair
            pair_data = self._process_image_pair(rgb_file, seg_file, mapping_data, 
                                               target_classes, seq_dir.name, step, camera_id)
            if pair_data:
                step_data.append(pair_data)
        
        return step_data
    
    def _process_image_pair(self, rgb_file: Path, seg_file: Path, mapping_data: Dict,
                           target_classes: List[str], sequence: str, step: str, camera_id: str) -> Optional[Dict]:
        """Process a single RGB-segmentation pair using Unity's color encoding"""
        
        # Load images
        rgb_img = cv2.imread(str(rgb_file))
        seg_img = cv2.imread(str(seg_file), cv2.IMREAD_UNCHANGED)
        
        if rgb_img is None or seg_img is None:
            print(f"⚠ Could not load images: {rgb_file.name}, {seg_file.name}")
            return None
        
        img_height, img_width = rgb_img.shape[:2]
        
        # Extract mappings
        instance_mapping = mapping_data.get('instance_mapping', {})
        color_mapping = mapping_data.get('color_mapping', {})
        
        print(f"\\nProcessing {seg_file.name}")
        print(f"Available color mappings: {list(color_mapping.keys())}")
        
        # Find valid instances using Unity's color encoding
        valid_instances = []
        
        if len(seg_img.shape) == 3:
            # Unity encodes instances as specific RGB colors
            # Create a mapping from each unique RGB combination to instance info
            
            # Get unique RGB combinations in the image
            rgb_img_reshaped = seg_img[:, :, :3].reshape(-1, 3)
            unique_colors = np.unique(rgb_img_reshaped, axis=0)
            
            print(f"Unique colors in image: {len(unique_colors)}")
            for color in unique_colors:
                color_tuple = tuple(color)
                pixel_count = np.sum(np.all(seg_img[:, :, :3] == color, axis=2))
                
                if color_tuple in color_mapping:
                    instance_info = color_mapping[color_tuple]
                    print(f"  Color {color_tuple}: {instance_info['label_name']} (ID: {instance_info['instance_id']}) - {pixel_count} pixels")
                else:
                    print(f"  Color {color_tuple}: Unknown - {pixel_count} pixels")
            
            # Process each known color
            for color_tuple, instance_info in color_mapping.items():
                # Create mask for this color
                color_array = np.array(color_tuple)
                mask = np.all(seg_img[:, :, :3] == color_array, axis=2).astype(np.uint8)
                pixel_count = np.sum(mask)
                
                if pixel_count < 50:  # Skip very small instances
                    continue
                
                instance_id = instance_info['instance_id']
                class_name = instance_info['label_name']
                
                # Check if this class is in our target classes
                if target_classes and class_name not in target_classes:
                    # Try fuzzy matching for common Unity class names
                    fuzzy_match = False
                    for target in target_classes:
                        if target.lower() in class_name.lower() or class_name.lower() in target.lower():
                            class_name = target  # Normalize to target class name
                            fuzzy_match = True
                            break
                    
                    if not fuzzy_match:
                        continue
                
                # Get bounding box
                y_coords, x_coords = np.where(mask)
                if len(x_coords) == 0:
                    continue
                
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                # Create polygon from mask contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                
                # Use largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Convert to normalized polygon coordinates
                polygon = []
                for point in simplified_contour:
                    x, y = point[0]
                    polygon.extend([x / img_width, y / img_height])
                
                # Need at least 3 points for a valid polygon
                if len(polygon) >= 6:
                    valid_instances.append({
                        'instance_id': int(instance_id),
                        'class_name': class_name,
                        'polygon': polygon,
                        'bbox': [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                        'pixel_count': int(pixel_count),
                        'mask': mask,
                        'color': color_tuple
                    })
                    
                    print(f"✓ Extracted {class_name} (ID: {instance_id}) with {pixel_count} pixels")
        
        if not valid_instances:
            print("❌ No valid instances found")
            return None
        
        return {
            'rgb_file': str(rgb_file),
            'seg_file': str(seg_file),
            'sequence': sequence,
            'step': step,
            'camera_id': camera_id,
            'image_size': [img_width, img_height],
            'instances': valid_instances,
            'rgb_image': rgb_img,
            'seg_image': seg_img
        }
    
    def create_yolo_dataset(self, 
                           extracted_data: List[Dict],
                           train_split: float = 0.8,
                           val_split: float = 0.15) -> bool:
        """Create YOLO dataset from extracted Unity instance data"""
        
        if not extracted_data:
            print("❌ No extracted data to process")
            return False
        
        # Get unique class names
        all_classes = set()
        for data in extracted_data:
            for instance in data['instances']:
                all_classes.add(instance['class_name'])
        
        self.class_names = sorted(list(all_classes))
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"✓ Found classes: {self.class_names}")
        
        # Create dataset structure
        for split in ['train', 'val', 'test']:
            os.makedirs(f"{self.dataset_root}/images/{split}", exist_ok=True)
            os.makedirs(f"{self.dataset_root}/labels/{split}", exist_ok=True)
        
        # Shuffle and split data
        np.random.shuffle(extracted_data)
        n_total = len(extracted_data)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_data = extracted_data[:n_train]
        val_data = extracted_data[n_train:n_train + n_val]
        test_data = extracted_data[n_train + n_val:]
        
        # Process each split
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            self._process_dataset_split(split_data, split_name)
        
        # Create dataset.yaml
        self._create_dataset_yaml()
        
        print(f"✓ YOLO dataset created:")
        print(f"  Train: {len(train_data)} images")
        print(f"  Val: {len(val_data)} images")
        print(f"  Test: {len(test_data)} images")
        print(f"  Classes: {len(self.class_names)}")
        
        return True
    
    def _process_dataset_split(self, split_data: List[Dict], split_name: str) -> None:
        """Process data for a specific dataset split"""
        
        for i, data in enumerate(split_data):
            # Create unique filename
            filename = f"{data['sequence']}_{data['step']}_{data['camera_id']}"
            
            # Save RGB image
            img_path = f"{self.dataset_root}/images/{split_name}/{filename}.png"
            cv2.imwrite(img_path, data['rgb_image'])
            
            # Create YOLO annotation
            img_width, img_height = data['image_size']
            yolo_lines = []
            
            for instance in data['instances']:
                class_idx = self.class_mapping[instance['class_name']]
                polygon = instance['polygon']
                
                # Clamp polygon coordinates to [0, 1]
                clamped_polygon = []
                for coord in polygon:
                    clamped_polygon.append(max(0, min(1, coord)))
                
                # YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
                if len(clamped_polygon) >= 6:  # At least 3 points
                    poly_str = ' '.join([f"{coord:.6f}" for coord in clamped_polygon])
                    yolo_lines.append(f"{class_idx} {poly_str}")
            
            # Save YOLO annotation file
            if yolo_lines:
                label_path = f"{self.dataset_root}/labels/{split_name}/{filename}.txt"
                with open(label_path, 'w') as f:
                    f.write('\\n'.join(yolo_lines))
        
        print(f"  Processed {len(split_data)} images for {split_name} split")
    
    def _create_dataset_yaml(self) -> None:
        """Create YOLO dataset configuration"""
        
        dataset_config = {
            'path': os.path.abspath(self.dataset_root),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = f"{self.dataset_root}/dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✓ Dataset config saved: {yaml_path}")
    
    def train_yolo_model(self) -> str:
        """Train YOLO model on Unity instance segmentation data"""
        
        print(f"🚀 Starting YOLO training on Unity instance segmentation data...")
        print(f"Model: YOLOv8{self.model_size}-seg")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.image_size}")
        
        # Load model
        model = YOLO(f'yolov8{self.model_size}-seg.pt')
        
        # Train
        results = model.train(
            data=f'{self.dataset_root}/dataset.yaml',
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            project=self.runs_dir,
            name=self.project_name,
            save=True,
            plots=True,
            device=self.device,
            patience=10,  # Early stopping
            save_period=5  # Save every 5 epochs
        )
        
        best_model = f'{self.runs_dir}/{self.project_name}/weights/best.pt'
        print(f"✓ Training completed! Best model: {best_model}")
        
        return best_model
    
    def visualize_training_data(self, extracted_data: List[Dict], num_samples: int = 5) -> None:
        """Visualize training data for debugging"""
        
        output_dir = Path(self.visualizations_dir) / "training_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        samples = np.random.choice(extracted_data, min(num_samples, len(extracted_data)), replace=False)
        
        for i, data in enumerate(samples):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original RGB
            rgb_img = data['rgb_image']
            axes[0].imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'RGB: {data["sequence"]} {data["step"]} {data["camera_id"]}')
            axes[0].axis('off')
            
            # Instance segmentation
            seg_img = data['seg_image']
            if len(seg_img.shape) == 3:
                instance_ids = seg_img[:, :, 0]
            else:
                instance_ids = seg_img
            
            unique_ids = np.unique(instance_ids)
            colored_seg = np.zeros((*instance_ids.shape, 3), dtype=np.uint8)
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
            for j, instance_id in enumerate(unique_ids):
                if instance_id == 0:
                    continue
                mask = instance_ids == instance_id
                colored_seg[mask] = (colors[j % len(colors)][:3] * 255).astype(np.uint8)
            
            axes[1].imshow(colored_seg)
            axes[1].set_title(f'Instance Segmentation: {len(data["instances"])} objects')
            axes[1].axis('off')
            
            # Overlay with labels
            overlay = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) * 0.7 + colored_seg * 0.3
            
            for instance in data['instances']:
                # Draw bounding box
                x, y, w, h = instance['bbox']
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 2)
                
                # Add label
                label = f"{instance['class_name']} (ID: {instance['instance_id']})"
                cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Labeled Objects')
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = output_dir / f"sample_{i:02d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Training data visualization saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Unity Instance Segmentation YOLO Training")
    parser.add_argument("--action", choices=['extract', 'train', 'visualize', 'full'], 
                       required=True, help="Action to perform")
    parser.add_argument("--solo_path", type=str, help="Path to Unity solo directory")
    parser.add_argument("--sequence", type=str, help="Path to specific sequence directory")
    parser.add_argument("--target_classes", type=str, nargs='+', 
                       default=['Module_Construction', 'block', 'cube'], 
                       help="Target class names")
    parser.add_argument("--max_sequences", type=int, default=10, 
                       help="Maximum number of sequences to process")
    parser.add_argument("--config", type=str, default=".env", help="Configuration file")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = UnityInstanceSegmentationYOLO(args.config)
    
    # Determine which path to use
    if args.sequence:
        # Use specific sequence directory
        sequence_path = args.sequence
        solo_path = str(Path(args.sequence).parent)
    else:
        # Construct path from Unity project path and dataset name
        if not trainer.unity_project_path:
            raise ValueError("UNITY_PROJECT_PATH environment variable not set")
        
        solo_path = args.solo_path or os.path.join(trainer.unity_project_path, trainer.unity_dataset_name)
        sequence_path = None
    
    if args.action == 'extract':
        print("🔄 Extracting Unity instance segmentation data...")
        
        if sequence_path:
            # Process single sequence
            print(f"Processing single sequence: {Path(sequence_path).name}")
            sequences = [Path(sequence_path)]
            extracted_data = []
            
            # Load instance mapping
            mapping_data = trainer._load_instance_mapping(sequences[0])
            
            # Find steps in this sequence
            steps = set()
            for file in sequences[0].glob("step*.png"):
                step = file.name.split('.')[0]
                steps.add(step)
            
            steps = sorted(list(steps))
            print(f"Found {len(steps)} steps in sequence")
            
            for step in steps:
                step_data = trainer._process_step(sequences[0], step, mapping_data, args.target_classes)
                if step_data:
                    extracted_data.extend(step_data)
        else:
            # Process multiple sequences from solo directory
            extracted_data = trainer.extract_unity_instance_data(solo_path, args.target_classes, args.max_sequences)
        
        if extracted_data:
            print("✅ Extraction completed!")
            print(f"Total images processed: {len(extracted_data)}")
            
            # Show sample data
            for i, data in enumerate(extracted_data[:3]):
                print(f"\\nSample {i+1}: {data['sequence']} {data['step']} {data['camera_id']}")
                print(f"  Instances: {len(data['instances'])}")
                for instance in data['instances']:
                    print(f"    {instance['class_name']}: {instance['pixel_count']} pixels")
        else:
            print("❌ No data extracted!")
    
    elif args.action == 'train':
        print("🔄 Full training pipeline...")
        
        # Extract data
        extracted_data = trainer.extract_unity_instance_data(solo_path, args.target_classes, args.max_sequences)
        if not extracted_data:
            print("❌ No data extracted!")
            return
        
        # Create dataset
        success = trainer.create_yolo_dataset(extracted_data)
        if not success:
            print("❌ Dataset creation failed!")
            return
        
        # Train model
        model_path = trainer.train_yolo_model()
        print(f"🎯 Training complete! Model saved: {model_path}")
    
    elif args.action == 'visualize':
        print("🔄 Visualizing Unity instance segmentation data...")
        
        if sequence_path:
            # Process single sequence
            sequences = [Path(sequence_path)]
            extracted_data = []
            
            # Load instance mapping
            mapping_data = trainer._load_instance_mapping(sequences[0])
            
            # Find steps in this sequence
            steps = set()
            for file in sequences[0].glob("step*.png"):
                step = file.name.split('.')[0]
                steps.add(step)
            
            steps = sorted(list(steps))
            
            for step in steps:
                step_data = trainer._process_step(sequences[0], step, mapping_data, args.target_classes)
                if step_data:
                    extracted_data.extend(step_data)
        else:
            # Process multiple sequences
            extracted_data = trainer.extract_unity_instance_data(solo_path, args.target_classes, args.max_sequences)
        
        if extracted_data:
            trainer.visualize_training_data(extracted_data, num_samples=10)
            print("✅ Visualization completed!")
        else:
            print("❌ No data to visualize!")
    
    elif args.action == 'full':
        print("🚀 Full Unity Instance Segmentation YOLO Pipeline")
        
        # Step 1: Extract data
        print("\\nStep 1: Extracting Unity instance segmentation data...")
        extracted_data = trainer.extract_unity_instance_data(solo_path, args.target_classes, args.max_sequences)
        if not extracted_data:
            print("❌ No data extracted!")
            return
        
        # Step 2: Visualize samples
        print("\\nStep 2: Creating visualizations...")
        trainer.visualize_training_data(extracted_data, num_samples=5)
        
        # Step 3: Create dataset
        print("\\nStep 3: Creating YOLO dataset...")
        success = trainer.create_yolo_dataset(extracted_data)
        if not success:
            print("❌ Dataset creation failed!")
            return
        
        # Step 4: Train model
        print("\\nStep 4: Training YOLO model...")
        model_path = trainer.train_yolo_model()
        
        print(f"\\n🎉 Full pipeline completed!")
        print(f"Model: {model_path}")
        print(f"Dataset: {trainer.dataset_root}")
        print(f"Visualizations: {trainer.visualizations_dir}")

if __name__ == "__main__":
    main()
