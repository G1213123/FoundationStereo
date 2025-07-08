"""
Unity YOLO training pipeline for automatically annotated data
Works with Unity's built-in annotation system
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

class UnityAutoYOLO:
    """YOLO training pipeline for Unity's automatic annotations"""
    
    def __init__(self, config_file: str = ".env"):
        # Load environment configuration
        if os.path.exists(config_file):
            load_dotenv(config_file)
        
        self.unity_project_path = os.getenv('UNITY_PROJECT_PATH', '')
        self.unity_solo_9_path = os.getenv('UNITY_SOLO_9_PATH', '')
        self.unity_solo_7_path = os.getenv('UNITY_SOLO_7_PATH', '')
        self.foundation_stereo_root = os.getenv('FOUNDATION_STEREO_ROOT', '.')
        self.python_env = os.getenv('PYTHON_ENV', 'python')
        
        self.project_name = os.getenv('PROJECT_NAME', 'unity_blocks')
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
    
    def extract_unity_annotations(self, 
                                 solo_path: str,
                                 target_classes: List[str] = None) -> List[Dict]:
        """
        Extract annotations from Unity's automatic annotation system
        
        Args:
            solo_path: Path to Unity solo directory (e.g., solo_9)
            target_classes: List of class names to extract
        
        Returns:
            List of annotation dictionaries
        """
        if not target_classes:
            target_classes = ['Module_Construction', 'block', 'cube', 'object']
        
        annotations = []
        solo_dir = Path(solo_path)
        
        if not solo_dir.exists():
            print(f"❌ Unity solo directory not found: {solo_path}")
            return annotations
        
        print(f"🔍 Searching for annotations in: {solo_path}")
        
        # Find all sequence directories
        sequence_dirs = [d for d in solo_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
        print(f"Found {len(sequence_dirs)} sequences")
        
        for seq_dir in sequence_dirs:
            # Look for frame data and annotation files
            json_files = list(seq_dir.glob("step*.frame_data.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract annotations for each camera
                    for capture in data.get('captures', []):
                        if capture.get('@type') == 'type.unity.com/unity.solo.RGBCamera':
                            image_filename = capture.get('filename', '')
                            image_path = seq_dir / image_filename
                            
                            if not image_path.exists():
                                continue
                            
                            # Extract bounding box annotations
                            bboxes = []
                            for annotation in capture.get('annotations', []):
                                if 'BoundingBox2DAnnotation' in annotation.get('@type', ''):
                                    # 2D bounding box annotations
                                    for value in annotation.get('values', []):
                                        class_name = value.get('labelName', 'unknown')
                                        
                                        if target_classes and class_name not in target_classes:
                                            continue
                                        
                                        # Extract 2D bounding box coordinates
                                        bbox_2d = value.get('boundingBox', {})
                                        if bbox_2d:
                                            x = bbox_2d.get('x', 0)
                                            y = bbox_2d.get('y', 0)
                                            width = bbox_2d.get('width', 0)
                                            height = bbox_2d.get('height', 0)
                                            
                                            bboxes.append({
                                                'class_name': class_name,
                                                'bbox_2d': [x, y, width, height],  # x, y, width, height
                                                'instance_id': value.get('instanceId', 0)
                                            })
                                
                                elif 'BoundingBox3DAnnotation' in annotation.get('@type', ''):
                                    # 3D bounding box annotations - convert to 2D projection
                                    for value in annotation.get('values', []):
                                        class_name = value.get('labelName', 'unknown')
                                        
                                        if target_classes and class_name not in target_classes:
                                            continue
                                        
                                        # For 3D boxes, we'll need to project to 2D
                                        # This is a simplified approach - you might need more sophisticated projection
                                        translation = value.get('translation', [0, 0, 0])
                                        size = value.get('size', [1, 1, 1])
                                        
                                        # Simplified 2D bounding box estimation
                                        # You would need proper camera projection here
                                        img_dims = capture.get('dimension', [650, 400])
                                        img_width, img_height = img_dims
                                        
                                        # Placeholder projection (needs proper implementation)
                                        center_x = img_width * 0.5  # Simplified
                                        center_y = img_height * 0.5
                                        bbox_width = min(img_width * 0.3, size[0] * 10)  # Scaled size
                                        bbox_height = min(img_height * 0.3, size[1] * 10)
                                        
                                        x = center_x - bbox_width / 2
                                        y = center_y - bbox_height / 2
                                        
                                        bboxes.append({
                                            'class_name': class_name,
                                            'bbox_2d': [x, y, bbox_width, bbox_height],
                                            'instance_id': value.get('instanceId', 0),
                                            'is_3d_projected': True
                                        })
                            
                            if bboxes:
                                annotations.append({
                                    'image_path': str(image_path),
                                    'sequence': seq_dir.name,
                                    'step': json_file.stem,
                                    'camera': capture['id'],
                                    'dimensions': capture.get('dimension', [650, 400]),
                                    'bboxes': bboxes
                                })
                
                except Exception as e:
                    print(f"⚠ Error processing {json_file}: {e}")
                    continue
        
        print(f"✓ Extracted {len(annotations)} annotated images")
        return annotations
    
    def convert_to_yolo_format(self, 
                              annotations: List[Dict],
                              train_split: float = 0.8,
                              val_split: float = 0.15) -> bool:
        """Convert Unity annotations to YOLO format"""
        
        if not annotations:
            print("❌ No annotations to convert")
            return False
        
        # Extract unique class names
        all_classes = set()
        for ann in annotations:
            for bbox in ann['bboxes']:
                all_classes.add(bbox['class_name'])
        
        self.class_names = sorted(list(all_classes))
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"✓ Found classes: {self.class_names}")
        
        # Create dataset structure
        for split in ['train', 'val', 'test']:
            os.makedirs(f"{self.dataset_root}/images/{split}", exist_ok=True)
            os.makedirs(f"{self.dataset_root}/labels/{split}", exist_ok=True)
        
        # Shuffle and split
        np.random.shuffle(annotations)
        n_total = len(annotations)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_anns = annotations[:n_train]
        val_anns = annotations[n_train:n_train + n_val]
        test_anns = annotations[n_train + n_val:]
        
        # Process each split
        for split_name, split_anns in [('train', train_anns), ('val', val_anns), ('test', test_anns)]:
            self._process_split(split_anns, split_name)
        
        # Create dataset.yaml
        self._create_dataset_yaml()
        
        print(f"✓ Dataset created:")
        print(f"  Train: {len(train_anns)} images")
        print(f"  Val: {len(val_anns)} images")
        print(f"  Test: {len(test_anns)} images")
        
        return True
    
    def _process_split(self, annotations: List[Dict], split_name: str) -> None:
        """Process annotations for a dataset split"""
        
        for ann in annotations:
            # Copy image
            src_image_path = ann['image_path']
            dst_image_path = f"{self.dataset_root}/images/{split_name}/{Path(src_image_path).name}"
            
            try:
                shutil.copy2(src_image_path, dst_image_path)
            except Exception as e:
                print(f"⚠ Could not copy {src_image_path}: {e}")
                continue
            
            # Convert annotations to YOLO format
            img_width, img_height = ann['dimensions']
            yolo_lines = []
            
            for bbox in ann['bboxes']:
                class_id = self.class_mapping.get(bbox['class_name'], 0)
                
                # Convert Unity bbox to YOLO format
                if 'is_3d_projected' in bbox and bbox['is_3d_projected']:
                    # Already processed 3D projection
                    x, y, width, height = bbox['bbox_2d']
                else:
                    # Unity 2D bounding box format
                    x, y, width, height = bbox['bbox_2d']
                
                # Convert to YOLO format (normalized center coordinates + width/height)
                center_x = (x + width / 2) / img_width
                center_y = (y + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # Clamp values to [0, 1]
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            # Save YOLO annotation file
            if yolo_lines:
                label_path = f"{self.dataset_root}/labels/{split_name}/{Path(src_image_path).stem}.txt"
                with open(label_path, 'w') as f:
                    f.write("\\n".join(yolo_lines))
    
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
    
    def train_model(self) -> str:
        """Train YOLO model with configured parameters"""
        
        print(f"🚀 Starting YOLO training...")
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
            device=self.device
        )
        
        best_model = f'{self.runs_dir}/{self.project_name}/weights/best.pt'
        print(f"✓ Training completed! Best model: {best_model}")
        
        return best_model
    
    def integrate_with_foundation_stereo(self, 
                                       model_path: str,
                                       left_image: str,
                                       right_image: str) -> List[Dict]:
        """
        Complete pipeline: YOLO segmentation + FoundationStereo depth
        """
        
        print("🔄 Running integrated YOLO + FoundationStereo pipeline...")
        
        # Step 1: Run FoundationStereo for depth
        depth_output = os.path.join(self.results_dir, f"depth_{Path(left_image).stem}")
        os.makedirs(depth_output, exist_ok=True)
        
        cmd = f'"{self.python_env}" scripts/run_demo.py --left_file "{left_image}" --right_file "{right_image}" --intrinsic_file "{self.intrinsics_file}" --out_dir "{depth_output}" --valid_iters 32'
        
        print("Running FoundationStereo...")
        result = os.system(cmd)
        if result != 0:
            print("❌ FoundationStereo failed")
            return []
        
        # Step 2: Run YOLO segmentation
        model = YOLO(model_path)
        yolo_results = model(left_image)
        
        # Step 3: Load depth data
        depth_file = os.path.join(depth_output, "depth_meter.npy")
        if not os.path.exists(depth_file):
            print("❌ Depth file not found")
            return []
        
        depth_map = np.load(depth_file)
        print(f"✓ Loaded depth map: {depth_map.shape}")
        
        # Step 4: Extract depth for each detected block
        image = cv2.imread(left_image)
        results = []
        
        for result in yolo_results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    confidence = box[4]
                    class_id = int(box[5])
                    class_name = self.class_names[class_id] if hasattr(self, 'class_names') and class_id < len(self.class_names) else 'block'
                    
                    # Resize mask to match depth map
                    mask_resized = cv2.resize(mask.astype(np.uint8), 
                                            (depth_map.shape[1], depth_map.shape[0]))
                    mask_bool = mask_resized.astype(bool)
                    
                    # Extract depth values
                    block_depths = depth_map[mask_bool]
                    valid_depths = block_depths[(block_depths > 0) & np.isfinite(block_depths)]
                    
                    if len(valid_depths) > 0:
                        results.append({
                            'block_id': i,
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'mean_depth': float(np.mean(valid_depths)),
                            'depth_range': [float(np.min(valid_depths)), float(np.max(valid_depths))],
                            'pixel_count': int(np.sum(mask_bool)),
                            'bbox': box[:4].tolist()
                        })
        
        # Step 5: Visualize and save results
        self._save_visualization(image, depth_map, yolo_results[0], results, 
                               os.path.join(self.visualizations_dir, f"result_{Path(left_image).stem}.png"))
        
        print(f"\\n✓ Found {len(results)} blocks with depth information:")
        for result in results:
            print(f"  {result['class_name']}: {result['mean_depth']:.3f}m (conf: {result['confidence']:.3f})")
        
        return results
    
    def _save_visualization(self, image, depth_map, yolo_result, depth_results, save_path):
        """Save visualization of results"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Depth map
        depth_vis = np.clip(depth_map, 0, np.percentile(depth_map[depth_map > 0], 95))
        im = axes[1].imshow(depth_vis, cmap='plasma')
        axes[1].set_title('Depth Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Segmentation + depth overlay
        overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if yolo_result.masks is not None:
            masks = yolo_result.masks.data.cpu().numpy()
            colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))
            
            for i, (mask, result) in enumerate(zip(masks, depth_results)):
                color = (colors[i][:3] * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]))
                mask_bool = mask_resized.astype(bool)
                overlay[mask_bool] = overlay[mask_bool] * 0.7 + color * 0.3
                
                # Add labels
                x1, y1, x2, y2 = [int(x) for x in result['bbox']]
                label = f"{result['class_name']}: {result['mean_depth']:.2f}m"
                cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Detected Blocks + Depth')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Unity Auto-YOLO Training Pipeline")
    parser.add_argument("--config", type=str, default=".env", help="Configuration file")
    parser.add_argument("--action", choices=['extract', 'train', 'test', 'integrate'], 
                       required=True, help="Action to perform")
    parser.add_argument("--target_classes", type=str, nargs='+', 
                       default=['Module_Construction'], help="Target class names")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--left_image", type=str, help="Left stereo image")
    parser.add_argument("--right_image", type=str, help="Right stereo image")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = UnityAutoYOLO(args.config)
    
    if args.action == 'extract':
        print("🔄 Extracting Unity annotations and preparing dataset...")
        
        # Extract annotations from Unity solo_9
        annotations = trainer.extract_unity_annotations(
            trainer.unity_solo_9_path,
            args.target_classes
        )
        
        if annotations:
            success = trainer.convert_to_yolo_format(annotations)
            if success:
                print("✅ Dataset preparation completed!")
            else:
                print("❌ Dataset preparation failed!")
        else:
            print("❌ No annotations found!")
    
    elif args.action == 'train':
        print("🔄 Training YOLO model...")
        model_path = trainer.train_model()
        print(f"🎯 Training complete! Model saved: {model_path}")
    
    elif args.action == 'test':
        if not args.model_path:
            print("❌ --model_path required for testing")
            return
        
        # Test on a sample image from the dataset
        test_images = list(Path(f"{trainer.dataset_root}/images/test").glob("*.png"))
        if test_images:
            model = YOLO(args.model_path)
            results = model(str(test_images[0]))
            results[0].save(f"{trainer.visualizations_dir}/test_result.jpg")
            print(f"✓ Test result saved to: {trainer.visualizations_dir}/test_result.jpg")
        else:
            print("❌ No test images found")
    
    elif args.action == 'integrate':
        if not all([args.model_path, args.left_image, args.right_image]):
            print("❌ --model_path, --left_image, --right_image required")
            return
        
        results = trainer.integrate_with_foundation_stereo(
            args.model_path, 
            args.left_image, 
            args.right_image
        )
        
        if results:
            print("✅ Integration completed successfully!")
        else:
            print("❌ Integration failed!")

if __name__ == "__main__":
    main()
