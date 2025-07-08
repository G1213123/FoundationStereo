import os
import cv2
import json
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import shutil

class UnityYOLOTrainer:
    """
    YOLO training pipeline for Unity-generated segmentation data
    """
    
    def __init__(self, project_name: str = "unity_block_segmentation"):
        self.project_name = project_name
        self.dataset_root = f"./datasets/{project_name}"
        self.class_names = []
        self.class_mapping = {}
        
    def setup_dataset_structure(self):
        """Create YOLO dataset directory structure"""
        directories = [
            f"{self.dataset_root}/images/train",
            f"{self.dataset_root}/images/val", 
            f"{self.dataset_root}/images/test",
            f"{self.dataset_root}/labels/train",
            f"{self.dataset_root}/labels/val",
            f"{self.dataset_root}/labels/test"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print(f"✓ Dataset structure created at: {self.dataset_root}")
    
    def extract_unity_annotations(self, 
                                  unity_sequences_path: str,
                                  target_classes: List[str] = None) -> List[Dict]:
        """
        Extract annotations from Unity JSON files
        
        Args:
            unity_sequences_path: Path to Unity solo_7 directory
            target_classes: List of class names to extract (e.g., ['Module_Construction'])
        
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        # Find all sequence directories
        sequences_dir = Path(unity_sequences_path)
        sequence_dirs = [d for d in sequences_dir.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
        
        print(f"Found {len(sequence_dirs)} sequences in {unity_sequences_path}")
        
        for seq_dir in sequence_dirs:
            # Look for frame_data.json files
            json_files = list(seq_dir.glob("step*.frame_data.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract image paths and annotations
                    for capture in data['captures']:
                        if capture['@type'] == 'type.unity.com/unity.solo.RGBCamera':
                            image_path = seq_dir / capture['filename']
                            
                            if not image_path.exists():
                                continue
                            
                            # Extract bounding box annotations
                            bboxes = []
                            for annotation in capture.get('annotations', []):
                                if annotation['@type'] == 'type.unity.com/unity.solo.BoundingBox3DAnnotation':
                                    for value in annotation.get('values', []):
                                        class_name = value.get('labelName', 'unknown')
                                        
                                        # Filter by target classes if specified
                                        if target_classes and class_name not in target_classes:
                                            continue
                                        
                                        # Note: Unity provides 3D bounding boxes, we need to project to 2D
                                        # For now, we'll use a placeholder - you'll need to implement 3D->2D projection
                                        # or use Unity's 2D bounding box annotations if available
                                        
                                        bbox_data = {
                                            'class_name': class_name,
                                            'translation': value.get('translation', [0, 0, 0]),
                                            'size': value.get('size', [1, 1, 1]),
                                            'rotation': value.get('rotation', [0, 0, 0, 1])
                                        }
                                        bboxes.append(bbox_data)
                            
                            if bboxes:  # Only add if we have annotations
                                annotations.append({
                                    'image_path': str(image_path),
                                    'sequence': seq_dir.name,
                                    'step': json_file.stem,
                                    'camera': capture['id'],
                                    'dimensions': capture['dimension'],
                                    'bboxes': bboxes
                                })
                
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue
        
        print(f"✓ Extracted {len(annotations)} annotated images")
        return annotations
    
    def create_manual_annotations(self, 
                                 unity_sequences_path: str,
                                 output_dir: str = "./manual_annotations") -> str:
        """
        Create a manual annotation tool for Unity images
        
        Returns:
            Path to annotation directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create annotation tool script
        annotation_tool_script = f"""
import cv2
import json
import os
from pathlib import Path
import argparse

class ManualAnnotator:
    def __init__(self, images_dir, output_dir):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.current_image = None
        self.current_image_path = None
        self.annotations = []
        self.drawing = False
        self.start_point = None
        self.temp_point = None
        self.class_name = "block"  # Default class name
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and (x, y) != self.start_point:
                # Add bounding box
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure correct order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                self.annotations.append({{
                    'class': self.class_name,
                    'bbox': [x1, y1, x2, y2]
                }})
                print(f"Added bbox: {{self.class_name}} [{x1}, {y1}, {x2}, {y2}]")
    
    def annotate_images(self):
        image_files = list(self.images_dir.glob("**/*.png"))
        
        for i, img_path in enumerate(image_files):
            print(f"\\nAnnotating {{i+1}}/{{len(image_files)}}: {{img_path.name}}")
            
            self.current_image_path = img_path
            self.current_image = cv2.imread(str(img_path))
            self.annotations = []
            
            if self.current_image is None:
                continue
                
            cv2.namedWindow('Annotator', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Annotator', self.mouse_callback)
            
            while True:
                display_img = self.current_image.copy()
                
                # Draw existing annotations
                for ann in self.annotations:
                    x1, y1, x2, y2 = ann['bbox']
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_img, ann['class'], (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw current selection
                if self.drawing and self.start_point and self.temp_point:
                    cv2.rectangle(display_img, self.start_point, self.temp_point, (255, 0, 0), 2)
                
                cv2.putText(display_img, f"Class: {{self.class_name}} | Press 'c' to change", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_img, "Draw bbox with mouse | 's'=save | 'n'=next | 'q'=quit", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Annotator', display_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):  # Save
                    self.save_annotations()
                    break
                elif key == ord('n'):  # Next without saving
                    break
                elif key == ord('q'):  # Quit
                    cv2.destroyAllWindows()
                    return
                elif key == ord('c'):  # Change class
                    new_class = input("Enter class name: ").strip()
                    if new_class:
                        self.class_name = new_class
                elif key == ord('z'):  # Undo last annotation
                    if self.annotations:
                        self.annotations.pop()
                        print("Removed last annotation")
            
            cv2.destroyAllWindows()
    
    def save_annotations(self):
        if not self.annotations:
            print("No annotations to save")
            return
            
        # Convert to YOLO format
        h, w = self.current_image.shape[:2]
        yolo_annotations = []
        
        for ann in self.annotations:
            x1, y1, x2, y2 = ann['bbox']
            
            # Convert to YOLO format (normalized center coordinates + width/height)
            center_x = (x1 + x2) / 2 / w
            center_y = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            class_id = 0  # For now, single class
            yolo_annotations.append(f"{{class_id}} {{center_x:.6f}} {{center_y:.6f}} {{width:.6f}} {{height:.6f}}")
        
        # Save annotation file
        output_file = self.output_dir / f"{{self.current_image_path.stem}}.txt"
        with open(output_file, 'w') as f:
            f.write("\\n".join(yolo_annotations))
        
        print(f"Saved {{len(yolo_annotations)}} annotations to {{output_file}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory containing images to annotate")
    parser.add_argument("--output", default="./annotations", help="Output directory for annotations")
    args = parser.parse_args()
    
    annotator = ManualAnnotator(args.images, args.output)
    annotator.annotate_images()
"""
        
        tool_path = os.path.join(output_dir, "manual_annotator.py")
        with open(tool_path, 'w') as f:
            f.write(annotation_tool_script)
        
        print(f"✓ Manual annotation tool created at: {tool_path}")
        print(f"Usage: python {tool_path} --images {unity_sequences_path} --output {output_dir}")
        
        return output_dir
    
    def convert_annotations_to_yolo(self, 
                                   annotations: List[Dict], 
                                   train_split: float = 0.8,
                                   val_split: float = 0.1) -> None:
        """
        Convert Unity annotations to YOLO format and split dataset
        """
        if not annotations:
            print("❌ No annotations provided")
            return
        
        # Extract unique class names
        all_classes = set()
        for ann in annotations:
            for bbox in ann['bboxes']:
                all_classes.add(bbox['class_name'])
        
        self.class_names = sorted(list(all_classes))
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"✓ Found classes: {self.class_names}")
        
        # Shuffle annotations
        np.random.shuffle(annotations)
        
        # Split dataset
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
        
        print(f"✓ Dataset created with {n_train} train, {len(val_anns)} val, {len(test_anns)} test images")
    
    def _process_split(self, annotations: List[Dict], split_name: str) -> None:
        """Process a single dataset split"""
        for ann in annotations:
            # Copy image
            src_image_path = ann['image_path']
            dst_image_path = f"{self.dataset_root}/images/{split_name}/{Path(src_image_path).name}"
            
            try:
                shutil.copy2(src_image_path, dst_image_path)
            except Exception as e:
                print(f"Warning: Could not copy {src_image_path}: {e}")
                continue
            
            # Create YOLO annotation
            img = cv2.imread(src_image_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            yolo_lines = []
            for bbox in ann['bboxes']:
                class_id = self.class_mapping.get(bbox['class_name'], 0)
                
                # For now, create a placeholder bounding box since Unity gives 3D data
                # You would need to implement proper 3D->2D projection here
                # This is a simplified example
                center_x = 0.5  # Placeholder - center of image
                center_y = 0.5
                width = 0.3     # Placeholder - 30% of image width
                height = 0.3    # Placeholder - 30% of image height
                
                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Save annotation file
            if yolo_lines:
                label_path = f"{self.dataset_root}/labels/{split_name}/{Path(src_image_path).stem}.txt"
                with open(label_path, 'w') as f:
                    f.write("\\n".join(yolo_lines))
    
    def _create_dataset_yaml(self) -> None:
        """Create YOLO dataset configuration file"""
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
        
        print(f"✓ Dataset config saved to: {yaml_path}")
    
    def train_model(self, 
                   model_size: str = 'n',  # n, s, m, l, x
                   epochs: int = 100,
                   imgsz: int = 640,
                   batch: int = 16,
                   device: str = '0') -> str:
        """
        Train YOLO model on the prepared dataset
        
        Returns:
            Path to trained model
        """
        # Load pre-trained YOLO model
        model = YOLO(f'yolov8{model_size}-seg.pt')
        
        # Train the model
        results = model.train(
            data=f'{self.dataset_root}/dataset.yaml',
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=f'./runs/{self.project_name}',
            name='train',
            save=True,
            plots=True
        )
        
        # Get path to best model
        best_model_path = f'./runs/{self.project_name}/train/weights/best.pt'
        
        print(f"✓ Training completed! Best model saved to: {best_model_path}")
        return best_model_path
    
    def evaluate_model(self, model_path: str) -> Dict:
        """Evaluate trained model"""
        model = YOLO(model_path)
        
        # Validate the model
        results = model.val(
            data=f'{self.dataset_root}/dataset.yaml',
            project=f'./runs/{self.project_name}',
            name='eval'
        )
        
        return results
    
    def test_inference(self, 
                      model_path: str, 
                      test_image_path: str,
                      save_path: str = None) -> List[Dict]:
        """
        Test inference on a single image
        
        Returns:
            List of detection results
        """
        model = YOLO(model_path)
        
        # Run inference
        results = model(test_image_path)
        
        detections = []
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                
                for mask, box in zip(masks, boxes):
                    class_id = int(box[5])
                    confidence = box[4]
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown',
                        'confidence': confidence,
                        'bbox': box[:4].tolist(),
                        'mask': mask
                    })
        
        # Save visualization if requested
        if save_path:
            result_img = results[0].plot()
            cv2.imwrite(save_path, result_img)
            print(f"✓ Result saved to: {save_path}")
        
        return detections

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on Unity data")
    parser.add_argument("--action", choices=['prepare', 'annotate', 'train', 'test'], 
                       required=True, help="Action to perform")
    parser.add_argument("--unity_path", type=str, 
                       help="Path to Unity solo_7 directory")
    parser.add_argument("--target_classes", type=str, nargs='+', 
                       default=['Module_Construction'], 
                       help="Target class names to extract")
    parser.add_argument("--model_size", choices=['n', 's', 'm', 'l', 'x'], 
                       default='n', help="YOLO model size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--test_image", type=str, help="Test image path")
    parser.add_argument("--project_name", type=str, default="unity_blocks", 
                       help="Project name")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = UnityYOLOTrainer(args.project_name)
    
    if args.action == 'prepare':
        print("🔄 Preparing dataset...")
        trainer.setup_dataset_structure()
        
        if args.unity_path:
            # Extract annotations from Unity data
            annotations = trainer.extract_unity_annotations(
                args.unity_path, 
                args.target_classes
            )
            
            if annotations:
                trainer.convert_annotations_to_yolo(annotations)
            else:
                print("❌ No annotations found. Consider using manual annotation tool.")
                trainer.create_manual_annotations(args.unity_path)
        else:
            print("❌ Unity path required for dataset preparation")
    
    elif args.action == 'annotate':
        print("🔄 Creating manual annotation tool...")
        if args.unity_path:
            trainer.create_manual_annotations(args.unity_path)
        else:
            print("❌ Unity path required for annotation tool")
    
    elif args.action == 'train':
        print("🔄 Training YOLO model...")
        model_path = trainer.train_model(
            model_size=args.model_size,
            epochs=args.epochs,
            batch=args.batch
        )
        
        # Evaluate model
        trainer.evaluate_model(model_path)
    
    elif args.action == 'test':
        print("🔄 Testing model...")
        if args.model_path and args.test_image:
            detections = trainer.test_inference(
                args.model_path,
                args.test_image,
                save_path=f"./test_result_{Path(args.test_image).stem}.jpg"
            )
            
            print(f"Found {len(detections)} detections:")
            for det in detections:
                print(f"  {det['class_name']}: {det['confidence']:.3f}")
        else:
            print("❌ Model path and test image required for testing")

if __name__ == "__main__":
    main()
