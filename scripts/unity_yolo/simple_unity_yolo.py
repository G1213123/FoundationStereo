"""
Simple YOLO training pipeline for Unity block segmentation
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
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class SimpleUnityYOLO:
    """Simplified YOLO training for Unity data"""
    
    def __init__(self, project_name: str = "unity_blocks"):
        self.project_name = project_name
        self.dataset_root = f"./datasets/{project_name}"
        
    def create_manual_annotation_tool(self, images_dir: str):
        """Create a simple manual annotation tool"""
        
        tool_script = '''
import cv2
import os
import numpy as np
from pathlib import Path

class SimpleAnnotator:
    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.annotations = []
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                if abs(x2-x1) > 10 and abs(y2-y1) > 10:  # Minimum size
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    self.annotations.append([x1, y1, x2, y2])
                    print(f"Added bbox: [{x1}, {y1}, {x2}, {y2}]")
            self.drawing = False
            self.start_point = None
    
    def annotate_directory(self, images_dir):
        os.makedirs("annotations", exist_ok=True)
        image_files = list(Path(images_dir).glob("**/*.png"))
        
        for i, img_path in enumerate(image_files):
            print(f"Annotating {i+1}/{len(image_files)}: {img_path.name}")
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            self.annotations = []
            
            cv2.namedWindow('Annotator')
            cv2.setMouseCallback('Annotator', self.mouse_callback)
            
            while True:
                display_img = img.copy()
                
                # Draw existing annotations
                for ann in self.annotations:
                    x1, y1, x2, y2 = ann
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.putText(display_img, "Draw bbox | 's'=save | 'n'=next | 'q'=quit | 'z'=undo", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Annotator', display_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    self.save_yolo_annotation(img_path, img.shape)
                    break
                elif key == ord('n'):
                    break
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('z') and self.annotations:
                    self.annotations.pop()
                    print("Removed last annotation")
        
        cv2.destroyAllWindows()
    
    def save_yolo_annotation(self, img_path, img_shape):
        if not self.annotations:
            return
            
        h, w = img_shape[:2]
        yolo_lines = []
        
        for x1, y1, x2, y2 in self.annotations:
            center_x = (x1 + x2) / 2 / w
            center_y = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        output_file = f"annotations/{Path(img_path).stem}.txt"
        with open(output_file, 'w') as f:
            f.write("\\n".join(yolo_lines))
        print(f"Saved {len(yolo_lines)} annotations to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python annotator.py <images_directory>")
        sys.exit(1)
    
    annotator = SimpleAnnotator()
    annotator.annotate_directory(sys.argv[1])
'''
        
        with open("manual_annotator.py", 'w') as f:
            f.write(tool_script)
        
        print("✓ Manual annotation tool created: manual_annotator.py")
        print(f"Usage: python manual_annotator.py {images_dir}")
        return "manual_annotator.py"
    
    def setup_dataset(self, 
                     images_dir: str, 
                     annotations_dir: str = "./annotations",
                     train_split: float = 0.8,
                     val_split: float = 0.15):
        """Setup YOLO dataset from images and annotations"""
        
        # Create dataset structure
        os.makedirs(f"{self.dataset_root}/images/train", exist_ok=True)
        os.makedirs(f"{self.dataset_root}/images/val", exist_ok=True)
        os.makedirs(f"{self.dataset_root}/images/test", exist_ok=True)
        os.makedirs(f"{self.dataset_root}/labels/train", exist_ok=True)
        os.makedirs(f"{self.dataset_root}/labels/val", exist_ok=True)
        os.makedirs(f"{self.dataset_root}/labels/test", exist_ok=True)
        
        # Get all image files
        image_files = list(Path(images_dir).glob("**/*.png"))
        image_files = [f for f in image_files if (Path(annotations_dir) / f"{f.stem}.txt").exists()]
        
        print(f"Found {len(image_files)} images with annotations")
        
        if len(image_files) == 0:
            print("❌ No annotated images found!")
            return False
        
        # Shuffle and split
        np.random.shuffle(image_files)
        n_train = int(len(image_files) * train_split)
        n_val = int(len(image_files) * val_split)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to dataset structure
        for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for img_file in files:
                # Copy image
                dst_img = f"{self.dataset_root}/images/{split_name}/{img_file.name}"
                shutil.copy2(img_file, dst_img)
                
                # Copy annotation
                ann_file = Path(annotations_dir) / f"{img_file.stem}.txt"
                if ann_file.exists():
                    dst_ann = f"{self.dataset_root}/labels/{split_name}/{img_file.stem}.txt"
                    shutil.copy2(ann_file, dst_ann)
        
        # Create dataset.yaml
        dataset_config = {
            'path': os.path.abspath(self.dataset_root),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': 1,  # Number of classes
            'names': ['block']  # Class names
        }
        
        with open(f"{self.dataset_root}/dataset.yaml", 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✓ Dataset created:")
        print(f"  Train: {len(train_files)} images")
        print(f"  Val: {len(val_files)} images") 
        print(f"  Test: {len(test_files)} images")
        print(f"  Config: {self.dataset_root}/dataset.yaml")
        
        return True
    
    def train_model(self, 
                   model_size: str = 'n',
                   epochs: int = 50,
                   batch: int = 8,
                   imgsz: int = 640):
        """Train YOLO model"""
        
        print(f"🚀 Starting YOLO training...")
        print(f"Model: YOLOv8{model_size}-seg")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch}")
        print(f"Image size: {imgsz}")
        
        # Load model
        model = YOLO(f'yolov8{model_size}-seg.pt')
        
        # Train
        results = model.train(
            data=f'{self.dataset_root}/dataset.yaml',
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=f'./runs/train',
            name=self.project_name,
            save=True,
            plots=True,
            device='0' if os.system('nvidia-smi') == 0 else 'cpu'
        )
        
        best_model = f'./runs/train/{self.project_name}/weights/best.pt'
        print(f"✓ Training completed! Best model: {best_model}")
        
        return best_model
    
    def test_model(self, model_path: str, test_image: str, save_result: bool = True):
        """Test trained model on an image"""
        
        model = YOLO(model_path)
        
        # Run inference
        results = model(test_image)
        
        # Process results
        detections = []
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                
                for mask, box in zip(masks, boxes):
                    confidence = box[4]
                    detections.append({
                        'confidence': float(confidence),
                        'bbox': box[:4].tolist(),
                        'mask': mask
                    })
        
        print(f"Found {len(detections)} blocks with confidences: {[d['confidence'] for d in detections]}")
        
        # Save visualization
        if save_result:
            result_img = results[0].plot()
            output_path = f"detection_result_{Path(test_image).stem}.jpg"
            cv2.imwrite(output_path, result_img)
            print(f"✓ Result saved to: {output_path}")
        
        return detections
    
    def integrate_with_depth(self, 
                           model_path: str,
                           left_image: str,
                           right_image: str, 
                           intrinsics_file: str):
        """Integrate YOLO segmentation with FoundationStereo depth"""
        
        print("🔄 Running integrated pipeline...")
        
        # Step 1: Run FoundationStereo
        depth_output = f"./depth_output_{Path(left_image).stem}"
        os.makedirs(depth_output, exist_ok=True)
        
        cmd = f"""C:/Users/1213123/Documents/Scripts/FoundationStereo/.venv/Scripts/python.exe scripts/run_demo.py \
            --left_file "{left_image}" \
            --right_file "{right_image}" \
            --intrinsic_file "{intrinsics_file}" \
            --out_dir "{depth_output}" \
            --valid_iters 32"""
        
        print("Running FoundationStereo...")
        os.system(cmd)
        
        # Step 2: Run YOLO segmentation
        model = YOLO(model_path)
        yolo_results = model(left_image)
        
        # Step 3: Load depth data
        depth_file = f"{depth_output}/depth_meter.npy"
        if os.path.exists(depth_file):
            depth_map = np.load(depth_file)
            print(f"✓ Loaded depth map: {depth_map.shape}")
        else:
            print("❌ Depth file not found")
            return None
        
        # Step 4: Extract depth for each detected block
        image = cv2.imread(left_image)
        results = []
        
        for result in yolo_results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    confidence = box[4]
                    
                    # Resize mask to match depth map
                    mask_resized = cv2.resize(mask.astype(np.uint8), 
                                            (depth_map.shape[1], depth_map.shape[0]))
                    mask_bool = mask_resized.astype(bool)
                    
                    # Extract depth values for this block
                    block_depths = depth_map[mask_bool]
                    valid_depths = block_depths[(block_depths > 0) & np.isfinite(block_depths)]
                    
                    if len(valid_depths) > 0:
                        mean_depth = np.mean(valid_depths)
                        min_depth = np.min(valid_depths)
                        max_depth = np.max(valid_depths)
                        
                        results.append({
                            'block_id': i,
                            'confidence': float(confidence),
                            'mean_depth': float(mean_depth),
                            'depth_range': [float(min_depth), float(max_depth)],
                            'pixel_count': int(np.sum(mask_bool)),
                            'bbox': box[:4].tolist()
                        })
        
        # Step 5: Visualize results
        self._visualize_depth_segmentation(image, depth_map, yolo_results[0], results)
        
        print(f"\\n✓ Found {len(results)} blocks with depth information:")
        for i, result in enumerate(results):
            print(f"  Block {i+1}: {result['mean_depth']:.3f}m (conf: {result['confidence']:.3f})")
        
        return results
    
    def _visualize_depth_segmentation(self, image, depth_map, yolo_result, depth_results):
        """Create visualization of segmentation + depth"""
        
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
        
        # Segmentation overlay
        overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if yolo_result.masks is not None:
            masks = yolo_result.masks.data.cpu().numpy()
            colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))
            
            for i, (mask, result) in enumerate(zip(masks, depth_results)):
                color = (colors[i][:3] * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]))
                mask_bool = mask_resized.astype(bool)
                overlay[mask_bool] = overlay[mask_bool] * 0.7 + color * 0.3
                
                # Add depth info
                x1, y1, x2, y2 = [int(x) for x in result['bbox']]
                label = f"Block {i+1}: {result['mean_depth']:.2f}m"
                cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Blocks + Depth Info')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('depth_segmentation_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualization saved as: depth_segmentation_result.png")

def main():
    parser = argparse.ArgumentParser(description="Train YOLO for Unity block segmentation")
    parser.add_argument("--action", choices=['annotate', 'setup', 'train', 'test', 'integrate'], 
                       required=True, help="Action to perform")
    parser.add_argument("--images_dir", type=str, help="Directory containing Unity images")
    parser.add_argument("--annotations_dir", type=str, default="./annotations", help="Annotations directory")
    parser.add_argument("--model_size", choices=['n', 's', 'm', 'l', 'x'], default='n', help="YOLO model size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--test_image", type=str, help="Test image path")
    parser.add_argument("--left_image", type=str, help="Left stereo image")
    parser.add_argument("--right_image", type=str, help="Right stereo image")
    parser.add_argument("--intrinsics", type=str, help="Camera intrinsics file")
    
    args = parser.parse_args()
    
    trainer = SimpleUnityYOLO()
    
    if args.action == 'annotate':
        if not args.images_dir:
            print("❌ --images_dir required for annotation")
            return
        trainer.create_manual_annotation_tool(args.images_dir)
    
    elif args.action == 'setup':
        if not args.images_dir:
            print("❌ --images_dir required for setup")
            return
        success = trainer.setup_dataset(args.images_dir, args.annotations_dir)
        if not success:
            print("❌ Dataset setup failed")
    
    elif args.action == 'train':
        model_path = trainer.train_model(args.model_size, args.epochs, args.batch)
        print(f"🎯 Training complete! Use model: {model_path}")
    
    elif args.action == 'test':
        if not args.model_path or not args.test_image:
            print("❌ --model_path and --test_image required for testing")
            return
        trainer.test_model(args.model_path, args.test_image)
    
    elif args.action == 'integrate':
        if not all([args.model_path, args.left_image, args.right_image, args.intrinsics]):
            print("❌ --model_path, --left_image, --right_image, --intrinsics required")
            return
        trainer.integrate_with_depth(args.model_path, args.left_image, args.right_image, args.intrinsics)

if __name__ == "__main__":
    main()
