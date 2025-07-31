import numpy as np
import cv2
import torch
import argparse
import os
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class DepthSegmentResult:
    """Container for segmentation and depth results"""
    mask: np.ndarray
    depth_values: np.ndarray
    mean_depth: float
    min_depth: float
    max_depth: float
    pixel_count: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2

class DepthSegmentationPipeline:
    """
    Pipeline for segmenting objects and extracting depth information
    """
    
    def __init__(self, segmentation_model='sam2', device='cuda'):
        self.device = device
        self.segmentation_model = segmentation_model
        self.predictor = None
        
        self._load_segmentation_model()
    
    def _load_segmentation_model(self):
        """Load the specified segmentation model"""
        if self.segmentation_model == 'sam2':
            self._load_sam2()
        elif self.segmentation_model == 'yolo':
            self._load_yolo()
        elif self.segmentation_model == 'detectron2':
            self._load_detectron2()
        else:
            raise ValueError(f"Unsupported model: {self.segmentation_model}")
    
    def _load_sam2(self):
        """Load SAM2 model"""
        try:
            from sam2 import SAM2ImagePredictor
            self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            print("✓ SAM2 model loaded successfully")
        except ImportError:
            print("❌ SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            raise
    
    def _load_yolo(self):
        """Load YOLO segmentation model"""
        try:
            from ultralytics import YOLO
            self.predictor = YOLO('yolov8n-seg.pt')  # You can use larger models: yolov8s-seg.pt, yolov8m-seg.pt, etc.
            print("✓ YOLO segmentation model loaded successfully")
        except ImportError:
            print("❌ Ultralytics not installed. Install with: pip install ultralytics")
            raise
    
    def _load_detectron2(self):
        """Load Detectron2 model"""
        try:
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.DEVICE = self.device
            self.predictor = DefaultPredictor(cfg)
            print("✓ Detectron2 model loaded successfully")
        except ImportError:
            print("❌ Detectron2 not installed. Install with: pip install detectron2")
            raise
    
    def segment_image(self, image: np.ndarray, prompts: Optional[List] = None) -> List[Dict]:
        """
        Segment objects in the image
        
        Args:
            image: Input image (H, W, 3)
            prompts: For SAM2 - list of (x, y, label) tuples where label=1 for foreground
        
        Returns:
            List of segmentation results
        """
        if self.segmentation_model == 'sam2':
            return self._segment_sam2(image, prompts)
        elif self.segmentation_model == 'yolo':
            return self._segment_yolo(image)
        elif self.segmentation_model == 'detectron2':
            return self._segment_detectron2(image)
    
    def _segment_sam2(self, image: np.ndarray, prompts: List[Tuple[int, int, int]]) -> List[Dict]:
        """Segment using SAM2"""
        if prompts is None:
            raise ValueError("SAM2 requires prompts (x, y, label) where label=1 for foreground")
        
        results = []
        for i, (x, y, label) in enumerate(prompts):
            masks = self.predictor.predict(image, point_prompts=[(x, y, label)])
            if len(masks) > 0:
                mask = masks[0]  # Take the first/best mask
                bbox = self._mask_to_bbox(mask)
                results.append({
                    'mask': mask,
                    'class_name': f'object_{i}',
                    'confidence': 1.0,  # SAM2 doesn't provide confidence scores
                    'bbox': bbox
                })
        return results
    
    def _segment_yolo(self, image: np.ndarray) -> List[Dict]:
        """Segment using YOLO"""
        results_raw = self.predictor(image)
        results = []
        
        for result in results_raw:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    # Resize mask to image size
                    mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
                    mask_resized = mask_resized.astype(bool)
                    
                    class_id = int(box[5])
                    confidence = box[4]
                    bbox = tuple(box[:4].astype(int))
                    
                    results.append({
                        'mask': mask_resized,
                        'class_name': self.predictor.names[class_id],
                        'confidence': confidence,
                        'bbox': bbox
                    })
        return results
    
    def _segment_detectron2(self, image: np.ndarray) -> List[Dict]:
        """Segment using Detectron2"""
        outputs = self.predictor(image)
        results = []
        
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        
        for mask, box, class_id, score in zip(masks, boxes, classes, scores):
            bbox = tuple(box.astype(int))
            results.append({
                'mask': mask,
                'class_name': f'class_{class_id}',  # You can map this to actual class names
                'confidence': score,
                'bbox': bbox
            })
        
        return results
    
    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert mask to bounding box"""
        coords = np.where(mask)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        return (x_min, y_min, x_max, y_max)
    
    def extract_depth_from_segments(self, 
                                   image: np.ndarray,
                                   depth_map: np.ndarray,
                                   segments: List[Dict],
                                   filter_classes: Optional[List[str]] = None) -> List[DepthSegmentResult]:
        """
        Extract depth information for each segment
        
        Args:
            image: RGB image
            depth_map: Depth map from FoundationStereo
            segments: Segmentation results
            filter_classes: Only process these class names (None for all)
        
        Returns:
            List of DepthSegmentResult objects
        """
        results = []
        
        for segment in segments:
            class_name = segment['class_name']
            
            # Skip if filtering and this class not in filter list
            if filter_classes and class_name not in filter_classes:
                continue
            
            mask = segment['mask']
            confidence = segment['confidence']
            bbox = segment['bbox']
            
            # Extract depth values for this segment
            masked_depth = depth_map[mask]
            
            # Filter out invalid depth values (inf, nan, zero)
            valid_depths = masked_depth[(masked_depth > 0) & 
                                      np.isfinite(masked_depth) & 
                                      (masked_depth < 1000)]  # Reasonable depth limit
            
            if len(valid_depths) == 0:
                continue  # Skip if no valid depth values
            
            # Calculate statistics
            mean_depth = np.mean(valid_depths)
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            pixel_count = np.sum(mask)
            
            result = DepthSegmentResult(
                mask=mask,
                depth_values=valid_depths,
                mean_depth=mean_depth,
                min_depth=min_depth,
                max_depth=max_depth,
                pixel_count=pixel_count,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox
            )
            
            results.append(result)
        
        return results
    
    def visualize_results(self, 
                         image: np.ndarray,
                         depth_map: np.ndarray,
                         results: List[DepthSegmentResult],
                         save_path: Optional[str] = None):
        """
        Visualize segmentation and depth results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Depth map
        depth_vis = np.clip(depth_map, 0, np.percentile(depth_map[depth_map > 0], 95))
        im1 = axes[0, 1].imshow(depth_vis, cmap='plasma')
        axes[0, 1].set_title('Depth Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Segmentation overlay
        overlay = image.copy()
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for i, result in enumerate(results):
            color = (colors[i][:3] * 255).astype(np.uint8)
            overlay[result.mask] = overlay[result.mask] * 0.7 + color * 0.3
            
            # Add bounding box and label
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)
            label = f"{result.class_name}: {result.mean_depth:.2f}m"
            cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Segmentation + Depth Info')
        axes[1, 0].axis('off')
        
        # Depth statistics
        axes[1, 1].axis('off')
        text_info = "Depth Statistics:\\n\\n"
        for i, result in enumerate(results):
            text_info += f"{result.class_name}:\\n"
            text_info += f"  Mean: {result.mean_depth:.3f}m\\n"
            text_info += f"  Range: {result.min_depth:.3f}-{result.max_depth:.3f}m\\n"
            text_info += f"  Pixels: {result.pixel_count}\\n"
            text_info += f"  Confidence: {result.confidence:.2f}\\n\\n"
        
        axes[1, 1].text(0.1, 0.9, text_info, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def process_unity_sequence(self,
                              unity_sequence_path: str,
                              intrinsics_file: str,
                              target_classes: List[str] = None,
                              prompts: List[Tuple[int, int, int]] = None) -> List[DepthSegmentResult]:
        """
        Process a Unity sequence with FoundationStereo + Segmentation
        
        Args:
            unity_sequence_path: Path to Unity sequence directory
            intrinsics_file: Camera intrinsics file
            target_classes: Classes to segment (for YOLO/Detectron2)
            prompts: Click prompts for SAM2 [(x, y, 1), ...]
        
        Returns:
            Combined segmentation and depth results
        """
        # TODO: Implement full pipeline integration
        pass

def main():
    parser = argparse.ArgumentParser(description="Segment objects and extract depth information")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--depth", type=str, required=True, help="Depth map (.npy file)")
    parser.add_argument("--model", type=str, choices=['sam2', 'yolo', 'detectron2'], 
                       default='yolo', help="Segmentation model to use")
    parser.add_argument("--prompts", type=str, help="For SAM2: comma-separated x,y,label tuples")
    parser.add_argument("--classes", type=str, help="Comma-separated class names to filter")
    parser.add_argument("--output", type=str, default="./depth_segmentation_results.png", 
                       help="Output visualization path")
    
    args = parser.parse_args()
    
    # Load image and depth map
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_map = np.load(args.depth)
    
    # Initialize pipeline
    pipeline = DepthSegmentationPipeline(segmentation_model=args.model)
    
    # Parse prompts for SAM2
    prompts = None
    if args.prompts and args.model == 'sam2':
        prompt_parts = args.prompts.split(',')
        prompts = [(int(prompt_parts[i]), int(prompt_parts[i+1]), int(prompt_parts[i+2])) 
                  for i in range(0, len(prompt_parts), 3)]
    
    # Parse target classes
    target_classes = None
    if args.classes:
        target_classes = [cls.strip() for cls in args.classes.split(',')]
    
    # Run segmentation
    segments = pipeline.segment_image(image, prompts)
    
    # Extract depth information
    results = pipeline.extract_depth_from_segments(image, depth_map, segments, target_classes)
    
    # Visualize results
    pipeline.visualize_results(image, depth_map, results, args.output)
    
    # Print results
    print("\\nDepth Segmentation Results:")
    print("=" * 50)
    for result in results:
        print(f"Class: {result.class_name}")
        print(f"  Mean depth: {result.mean_depth:.3f}m")
        print(f"  Depth range: {result.min_depth:.3f} - {result.max_depth:.3f}m")
        print(f"  Pixel count: {result.pixel_count}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Bbox: {result.bbox}")
        print()

if __name__ == "__main__":
    main()
