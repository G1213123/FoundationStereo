import argparse
import cv2
import os
from pathlib import Path
from ultralytics import YOLO

def run_inference_single(image_path, model, output_dir, conf_threshold=0.5):
    """
    Run YOLO inference on a single image and save the results.
    """
    
    print(f"🔍 Running inference on: {Path(image_path).name}")
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,  # We will save manually to control output path
        retina_masks=True
    )
    
    # Process results
    result = results[0]
    
    # Plot results
    # plot() returns a BGR numpy array
    im_bgr = result.plot()
    
    # Construct output path
    image_name = Path(image_path).name
    output_path = os.path.join(output_dir, f"pred_{image_name}")
    
    # Save image
    cv2.imwrite(output_path, im_bgr)
    print(f"✅ Result saved to: {output_path}")
    
    # Print detection info
    if result.masks:
        print(f"  Found {len(result.masks)} instances")
        for i, (box, cls) in enumerate(zip(result.boxes, result.boxes.cls)):
            class_name = result.names[int(cls)]
            conf = float(box.conf)
            print(f"    {i+1}. {class_name} ({conf:.2f})")
    else:
        print("  No instances found.")
    
    return result

def run_inference(image_path, model_path, output_dir, conf_threshold=0.5):
    """
    Run YOLO inference on image(s) and save the results.
    Supports both single image and folder of images.
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check if path is a directory or file
    path = Path(image_path)
    
    if path.is_dir():
        # Process all images in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = [f for f in path.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No image files found in: {image_path}")
            return
        
        print(f"📁 Found {len(image_files)} images in folder")
        print(f"Processing batch inference...\n")
        
        results_summary = []
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
            result = run_inference_single(str(img_file), model, output_dir, conf_threshold)
            num_detections = len(result.masks) if result.masks else 0
            results_summary.append((img_file.name, num_detections))
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 BATCH INFERENCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {len(image_files)}")
        print(f"Output directory: {output_dir}")
        print(f"\nDetection counts:")
        for img_name, count in results_summary:
            print(f"  {img_name}: {count} instances")
        print(f"{'='*60}")
        
    elif path.is_file():
        # Process single image
        if not path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        run_inference_single(str(path), model, output_dir, conf_threshold)
    else:
        print(f"❌ Path not found: {image_path}")
        return

def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on image(s)")
    parser.add_argument("--image_path", type=str, required=True, 
                       help="Path to the input image or folder containing images")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained YOLO model (.pt)")
    parser.add_argument("--output_dir", type=str, default="output/yolo_inference", 
                       help="Directory to save results")
    parser.add_argument("--conf_threshold", type=float, default=0.5, 
                       help="Confidence threshold")
    
    args = parser.parse_args()
    
    run_inference(args.image_path, args.model_path, args.output_dir, args.conf_threshold)

if __name__ == "__main__":
    main()
