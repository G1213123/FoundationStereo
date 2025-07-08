"""
Unity Auto-YOLO Pipeline Usage Examples
This script demonstrates how to use Unity's automatic annotations for training YOLO
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

def run_extract_annotations():
    """Extract annotations from Unity's solo_9 directory using instance segmentation PNG files"""
    print("=" * 60)
    print("STEP 1: Extract Unity Instance Segmentation Data")
    print("=" * 60)
    
    cmd = 'python scripts/unity_instance_seg_yolo.py --action extract --target_classes Module_Construction block cube'
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print("✅ Instance segmentation extraction completed!")
        return True
    else:
        print("❌ Instance segmentation extraction failed!")
        return False

def run_train_model():
    """Train YOLO model on extracted instance segmentation data"""
    print("=" * 60)
    print("STEP 2: Train YOLO Model on Instance Segmentation")
    print("=" * 60)
    
    cmd = 'python scripts/unity_instance_seg_yolo.py --action train --target_classes Module_Construction block cube'
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print("✅ Model training completed!")
        return True
    else:
        print("❌ Model training failed!")
        return False

def run_test_model():
    """Test the trained model"""
    print("=" * 60)
    print("STEP 3: Test Trained Model")
    print("=" * 60)
    
    # Find the latest trained model
    runs_dir = Path("./runs")
    if runs_dir.exists():
        project_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "unity_blocks_auto" in d.name]
        if project_dirs:
            # Get the most recent project directory
            latest_project = max(project_dirs, key=lambda x: x.stat().st_mtime)
            model_path = latest_project / "weights" / "best.pt"
            
            if model_path.exists():
                cmd = f'python scripts/unity_instance_seg_yolo.py --action visualize'
                print(f"Running: {cmd}")
                result = os.system(cmd)
                
                if result == 0:
                    print("✅ Model testing completed!")
                    return True
                else:
                    print("❌ Model testing failed!")
                    return False
            else:
                print(f"❌ Model file not found: {model_path}")
                return False
        else:
            print("❌ No trained models found in runs directory")
            return False
    else:
        print("❌ Runs directory not found - train a model first")
        return False

def run_integration_test():
    """Run integrated YOLO + FoundationStereo pipeline"""
    print("=" * 60)
    print("STEP 4: Integration Test (YOLO + FoundationStereo)")
    print("=" * 60)
    
    # Find trained model
    runs_dir = Path("./runs")
    model_path = None
    
    if runs_dir.exists():
        project_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "unity_instance" in d.name]
        if project_dirs:
            latest_project = max(project_dirs, key=lambda x: x.stat().st_mtime)
            model_path = latest_project / "weights" / "best.pt"
    
    if not model_path or not model_path.exists():
        print("❌ No trained model found - train a model first")
        return False
    
    # Use Unity solo_7 images for testing
    unity_solo_7 = os.getenv('UNITY_SOLO_7_PATH', 'C:\\Users\\1213123\\AppData\\LocalLow\\DefaultCompany\\My project (1)\\solo_7')
    
    # Find a stereo pair
    solo_7_path = Path(unity_solo_7)
    if not solo_7_path.exists():
        print(f"❌ Unity solo_7 directory not found: {unity_solo_7}")
        print("Please update the UNITY_SOLO_7_PATH in your .env file")
        return False
    
    # Look for sequence directories
    sequence_dirs = [d for d in solo_7_path.iterdir() if d.is_dir() and d.name.startswith('sequence.')]
    
    if not sequence_dirs:
        print("❌ No sequence directories found in solo_7")
        return False
    
    # Find stereo images
    left_image = None
    right_image = None
    
    for seq_dir in sequence_dirs[:1]:  # Just test with first sequence
        images = list(seq_dir.glob("*.png"))
        if len(images) >= 2:
            images.sort()
            # Assume first two images are stereo pair
            left_image = str(images[0])
            right_image = str(images[1])
            break
    
    if not left_image or not right_image:
        print("❌ Could not find stereo image pair in Unity data")
        return False
    
    print(f"Testing with:")
    print(f"  Left image: {Path(left_image).name}")
    print(f"  Right image: {Path(right_image).name}")
    print(f"  Model: {model_path.name}")
    
    cmd = f'python scripts/unity_instance_seg_yolo.py --action integrate --model_path "{model_path}" --left_image "{left_image}" --right_image "{right_image}"'
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print("✅ Integration test completed!")
        print("Check the visualizations directory for results")
        return True
    else:
        print("❌ Integration test failed!")
        return False

def check_environment():
    """Check if environment is properly configured"""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please copy .env.example to .env and update the paths")
        return False
    
    print("✅ .env file found")
    
    # Check Unity paths
    from dotenv import load_dotenv
    load_dotenv()
    
    unity_solo_9 = os.getenv('UNITY_SOLO_9_PATH', '')
    unity_solo_7 = os.getenv('UNITY_SOLO_7_PATH', '')
    
    if not unity_solo_9:
        print("❌ UNITY_SOLO_9_PATH not set in .env file")
        return False
    
    if not Path(unity_solo_9).exists():
        print(f"❌ Unity solo_9 directory not found: {unity_solo_9}")
        print("Please check the path in your .env file")
        return False
    
    print(f"✅ Unity solo_9 path: {unity_solo_9}")
    
    if unity_solo_7 and Path(unity_solo_7).exists():
        print(f"✅ Unity solo_7 path: {unity_solo_7}")
    else:
        print("⚠ Unity solo_7 path not configured or not found")
        print("This is needed for testing the integration")
    
    # Check intrinsics file
    intrinsics_file = os.getenv('INTRINSICS_FILE', './unity_seq49_intrinsics_cm.txt')
    if Path(intrinsics_file).exists():
        print(f"✅ Intrinsics file: {intrinsics_file}")
    else:
        print(f"⚠ Intrinsics file not found: {intrinsics_file}")
        print("You may need to run the intrinsics extraction script first")
    
    return True

def main():
    """Run the complete Unity Auto-YOLO pipeline"""
    print("🚀 Unity Auto-YOLO Pipeline")
    print("Automatically train YOLO on Unity's pre-generated annotations")
    print()
    
    # Check environment
    if not check_environment():
        print("\\n❌ Environment check failed. Please fix the issues above before continuing.")
        return
    
    print("\\n🎯 Starting pipeline...")
    
    # Step 1: Extract annotations
    if not run_extract_annotations():
        print("\\n❌ Pipeline failed at annotation extraction")
        return
    
    # Step 2: Train model
    if not run_train_model():
        print("\\n❌ Pipeline failed at model training")
        return
    
    # Step 3: Test model
    if not run_test_model():
        print("\\n❌ Pipeline failed at model testing")
        return
    
    # Step 4: Integration test
    if not run_integration_test():
        print("\\n❌ Pipeline failed at integration testing")
        return
    
    print("\\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\\nResults:")
    print("- Trained YOLO model: ./runs/unity_instance_seg*/weights/best.pt")
    print("- Dataset: ./datasets/unity_instance_seg/")
    print("- Test visualizations: ./visualizations/")
    print("- Integration results: ./results/")
    print("\\nYou can now use the trained model for block detection and depth estimation!")

if __name__ == "__main__":
    main()
