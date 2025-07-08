# Unity YOLO Scripts

This folder contains all scripts related to Unity instance segmentation and YOLO training/inference.

## Environment Configuration

All scripts use environment variables from `.env` file for path configuration:

```properties
# Unity Project Paths - NEW FLEXIBLE CONFIGURATION
UNITY_PROJECT_PATH=C:\Users\1213123\AppData\LocalLow\DefaultCompany\My project (1)
UNITY_DATASET_NAME=solo_9

# Other paths and settings...
```

The Unity data directory is constructed dynamically as: `UNITY_PROJECT_PATH/UNITY_DATASET_NAME`

## Main Scripts

### Training and Pipeline
- `unity_instance_seg_yolo.py` - Main pipeline for training YOLO on Unity instance segmentation data
- `run_unity_auto_pipeline.py` - Automated pipeline runner

### Testing and Inference  
- `test_yolo_best_only.py` - **Main inference script** - Returns only best detections
- `get_best_detection.py` - Analyzes and recommends optimal confidence thresholds
- `test_complete_pipeline.py` - Tests the complete pipeline
- `test_new_model.py` - Test newly trained models

### Analysis and Debugging
- `debug_detection.py` - Debug why YOLO detects no objects
- `test_confidence_thresholds.py` - Test different confidence thresholds
- `test_training_vs_test.py` - Compare performance on different sequences

### Unity Data Processing
- `unity_instance_segmentation.py` - Extract objects from Unity PNG files
- `analyze_unity_json.py` - Analyze Unity JSON metadata
- `extract_unity_intrinsics.py` - Extract camera parameters
- `find_unity_objects.py` - Find and analyze Unity objects

### Legacy/Alternative Scripts
- `unity_auto_yolo.py` - Alternative YOLO training approach
- `simple_unity_yolo.py` - Simplified version
- `unity_yolo_trainer.py` - Alternative trainer

## Current Model Performance

- **Best Model**: `runs/unity_blocks_auto6/weights/best.pt`
- **Confidence Range**: 0.83-0.99 for high-quality detections
- **Recommended Threshold**: 0.50 for balanced precision/recall
- **Training Data**: 50 sequences from Unity automatic annotations

## Path Configuration Benefits

✅ **Flexible dataset switching**: Change `UNITY_DATASET_NAME` to switch between solo_9, solo_7, etc.
✅ **No hardcoded paths**: All paths configured via environment variables
✅ **Easy deployment**: Copy `.env` file and update paths for different systems
✅ **Maintainable code**: Single source of truth for configuration

## Usage

### Quick Inference
```bash
python scripts/unity_yolo/test_yolo_best_only.py
```

### Train New Model
```bash
python scripts/unity_yolo/unity_instance_seg_yolo.py --action full --max_sequences 50
```

### Switch Dataset
```properties
# In .env file
UNITY_DATASET_NAME=solo_7  # Switch from solo_9 to solo_7
```

## Results

All results are saved to `./results/` folder with annotated images showing detected objects.
