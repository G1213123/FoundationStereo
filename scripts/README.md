# Scripts Organization

This folder contains scripts organized by model type and functionality.

## Structure

```
scripts/
├── unity_yolo/          # Unity instance segmentation + YOLO training/inference
│   ├── README.md        # Detailed documentation for Unity YOLO scripts
│   ├── unity_instance_seg_yolo.py  # Main training pipeline
│   ├── test_yolo_best_only.py      # Main inference script
│   └── ...              # Other Unity/YOLO related scripts
│
├── foundation_stereo/   # Foundation Stereo depth estimation
│   ├── README.md        # Detailed documentation for Foundation Stereo scripts
│   ├── run_demo.py      # Main demo script
│   └── ...              # Other Foundation Stereo related scripts
│
└── README.md            # This file
```

## Quick Start

### Unity YOLO (Object Detection)
```bash
# Run inference on Unity images
python scripts/unity_yolo/test_yolo_best_only.py

# Train new model
python scripts/unity_yolo/unity_instance_seg_yolo.py --action train --target_classes Module_Construction --max_sequences 50
```

### Foundation Stereo (Depth Estimation)
```bash
# Run depth estimation demo
python scripts/foundation_stereo/run_demo.py
```

## Results

- Unity YOLO results: `./results/` folder
- Foundation Stereo results: Check individual script documentation

See individual README files in each subfolder for detailed documentation.
