# Foundation Stereo Scripts

This folder contains all scripts related to the Foundation Stereo depth estimation model.

## Main Scripts

### Demo and Inference
- `run_demo.py` - Main demo script for Foundation Stereo depth estimation
- `run_demo_tensorrt.py` - TensorRT optimized demo

### Model Conversion and Optimization
- `make_onnx.py` - Convert model to ONNX format

### Data Processing and Visualization
- `depth_segmentation_pipeline.py` - Combined depth estimation and segmentation pipeline
- `view_pointcloud.py` - Visualize depth maps as point clouds
- `vis_dataset.py` - Visualize training datasets

## Usage

To run the main Foundation Stereo demo from the project root:
```bash
python scripts/foundation_stereo/run_demo.py
```

To create point cloud visualizations:
```bash
python scripts/foundation_stereo/view_pointcloud.py
```

## Purpose

These scripts handle:
- Stereo depth estimation using Foundation Stereo model
- Point cloud generation and visualization
- Model optimization and conversion
- Dataset visualization and analysis
