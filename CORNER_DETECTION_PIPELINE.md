# Corner Detection Pipeline (CDL) Methodology

This document outlines the algorithmic steps for the Corner Detection Pipeline (CDL), designed to localize 3D box corners from depth and edge maps. The pipeline integrates 2D image processing with 3D geometric reconstruction to achieve robust corner estimation.

## 1. Data Ingestion and Preprocessing
The pipeline initiates by loading raw depth maps ($D$) and binary edge maps ($E$) from the input directory.
- **Depth Map**: Loaded from `.npy` or `.exr` formats. Zeros are treated as invalid or background.
- **Edge Map**: Loaded from `.npy` or `.png` formats.
- **Instance Filtering**: If instance segmentation metadata is available (e.g., from YOLO), the edge map is masked to retain only the edges corresponding to the highest-confidence object instance.

## 2. Region of Interest (ROI) Extraction
To focus computational resources and reduce noise, a Region of Interest (ROI) is computed.
- **Edge Buffering**: The binary edge map $E$ is dilated using a morphological rectangular kernel defined by `buffer_px`.
- **Bounding Box**: An axis-aligned bounding box is computed around the buffered edges, with an optional margin `roi_margin_px`.
- **Cropping**: The depth map is cropped to this ROI for subsequent processing.

## 3. Depth Edge Detection
Edges are detected directly from the depth information to complement the semantic edges.
- **Depth Smoothing**: The depth ROI undergoes a multi-stage smoothing process to mitigate sensor noise:
    1.  **Invalid Filling**: Invalid pixels are filled with the median value of valid pixels.
    2.  **Outlier Removal**: Pixels outside the 5th-95th percentile range are clipped.
    3.  **Morphological Smoothing**: A morphological opening/closing sequence (erosion followed by dilation) with an elliptical kernel is applied to reduce noise while preserving structural boundaries.
    4.  **Sigma Clipping**: Statistical outlier removal is performed based on the mean and standard deviation ($\mu \pm 3\sigma$) of the depth values.
- **Edge Extraction**: A gradient-based edge detection method (Canny, Sobel, or Laplacian) is applied to the smoothed depth map. The Canny edge detector is the default, utilizing dynamic thresholding based on image statistics.

## 4. Edge Filtering and Loop Closure
The detected depth edges are refined to isolate the object boundary.
- **Proximity Filtering**: Depth edges are filtered based on their Euclidean distance to the original semantic edges. A Distance Transform is used to retain only those depth edges within `max_edge_dist_px`.
- **Closed Loop Detection**: An iterative morphological closing operation is applied to the filtered edges to form a continuous closed loop. Contours are analyzed at each iteration to identify a closed polygon that satisfies minimum area and vertex constraints.

## 5. 3D Point Cloud Generation
The 2D region defined by the closed loop is lifted into 3D space.
- **Boundary Extraction**: Pixels within the closed loop polygon are extracted.
- **Back-projection**: Using the camera intrinsic parameters ($f_x, f_y, c_x, c_y$), the 2D pixel coordinates $(u, v)$ and their corresponding depth values $Z$ are back-projected into 3D camera space $(X_c, Y_c, Z_c)$.
- **Point Cloud Processing**:
    - **Downsampling**: Voxel grid downsampling is applied to regularize point density.
    - **Outlier Removal**: Statistical outlier removal filters points based on local neighborhood distances.
    - **Normal Estimation**: Surface normals are estimated for each point.

## 6. Surface Reconstruction and Segmentation
The 3D point cloud is processed to identify planar surfaces.
- **Mesh Generation**: A triangle mesh is reconstructed from the point cloud using the Ball Pivoting Algorithm (BPA).
- **Plane Segmentation**: The RANSAC (Random Sample Consensus) algorithm is iteratively applied to the mesh vertices to segment dominant planar surfaces.

## 7. Corner Localization
The final corner position is derived from the intersection of the segmented planes.
- **Orthogonality Check**: Planes are filtered based on their orthogonality to the "prime plane" (the plane with the most inliers). Planes deviating significantly from orthogonality are discarded.
- **Intersection Computation**:
    - **3 Planes**: The unique intersection point is computed by solving the linear system $N \mathbf{x} = -d$, where $N$ is the matrix of normal vectors and $d$ is the vector of plane distances.
    - **2 Planes**: The intersection line is computed, and the corner is estimated by projecting the inlier points onto this line.
    - **1 Plane**: A bounding rectangle is fitted to the projected points on the plane, and corners are derived from the rectangle vertices.

## 8. Evaluation
The detected corner is compared against ground truth data (if available).
- **Coordinate Transformation**: Detected corners are transformed from camera space to world space using the camera's extrinsic parameters (position and rotation).
- **Metric Comparison**: The Euclidean distance between the detected corner and the nearest ground truth corner is calculated to quantify accuracy.
