import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion (x,y,z,w) to a 3x3 rotation matrix.

    Assumes right-handed coordinates. Normalizes quaternion if needed.
    """
    q = np.array([qx, qy, qz, qw], dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-length quaternion")
    q = q / norm
    x, y, z, w = q
    # Rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)],
    ], dtype=float)
    return R


def euler_xyz_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Build rotation matrix from intrinsic XYZ Euler angles in degrees.

    Applies rotations about local X, then Y, then Z: R = Rz @ Ry @ Rx
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx),  math.cos(rx)],
    ], dtype=float)

    Ry = np.array([
        [ math.cos(ry), 0, math.sin(ry)],
        [ 0,            1, 0          ],
        [-math.sin(ry), 0, math.cos(ry)],
    ], dtype=float)

    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz),  math.cos(rz), 0],
        [0,             0,            1],
    ], dtype=float)

    # Intrinsic rotations X->Y->Z
    return Rz @ Ry @ Rx


def find_first_bbox(values: List[Dict]) -> Optional[Dict]:
    for v in values:
        if all(k in v for k in ("translation", "rotation")):
            return v
    return None


def extract_pose_from_frame_json(frame_json: Dict, camera_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (translation, rotation_matrix) for the first 3D bbox on a camera.

    - translation: ndarray shape (3,), assumed to be the box corner or center depending on dataset; SOLO uses bbox center in camera frame.
    - rotation_matrix: ndarray shape (3,3), from quaternion (x,y,z,w) provided.
    """
    captures = frame_json.get("captures", [])
    for cap in captures:
        if camera_id is not None and cap.get("id") != camera_id:
            continue
        anns = cap.get("annotations", [])
        for ann in anns:
            if ann.get("@type", "").endswith("BoundingBox3DAnnotation"):
                values = ann.get("values", [])
                bbox = find_first_bbox(values)
                if bbox is None:
                    continue
                t = np.array(bbox["translation"], dtype=float)
                # Quaternion order in SOLO appears to be [x,y,z,w]
                q = np.array(bbox["rotation"], dtype=float)
                if q.shape[0] != 4:
                    raise ValueError("Expected quaternion [x,y,z,w]")
                R = quat_to_matrix(q[0], q[1], q[2], q[3])
                return t, R
    raise ValueError("No BoundingBox3DAnnotation with translation/rotation found (camera_id=%s)" % (camera_id,))


def compute_corners_from_corner(base_corner_world: np.ndarray,
                                R_axes_world: np.ndarray,
                                size_xyz: Tuple[float, float, float]) -> Dict[str, List[float]]:
    """Given a base corner position and oriented axes, generate 8 box corners.

    base_corner_world: (3,) numpy array (this is one corner of the box)
    R_axes_world: (3,3) rotation matrix whose columns are unit vectors for +X, +Y, +Z axes directions in world
    size_xyz: (Lx, Ly, Lz) edge lengths along those axes

    Returns dict of corner_name -> [x,y,z]
    Names: c000, c100, c010, c110, c001, c101, c011, c111 where 1 indicates adding that axis length.
    """
    Lx, Ly, Lz = size_xyz
    ux = R_axes_world[:, 0]
    uy = R_axes_world[:, 1]
    uz = R_axes_world[:, 2]

    corners = {}
    # bit pattern: (a,b,c) in {0,1}^3
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                name = f"c{a}{b}{c}"
                offset = (a * Lx) * ux + (b * Ly) * uy + (c * Lz) * uz
                p = base_corner_world + offset
                corners[name] = [float(p[0]), float(p[1]), float(p[2])]
    return corners


def main():
    parser = argparse.ArgumentParser(description="Compute 3D box corners from Unity SOLO frame_data.json with additional transform.")
    parser.add_argument("frame_json", type=Path, help="Path to frame_data.json")
    parser.add_argument("--camera-id", type=str, default=None, help="Filter by camera id (optional)")
    parser.add_argument("--offset-translation", type=float, nargs=3, default=[9.0, -5.7, 11.9319], metavar=("TX","TY","TZ"),
                        help="Additional local translation to the chosen corner (x y z)")
    parser.add_argument("--offset-rotation", type=float, nargs=3, default=[-90.0, 90.0, -90.0], metavar=("RX","RY","RZ"),
                        help="Additional intrinsic XYZ Euler rotation in degrees applied to local axes")
    parser.add_argument("--size", type=float, nargs=3, default=[10.4, 6.0, 2.8], metavar=("SX","SY","SZ"),
                        help="Box dimensions along x y z (edge lengths)")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write corners as JSON")

    args = parser.parse_args()

    # Load JSON
    with args.frame_json.open("r", encoding="utf-8") as f:
        frame = json.load(f)

    # Extract bbox pose (center + orientation) in camera/world frame
    center_t, R_box = extract_pose_from_frame_json(frame, camera_id=args.camera_id)

    # Build additional rotation and translation (local)
    R_off = euler_xyz_to_matrix(*args.offset_rotation)
    t_off = np.array(args.offset_translation, dtype=float)

    # Compute base corner in world: center + R_box * (R_off * t_off)
    # Here, R_off defines local axes orientation relative to bbox; translating by t_off after R_off yields a specific corner in bbox local frame.
    base_corner_world = center_t + R_box @ (R_off @ t_off)

    # Axes directions in world for the box edges: columns of R_world = R_box @ R_off
    R_axes_world = R_box @ R_off

    corners = compute_corners_from_corner(base_corner_world, R_axes_world, tuple(args.size))

    result = {
        "frame_json": str(args.frame_json),
        "camera_id": args.camera_id,
        "center": [float(center_t[0]), float(center_t[1]), float(center_t[2])],
        "base_corner": [float(base_corner_world[0]), float(base_corner_world[1]), float(base_corner_world[2])],
        "size": list(map(float, args.size)),
        "offset_translation": list(map(float, args.offset_translation)),
        "offset_rotation_xyz_deg": list(map(float, args.offset_rotation)),
        "corners": corners,
    }

    # Pretty print to stdout
    print(json.dumps(result, indent=2))

    # Optionally write to file
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote corners to: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
