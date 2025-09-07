#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from PIL import Image as PilImage
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import ObjectData, CameraData
from megapose.lib3d.transform import Transform
from megapose.utils.logging import get_logger
from typing import Tuple, List, Dict

logger = get_logger(__name__)


def convert_camera_info(camera_info_msg, resize_factor: int = 1):
    """
    Convert a ROS CameraInfo message into the dict format MegaPose expects.
    The intrinsics and resolution are pre-scaled by resize_factor.
    """
    K = np.array(camera_info_msg.k, dtype=np.float32).reshape(3, 3)

    if resize_factor != 1:
        K[0, 0] /= resize_factor
        K[1, 1] /= resize_factor
        K[0, 2] /= resize_factor
        K[1, 2] /= resize_factor

    return {
        "K": K.tolist(),
        "resolution": [
            int(camera_info_msg.height / resize_factor),
            int(camera_info_msg.width  / resize_factor),
        ],
    }


def convert_yolo_detections(
    detection_array_msg,
    camera_res: Tuple[int, int],
    crop_size: Tuple[int, int] = (1024, 1024),
    resize_factor: int = 1,
    cropped: bool = True
) -> List[Dict]:
    """
    Args:
        detection_array_msg: YOLO-ROS DetectionArray (coords on 1024×1024)
        camera_res:          (orig_h, orig_w) of the raw camera image
        crop_size:           (crop_h, crop_w) of the patch (default 1024×1024)
        resize_factor:       same factor you use in image_cb to downsample
        cropped:             if False, assume patch origin = (0,0)

    Returns:
        List of { "label": str, "bbox_modal": [x1,y1,x2,y2] }
    """
    orig_h, orig_w = camera_res
    crop_h, crop_w = crop_size

    if cropped:
        x0 = max((orig_w - crop_w) // 2, 0)
        y0 = max((orig_h - crop_h) // 2, 0)
    else:
        x0 = 0
        y0 = 0

    dets: List[Dict] = []
    for d in detection_array_msg.detections:
        cx_crop = d.bbox.center.position.x
        cy_crop = d.bbox.center.position.y
        w_crop  = d.bbox.size.x
        h_crop  = d.bbox.size.y

        cx_full = cx_crop + x0
        cy_full = cy_crop + y0
        w_full  = w_crop
        h_full  = h_crop

        x1_full = cx_full - w_full / 2.0
        y1_full = cy_full - h_full / 2.0
        x2_full = cx_full + w_full / 2.0
        y2_full = cy_full + h_full / 2.0

        x1 = x1_full / resize_factor
        y1 = y1_full / resize_factor
        x2 = x2_full / resize_factor
        y2 = y2_full / resize_factor

        dets.append({
            "label": d.class_name,
            "bbox_modal": [x1, y1, x2, y2],
        })

    return dets

def save_inputs(example_dir, rgb_image, cam_info_dict, detections_list):
    """Save input data to disk for debugging or later use."""
    example_dir = Path(example_dir)
    example_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_path = example_dir / "image_rgb.png"
    PilImage.fromarray(rgb_image).save(rgb_path)
    
    camera_data_path = example_dir / "camera_data.json"
    with open(camera_data_path, 'w') as f:
        json.dump(cam_info_dict, f, indent=2)
    
    inputs_dir = example_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    object_data_path = inputs_dir / "object_data.json"
    with open(object_data_path, 'w') as f:
        json.dump(detections_list, f, indent=2)
    
    logger.info(f"Saved input data to {example_dir}")


def save_predictions(example_dir, pose_estimates):
    """Save MegaPose predictions to disk."""
    example_dir = Path(example_dir)
    
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    
    output_dir = example_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    object_data_list = []
    for label, pose in zip(labels, poses):
        transform = Transform(pose)
        od = ObjectData(label=label, TWO=transform)
        object_data_list.append(od.to_json())
    
    output_fn = output_dir / "object_data.json"
    with open(output_fn, 'w') as f:
        json.dump(object_data_list, f, indent=2)
    
    logger.info(f"Saved predictions to {output_fn}")


def make_object_dataset(example_dir):
    """Create a RigidObjectDataset from meshes in the specified directory."""
    example_dir = Path(example_dir)
    mesh_units = "mm"
    rigid_objects = []
    
    meshes_dir = example_dir / "meshes"
    if not meshes_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found at '{meshes_dir}'")
    
    mesh_count = 0
    
    mesh_extensions = ["*.obj", "*.OBJ", "*.ply", "*.PLY"]
    
    for object_dir in meshes_dir.iterdir():
        if object_dir.is_dir():
            label = object_dir.name
            mesh_path = None
            
            for pattern in mesh_extensions:
                matches = list(object_dir.glob(pattern))
                if matches:
                    mesh_path = matches[0]
                    break
                    
            if not mesh_path:
                logger.warning(f"No mesh file found for {label}")
                continue
                
            rigid_objects.append(
                RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units)
            )
            mesh_count += 1
    
    for pattern in mesh_extensions:
        for mesh_path in meshes_dir.glob(pattern):
            if mesh_path.is_file():
                label = mesh_path.stem
                
                if any(obj.label == label for obj in rigid_objects):
                    continue
                    
                rigid_objects.append(
                    RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units)
                )
                mesh_count += 1
    
    if not rigid_objects:
        raise FileNotFoundError(f"No valid mesh files found in '{meshes_dir}'")
    
    logger.info(f"Created dataset with {mesh_count} objects")
    return RigidObjectDataset(rigid_objects)