# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union
import torch
# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
import bokeh.io
from bokeh.io import output_file, save
# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay
import time


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple




logger = get_logger(__name__)


def load_observation(
    example_dir: Path,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    rgb = np.array(Image.open(example_dir / "image_rgb.png"), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path,
) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


# def make_detections_visualization(
#     example_dir: Path,
# ) -> None:
#     detections = load_detections(example_dir)
#     plotter = BokehPlotter()
#     fig_rgb = plotter.plot_image(rgb)
#     fig_det = plotter.plot_detections(fig_rgb, detections=detections)
#     output_fn = example_dir / "visualizations" / "detections.png"
#     output_fn.parent.mkdir(exist_ok=True)
#     fig_det.output_backend = "svg"  # or "canvas"
#     bokeh.io.save(fig_det, filename=str(output_fn).replace('.png', '.html'))
#     return

def plot_side_by_side(images: list, titles: Optional[list] = None, figsize: Tuple[int, int] = None) -> None:
    """Plot multiple images side by side using matplotlib"""
    n = len(images)
    if titles is None:
        titles = [f'Image {i+1}' for i in range(n)]
    
    if figsize is None:
        figsize = (5*n, 5)
        
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale
            ax.imshow(img, cmap='gray')
        else:  # RGB
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def make_detections_visualization(example_dir: Path) -> None:
    """Save detection visualization using matplotlib"""
    rgb, _, _ = load_observation(example_dir, load_depth=False)
    detections = load_detections(example_dir)
    
    # Create visualization directory if it doesn't exist
    output_dir = example_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Plot RGB image with detections
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    
    # Add bounding boxes for detections
    for detection in detections.bboxes:
        bbox = detection.cpu().numpy()
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        plt.gca().add_patch(plt.Rectangle((x1, y1), width, height,
                                        fill=False, color='red', linewidth=2))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / "detections.png"
    plt.savefig(str(output_path), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    logger.info(f"Saved detection visualization to {output_path}")

def save_predictions(
    example_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return


def run_inference(
    example_dir: Path,
    model_name: str,
) -> None:
    model_info = NAMED_MODELS[model_name]

    observation = load_observation_tensor(
        example_dir, load_depth=model_info["requires_depth"]
    ).cuda()
    detections = load_detections(example_dir).cuda()
    object_dataset = make_object_dataset(example_dir)

    logger.info(f"Loading model {model_name}.")
    first_time = time.time()

    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    logger.info(f"Running inference.")
    
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )
    second_time = time.time()
    elapsed_time = second_time - first_time
    logger.info(f"Time elapsed between first and second predictions: {elapsed_time:.2f} seconds")






    save_predictions(example_dir, output)
    return



def make_output_visualization(example_dir: Path) -> None:
    """Save output visualizations using matplotlib"""
    rgb, _, camera_data = load_observation(example_dir, load_depth=False)
    camera_data.TWC = Transform(np.eye(4))
    object_datas = load_object_data(example_dir / "outputs" / "object_data.json")
    object_dataset = make_object_dataset(example_dir)
    renderer = Panda3dSceneRenderer(object_dataset)
    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient", 
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    vis_dir = example_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Create contour overlay
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]

    # Create mesh overlay
    alpha = 0.5
    # Convert inputs to float32 and normalize to 0-1 range if needed
    rgb_float = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float32)
    render_float = renderings.rgb.astype(np.float32) / 255.0 if renderings.rgb.dtype == np.uint8 else renderings.rgb.astype(np.float32)
    
    # Perform the blending
    mesh_overlay = (1 - alpha) * rgb_float + alpha * render_float
    
    # Ensure values are in 0-1 range
    mesh_overlay = np.clip(mesh_overlay, 0, 1)

    # Plot all results together
    images = [rgb, contour_overlay, mesh_overlay]
    titles = ['Original', 'Contour Overlay', 'Mesh Overlay']
    
    fig = plot_side_by_side(images, titles, figsize=(15, 5))
    fig.savefig(str(vis_dir / "all_results.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Save individual images
    plt.imsave(str(vis_dir / "contour_overlay.png"), contour_overlay)
    plt.imsave(str(vis_dir / "mesh_overlay.png"), mesh_overlay)

    logger.info(f"Wrote visualizations to {vis_dir}")


# def make_mesh_visualization(RigidObject) -> List[Image]:
#     return


# def make_scene_visualization(CameraData, List[ObjectData]) -> List[Image]:
#     return


# def run_inference(example_dir, use_depth: bool = False):
#     return


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    parser.add_argument("--cuda-id", type=int, default=1, help="CUDA device ID (default: 0)")
    args = parser.parse_args()

    # Set CUDA device ID
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    logger.info(f"Using CUDA device: {args.cuda_id}")

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name

    if args.vis_detections:
        make_detections_visualization(example_dir)

    if args.run_inference:
        run_inference(example_dir, args.model)

    if args.vis_outputs:
        make_output_visualization(example_dir)

