# MegaPose ROS
ROS2 wrapper for MegaPose 6D object pose estimation based on the implementation of [megapose6d](https://github.com/megapose6d/megapose6d). It has been tested with ZED and RealSense cameras.

## Overview
This package provides:
- A ROS2 node for MegaPose 6D pose estimation(src/megapose/megapose/scripts/run_inference_ros.py)
- Service-based inference triggered on demand
- Configurable topics and parameters via launch file(src/megapose/launch/megapose_launch.launch.py)
- Integrated YOLO ROS package from [yolo_ros](https://github.com/mgonzs13/yolo_ros).

## Prerequisites
- ROS2 Jazzy (only tested version).
- CUDA-capable GPU with at least 6GB of VRAM, but **12BG** is recommended.
- CUDA 12.6 was used during development and testing(other CUDA versions might work as well).
- Realsense or Zed SDK installed, or a ros2 bag file.


## Installation

### 1. Clone the repository:

```bash
cd ~/megapose_ros/src
git clone https://github.com/igkini/megapose_ros.git
```

#### 2. Download the megapose [models](https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/) and place them under `src/megapose/megapose/local_data/megapose-models/`

#

<details> 

<summary style="font-size: 18px;"><strong>Build with Docker(Recommended)</strong></summary>

#

#### Prerequisites 

1. NVIDIA Container Toolkit 

#

#### Installation Steps:

1. Build the image:
  ```bash
    docker build -t megapose_ros .
  ```
2. Allow local Docker containers to connect to your X server:
  ```bash
    xhost +local:root
  ```  
3. Allow local Docker containers to connect to your X server:
  ```bash 
    docker run --gpus all -it --privileged -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$(pwd)":/workspace megapose_ros
  ```

#

</details> 

<details> 

<summary style="font-size: 18px;"><strong>Local Build</strong></summary>

#

#### Prerequisites
- CUDA 12.6 was used during development and testing (other CUDA versions might work as well)

#

#### Installation Steps

> **Important:** You will need to use `--break-system-packages` when installing Python dependencies to the system path

1. Install the main megapose package:
    ```bash
    cd megapose
    pip install -e . --break-system-packages
    cd ..
    ```

2. It is **recommended** to follow the usage steps up to: `ros2 launch megapose_ros megapose_launch.launch.py` and each time a missing module appears, run:
    
    ```bash
    pip install [module_name] --break-system-packages
    ```
    
    **Else** Install missing python modules using:
    
    ```bash
    pip install -r requirements-basic.txt --break-system-packages
    ```
    
3. To resolve the `ModuleNotFoundError: No module named 'bop_toolkit_lib'` error, run:
    ```bash
    cd src/megapose/megapose/deps/bop_toolkit_challenge/
    pip install -e .
    cd ../../../../..
    ```

</details> 

## Setup and Usage

### 1. Build the Package
```bash
cd ~/ros2_ws
colcon build
source install/setup.bash 
```

### 2. Object Meshes
The 3D meshes for objects are provided in the `megapose/megapose/local_data/custom_data/[label]/meshes/[label]/` directory. These meshes are used by MegaPose for pose estimation. The folder names within `custom_data/` must exactly match the class labels output by your YOLO detection model. For example, if YOLO detects an object with the label "class_3", MegaPose will look for the corresponding 3D mesh in `megapose/local_data/custom_data/class_3/meshes/class_3/.obj, .mtl, .png`.

#### Custom Mesh Requirements
- File format: `.obj` with accompanying `.mtl` and texture image files `.png, .jpeg, etc...`
- Acquisition methods:
  - Mobile scanner apps
  - Blender with UVgami plug-in for **UV-grid** and **texture image** acquisition
- Post-processing:
  - Models may need to be scaled and centered in Meshlab (x1000, or x0.001)
  - Ensure texture file path is correctly referenced in the `.mtl` file
  - **Performance optimization**: Remeshing CAD models with voxel > 0.6 will significantly increase resource capabilities

### 3. Camera Setup
The default configuration uses ZED 2i camera topics. Choose your camera and modify topics accordingly in the launch file:

#### ZED Camera
```bash
# Launch ZED 
ros2 launch zed_wrapper zed2i.launch.py
```

#### RealSense Camera
```bash
# Launch RealSense
ros2 launch realsense2_camera rs_launch.py
```

### 3. YOLO Detection Setup
You must train a YOLO model for your object and run the YOLO ROS node that publishes to the `/yolo/detections` topic using the `yolo_msgs/DetectionArray` message type.
```bash
# Launch your YOLO detection
ros2 launch yolo_bringup yolov11.launch.py
```

### 4. Launch MegaPose
```bash
# Launch with default parameters (ZED 2i)
ros2 launch megapose_ros megapose_launch.launch.py
# Or with custom camera topics
ros2 launch megapose_ros megapose_launch.launch.py \
 rgb_image_topic:=/camera/color/image_raw \
 camera_info_topic:=/camera/color/camera_info \
 resize_factor:=2
```

### 5. Trigger Inference
```bash
# Run inference using the first detected label by default
ros2 service call /run_inference megapose_interfaces/srv/LabelInference "{label: ''}"

# Or specify a particular label
ros2 service call /run_inference megapose_interfaces/srv/LabelInference "{label: 'port'}"
```

### 6. Visualize

To visualize the results run:

```bash
rviz2
```

## Topics

### Subscribed
- RGB images (sensor_msgs/Image)
- Depth images (sensor_msgs/Image)
- Camera info (sensor_msgs/CameraInfo)
- YOLO detections (yolo_msgs/DetectionArray)

### Published
- 6D pose estimates (vision_msgs/Detection3DArray)

## Services
- `/run_inference` (std_srvs/srv/Trigger): Triggers pose estimation with current data

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_name | string | megapose-1.0-RGB-multi-hypothesis-icp | MegaPose model to use. Defaults to ICP model when depth topic is available, falls back to megapose-1.0-RGB-multi-hypothesis if no depth topic is detected |
| rgb_topic | string | /zedm/zed_node/left/image_rect_color | RGB image input topic |
| depth_topic | string | /zedm/zed_node/depth/depth_registered | Depth image input topic |
| camera_info_topic | string | /zedm/zed_node/left/camera_info | Camera calibration info topic |
| detection_topic | string | /yolo/detections | Detection input topic |
| pose_output_topic | string | /megapose/estimation | 6D pose output topic |
| service_name | string | run_inference | Service name for inference |
| camera_frame_id | string | zedm_left_camera_frame | TF frame for camera |
| resize_factor | int | 1 | Factor to resize both depth and RGB frames. Automatically adjusts camera parameters and YOLO bounding box coordinates for **resource management** |
| cropped | bool | false | This parameter was introduced for a custom use case. Use false if the input image topic is the same for both YOLO and MegaPose. Enable when RGB topic resolution differs from YOLO detection image resolution (e.g., HD2K RGB input with 1024x1024 YOLO detections). |

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### Third-party Licenses

This package integrates the following third-party software:

- **MegaPose**: Licensed under Apache 2.0 License
  - Repository: https://github.com/megapose6d/megapose6d
- **YOLO ROS**: Licensed under GPL-3.0 License  
  - Repository: https://github.com/mgonzs13/yolo_ros
  - Author: Miguel Ángel González Santamarta