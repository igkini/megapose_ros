#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='megapose-1.0-RGB-multi-hypothesis-icp',
        description='MegaPose model to use for inference'
    )

    # /camera/camera/color/image_raw
    rgb_image_topic_arg = DeclareLaunchArgument(
        'rgb_image_topic',
        default_value='/zedm/zed_node/left/image_rect_color',
        description='Topic for RGB image input'
    )

    depth_image_topic_arg = DeclareLaunchArgument(
        'depth_image_topic',
        default_value='/zedm/zed_node/depth/depth_registered',
        description='Topic for depth image input'
    )
    
    # /camera/camera/color/camera_info
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/zedm/zed_node/left/camera_info',
        description='Topic for camera info input'
    )
    detection_topic_arg = DeclareLaunchArgument(
        'detection_topic',
        default_value='/yolo/detections',
        description='Topic for YOLO detections input'
    )
    
    pose_output_topic_arg = DeclareLaunchArgument(
        'pose_output_topic',
        default_value='/megapose/estimation',
        description='Topic for publishing MegaPose estimations'
    )
    
    service_name_arg = DeclareLaunchArgument(
        'service_name',
        default_value='run_inference',
        description='Service name for triggering inference'
    )
    
    camera_frame_id_arg = DeclareLaunchArgument(
        'camera_frame_id',
        default_value='zedm_left_camera_optical_frame',
        description='Frame ID of the camera for TF'
    )
    
    resize_factor_arg = DeclareLaunchArgument(
        'resize_factor',
        default_value='1',
        description='Integer factor by which to down-sample the long image edge '
                    'and scale intrinsics / bboxes (e.g. 2, 3 …)'
    )
    
    cropped_arg = DeclareLaunchArgument(
        'cropped',
        default_value='true',
        description='Whether YOLO detections are from cropped images'
    )
    
    # Create node
    megapose_node = Node(
        package='megapose',
        executable='run_inference_ros',
        name='megapose_launch',
        parameters=[{
            'model_name': LaunchConfiguration('model_name'),
            'rgb_topic': LaunchConfiguration('rgb_image_topic'),
            'depth_topic': LaunchConfiguration('depth_image_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'detection_topic': LaunchConfiguration('detection_topic'),
            'pose_output_topic': LaunchConfiguration('pose_output_topic'),
            'service_name': LaunchConfiguration('service_name'),
            'camera_frame_id': LaunchConfiguration('camera_frame_id'),
            'resize_factor': LaunchConfiguration('resize_factor'),
            'cropped': LaunchConfiguration('cropped'),
        }],
        output='screen'
    )
    
    return LaunchDescription([
        model_name_arg,
        rgb_image_topic_arg,
        depth_image_topic_arg,
        camera_info_topic_arg,
        detection_topic_arg,
        pose_output_topic_arg,
        service_name_arg,
        camera_frame_id_arg,
        resize_factor_arg,
        cropped_arg,
        megapose_node
    ])