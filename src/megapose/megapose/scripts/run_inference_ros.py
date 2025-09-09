#!/usr/bin/env python3



import rclpy
from rclpy.node import Node
import os
import gc
import cv2
from types import SimpleNamespace
 
from sensor_msgs.msg import Image, CameraInfo
from yolo_msgs.msg import DetectionArray
from megapose_interfaces.srv import MegaposeInference
 
import torch
import numpy as np
from cv_bridge import CvBridge
import tf_transformations
 
from megapose.config import PKG_LOCAL_DATA_DIR
from megapose.datasets.scene_dataset import ObjectData
from megapose.inference.types import ObservationTensor
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
 
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
 
from megapose.scripts.megapose_ros_utility import (
    convert_camera_info,
    convert_yolo_detections,
    save_inputs,
    save_predictions,
    make_object_dataset
)
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

logger = get_logger(__name__)
 
class MegaPoseNode(Node):
    def __init__(self):
        super().__init__('megapose_node')
 
        self.declare_parameter('model_name', 'megapose-1.0-RGB-multi-hypothesis')
        self.declare_parameter('rgb_topic', '/zedm/zed_node/left/image_rect_color')
        self.declare_parameter('camera_info_topic', '/zedm/zed_node/left/camera_info')
        self.declare_parameter('detection_topic', '/yolo/detections')
        self.declare_parameter('pose_output_topic', '/megapose/estimation')
        self.declare_parameter('service_name', 'run_inference')
        self.declare_parameter('camera_frame_id', 'zedm_left_camera_optical_frame')
        self.declare_parameter('depth_topic', '/zedm/zed_node/depth/depth_registered')
        self.declare_parameter('resize_factor', 1)
        self.declare_parameter('cropped', True)
        
        self.resize_factor = int(self.get_parameter('resize_factor').value)
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.pose_output_topic = self.get_parameter('pose_output_topic').get_parameter_value().string_value
        self.service_name = self.get_parameter('service_name').get_parameter_value().string_value
        self.camera_frame_id = self.get_parameter('camera_frame_id').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.cropped = self.get_parameter('cropped').get_parameter_value().bool_value

        
        if self.model_name not in NAMED_MODELS:
            self.get_logger().warn(
                f"Unknown model_name='{self.model_name}', defaulting to 'megapose-1.0-RGB-multi-hypothesis'"
            )
            self.model_name = "megapose-1.0-RGB-multi-hypothesis"
        
        self.model_info = NAMED_MODELS[self.model_name]

        if not torch.cuda.is_available():
            self.get_logger().error("CUDA not available. MegaPose requires CUDA to run.")
            raise RuntimeError("CUDA is required for MegaPose inference but not available")

        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_tf = None
        self.tf_timer=self.create_timer(0.5, self.republish_tf())

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.get_logger().info("Enabled expandable segments for CUDA memory")

        self.pose_estimator = None
        self.loaded_dir_label = None
        self.orig_shape = None
        self.rgb_image = None
        self.depth_image = None
        self.camera_info_dict = None
        self.yolo_detections_list = None

        self.bridge = CvBridge()
        set_logging_level("info")
 
        self.create_subscription(Image, self.rgb_topic, self.image_cb, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, 10)
        self.create_subscription(DetectionArray, self.detection_topic, self.detection_cb, 10)
 
        self.inference_service = self.create_service(
            MegaposeInference, self.service_name, self.inference_cb)
 
        self.pose_pub = self.create_publisher(Detection3DArray, self.pose_output_topic, 10)
 
    def image_cb(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.orig_shape = cv_image.shape[:2]
        
        if self.resize_factor != 1:
            h, w = cv_image.shape[:2]
            cv_image = cv2.resize(
                cv_image,
                (int(w / self.resize_factor), int(h / self.resize_factor)),
                interpolation=cv2.INTER_AREA,
            )
        
        rgb_image = cv_image[..., ::-1]
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image)
        self.rgb_image = rgb_image
    
    def depth_cb(self, msg):
        if msg.encoding == '32FC1':
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1').astype(np.float32)
            depth = depth_raw
        # elif msg.encoding == '16UC1':
        #     depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1').astype(np.uint16)
        #     depth_m = depth_raw.astype(np.float32) / 1000.0
        else:
            self.get_logger().warn(f"Unexpected depth encoding '{msg.encoding}'; attempting passthrough.")
            try:
                depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                depth_m = depth_raw.astype(np.float32) / 1000.0
            except Exception as e:
                self.get_logger().error(f"Failed to process depth image: {e}")
                return

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth <= 0.0] = 0.0

        if self.resize_factor != 1:
            h, w = depth.shape[:2]
            new_w = int(w / self.resize_factor)
            new_h = int(h / self.resize_factor)
            depth = cv2.resize(
                depth,
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            )
        
        if not depth.flags['C_CONTIGUOUS']:
            depth = np.ascontiguousarray(depth)
        self.depth_image = depth

    def camera_info_cb(self, msg):
        self.camera_info_dict = convert_camera_info(msg, self.resize_factor)
    
    def detection_cb(self, msg):
        if self.orig_shape is None:
            self.get_logger().warn("Received detections before any image; skipping conversion.")
            return
        
        cam_res = self.orig_shape
        crop_size = np.array([1024, 1024])
        self.yolo_detections_list = convert_yolo_detections(msg, cam_res, crop_size, self.resize_factor, self.cropped)
        
    def inference_cb(self, request, response):
        requested_label = request.label if request.label else None
        result = self.process_pipeline(requested_label)
        response.success = result.success
        response.message = result.message
        
        return response
        
    def process_pipeline(self, requested_label=None):
        response = SimpleNamespace(success=False, message='')
        
        if self.rgb_image is None or self.orig_shape is None:
            response.message = "No image received yet."
            return response
        
        if self.depth_image is None and self.model_info["requires_depth"]:
            response.message = "No depth received yet."
            return response
            
        if self.camera_info_dict is None:
            response.message = "No camera_info received yet."
            return response
            
        if not self.yolo_detections_list:
            response.message = "No detections received yet."
            return response
 
        filtered_detections = self.yolo_detections_list
        if requested_label:
            filtered_detections = [
                d for d in self.yolo_detections_list if d["label"] == requested_label
            ]
            if not filtered_detections:
                response.message = f"No detections with label '{requested_label}' found."
                return response
            self.get_logger().info(f"Label found '{requested_label}'")
 
            if len(filtered_detections) > 1:
                filtered_detections = [
                    max(filtered_detections, key=lambda d: d.get("score", 0.0))
                ]
                self.get_logger().info(
                    f"Multiple detections for '{requested_label}' – using highest score one."
                )

        try:
            if requested_label:
                label_for_dir = requested_label
            else:
                label_for_dir = filtered_detections[0]["label"]
            
            example_dir = PKG_LOCAL_DATA_DIR / "custom_data" / label_for_dir
            meshes_dir = example_dir / "meshes"
            if not meshes_dir.exists():
                response.message = (
                    f"Mesh directory not found for label '{label_for_dir}'. "
                    f"Please ensure '{meshes_dir}' exists."
                )
                return response
            
            self.run_inference(filtered_detections, label_for_dir)
            response.success = True
            response.message = (
                f"Inference complete for label '{label_for_dir}'. Results published & saved."
            )
            
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            gc.collect()
            response.message = f"CUDA out of memory: {str(e)}"
            self.get_logger().error(response.message)
        except FileNotFoundError as e:
            response.message = str(e)
            self.get_logger().error(response.message)
        except Exception as e:
            response.message = f"Inference failure: {e}"
            self.get_logger().error(response.message)
 
        return response
 
    def run_inference(self, detections_list, dir_label):
        example_dir = PKG_LOCAL_DATA_DIR / "custom_data" / dir_label
        example_dir.mkdir(parents=True, exist_ok=True)
        
        save_inputs(example_dir, self.rgb_image, self.camera_info_dict, detections_list)
 
        K = np.array(self.camera_info_dict["K"], dtype=np.float32)
        observation = ObservationTensor.from_numpy(self.rgb_image, self.depth_image, K)
        
        if self.model_info["requires_depth"] and self.depth_image is not None:
            self.get_logger().info(f"Using depth image with shape: {self.depth_image.shape}")
        
        observation = observation.cuda()
 
        object_data_list = []
        for d in detections_list:
            bbox_modal = np.array(d["bbox_modal"], dtype=np.float32) if d["bbox_modal"] else None
            od = ObjectData(label=d["label"], bbox_modal=bbox_modal, TWO=None)
            object_data_list.append(od)
 
        detections = make_detections_from_object_data(object_data_list)
        detections = detections.cuda()
 
        object_dataset = make_object_dataset(example_dir)

        load_needed = self.pose_estimator is None or self.loaded_dir_label != dir_label
        if load_needed:
            if self.pose_estimator is not None:
                del self.pose_estimator
                self.pose_estimator = None
                torch.cuda.empty_cache()
                gc.collect()

            self.get_logger().info(
                f"Loading estimator '{self.model_name}' for label '{dir_label}'"
            )
            self.pose_estimator = load_named_model(self.model_name, object_dataset).cuda()
            self.loaded_dir_label = dir_label
        else:
            self.get_logger().info(f"Using cached model for '{self.loaded_dir_label}'")

        self.get_logger().info("Running inference")
        output, _ = self.pose_estimator.run_inference_pipeline(
            observation, detections=detections, **self.model_info["inference_parameters"]
        )

        del observation, detections

        save_predictions(example_dir, output)
        self.publish_pose(output)
        self.get_logger().info(
            f"Inference done. Results published and saved to {example_dir}"
        )
 
    def publish_pose(self, output):
        pose_cpu = output.poses.cpu().numpy()[0]
        label = output.infos["label"][0]
        now = self.get_clock().now().to_msg()

        quaternion = tf_transformations.quaternion_from_matrix(pose_cpu)
        translation = pose_cpu[:3, 3]
        
        self.get_logger().info(
            f"Publishing pose for '{label}': t=[{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]"
        )

        det_array = Detection3DArray()
        det_array.header.stamp = now
        det_array.header.frame_id = self.camera_frame_id

        det_obj = Detection3D()
        det_obj.id = label
    
        obj_hyp = ObjectHypothesisWithPose()
        obj_hyp.pose.pose.position.x = float(translation[0])
        obj_hyp.pose.pose.position.y = float(translation[1])
        obj_hyp.pose.pose.position.z = float(translation[2])
        obj_hyp.pose.pose.orientation.x = float(quaternion[0])
        obj_hyp.pose.pose.orientation.y = float(quaternion[1])
        obj_hyp.pose.pose.orientation.z = float(quaternion[2])
        obj_hyp.pose.pose.orientation.w = float(quaternion[3])

        det_obj.results = [obj_hyp]
        det_array.detections = [det_obj]

        self.pose_pub.publish(det_array)

        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = now
        tf_stamped.header.frame_id = self.camera_frame_id
        tf_stamped.child_frame_id = f"{label}_frame"
        tf_stamped.transform.translation.x = float(translation[0])
        tf_stamped.transform.translation.y = float(translation[1])
        tf_stamped.transform.translation.z = float(translation[2])
        tf_stamped.transform.rotation.x = float(quaternion[0])
        tf_stamped.transform.rotation.y = float(quaternion[1])
        tf_stamped.transform.rotation.z = float(quaternion[2])
        tf_stamped.transform.rotation.w = float(quaternion[3])

        self.tf_broadcaster.sendTransform(tf_stamped)
        self.latest_tf= tf_stamped
        self.get_logger().info(f"Broadcast static TF: {tf_stamped.child_frame_id}")

    def republish_tf(self):
        if not self.latest_tf:
            return
        self.latest_tf.header.stamp=self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.latest_tf)

    def destroy_node(self):
        if self.pose_estimator is not None:
            del self.pose_estimator
            self.pose_estimator = None
            torch.cuda.empty_cache()
        super().destroy_node()
 
def main(args=None):
    rclpy.init(args=args)
    node = MegaPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
if __name__ == '__main__':
    main()