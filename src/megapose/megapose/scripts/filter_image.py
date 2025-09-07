#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from yolo_msgs.msg import DetectionArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber


# ---------- Tunables ----------
_INT_GAIN   = 0.8
_INT_BIAS   = 5
_INT_GAMMA  = 0.8
_INT_SAT    = 1.3
_INT_SHARP  = 0.7
_INT_EXPOSURE = 0.5  # New tunable for exposure (1.0 = no change, >1.0 = brighter, <1.0 = darker)
_USE_CLAHE  = True
_CLAHE_CLIP = 2.0
_CLAHE_TILE = (8, 8)
# ------------------------------


def _gamma_correct(img, gamma: float):
    if gamma is None or gamma == 1.0:
        return img
    inv = 1.0 / gamma
    table = (np.arange(256, dtype=np.float32) / 255.0) ** inv
    table = np.clip(table * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


def _clahe_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILE)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _saturation_boost(img_bgr, scale: float):
    if scale is None or scale == 1.0:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    hsv2 = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)


def _exposure_adjust(img_bgr, scale: float):
    if scale is None or scale == 1.0:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    hsv2 = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)


def _unsharp(img_bgr, amount: float, blur_ksize=0, sigma=1.0):
    if amount is None or amount <= 0.0:
        return img_bgr
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # sharpened = original*(1+amt) + blurred*(-amt)
    return cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)


def intensify_roi(roi_bgr: np.ndarray) -> np.ndarray:
    out = roi_bgr
    out = _exposure_adjust(out, _INT_EXPOSURE)  # Apply exposure adjustment first
    out = _gamma_correct(out, _INT_GAMMA)
    if _USE_CLAHE:
        out = _clahe_lab(out)
    out = _saturation_boost(out, _INT_SAT)
    out = _unsharp(out, _INT_SHARP, sigma=1.0)
    out = cv2.convertScaleAbs(out, alpha=_INT_GAIN, beta=_INT_BIAS)
    return out


class IntensifyBBoxes(Node):
    def __init__(self):
        super().__init__('intensify_bboxes')

        self.declare_parameter('image_topic', '/zedm/zed_node/left/image_rect_color_cropped')
        self.declare_parameter('det_topic',  '/yolo/detections')
        self.declare_parameter('output_topic', '/image_intense')

        img_topic     = self.get_parameter('image_topic').get_parameter_value().string_value
        det_topic     = self.get_parameter('det_topic').get_parameter_value().string_value
        output_topic  = self.get_parameter('output_topic').get_parameter_value().string_value

        sub_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE
        )
        pub_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE
        )

        self.pub_img = self.create_publisher(Image, output_topic, pub_qos)

        self.bridge = CvBridge()
        sub_img = Subscriber(self, Image, img_topic, qos_profile=sub_qos)
        sub_det = Subscriber(self, DetectionArray, det_topic, qos_profile=sub_qos)

        self.sync = ApproximateTimeSynchronizer(
            [sub_img, sub_det],
            queue_size=10,
            slop=0.05,
            allow_headerless=False
        )
        self.sync.registerCallback(self.callback)

        self.get_logger().info(
            f'intensify_bboxes started.\n'
            f'  Subscribing image: {img_topic}\n'
            f'  Subscribing dets : {det_topic}\n'
            f'  Publishing       : {output_topic} (RELIABLE)\n'
            f'  Tunables: gain={_INT_GAIN} bias={_INT_BIAS} gamma={_INT_GAMMA} '
            f'sat={_INT_SAT} sharp={_INT_SHARP} exposure={_INT_EXPOSURE} clahe={_USE_CLAHE}'
        )

    def callback(self, img_msg: Image, det_msg: DetectionArray):
        # Force BGR8 for simplicity; adjust if you need passthrough
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        h, w = cv_img.shape[:2]

        for det in det_msg.detections:
            cx = int(round(det.bbox.center.position.x))
            cy = int(round(det.bbox.center.position.y))
            bw = int(round(det.bbox.size.x))
            bh = int(round(det.bbox.size.y))
            if bw <= 0 or bh <= 0:
                continue

            x1 = max(cx - bw // 2, 0)
            y1 = max(cy - bh // 2, 0)
            x2 = min(cx + bw // 2, w - 1)
            y2 = min(cy + bh // 2, h - 1)
            if x1 >= x2 or y1 >= y2:
                continue

            roi = cv_img[y1:y2, x1:x2]
            cv_img[y1:y2, x1:x2] = intensify_roi(roi)

        out_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        out_msg.header = img_msg.header
        self.pub_img.publish(out_msg)


def main():
    rclpy.init()
    node = IntensifyBBoxes()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()