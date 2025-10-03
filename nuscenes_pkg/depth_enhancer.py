# 必要なライブラリをインポート
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import os
import sys
import traceback

import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from .depth_anything_v2.dpt import DepthAnythingV2

class DepthEnhancerNode(Node):
    """
    密な深度マップを生成するノード。
    """
    def __init__(self):
        # --- [修正点] ---
        # 1. 最初にすべてのインスタンス変数を宣言する
        self.initialization_successful = False
        self.bridge = None
        self.device = None
        self.model = None
        self.cameras = []
        self.depth_subscribers = []
        self.depth_publishers = {}

        super().__init__('depth_enhancer')
        try:
            # 3. これでNodeの機能を安全に使いつつ、宣言済みの変数に値を設定できる
            self.get_logger().info("Depth Enhancer Node (Original Author's Method) - Initializing...")

            self.declare_parameter('checkpoint_path', '')
            checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value

            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                self.get_logger().fatal(f"Mandatory parameter 'checkpoint_path' is not set or the file does not exist: {checkpoint_path}")
                raise RuntimeError("Invalid checkpoint path.")

            # CV Bridgeの初期化
            self.bridge = CvBridge()

            # GPUが利用可能かチェック
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.get_logger().info(f"Using device: {self.device}")

            self.get_logger().info(f"Loading model checkpoint from: {checkpoint_path}")
            self.model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info("Model loaded successfully.")

            # カメラのリスト
            self.cameras = [
                'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ]

            qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

            for cam in self.cameras:
                cam_lower = cam.lower()
                pub_topic = f'/nuscenes/{cam_lower}/dense_depth'
                self.depth_publishers[cam] = self.create_publisher(Image, pub_topic, 10)
                image_sub_topic = f'/nuscenes/{cam_lower}/image_raw'
                sparse_depth_sub_topic = f'/nuscenes/{cam_lower}/depth'
                image_sub = message_filters.Subscriber(self, Image, image_sub_topic, qos_profile=qos_profile)
                sparse_depth_sub = message_filters.Subscriber(self, Image, sparse_depth_sub_topic, qos_profile=qos_profile)
                ts = message_filters.ApproximateTimeSynchronizer([image_sub, sparse_depth_sub], queue_size=10, slop=0.2)
                ts.registerCallback(lambda img, depth, cam_name=cam: self.callback(img, depth, cam_name))
                self.depth_subscribers.append((image_sub, sparse_depth_sub, ts))

            self.get_logger().info("Initialization complete. Waiting for data...")
            self.initialization_successful = True

        except Exception as e:
            self.get_logger().fatal(f"An exception occurred during node initialization: {e}")
            self.get_logger().fatal(traceback.format_exc()) # スタックトレース全体を出力
            self.get_logger().fatal("Shutting down due to initialization failure.")
            # self.context.try_shutdown() は不要

    def callback(self, image_msg, sparse_depth_msg, cam_name):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            sparse_depth = self.bridge.imgmsg_to_cv2(sparse_depth_msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f"Failed to convert messages for {cam_name}: {e}")
            return

        with torch.no_grad():
            dense_relative_depth = self.model.infer_image(bgr_image)

        valid_mask = sparse_depth > 0.01
        lidar_values = sparse_depth[valid_mask]
        model_values = dense_relative_depth[valid_mask]

        if lidar_values.size > 10:
            model_values_safe = np.where(model_values > 1e-6, model_values, 1e-6)
            scales = lidar_values / model_values_safe
            scale_factor = np.median(scales)
            dense_aligned_depth = dense_relative_depth * scale_factor
            dense_aligned_depth[valid_mask] = lidar_values
        else:
            self.get_logger().warn(f"Not enough LiDAR points for {cam_name}. Using relative depth.")
            dense_aligned_depth = dense_relative_depth

        try:
            final_depth_msg = self.bridge.cv2_to_imgmsg(dense_aligned_depth.astype(np.float32), "32FC1")
            final_depth_msg.header = image_msg.header
            self.depth_publishers[cam_name].publish(final_depth_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish dense depth for {cam_name}: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthEnhancerNode()

    if node.initialization_successful:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            # rclpy.spin()が終了した後にクリーンアップ
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
