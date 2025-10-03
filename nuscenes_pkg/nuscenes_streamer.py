import rclpy
from rclpy.node import Node
from rclpy.time import Time

import os
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
from pyquaternion import Quaternion
import gc # Import garbage collector interface

# ROS 2 メッセージ
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion as QuaternionMsg, Twist, Vector3
from std_msgs.msg import Header

# NuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

class NuScenesRos2Streamer(Node):
    """
    NuScenesデータセットをROS 2トピックとしてストリーミングするノード
    """

    def __init__(self):
        super().__init__('nuscenes_streamer')

        # パラメータ宣言
        self.declare_parameter('dataroot', '/path/to/your/nuscenes/data')
        self.declare_parameter('version', 'v1.0-mini')
        self.declare_parameter('scene_names', ['scene-0061']) # 再生したいシーン名のリスト
        self.declare_parameter('play_all_scenes', False) # 全シーンを再生するかどうかのフラグ
        self.declare_parameter('lidar_sensor', 'LIDAR_TOP')
        self.declare_parameter('publish_rate_hz', 10.0) # デフォルトのレート

        # パラメータ取得
        self.dataroot = self.get_parameter('dataroot').get_parameter_value().string_value
        self.version = self.get_parameter('version').get_parameter_value().string_value
        self.scene_names = self.get_parameter('scene_names').get_parameter_value().string_array_value
        self.play_all_scenes = self.get_parameter('play_all_scenes').get_parameter_value().bool_value
        self.lidar_sensor = self.get_parameter('lidar_sensor').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self.get_logger().info(f"NuScenes dataroot: {self.dataroot}")
        self.get_logger().info(f"NuScenes version: {self.version}")

        # NuScenes SDKの初期化
        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=True)

        # カメラのリスト
        self.cameras = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        # Publisherの作成
        self.cv_bridge = CvBridge()
        self.image_publishers = {}
        self.depth_publishers = {}
        for cam in self.cameras:
            # カラー画像用
            topic_name = f'/nuscenes/{cam.lower()}/image_raw'
            self.image_publishers[cam] = self.create_publisher(Image, topic_name, 10)
            self.get_logger().info(f"Publishing {cam} images to {topic_name}")

            # 深度画像用
            depth_topic_name = f'/nuscenes/{cam.lower()}/depth'
            self.depth_publishers[cam] = self.create_publisher(Image, depth_topic_name, 10)
            self.get_logger().info(f"Publishing {cam} depth to {depth_topic_name}")


        # オドメトリ用Publisher
        self.odom_publisher = self.create_publisher(Odometry, '/nuscenes/odom', 10)
        self.get_logger().info("Publishing odometry to /nuscenes/odom")

        # 再生するシーンのリストを取得
        self.scenes_to_play = []
        if self.play_all_scenes:
            self.scenes_to_play = self.nusc.scene
            self.get_logger().info(f"Preparing to play all {len(self.scenes_to_play)} scenes.")
        else:
            for name in self.scene_names:
                scene = self._get_scene_by_name(name)
                if scene:
                    self.scenes_to_play.append(scene)
                else:
                    self.get_logger().warn(f"Scene '{name}' not found and will be skipped.")

        if not self.scenes_to_play:
            self.get_logger().error("No valid scenes found to play. Shutting down.")
            rclpy.shutdown()
            return

        # 再生状態の初期化
        self.current_scene_index = 0
        self.current_scene = self.scenes_to_play[self.current_scene_index]
        self.current_sample_token = self.current_scene['first_sample_token']
        self.get_logger().info(f"Starting with scene '{self.current_scene['name']}' ({self.current_scene_index + 1}/{len(self.scenes_to_play)}).")

        # メインループのタイマーを作成
        self.timer = self.create_timer(1.0 / self.publish_rate, self.stream_data)
        self.last_timestamp = None


    def _get_scene_by_name(self, scene_name):
        """指定された名前のシーンを取得"""
        for scene in self.nusc.scene:
            if scene['name'] == scene_name:
                return scene
        return None

    def _go_to_next_scene(self):
        """プレイリストの次のシーンに切り替える"""
        self.get_logger().info(f"Finished scene '{self.current_scene['name']}'.")
        self.current_scene_index += 1

        if self.current_scene_index < len(self.scenes_to_play):
            # 次のシーンがある場合
            self.current_scene = self.scenes_to_play[self.current_scene_index]
            self.current_sample_token = self.current_scene['first_sample_token']
            self.get_logger().info(f"Switching to next scene '{self.current_scene['name']}' ({self.current_scene_index + 1}/{len(self.scenes_to_play)}).")
        else:
            # 全てのシーンの再生が完了した場合
            self.get_logger().info("All scenes have been played. Stopping publisher.")
            self.current_sample_token = ''  # ループを終了させる
            self.timer.cancel()

    def stream_data(self):
        """
        メインのデータストリーミング関数
        """
        if not self.current_sample_token:
            # これ以上再生するサンプルがない場合（全てのシーンが終了した）
            return

        sample = self.nusc.get('sample', self.current_sample_token)
        timestamp_ns = sample['timestamp'] * 1e3 # マイクロ秒からナノ秒へ
        ros_timestamp = Time(seconds=int(timestamp_ns // 1e9), nanoseconds=int(timestamp_ns % 1e9)).to_msg()

        # publish odom, images, and depth
        self.publish_odometry(sample, ros_timestamp)
        self.publish_images(sample, ros_timestamp)
        self.publish_depth_images(sample, ros_timestamp)


        # 次のサンプルへ
        next_token = sample['next']
        if next_token:
            self.current_sample_token = next_token
        else:
            # 現在のシーンの終端に到達したので、次のシーンへ
            self._go_to_next_scene()

        # === MEMORY LEAK FIX START ===
        # Periodically run garbage collection to free up memory.
        gc.collect()
        # === MEMORY LEAK FIX END ===


    def publish_odometry(self, sample, timestamp):
        """
        Ego Poseからオドメトリをパブリッシュする
        """
        lidar_data = self.nusc.get('sample_data', sample['data'][self.lidar_sensor])
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Pose
        translation = ego_pose['translation']
        rotation_q = ego_pose['rotation']
        odom_msg.pose.pose.position = Point(x=translation[0], y=translation[1], z=translation[2])
        odom_msg.pose.pose.orientation = QuaternionMsg(x=rotation_q[1], y=rotation_q[2], z=rotation_q[3], w=rotation_q[0])

        # Twist (NuScenesの基本データセットには速度情報がないため、0で埋める)
        odom_msg.twist.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
        odom_msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)

        self.odom_publisher.publish(odom_msg)

    def publish_images(self, sample, timestamp):
        """
        各カメラの画像をパブリッシュする
        """
        for cam in self.cameras:
            cam_data = self.nusc.get('sample_data', sample['data'][cam])
            img_path = self.nusc.get_sample_data_path(cam_data['token'])

            if not os.path.exists(img_path):
                self.get_logger().warn(f"Image file not found: {img_path}")
                continue

            img = cv2.imread(img_path)
            img_msg = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")
            img_msg.header.stamp = timestamp
            img_msg.header.frame_id = cam.lower()

            self.image_publishers[cam].publish(img_msg)

            # === MEMORY LEAK FIX START ===
            # Explicitly delete large objects after use to help the garbage collector.
            del img
            del img_msg
            # === MEMORY LEAK FIX END ===

    def publish_depth_images(self, sample, timestamp):
        """
        LIDAR点群から深度画像を生成してパブリッシュする
        """
        # LIDARデータを取得
        lidar_token = sample['data'][self.lidar_sensor]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        pc_path = self.nusc.get_sample_data_path(lidar_token)
        pc = LidarPointCloud.from_file(pc_path)

        # LIDARセンサーの情報を取得
        lidar_cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        for cam in self.cameras:
            cam_data = self.nusc.get('sample_data', sample['data'][cam])
            cam_cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

            # === ERROR FIX START ===
            # Create a new LidarPointCloud object with a copy of the points
            pc_copy = LidarPointCloud(pc.points.copy())
            # === ERROR FIX END ===

            # 点群をLIDAR座標系からEgo Vehicle座標系へ変換
            pc_copy.rotate(Quaternion(lidar_cs_record['rotation']).rotation_matrix)
            pc_copy.translate(np.array(lidar_cs_record['translation']))

            # 点群をEgo Vehicle座標系からカメラ座標系へ変換
            pc_copy.translate(-np.array(cam_cs_record['translation']))
            pc_copy.rotate(Quaternion(cam_cs_record['rotation']).inverse.rotation_matrix)

            # 画像平面への投影
            depths = pc_copy.points[2, :]
            points_cam = view_points(pc_copy.points[:3, :], np.array(cam_cs_record['camera_intrinsic']), normalize=True)

            # 画像の範囲内にあり、かつ前方にある点のみをフィルタリング
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > 0)
            mask = np.logical_and(mask, points_cam[0, :] > 0)
            mask = np.logical_and(mask, points_cam[0, :] < cam_data['width'])
            mask = np.logical_and(mask, points_cam[1, :] > 0)
            mask = np.logical_and(mask, points_cam[1, :] < cam_data['height'])

            points_cam = points_cam[:, mask]
            depths = depths[mask]

            # 深度画像を生成 (CV_32FC1)
            depth_img = np.zeros((cam_data['height'], cam_data['width']), dtype=np.float32)
            for i in range(points_cam.shape[1]):
                x, y = int(points_cam[0, i]), int(points_cam[1, i])
                # To prevent overwriting closer points with farther ones, check existing depth
                if depth_img[y, x] == 0 or depths[i] < depth_img[y, x]:
                    depth_img[y, x] = depths[i]

            # ROSメッセージに変換してパブリッシュ
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_img, "32FC1")
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = cam.lower()
            self.depth_publishers[cam].publish(depth_msg)

            # === MEMORY LEAK FIX START ===
            # Explicitly delete large objects created within the loop.
            del pc_copy
            del depth_img
            del depth_msg
            # === MEMORY LEAK FIX END ===

        # === MEMORY LEAK FIX START ===
        # Delete the main point cloud object after it's been used for all cameras.
        del pc
        # === MEMORY LEAK FIX END ===


def main(args=None):
    rclpy.init(args=args)
    node = NuScenesRos2Streamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
