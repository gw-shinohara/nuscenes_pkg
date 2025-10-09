from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """
    nuscenes_streamer と depth_enhancer ノードを同時に起動するための
    LaunchDescriptionを生成します。
    """

    # --- nuscenes_streamer の起動引数 ---
    dataroot_arg = DeclareLaunchArgument(
        'dataroot',
        default_value='/root/data/nuscenes/v1.0-mini', # <-- ご自身の環境に合わせて変更してください
        description='Path to the NuScenes dataroot.'
    )

    version_arg = DeclareLaunchArgument(
        'version',
        default_value='v1.0-mini',
        description='NuScenes version (e.g., v1.0-mini, v1.0-trainval).'
    )

    scene_names_arg = DeclareLaunchArgument(
        'scene_names',
        default_value="['scene-0061']",
        description='List of NuScenes scenes to stream.'
    )

    play_all_scenes_arg = DeclareLaunchArgument(
        'play_all_scenes',
        default_value='True',
        description='If True, plays all scenes and ignores scene_names.'
    )

    rate_arg = DeclareLaunchArgument(
        'publish_rate_hz',
        default_value='0.5', #dense depth modelの処理速度に合わせる
        description='Publishing rate in Hz. Only float values are supported.'
    )

    # --- depth_enhancer の起動引数 (必須) ---
    checkpoint_path_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value='/root/weights/nuscenes_pkgs/Depth-Anything-V2-Small/depth_anything_v2_vits.pth',
        description='Path to the local directory with the Depth Anything V2 model. THIS IS MANDATORY.'
    )

    # --- ノードの定義 ---

    # 1. NuScenes Streamer Node
    nuscenes_streamer_node = Node(
        package='nuscenes_pkg',
        executable='nuscenes_streamer',
        name='nuscenes_streamer',
        output='screen',
        parameters=[{
            'dataroot': LaunchConfiguration('dataroot'),
            'version': LaunchConfiguration('version'),
            'scene_names': LaunchConfiguration('scene_names'),
            'play_all_scenes': LaunchConfiguration('play_all_scenes'),
            'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
        }]
    )

    # 2. Depth Enhancer Node
    depth_enhancer_node = Node(
        package='nuscenes_pkg',
        executable='depth_enhancer',
        name='depth_enhancer',
        output='screen',
        parameters=[{
            'checkpoint_path': LaunchConfiguration('checkpoint_path'),
        }]
    )

    return LaunchDescription([
        # 引数の宣言
        dataroot_arg,
        version_arg,
        scene_names_arg,
        play_all_scenes_arg,
        rate_arg,
        checkpoint_path_arg,

        # 起動するノード
        nuscenes_streamer_node,
        depth_enhancer_node
    ])
