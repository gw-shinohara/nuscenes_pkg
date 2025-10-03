# streaming.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """
    nuscenes_streamerノードを起動するためのLaunchDescriptionを生成します。
    """

    dataroot_arg = DeclareLaunchArgument(
        'dataroot',
        default_value='/root/data/nuscenes/v1.0-mini',
        description='Path to the NuScenes dataroot.'
    )

    version_arg = DeclareLaunchArgument(
        'version',
        default_value='v1.0-mini',
        description='NuScenes version (e.g., v1.0-mini, v1.0-trainval).'
    )

    # scene_nameをscene_namesに変更し、デフォルト値をリスト形式に
    scene_names_arg = DeclareLaunchArgument(
        'scene_names',
        default_value="['scene-0061', 'scene-0103']",
        description='List of NuScenes scenes to stream.'
    )

    # play_all_scenesパラメータを追加
    play_all_scenes_arg = DeclareLaunchArgument(
        'play_all_scenes',
        default_value='False',
        description='If True, plays all scenes and ignores scene_names.'
    )

    rate_arg = DeclareLaunchArgument(
        'publish_rate_hz',
        default_value='10.0',
        description='Publishing rate in Hz.'
    )

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

    return LaunchDescription([
        dataroot_arg,
        version_arg,
        scene_names_arg,
        play_all_scenes_arg,
        rate_arg,
        nuscenes_streamer_node
    ])
