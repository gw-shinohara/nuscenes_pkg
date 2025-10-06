from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nuscenes_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gw-shinohara',
    maintainer_email='shinohara@globalwalkers.co.jp',
    description='A ROS node for streaming nuScenes image and depth data.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nuscenes_streamer = nuscenes_pkg.nuscenes_streamer:main',
            'depth_enhancer = nuscenes_pkg.depth_enhancer:main',
        ],
    },
)
