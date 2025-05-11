from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'astro_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # URDF
        (f'share/{package_name}/urdf', ['urdf/rover.urdf']),

        # ament index registration
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),

        # package manifest
        (f'share/{package_name}', ['package.xml']),

        # all launch scripts
        (f'share/{package_name}/launch', glob('launch/*.py')),

        # all param files
        (f'share/{package_name}/params', glob('params/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'pyserial',
    ],
    zip_safe=True,
    maintainer='astrobotics',
    maintainer_email='astrobotics@todo.todo',
    description='Launch nodes (navigation, localization, opc, routine)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_command_node = astro_launch.motor_command_node:main',
            'beacon_localization = astro_launch.beacon_localization:main',
            'localization_testing = astro_launch.localization_testing:main',
            'movavg_beac_loc = astro_launch.movavg_beac_loc:main',
            'ewma_beac_loc = astro_launch.ewma_beac_loc:main',
        ],
    },
)
