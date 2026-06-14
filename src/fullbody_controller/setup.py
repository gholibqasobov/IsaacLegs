from glob import glob

from setuptools import find_packages, setup

package_name = 'fullbody_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ship the trained policy + IO descriptors so the node can locate them at runtime
        # via ament_index_python regardless of where ``ros2 run`` is invoked from.
        (f'share/{package_name}/policy/g1_locomotion',
            glob('policy/g1_locomotion/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qasob',
    maintainer_email='qasobovgholib@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
