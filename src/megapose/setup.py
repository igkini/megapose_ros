from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'megapose'

def add_data_tree(data_files_list, start_dir, rel_base):
    for root, _, files in os.walk(start_dir):
        if not files:
            continue
        rel = os.path.relpath(root, start=rel_base)
        dest = os.path.join('share', package_name, rel)
        srcs = [os.path.join(root, f) for f in files if os.path.isfile(os.path.join(root, f))]
        if srcs:
            data_files_list.append((dest, srcs))

launch_files = glob('launch/*.launch.py')

data_files = [
    ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
    (f'share/{package_name}', ['package.xml']),
    (os.path.join('share', package_name, 'launch'), launch_files),
]

add_data_tree(data_files, start_dir=os.path.join('megapose', 'local_data', 'custom_data'),
              rel_base='megapose')
add_data_tree(data_files, start_dir=os.path.join('megapose', 'local_data', 'megapose-models'),
              rel_base='megapose')

data_files.append(
    (os.path.join('share', package_name, 'data'),
     glob(os.path.join('megapose', 'data', '*')))
)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='giannis',
    maintainer_email='giannis@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'run_inference_ros = megapose.scripts.run_inference_ros:main',
            'filter_image = megapose.scripts.filter_image:main',
        ],
    },
)
