from setuptools import find_packages
from distutils.core import setup

setup(
    name='dlabsim',
    version='1.2.7',
    author='Yufei Jia',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='jyf23@mails.tsinghua.edu.cn',
    description='Mujoco environments for Airbot play',
    install_requires=['mujoco', 'opencv-python']
)
