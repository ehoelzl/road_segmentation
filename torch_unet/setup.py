#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='torch_unet',
      author='Edoardo Holzl',
      version='0.1.0',
      packages=find_packages(where='.'),
      zip_safe=False,
      install_requires=[
          "tqdm",
          "tensorboard=2.0.0"
          "matplotlib==3.1.1",
          "numpy==1.17.4",
          "torch==1.3.1",
          "opencv-python==4.1.2.30",
          "scikit-image==0.16.2",
          "click==7.0",
          "albumentations==0.4.3",
          ])
