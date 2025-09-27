from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension,CUDAExtension
import torch
import math
from torch.utils.cpp_extension import load
import os, time


setup(
    name="custom_extension",
    ext_modules=[
        CUDAExtension(
            'custom_extension',
            ['custom_extension.cpp', 'custom_cuda.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)



    
