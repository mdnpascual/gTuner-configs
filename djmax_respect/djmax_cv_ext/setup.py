"""
Build: python setup.py build_ext --inplace

Prerequisites:
1. Visual Studio 2022 (you have this)
2. OpenCV C++ SDK - download from https://opencv.org/releases/
   Extract to C:\opencv so you have C:\opencv\build\include and C:\opencv\build\x64\vc16\lib
   OR set OPENCV_DIR env var to your opencv\build path
3. numpy (pip install numpy)

After build, copy the resulting djmax_cv.cp311-win_amd64.pyd to this directory.
"""
from setuptools import setup, Extension
import numpy as np
import os
import glob

# Find OpenCV C++ SDK
opencv_dir = os.environ.get("OPENCV_DIR", r"C:\opencv\opencv\build")
opencv_include = os.path.join(opencv_dir, "include")
opencv_lib_dir = os.path.join(opencv_dir, "x64", "vc16", "lib")

# Fallback: try vc15, vc17
if not os.path.isdir(opencv_lib_dir):
    for vc in ["vc17", "vc16", "vc15", "vc14"]:
        candidate = os.path.join(opencv_dir, "x64", vc, "lib")
        if os.path.isdir(candidate):
            opencv_lib_dir = candidate
            break

# Auto-detect opencv_world lib
opencv_libs = []
if os.path.isdir(opencv_lib_dir):
    for f in os.listdir(opencv_lib_dir):
        if f.startswith("opencv_world") and f.endswith(".lib") and "d" not in f[-6:]:
            opencv_libs.append(f[:-4])
            break

if not opencv_libs:
    print(f"WARNING: Could not find opencv_world*.lib in {opencv_lib_dir}")
    print("Set OPENCV_DIR to your opencv/build directory")
    opencv_libs = ["opencv_world4100"]

print(f"OpenCV include: {opencv_include}")
print(f"OpenCV lib dir: {opencv_lib_dir}")
print(f"OpenCV libs: {opencv_libs}")

ext = Extension(
    "djmax_cv",
    sources=["djmax_cv.cpp"],
    include_dirs=[np.get_include(), opencv_include],
    library_dirs=[opencv_lib_dir],
    libraries=opencv_libs,
    language="c++",
    extra_compile_args=["/std:c++17", "/O2", "/EHsc"],
)

setup(
    name="djmax_cv",
    version="1.0",
    ext_modules=[ext],
)
