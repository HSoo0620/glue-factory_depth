"""
Alternative build via pip: pip install -e .
Works when pkg-config can find PCL.
"""

import subprocess
import os
from setuptools import setup, Extension
import pybind11


def get_pcl_flags():
    """Get PCL compile/link flags via pkg-config."""
    components = [
        "pcl_common", "pcl_features", "pcl_filters",
        "pcl_kdtree", "pcl_search", "pcl_octree",
    ]

    # Try PCL versions 1.15 -> 1.10
    for ver in ["1.15", "1.14", "1.13", "1.12", "1.11", "1.10"]:
        pkgs = [f"{c}-{ver}" for c in components]
        try:
            subprocess.check_output(["pkg-config", "--exists"] + pkgs,
                                    stderr=subprocess.DEVNULL)
            cflags = subprocess.check_output(
                ["pkg-config", "--cflags"] + pkgs
            ).decode().strip().split()
            libs = subprocess.check_output(
                ["pkg-config", "--libs"] + pkgs
            ).decode().strip().split()
            print(f"Found PCL {ver}")
            return cflags, libs
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    # Fallback: common Linux paths
    print("pkg-config not found for PCL, using fallback paths")
    cflags = [
        "-I/usr/include/pcl-1.14",
        "-I/usr/include/pcl-1.13",
        "-I/usr/include/pcl-1.12",
        "-I/usr/include/eigen3",
    ]
    libs = [
        "-lpcl_common", "-lpcl_features", "-lpcl_filters",
        "-lpcl_kdtree", "-lpcl_search", "-lpcl_octree",
        "-lflann_cpp",
    ]
    return cflags, libs


pcl_cflags, pcl_libs = get_pcl_flags()

# Separate -I (include dirs) and -L/-l (library dirs/libs)
include_dirs = [pybind11.get_include()]
extra_compile_args = ["-std=c++17", "-O2", "-fopenmp",
                      "-DNOMINMAX", "-DEIGEN_MAX_ALIGN_BYTES=32"]
extra_link_args = ["-fopenmp"]

for flag in pcl_cflags:
    if flag.startswith("-I"):
        include_dirs.append(flag[2:])
    else:
        extra_compile_args.append(flag)

for flag in pcl_libs:
    extra_link_args.append(flag)

setup(
    name="shot_module",
    version="1.0",
    ext_modules=[
        Extension(
            "shot_module",
            sources=["shot_module.cpp"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++",
        )
    ],
)
