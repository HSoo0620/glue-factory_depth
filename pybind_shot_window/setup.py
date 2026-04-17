from setuptools import setup, Extension
import pybind11

setup(
    name="shot_module",
    version="1.0",
    ext_modules=[
        Extension(
            "shot_module",
            sources=["shot_module.cpp"],
            include_dirs=[
                pybind11.get_include(),
                r"C:\Program Files\PCL 1.15.1\include\pcl-1.15",
                r"C:\Program Files\PCL 1.15.1\3rdParty\Eigen3\include\eigen3",
                r"C:\Program Files\PCL 1.15.1\3rdParty\Boost\include\boost-1_87",
                r"C:\Program Files\PCL 1.15.1\3rdParty\FLANN\include",
            ],
            library_dirs=[
                r"C:\Program Files\PCL 1.15.1\lib",
                r"C:\Program Files\PCL 1.15.1\3rdParty\Boost\lib",
                r"C:\Program Files\PCL 1.15.1\3rdParty\FLANN\lib",
            ],
            libraries=[
                "pcl_common",
                "pcl_features",
                "pcl_filters",
                "pcl_kdtree",
                "pcl_search",
                "pcl_octree",
                "flann_cpp",
            ],
            extra_compile_args=[
                "/std:c++17",
                "/openmp",
                "/O2",
                "/DNOMINMAX",
                "/D_CRT_SECURE_NO_WARNINGS",
                "/DEIGEN_MAX_ALIGN_BYTES=32",
                "/EHsc",
            ],
            language="c++",
        )
    ],
)
