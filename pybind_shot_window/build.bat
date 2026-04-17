@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

cd /d "%~dp0"

set PYTHON_INC=C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0\Include
set PYTHON_LIBS=C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0\libs

for /f "delims=" %%i in ('python -c "import pybind11; print(pybind11.get_include())"') do set PYBIND_INC=%%i

set PCL=C:\Program Files\PCL 1.15.1
set PCL_INC=%PCL%\include\pcl-1.15
set EIGEN_INC=%PCL%\3rdParty\Eigen3\include\eigen3
set BOOST_INC=%PCL%\3rdParty\Boost\include\boost-1_87
set FLANN_INC=%PCL%\3rdParty\FLANN\include
set PCL_LIB=%PCL%\lib
set BOOST_LIB=%PCL%\3rdParty\Boost\lib
set FLANN_LIB=%PCL%\3rdParty\FLANN\lib

echo.
echo ========== Building shot_module.pyd ==========
echo.

cl.exe /O2 /EHsc /std:c++17 /openmp /MD /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /DEIGEN_MAX_ALIGN_BYTES=32 ^
    /I"%PYTHON_INC%" /I"%PYBIND_INC%" ^
    /I"%PCL_INC%" /I"%EIGEN_INC%" /I"%BOOST_INC%" /I"%FLANN_INC%" ^
    /LD shot_module.cpp ^
    /Fe:shot_module.cp313-win_amd64.pyd ^
    /link ^
    /LIBPATH:"%PYTHON_LIBS%" /LIBPATH:"%PCL_LIB%" /LIBPATH:"%BOOST_LIB%" /LIBPATH:"%FLANN_LIB%" ^
    python313.lib pcl_common.lib pcl_features.lib pcl_filters.lib pcl_kdtree.lib pcl_search.lib pcl_octree.lib flann_cpp.lib

if %ERRORLEVEL% == 0 (
    echo.
    echo ========== BUILD SUCCESS ==========
    dir shot_module.cp313-win_amd64.pyd
) else (
    echo.
    echo ========== BUILD FAILED ==========
)

pause
