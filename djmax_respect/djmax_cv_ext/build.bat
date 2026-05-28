@echo off
REM Build djmax_cv.pyd extension
REM Prerequisites: 
REM   1. Visual Studio 2022 (with C++ workload)
REM   2. OpenCV C++ SDK extracted to C:\opencv (or set OPENCV_DIR)
REM   3. Python 3.8+ with numpy

REM Set up VS environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Build
cd /d "%~dp0"
python setup.py build_ext --inplace

REM Copy .pyd to parent directory for easy import
for %%f in (djmax_cv*.pyd) do copy "%%f" "..\djmax_cv_ext\%%f"

echo.
echo Done! If successful, djmax_cv*.pyd should be in this directory.
pause
