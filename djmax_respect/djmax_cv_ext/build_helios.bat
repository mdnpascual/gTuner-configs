@echo off
REM Build DJMaxCV.dll (release) and DJMax_Debug.dll (with overlay) from same source

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set OPENCV_INC=C:\opencv\opencv\build\include
set HELIOS_SDK=C:\Users\MDuh\Desktop\Projects\Helios\versions\1.0.7.9\lib\cv_cpp
set OUT_DIR=%HELIOS_SDK%

echo === Building DJMaxCV.dll (release, no overlay) ===
cl /std:c++17 /O2 /EHsc /MD /LD ^
    /I"%OPENCV_INC%" /I"%HELIOS_SDK%" ^
    DJMaxCV.cpp ^
    /link /LIBPATH:"%HELIOS_SDK%" opencv_core4.lib opencv_imgproc4.lib opencv_imgcodecs4.lib ^
    /OUT:"%OUT_DIR%\DJMaxCV.dll"
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo === Building DJMax_Debug.dll (with debug overlay) ===
cl /std:c++17 /O2 /EHsc /MD /LD /DDJMAX_DEBUG ^
    /I"%OPENCV_INC%" /I"%HELIOS_SDK%" ^
    DJMaxCV.cpp ^
    /FoDJMaxCV_dbg.obj ^
    /link /LIBPATH:"%HELIOS_SDK%" opencv_core4.lib opencv_imgproc4.lib opencv_imgcodecs4.lib ^
    /OUT:"%OUT_DIR%\DJMax_Debug.dll"
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo SUCCESS: Both DLLs built in %OUT_DIR%
echo   DJMaxCV.dll     - production (no overlay)
echo   DJMax_Debug.dll - debug (FPS + lane info + timing)
goto :end

:fail
echo BUILD FAILED
:end
pause
