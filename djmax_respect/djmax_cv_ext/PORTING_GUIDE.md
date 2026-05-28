# Porting GTuner IV Python CV to Helios 2 C++ CV

## Overview

This document describes how the DJMax Respect auto-play computer vision script was ported from a GTuner IV Python script (`pc_auto6B.py`) to a native C++ DLL (`DJMaxCV.dll`) for Helios 2's CvCpp plugin, achieving **4.7ms processing time** (vs ~28ms in Python) at **120 FPS** (vs 33 FPS capped by GTuner IV).

## Architecture Comparison

### GTuner IV (Original)

```
Screen Capture (BitBlt, ~15ms)
    → Gpython.exe (embedded Python 3.8)
        → GCVWorker.process(frame)
            → 8 Python threads (GIL contention)
            → cv2.matchTemplate per lane
        → returns (frame, bytearray)
    → Gtuner IV reads bytearray
    → USB → Titan Two → gcv_read() in GPC
```

- **Bottleneck**: GTuner IV's capture loop caps at ~33 FPS
- **Detection time**: ~28ms (Python threading overhead + GIL)
- **Communication**: `process()` returns a `bytearray`; GTuner reads it and forwards to GPC via `gcv_read(offset, &var)`

### Helios 2 (Ported)

```
Screen Capture (DXGI Desktop Duplication, ~1ms)
    → CVCppWrapper.exe loads DJMaxCV.dll
        → CCVWorkerBase::process(frame)
            → Crop 1920x1080 → resize to 1003x158
            → 8 native C++ threads (no GIL)
            → cv::matchTemplate per lane
        → calls send_gcvdata(data)
        → returns {frame, vector<uint8_t>}
    → Helios forwards to GPC3 script
```

- **Capture**: DXGI duplication at monitor refresh rate
- **Detection time**: ~4.7ms (native C++ threads, zero interpreter overhead)
- **Communication**: `send_gcvdata()` pushes data to GPC; also returned from `process()`

## Key Differences

| Aspect | GTuner IV Python | Helios 2 C++ |
|--------|-----------------|--------------|
| Entry point | `class GCVWorker` with `process(self, frame)` | `class CCVWorkerBase` with `process(cv::Mat& frame)` |
| Return type | `(frame, bytearray)` | `std::pair<cv::Mat, std::vector<uint8_t>>` |
| Data to GPC | Implicit via returned bytearray | Explicit `send_gcvdata(data)` call |
| Frame input | Pre-cropped by GTuner config | Full screen capture; you crop yourself |
| Threading | Python threads (GIL-limited) | Native `std::thread` (true parallelism) |
| Template loading | `cv2.imdecode(np.fromfile(...))` | `cv::imread(path)` |
| DLL exports | N/A (Python class) | `createWorker()`, `destroyWorker()`, `getVersion()` |

## Porting Steps

### 1. Understand the Python Detection Logic

The original script:
- Loads 8 template images (white note, color note, hold start/mid/end variants)
- For each frame, crops to 8 lane regions by X coordinates
- Runs `cv2.matchTemplate` with `TM_CCOEFF_NORMED` (method 5)
- If match > 0.75 threshold, marks that lane as active
- Side lanes use color mean detection instead of template matching
- Returns a 10-byte array: `0x01` = note detected, `0x00` = no note

### 2. Map to Helios C++ SDK

Created `DJMaxCV.cpp`:
- Inherits from `CCVWorkerBase` (defined in `GCVCppSDK.h`)
- `createWorker()` export instantiates the class
- Constructor loads templates via `cv::imread`
- `process()` crops the frame, runs detection, calls `send_gcvdata()`

### 3. Handle Frame Resolution Difference

GTuner IV pre-cropped the frame to 1003x158 based on Video Input Configuration (`2697:0:1003:158` at 4K). Helios sends the full 1920x1080 capture, so the DLL must:
1. Crop to the game region (scaled from 4K to 1080p: `1348, 0, 502, 79`)
2. Resize to 1003x158 to match template dimensions

### 4. Build Against Helios's OpenCV

Helios ships split OpenCV 4 modules (`opencv_core4.dll`, `opencv_imgproc4.dll`, `opencv_imgcodecs4.dll`) without import libraries. To link:
1. Generated `.def` files from DLL exports using `dumpbin /exports`
2. Created `.lib` import libraries using `lib /def:file.def /machine:x64`
3. Linked against these at build time

### 5. Build Command

```bat
cl /std:c++17 /O2 /EHsc /MD /LD ^
    /I"C:\opencv\opencv\build\include" ^
    /I"<helios>\lib\cv_cpp" ^
    DJMaxCV.cpp ^
    /link /LIBPATH:"<helios>\lib\cv_cpp" ^
    opencv_core4.lib opencv_imgproc4.lib opencv_imgcodecs4.lib ^
    /OUT:"<helios>\lib\cv_cpp\DJMaxCV.dll"
```

Requires:
- Visual Studio 2022 (MSVC v143)
- OpenCV 4.x C++ headers (from opencv.org SDK download)
- Generated import libs from Helios's OpenCV DLLs

## File Structure

```
djmax_cv_ext/
├── DJMaxCV.cpp          ← Helios C++ DLL source
├── build_helios.bat     ← Build script
├── djmax_cv.cpp         ← Python .pyd extension (GTuner fallback)
├── setup.py             ← Python extension build
└── build.bat            ← Python extension build script
```

## Performance Results

| Metric | GTuner IV Python | Helios 2 C++ |
|--------|-----------------|--------------|
| Frame rate | 33 FPS (drops to 25) | 120 FPS stable |
| Detection time | ~28ms | ~4.7ms |
| Jitter | ±15ms | ±0.5ms |
| Timing variance | High (missed notes) | Minimal |
