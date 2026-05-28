# Helios 2 CvCpp SDK Reference

Reverse-engineered from `GCVCppSDK.h`, `HeliosInferenceSDK.hpp`, `HeliosInferenceHostContextABI.h`, `HeliosInferenceResultsABI.h`, and `CVTest.dll` binary analysis.

## Overview

Helios 2 supports two CV script modes:
- **CvPython** — Python scripts using the `GCVWorker` class (GTuner IV compatible)
- **CvCpp** — Native C++ DLLs using the `CCVWorkerBase` class (this document)

C++ DLLs are loaded by `CVCppWrapper.exe` (in `lib/cv_cpp/`). The host provides screen frames and forwards detection results to the GPC3 script.

---

## Required DLL Exports

Your DLL must export these three functions:

```cpp
// Creates and returns your worker instance
extern "C" __declspec(dllexport) CCVWorkerBase* createWorker();

// Destroys the worker (use CCV_DEFAULT_DESTROY_WORKER macro)
extern "C" __declspec(dllexport) void destroyWorker(CCVWorkerBase* worker);

// Returns version string (use CCV_DEFAULT_GET_VERSION macro)
extern "C" __declspec(dllexport) const char* getVersion();
```

### Convenience Macros

```cpp
CCV_DEFAULT_DESTROY_WORKER   // Expands to: void destroyWorker(...) { delete worker; }
CCV_DEFAULT_GET_VERSION      // Expands to: const char* getVersion() { return "1.0.0"; }
```

---

## Base Class: `CCVWorkerBase`

```cpp
#include "GCVCppSDK.h"

class CCVWorkerBase {
public:
    virtual ~CCVWorkerBase() = default;
    virtual std::pair<cv::Mat, std::vector<uint8_t>> process(cv::Mat& frame) = 0;
};
```

### `process(cv::Mat& frame)`

Called every frame by the host. The frame is BGR, full capture resolution (e.g., 1920x1080).

**Parameters:**
- `frame` — Current screen capture as `cv::Mat` (BGR, 8-bit). You may draw on it for the preview display.

**Returns:**
- `first` — The frame (optionally with debug overlays drawn on it) for Helios's preview
- `second` — A `vector<uint8_t>` of arbitrary data sent to the GPC script via `gcv_read()`

---

## Communication Functions

### `send_gcvdata(const unsigned char* data, size_t size)`

Sends a byte array to the GPC3 script. The GPC reads it with `gcv_read(offset, &variable)`.

```cpp
// Send 10 bytes to GPC
std::vector<uint8_t> data = {0x01, 0x00, 0x01, ...};
send_gcvdata(data);

// Or with raw pointer
unsigned char buf[10] = {0};
send_gcvdata(buf, 10);
```

**Note:** Also available as `CCV::send_gcvdata()`. Both the return value from `process()` and explicit `send_gcvdata()` calls deliver data to the GPC.

---

### `get_actual(int identifier)`

Reads the **input** state of a controller button/axis (what the physical controller is doing).

```cpp
float value = get_actual(CCV::Controller::BUTTON_5); // Right trigger
float stickX = get_actual(CCV::Controller::STICK_1_X);
```

**Returns:** `float` — Range depends on identifier:
- Buttons: 0.0 (released) to 100.0 (fully pressed)
- Sticks: -100.0 to 100.0

---

### `get_val(int identifier)`

Reads the **output** state (what's being sent to the console/PC after GPC processing).

```cpp
float value = get_val(CCV::Controller::BUTTON_14); // Cross/A button output
```

---

### `controller_input_state()` / `controller_report_state()`

Returns a pointer to the full controller state struct.

```cpp
const CCV::ControllerData* input = CCV::controller_input_state();
if (input && input->state.isConnected) {
    float trigger = input->state.buttons[CCV::Controller::BUTTON_5];
}
```

---

### `keyboard_input_state()` / `keyboard_report_state()`

Returns the keyboard state.

```cpp
const CCV::KeyboardData* kb = CCV::keyboard_input_state();
if (kb && kb->isKeyDown(0x53)) { // VK_S
    // S key is pressed
}
```

---

### `mouse_input_state()` / `mouse_report_state()`

Returns the mouse state.

```cpp
const CCV::MouseData* mouse = CCV::mouse_input_state();
if (mouse && mouse->isButtonDown(CCV::MouseButton::LEFT)) {
    int dx = mouse->deltaX;
    int dy = mouse->deltaY;
}
```

---

## Controller Identifiers (`CCV::Controller`)

| Constant | Value | Description |
|----------|-------|-------------|
| `BUTTON_1` – `BUTTON_21` | 0–20 | Controller buttons |
| `STICK_1_X` | 21 | Right stick X |
| `STICK_1_Y` | 22 | Right stick Y |
| `STICK_2_X` | 23 | Left stick X |
| `STICK_2_Y` | 24 | Left stick Y |
| `POINT_1_X/Y` | 25–26 | Touch point 1 |
| `POINT_2_X/Y` | 27–28 | Touch point 2 |
| `ACCEL_1_X/Y/Z` | 29–31 | Accelerometer 1 |
| `ACCEL_2_X/Y/Z` | 32–34 | Accelerometer 2 |
| `GYRO_1_X/Y/Z` | 35–37 | Gyroscope |
| `PADDLE_1` – `PADDLE_4` | 38–41 | Back paddles |

---

## Keyboard Modifiers (`CCV::KeyboardModifier`)

| Constant | Bit |
|----------|-----|
| `SHIFT_LEFT` | 0 |
| `SHIFT_RIGHT` | 1 |
| `CTRL_LEFT` | 2 |
| `CTRL_RIGHT` | 3 |
| `ALT_LEFT` | 4 |
| `ALT_RIGHT` | 5 |
| `WIN_LEFT` | 6 |
| `WIN_RIGHT` | 7 |
| `CAPS_LOCK` | 8 |
| `NUM_LOCK` | 9 |
| `SCROLL_LOCK` | 10 |

---

## Mouse Buttons (`CCV::MouseButton`)

| Constant | Bit |
|----------|-----|
| `LEFT` | 0 |
| `RIGHT` | 1 |
| `MIDDLE` | 2 |
| `X1` | 3 |
| `X2` | 4 |

---

## Data Structures

### `ControllerData` (256 bytes, 64-byte aligned)

```cpp
struct ControllerData {
    SharedMemoryControllerState state;  // buttons[21], axes[21], timestamp, isConnected
    float leftMotor;
    float rightMotor;
    float leftTriggerMotor;
    float rightTriggerMotor;
    float haptic3;
    float haptic4;
    bool rumbleEnabled;
};
```

### `KeyboardData` (384 bytes, 64-byte aligned)

```cpp
struct KeyboardData {
    unsigned char keys[320];    // Virtual key states (0 = up, non-zero = down)
    unsigned int modifiers;     // Bitmask of KeyboardModifier flags
    bool capsLock, numLock, scrollLock;

    bool isKeyDown(unsigned int virtualKey) const;
};
```

### `MouseData` (128 bytes, 64-byte aligned)

```cpp
struct MouseData {
    int deltaX, deltaY;         // Relative movement
    short wheelDeltaY, wheelDeltaX;
    unsigned int buttons;       // Bitmask of MouseButton flags
    int absoluteX, absoluteY;   // Absolute position (if coordMode == Absolute)
    MouseCoordMode coordMode;   // Relative or Absolute
    unsigned int dpi;
    unsigned int screenWidth, screenHeight;

    bool isButtonDown(unsigned int buttonMask) const;
};
```

---

## Inference Engine (AI/YOLO Object Detection)

Helios 2 includes an inference engine for GPU-accelerated object detection (YOLO models). This is separate from the CV frame processing but can be used alongside it.

### Key Classes

```cpp
#include "HeliosInferenceSDK.hpp"
using namespace Helios::Inference;
```

### `Session`

Manages an inference engine instance.

```cpp
Session session = Session::create("model-uuid", true); // true = use TensorRT
session.start();                    // Begin inference on video frames
auto results = session.waitForResults(35); // Wait up to 35ms for detections
session.stop();
session.destroy();
```

### `HeliosInferenceDetectionV1`

Each detection result contains:
- Class ID
- Confidence score
- Bounding box (x, y, width, height as floats 0.0–1.0)
- Polar coordinates relative to origin
- Keypoints (for pose models)

### Session Configuration

```cpp
session.setConfidenceThreshold(0.5f);
session.setNmsThreshold(0.45f);
session.setRoi(0.5f, 0.5f, 0.3f, 16.0f, 9.0f); // Center ROI
session.setDrawDetections(true);
session.setDrawBenchmarks(true);
session.setPolarOrigin(0.5f, 0.5f);  // Screen center as aim point
```

### `HostContext`

Provides host environment information:

```cpp
uint64_t timestamp = HostContext::currentFrameTimestampNs();
uint32_t pid = HostContext::heliosProcessId();
std::string bufferName = HostContext::videoRingBufferName();
```

---

## Minimal Example

```cpp
#define NOMINMAX
#include "GCVCppSDK.h"

class MyWorker : public CCVWorkerBase {
public:
    MyWorker() {
        // Initialize resources
    }

    std::pair<cv::Mat, std::vector<uint8_t>> process(cv::Mat& frame) override {
        std::vector<uint8_t> data(4, 0x00);

        // Example: detect if a region is bright
        cv::Mat roi = frame(cv::Rect(100, 100, 50, 50));
        cv::Scalar mean = cv::mean(roi);
        if (mean[0] > 200) data[0] = 0x01;

        // Send to GPC
        send_gcvdata(data);

        // Draw debug overlay
        cv::putText(frame, "Active", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1, {0,255,0}, 2);

        return {frame, data};
    }
};

extern "C" __declspec(dllexport) CCVWorkerBase* createWorker() {
    return new MyWorker();
}

CCV_DEFAULT_DESTROY_WORKER
CCV_DEFAULT_GET_VERSION
```

---

## Build Requirements

- **Compiler**: MSVC v143+ (Visual Studio 2022)
- **C++ Standard**: C++17
- **Headers**: OpenCV 4.x headers (from opencv.org SDK)
- **Link against**: Helios's split OpenCV libs (`opencv_core4.lib`, `opencv_imgproc4.lib`, `opencv_imgcodecs4.lib`)
- **Runtime**: DLL is loaded by `CVCppWrapper.exe`; OpenCV DLLs must be findable (they're in `lib/cv_cpp/` and `lib/`)

### Generating Import Libraries

Helios ships OpenCV as DLLs without `.lib` files. Generate them:

```powershell
# Extract exports
dumpbin /exports opencv_core4.dll | Select-String "^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)" |
    ForEach-Object { $_.Matches[0].Groups[1].Value } > exports.txt

# Create .def
echo "LIBRARY opencv_core4`nEXPORTS" > opencv_core4.def
cat exports.txt >> opencv_core4.def

# Generate .lib
lib /def:opencv_core4.def /out:opencv_core4.lib /machine:x64
```

Repeat for `opencv_imgproc4` and `opencv_imgcodecs4`.

---

## GPC Side (Reading CV Data)

## Important: Frame Color Channel Order

Helios's CvCpp host passes frames from DXGI Desktop Duplication. The channel order may differ from what you expect:

- **GTuner IV (Python)**: Frames are BGR (standard OpenCV convention). `cv2.mean()` returns `(B, G, R)`.
- **Helios CvCpp**: Frames appear to be **RGB** (DirectX/DXGI convention), but OpenCV's `cv::Mat` still interprets them as BGR. This means channel 0 = R, channel 1 = G, channel 2 = B — the opposite of what `cv::Scalar` labels suggest.

**Practical impact**: If you're doing color-based detection (e.g., checking if a region is a specific color), the values from `cv::mean()` will have R and B swapped compared to the same detection in GTuner IV's Python environment.

**Recommendation**: Calibrate color boundaries empirically using debug output rather than assuming BGR order. Use the debug DLL to print actual channel values and set thresholds based on what you observe.

**HDR note**: HDR and SDR captures produce different color values for the same on-screen color. Design your boundaries wide enough to cover both, or detect the mode and switch thresholds.

---

## GPC Side (Reading CV Data)

In the GPC3 script, read the data sent by `send_gcvdata()`:

```c
if (gcv_ready()) {
    uint8 value;
    gcv_read(0, &value);  // Read byte at offset 0
    gcv_read(1, &value);  // Read byte at offset 1
    // ...
}
```

The byte offsets correspond to the indices in the `vector<uint8_t>` returned/sent from C++.
