#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>
#include <string>
#include <mutex>

static const double THRESHOLD = 0.75;

// Lane definitions: {xStart, xEnd}
static const int WHITE_LANES[4][2] = {
    {20, 183}, {340, 503}, {503, 666}, {822, 985}
};
static const int COLOR_LANES[2][2] = {
    {181, 344}, {662, 825}
};
static const int SIDE_LANES[2][2] = {
    {23, 236}, {503, 716}
};

// Side color boundary (BGR order in OpenCV)
static const double SIDE_LO[3] = {25, 169, 177};  // B, G, R
static const double SIDE_HI[3] = {95, 220, 215};

struct DetectorState {
    cv::Mat whiteNote, colorNote;
    cv::Mat whiteHoldS, whiteHoldM, whiteHoldE;
    cv::Mat colorHoldS, colorHoldM, colorHoldE;
    bool loaded = false;
};

static DetectorState g_state;

static double matchBest(const cv::Mat& region, const cv::Mat& tmpl) {
    if (region.cols < tmpl.cols || region.rows < tmpl.rows) return 0.0;
    cv::Mat result;
    cv::matchTemplate(region, tmpl, result, cv::TM_CCOEFF_NORMED);
    double maxVal;
    cv::minMaxLoc(result, nullptr, &maxVal, nullptr, nullptr);
    return maxVal;
}

static double detectWhiteLane(const cv::Mat& frame, int xStart, int xEnd) {
    cv::Mat roi = frame(cv::Range(0, frame.rows), cv::Range(xStart, xEnd));
    double v = matchBest(roi, g_state.whiteHoldS);
    if (v >= THRESHOLD) return v;
    v = matchBest(roi, g_state.whiteHoldM);
    if (v >= THRESHOLD) return v;
    v = matchBest(roi, g_state.whiteHoldE);
    if (v >= THRESHOLD) return v;
    return matchBest(roi, g_state.whiteNote);
}

static double detectColorLane(const cv::Mat& frame, int xStart, int xEnd) {
    cv::Mat roi = frame(cv::Range(0, frame.rows), cv::Range(xStart, xEnd));
    double v = matchBest(roi, g_state.colorHoldS);
    if (v >= THRESHOLD) return v;
    v = matchBest(roi, g_state.colorHoldM);
    if (v >= THRESHOLD) return v;
    v = matchBest(roi, g_state.colorHoldE);
    if (v >= THRESHOLD) return v;
    return matchBest(roi, g_state.colorNote);
}

static double detectSide(const cv::Mat& frame, int xStart, int xEnd) {
    cv::Mat roi = frame(cv::Range(60, 80), cv::Range(xStart, xEnd));
    cv::Scalar mean = cv::mean(roi);
    // mean is BGR
    double b = mean[0], g = mean[1], r = mean[2];
    if (b > SIDE_LO[0] && b < SIDE_HI[0] &&
        g > SIDE_LO[1] && g < SIDE_HI[1] &&
        r > SIDE_LO[2] && r < SIDE_HI[2]) {
        return 1.0;
    }
    return 0.0;
}

static PyObject* py_load_templates(PyObject* self, PyObject* args) {
    const char* templateDir;
    if (!PyArg_ParseTuple(args, "s", &templateDir)) return NULL;

    std::string dir(templateDir);
    if (dir.back() != '/' && dir.back() != '\\') dir += '/';

    g_state.whiteNote  = cv::imread(dir + "whiteNote.jpg", cv::IMREAD_COLOR);
    g_state.colorNote  = cv::imread(dir + "colorNote.jpg", cv::IMREAD_COLOR);
    g_state.whiteHoldS = cv::imread(dir + "whiteHoldS.jpg", cv::IMREAD_COLOR);
    g_state.whiteHoldM = cv::imread(dir + "whiteHoldM.jpg", cv::IMREAD_COLOR);
    g_state.whiteHoldE = cv::imread(dir + "whiteHoldE.jpg", cv::IMREAD_COLOR);
    g_state.colorHoldS = cv::imread(dir + "colorHoldS.jpg", cv::IMREAD_COLOR);
    g_state.colorHoldM = cv::imread(dir + "colorHoldM.jpg", cv::IMREAD_COLOR);
    g_state.colorHoldE = cv::imread(dir + "colorHoldE.jpg", cv::IMREAD_COLOR);

    if (g_state.whiteNote.empty() || g_state.colorNote.empty()) {
        PyErr_SetString(PyExc_FileNotFoundError, "Failed to load template images");
        return NULL;
    }
    g_state.loaded = true;
    Py_RETURN_NONE;
}

static PyObject* py_detect(PyObject* self, PyObject* args) {
    PyArrayObject* frameArr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &frameArr)) return NULL;

    if (!g_state.loaded) {
        PyErr_SetString(PyExc_RuntimeError, "Templates not loaded. Call load_templates() first.");
        return NULL;
    }

    // Ensure contiguous array
    PyArrayObject* contArr = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)frameArr, NPY_UINT8, 3, 3);
    if (!contArr) return NULL;

    npy_intp* shape = PyArray_DIMS(contArr);
    int height = (int)shape[0];
    int width = (int)shape[1];
    int channels = (int)shape[2];

    if (channels != 3) {
        Py_DECREF(contArr);
        PyErr_SetString(PyExc_ValueError, "Frame must have 3 channels");
        return NULL;
    }

    cv::Mat frame(height, width, CV_8UC3, PyArray_DATA(contArr));

    double results[10] = {0};

    // Clamp lane coordinates to frame bounds
    auto clampLane = [width](int xStart, int xEnd) -> std::pair<int,int> {
        return {std::max(0, std::min(xStart, width-1)), std::max(0, std::min(xEnd, width))};
    };

    std::thread workers[8];

    for (int i = 0; i < 4; i++) {
        workers[i] = std::thread([&results, &frame, i, width, height, &clampLane]() {
            auto [xs, xe] = clampLane(WHITE_LANES[i][0], WHITE_LANES[i][1]);
            if (xe - xs > 10 && height > 10)
                results[i] = detectWhiteLane(frame, xs, xe);
        });
    }
    for (int i = 0; i < 2; i++) {
        workers[4 + i] = std::thread([&results, &frame, i, width, height, &clampLane]() {
            auto [xs, xe] = clampLane(COLOR_LANES[i][0], COLOR_LANES[i][1]);
            if (xe - xs > 10 && height > 10)
                results[4 + i] = detectColorLane(frame, xs, xe);
        });
    }
    for (int i = 0; i < 2; i++) {
        workers[6 + i] = std::thread([&results, &frame, i, width, height, &clampLane]() {
            auto [xs, xe] = clampLane(SIDE_LANES[i][0], SIDE_LANES[i][1]);
            if (xe - xs > 10 && height > 80)
                results[6 + i] = detectSide(frame, xs, xe);
        });
    }

    for (int i = 0; i < 8; i++) workers[i].join();

    Py_DECREF(contArr);

    char out[10] = {0};
    for (int i = 0; i < 10; i++) {
        out[i] = (results[i] >= THRESHOLD) ? 0x01 : 0x00;
    }

    return PyBytes_FromStringAndSize(out, 10);
}

static PyMethodDef methods[] = {
    {"load_templates", py_load_templates, METH_VARARGS, "Load template images from directory"},
    {"detect", py_detect, METH_VARARGS, "Detect notes in frame, returns 10-byte result"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "djmax_cv", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_djmax_cv(void) {
    import_array();
    return PyModule_Create(&module);
}
