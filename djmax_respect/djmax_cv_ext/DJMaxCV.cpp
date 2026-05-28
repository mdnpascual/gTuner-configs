#define NOMINMAX
#include "GCVCppSDK.h"
#include <string>
#include <thread>
#include <vector>
#include <chrono>

static const double THRESHOLD = 0.75;

static const int WHITE_LANES[4][2] = {{20,183},{340,503},{503,666},{822,985}};
static const int COLOR_LANES[2][2] = {{181,344},{662,825}};
static const int SIDE_LANES[2][2] = {{23,236},{503,716}};
static const double SIDE_LO[3] = {150, 200, 0};   // B, G, R min
static const double SIDE_HI[3] = {240, 255, 60};  // B, G, R max

static cv::Mat g_whiteNote, g_colorNote;
static cv::Mat g_whiteHoldS, g_whiteHoldM, g_whiteHoldE;
static cv::Mat g_colorHoldS, g_colorHoldM, g_colorHoldE;

static double matchBest(const cv::Mat& region, const cv::Mat& tmpl) {
    if (region.cols < tmpl.cols || region.rows < tmpl.rows) return 0.0;
    cv::Mat result;
    cv::matchTemplate(region, tmpl, result, cv::TM_CCOEFF_NORMED);
    double maxVal;
    cv::minMaxLoc(result, nullptr, &maxVal, nullptr, nullptr);
    return maxVal;
}

static double detectWhiteLane(const cv::Mat& frame, int xs, int xe) {
    cv::Mat roi = frame(cv::Range(0, frame.rows), cv::Range(xs, xe));
    double v = matchBest(roi, g_whiteHoldS);
    if (v >= THRESHOLD) return v;
    v = matchBest(roi, g_whiteHoldM);
    if (v >= THRESHOLD) return v;
    v = matchBest(roi, g_whiteHoldE);
    if (v >= THRESHOLD) return v;
    return matchBest(roi, g_whiteNote);
}

static double detectColorLane(const cv::Mat& frame, int xs, int xe) {
    cv::Mat roi = frame(cv::Range(0, frame.rows), cv::Range(xs, xe));
    double v = matchBest(roi, g_colorHoldS);
    if (v >= THRESHOLD) return v;
    v = matchBest(roi, g_colorHoldM);
    if (v >= THRESHOLD) return v;
    v = matchBest(roi, g_colorHoldE);
    if (v >= THRESHOLD) return v;
    return matchBest(roi, g_colorNote);
}

static double detectSide(const cv::Mat& frame, int xs, int xe
#ifdef DJMAX_DEBUG
    , double* dbgB = nullptr, double* dbgG = nullptr, double* dbgR = nullptr
#endif
) {
    if (frame.rows < 80) return 0.0;
    cv::Mat roi = frame(cv::Range(60, 80), cv::Range(xs, xe));
    cv::Scalar mean = cv::mean(roi);
    double b = mean[0], g = mean[1], r = mean[2];
#ifdef DJMAX_DEBUG
    if (dbgB) *dbgB = b;
    if (dbgG) *dbgG = g;
    if (dbgR) *dbgR = r;
#endif
    if (b > SIDE_LO[0] && b < SIDE_HI[0] &&
        g > SIDE_LO[1] && g < SIDE_HI[1] &&
        r > SIDE_LO[2] && r < SIDE_HI[2])
        return 1.0;
    return 0.0;
}

class GCVWorker : public CCVWorkerBase {
    bool m_loaded = false;
    int m_frameCount = 0;
    double m_fps = 0.0;
    std::chrono::steady_clock::time_point m_lastTime = std::chrono::steady_clock::now();
public:
    GCVWorker() {
        // Load templates - adjust path as needed
        std::string dir = "C:/Users/MDuh/Desktop/Projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/";
        g_whiteNote  = cv::imread(dir + "whiteNote.jpg");
        g_colorNote  = cv::imread(dir + "colorNote.jpg");
        g_whiteHoldS = cv::imread(dir + "whiteHoldS.jpg");
        g_whiteHoldM = cv::imread(dir + "whiteHoldM.jpg");
        g_whiteHoldE = cv::imread(dir + "whiteHoldE.jpg");
        g_colorHoldS = cv::imread(dir + "colorHoldS.jpg");
        g_colorHoldM = cv::imread(dir + "colorHoldM.jpg");
        g_colorHoldE = cv::imread(dir + "colorHoldE.jpg");
        m_loaded = !g_whiteNote.empty() && !g_colorNote.empty();
    }

    std::pair<cv::Mat, std::vector<uint8_t>> process(cv::Mat& frame) override {
        std::vector<uint8_t> data(10, 0x00);
        if (!m_loaded || frame.empty()) return {frame, data};

        // Helios captures at 1920x1080. Original config was 2697:0:1003:158 at 4K (3840x2160).
        // Scale factor: 1920/3840 = 0.5
        // Crop at 1080p: x=1348, y=0, w=502, h=79, then resize to 1003x158 for templates
        int cropX = 1348, cropY = 0, cropW = 502, cropH = 79;
        if (frame.cols < cropX + cropW || frame.rows < cropY + cropH)
            return {frame, data};
        cv::Mat crop = frame(cv::Rect(cropX, cropY, cropW, cropH));
        cv::Mat cropped;
        cv::resize(crop, cropped, cv::Size(1003, 158));

        int width = cropped.cols;
        int height = cropped.rows;
        double results[10] = {0};
#ifdef DJMAX_DEBUG
        double timings[8] = {0};
        double sideBGR[2][3] = {{0}};
        auto processStart = std::chrono::steady_clock::now();
#endif

        std::thread workers[8];
        for (int i = 0; i < 4; i++) {
            int xs = std::max(0, std::min(WHITE_LANES[i][0], width-1));
            int xe = std::max(0, std::min(WHITE_LANES[i][1], width));
            if (xe - xs > 10)
                workers[i] = std::thread([&results,
#ifdef DJMAX_DEBUG
                    &timings,
#endif
                    &cropped, i, xs, xe]() {
#ifdef DJMAX_DEBUG
                    auto t0 = std::chrono::steady_clock::now();
#endif
                    results[i] = detectWhiteLane(cropped, xs, xe);
#ifdef DJMAX_DEBUG
                    timings[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
#endif
                });
        }
        for (int i = 0; i < 2; i++) {
            int xs = std::max(0, std::min(COLOR_LANES[i][0], width-1));
            int xe = std::max(0, std::min(COLOR_LANES[i][1], width));
            if (xe - xs > 10)
                workers[4+i] = std::thread([&results,
#ifdef DJMAX_DEBUG
                    &timings,
#endif
                    &cropped, i, xs, xe]() {
#ifdef DJMAX_DEBUG
                    auto t0 = std::chrono::steady_clock::now();
#endif
                    results[4+i] = detectColorLane(cropped, xs, xe);
#ifdef DJMAX_DEBUG
                    timings[4+i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
#endif
                });
        }
        for (int i = 0; i < 2; i++) {
            int xs = std::max(0, std::min(SIDE_LANES[i][0], width-1));
            int xe = std::max(0, std::min(SIDE_LANES[i][1], width));
            if (xe - xs > 10 && height > 80)
                workers[6+i] = std::thread([&results,
#ifdef DJMAX_DEBUG
                    &timings, &sideBGR,
#endif
                    &cropped, i, xs, xe]() {
#ifdef DJMAX_DEBUG
                    auto t0 = std::chrono::steady_clock::now();
                    results[6+i] = detectSide(cropped, xs, xe, &sideBGR[i][0], &sideBGR[i][1], &sideBGR[i][2]);
                    timings[6+i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
#else
                    results[6+i] = detectSide(cropped, xs, xe);
#endif
                });
        }
        for (int i = 0; i < 8; i++)
            if (workers[i].joinable()) workers[i].join();

#ifdef DJMAX_DEBUG
        auto processEnd = std::chrono::steady_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(processEnd - processStart).count();
#endif

        for (int i = 0; i < 10; i++)
            data[i] = (results[i] >= THRESHOLD) ? 0x01 : 0x00;

        // FPS counter
#ifdef DJMAX_DEBUG
        m_frameCount++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - m_lastTime).count();
        if (elapsed >= 1.0) {
            m_fps = m_frameCount / elapsed;
            m_frameCount = 0;
            m_lastTime = now;
        }
        char buf[128];
        // Position text left of the crop region, vertically centered on frame
        int textX = cropX - 280;
        int textY = 540 - 100; // center of 1080p minus half the text block height

        snprintf(buf, sizeof(buf), "FPS: %.1f", m_fps);
        cv::putText(frame, buf, {textX, textY}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);

        // Per-lane debug text
        const char* laneNames[] = {"6B1W", "6B2W", "6B3W", "6B4W", "6B1C", "6B2C", "SideL", "SideR", "", ""};
        for (int i = 0; i < 8; i++) {
            cv::Scalar color = (results[i] >= THRESHOLD) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 255);
            snprintf(buf, sizeof(buf), "%s: %.2f  %.3fms", laneNames[i], results[i], timings[i]);
            cv::putText(frame, buf, {textX, textY + 25 + i * 20}, cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
        }
        snprintf(buf, sizeof(buf), "Total: %.2fms", totalMs);
        cv::putText(frame, buf, {textX, textY + 25 + 8 * 20}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 200, 255}, 1, cv::LINE_AA);
#endif

        // Send data to GPC
        send_gcvdata(data);

        return {frame, data};
    }
};

extern "C" __declspec(dllexport) CCVWorkerBase* createWorker() {
    return new GCVWorker();
}

CCV_DEFAULT_DESTROY_WORKER
CCV_DEFAULT_GET_VERSION
