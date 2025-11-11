#include "CPUFilters.hpp"
#include <algorithm>

namespace CPUFilters {

void pixelate(cv::Mat& src, cv::Mat& dst, int pixelSize) {
    dst = src.clone();
    for (int y = 0; y < src.rows; y += pixelSize) {
        for (int x = 0; x < src.cols; x += pixelSize) {
            cv::Rect rect(x, y, pixelSize, pixelSize);
            rect &= cv::Rect(0, 0, src.cols, src.rows);
            cv::Scalar color = cv::mean(src(rect));
            cv::rectangle(dst, rect, color, cv::FILLED);
        }
    }
}

void sinCity(cv::Mat& src, cv::Mat& dst) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, dst, cv::COLOR_GRAY2BGR);
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b color = src.at<cv::Vec3b>(y, x);
            if (color[2] > 150 && color[2] > color[1] * 1.3 && color[2] > color[0] * 1.3) {
                dst.at<cv::Vec3b>(y, x) = color;  // Keep red-ish pixels
            }
        }
    }
}

}
