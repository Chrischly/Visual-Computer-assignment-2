#pragma once
#include <opencv2/opencv.hpp>

namespace CPUFilters {

    // Simple pixelation filter using block averaging
    void pixelate(cv::Mat& src, cv::Mat& dst, int pixelSize = 10);

    // Sin City filter: grayscale + keep red tones
    void sinCity(cv::Mat& src, cv::Mat& dst);

}
