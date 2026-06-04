#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // --- Chessboard settings ---
    const int boardWidth = 9;        // internal corners
    const int boardHeight = 6;
    const float squareSize = 0.02f;  // meters (20 mm)

    cv::Size patternSize(boardWidth, boardHeight);

    // Prepare object points (0,0,0), (1,0,0), ...
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < boardHeight; i++) {
        for (int j = 0; j < boardWidth; j++) {
            objp.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }

    std::vector<std::vector<cv::Point3f>> objectPoints;  // 3D points in world space
    std::vector<std::vector<cv::Point2f>> imagePoints;   // 2D detected corners

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Could not open webcam!" << std::endl;
        return -1;
    }

    std::cout << "Calibration started." << std::endl;
    std::cout << "Press SPACE to capture frame with detected chessboard." << std::endl;
    std::cout << "Press ESC to finish and compute calibration." << std::endl;

    cv::Mat frame, gray;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, patternSize, corners,
                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

            cv::drawChessboardCorners(frame, patternSize, corners, found);
        }

        cv::imshow("Calibration", frame);
        int key = cv::waitKey(1);

        if (key == 27) { // ESC → finish calibration
            break;
        } 
        else if (key == ' ' && found) { // SPACE → save frame
            imagePoints.push_back(corners);
            objectPoints.push_back(objp);
            std::cout << "Captured frame (" << imagePoints.size() << " samples)." << std::endl;
        }
    }

    if (imagePoints.size() < 8) {
        std::cout << "Not enough samples collected! Need at least 8." << std::endl;
        return -1;
    }

    std::cout << "Running calibration..." << std::endl;

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(
        objectPoints,
        imagePoints,
        frame.size(),
        cameraMatrix,
        distCoeffs,
        rvecs,
        tvecs
    );

    std::cout << "Calibration RMS error = " << rms << std::endl;
    std::cout << "Camera matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients:\n" << distCoeffs << std::endl;

    // Save to file
    cv::FileStorage fs("camera_calibration.yml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs.release();

    std::cout << "Saved calibration to camera_calibration.yml" << std::endl;

    return 0;
}
