#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

// Directory containing the running executable (with trailing separator).
static std::string executableDir() {
#ifdef _WIN32
    char buf[MAX_PATH] = {0};
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::string path(buf, n);
        size_t slash = path.find_last_of("\\/");
        if (slash != std::string::npos) return path.substr(0, slash + 1);
    }
#endif
    return std::string();
}

int main() {
    // ---- Chessboard settings ----
    // The board has boardWidth x boardHeight internal corners. squareSize is the real-world
    // edge length of a square.
    const int boardWidth = 9;
    const int boardHeight = 6;
    const float squareSize = 0.02f;  // 20 mm squares

    cv::Size patternSize(boardWidth, boardHeight);

    // One set of 3D object points for the board, laid out on the z = 0 plane.
    // The same set is reused for every captured view.
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < boardHeight; i++) {
        for (int j = 0; j < boardWidth; j++) {
            objp.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }

    std::vector<std::vector<cv::Point3f>> objectPoints;  // 3D board points per view
    std::vector<std::vector<cv::Point2f>> imagePoints;   // detected 2D corners per view

    // ---- Camera ----
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Could not open webcam!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    std::cout << "Calibration started." << std::endl;
    std::cout << "Capture resolution: "
              << (int)cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Press SPACE to capture a frame with a detected chessboard." << std::endl;
    std::cout << "Press ESC to finish and compute the calibration." << std::endl;

    cv::Mat frame, gray;

    // - Capture loop: collect chessboard views -
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, patternSize, corners,
                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // Refine corner locations to sub-pixel accuracy for a better fit.
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

            cv::drawChessboardCorners(frame, patternSize, corners, found);
        }

        cv::imshow("Calibration", frame);
        int key = cv::waitKey(1);

        if (key == 27) {                       // ESC: finish and calibrate
            break;
        }
        else if (key == ' ' && found) {        // SPACE: keep this view
            imagePoints.push_back(corners);
            objectPoints.push_back(objp);
            std::cout << "Captured frame (" << imagePoints.size() << " samples)." << std::endl;
        }
    }

    if (imagePoints.size() < 8) {
        std::cout << "Not enough samples collected! Need at least 8." << std::endl;
        return -1;
    }

    // - Solve for the intrinsics -
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

    // - Save -
    std::string ymlPath = executableDir() + "camera_calibration.yml";
    cv::FileStorage fs(ymlPath, cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs.release();

    std::cout << "Saved calibration to " << ymlPath << std::endl;

    return 0;
}
