#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <iomanip>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <common/Shader.hpp>
#include <common/TextureShader.hpp>
#include <common/Quad.hpp>
#include <common/Texture.hpp>
#include <common/Scene.hpp>
#include <common/Camera.hpp>
#include <common/filters/CPUFilters.hpp>
#include <common/Cube.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

// Returns the directory that contains the running executable (e.g. build\Debug),
// with a trailing separator. Falls back to "" (current dir) if it can't be found.
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

using namespace std;

// ---------------------- Globals ----------------------
GLFWwindow* window = nullptr;

float rotateAngle = 0.0f;
float translateX = 0.0f, translateY = 0.0f;
float scaleFactor = 1.0f;

bool mousePressed = false;
double lastX = 0.0, lastY = 0.0;

enum FilterType { FILTER_NONE, FILTER_PIXELATE, FILTER_SINCITY };
FilterType activeFilter = FILTER_NONE;
bool useGPU = true;

std::atomic<bool> batchRequested(false);
std::atomic<bool> batchRunning(false);


bool warmupCamera(cv::VideoCapture &cap, const int maxAttempts = 80, int msBetween = 15) {
    cv::Mat tmp;
    for (int i = 0; i < maxAttempts; ++i) {
        cap >> tmp;
        if (!tmp.empty()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(msBetween));
    }
    return false;
}

// Safe frame grab to avoid using frames with inconsistent step
bool grabSafeFrame(cv::VideoCapture &cap, cv::Mat &frame) {
    cap >> frame;
    if (frame.empty()) {
        return false;
    }
    if (frame.step < (size_t)frame.cols * (size_t)frame.elemSize1() * (size_t)frame.channels()) {
        // invalid frame, consider it bad
        return false;
    }
    return true;
}

// Simple debounce helper to return true when key went from not pressed to pressed
bool keyPressedOnce(int key) {
    static std::unordered_map<int, bool> prev;
    int state = glfwGetKey(window, key);
    bool now = (state == GLFW_PRESS);
    bool was = prev[key];
    prev[key] = now;
    return now && !was;
}

// -- Window + Input --
bool initWindow(const std::string& name) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1024, 768, name.c_str(), nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);

    // mouse callbacks
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) mousePressed = (action == GLFW_PRESS);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double xpos, double ypos) {
        if (mousePressed) {
            float dx = static_cast<float>(xpos - lastX) / 500.0f;
            float dy = static_cast<float>(ypos - lastY) / 500.0f;
            translateX += dx;
            translateY -= dy;
        }
        lastX = xpos;
        lastY = ypos;
    });

    glfwSetScrollCallback(window, [](GLFWwindow* w, double xoffset, double yoffset) {
        scaleFactor *= (1.0f + 0.1f * static_cast<float>(yoffset));
        if (scaleFactor < 0.05f) scaleFactor = 0.05f;
    });

    return true;
}

void processInput() {
    // Movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) translateY += 0.01f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) translateY -= 0.01f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) translateX -= 0.01f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) translateX += 0.01f;

    // Rotation
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) rotateAngle -= 1.0f;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) rotateAngle += 1.0f;

    // Zoom
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) scaleFactor *= 1.01f;
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) scaleFactor *= 0.99f;

    // Filters
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) activeFilter = FILTER_NONE;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) activeFilter = FILTER_PIXELATE;
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) activeFilter = FILTER_SINCITY;

    // Backend
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) useGPU = true;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) useGPU = false;

    // Batch experiments
    if (keyPressedOnce(GLFW_KEY_T)) {
        if (!batchRunning.load()) {
            batchRequested = true;
        }
    }

    // Exit
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

// -- Batch experiments --
// Runs a set of experiments, logs averaged FPS per run to a experiments.csv file.
void runBatchExperiments(
    cv::VideoCapture &cap,
    Texture* videoTexture,
    Quad* quad,
    Scene* scene,
    Camera* cam,
    TextureShader* defaultShader,
    TextureShader* pixelateShader,
    TextureShader* sinCityShader
) {
    batchRunning = true;
    std::cout << "[MAIN] Running automatic experiments (T pressed)\n";
    // Config
    const vector<pair<int,int>> resolutions = { {1280,720}, {1024,576}, {640,360} };
    const vector<int> backends = { 0 /*GPU*/, 1 /*CPU*/ };
    const vector<FilterType> filters = { FILTER_NONE, FILTER_PIXELATE, FILTER_SINCITY };
    const vector<bool> transformFlags = { false, true };

    const int runSeconds = 8;
    const int warmupMs = 400; 
    const string csvName = "experiments.csv";

    double origW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double origH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // open CSV
    ofstream csv(csvName, ios::app);
    if (!csv.is_open()) {
        std::cerr << "[BATCH] Cannot open " << csvName << " for writing\n";
        batchRunning = false;
        return;
    }
    // write header if new file
    csv.seekp(0, ios::end);
    if (csv.tellp() == 0) {
        csv << "resolution_w,resolution_h,backend,filter,transform,avg_fps,run_seconds,build_type,avg_frame_time_ms\n";
    }

    #ifdef NDEBUG
    const string build_type = "Release";
    #else
    const string build_type = "Debug";
    #endif

    // iterate configs
    for (auto res : resolutions) {
        int w = res.first, h = res.second;
        // set camera resolution
        cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);

      
        std::this_thread::sleep_for(std::chrono::milliseconds(warmupMs));
        for (int d=0; d<6; ++d) {
            cv::Mat tmp; cap >> tmp;
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }

        for (int backend : backends) {
            for (auto f : filters) {
                for (bool transformActive : transformFlags) {
                    if (glfwWindowShouldClose(window)) break;

                    cout << "[BATCH] Running: " << w << "x" << h
                         << " backend=" << (backend==0 ? "GPU" : "CPU")
                         << " filter=" << (f==FILTER_NONE ? "NONE" : (f==FILTER_PIXELATE ? "PIXELATE" : "SINCITY"))
                         << " transform=" << (transformActive ? "ON" : "OFF")
                         << " for " << runSeconds << "s\n";

                    // prepare run variables
                    bool localUseGPU = (backend == 0);
                    // representative transform for transform ON:
                    const float txNorm = transformActive ? 0.10f : 0.0f;
                    const float tyNorm = transformActive ? 0.05f : 0.0f;
                    const float rotDeg  = transformActive ? 15.0f : 0.0f;
                    const float scl     = transformActive ? 0.9f : 1.0f;

                    // per-run stats
                    uint64_t frames = 0;
                    double totalFrameMs = 0.0;

                    auto tEnd = chrono::high_resolution_clock::now() + chrono::seconds(runSeconds);

                    // run loop
                    while (chrono::high_resolution_clock::now() < tEnd) {
                        auto frameStart = chrono::high_resolution_clock::now();

                        cv::Mat frame;
                        cap >> frame;
                        if (frame.empty()) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                            continue;
                        }

                        if (localUseGPU) {
                            cv::flip(frame, frame, 0);
                            videoTexture->update(frame.data, frame.cols, frame.rows, true);
                            // shader selection
                            if (f == FILTER_PIXELATE) quad->setShader(pixelateShader);
                            else if (f == FILTER_SINCITY) quad->setShader(sinCityShader);
                            else quad->setShader(defaultShader);
                            // apply transform to quad as normalized values
                            quad->setTranslate(glm::vec3(txNorm, tyNorm, 0.0f));
                            quad->setRotate(rotDeg);
                            quad->setScale(scl);

                        } else {
                            // CPU path: filter + warpAffine if transformActive
                            cv::Mat processed;
                            if (f == FILTER_PIXELATE) CPUFilters::pixelate(frame, processed, 10);
                            else if (f == FILTER_SINCITY) CPUFilters::sinCity(frame, processed);
                            else processed = frame.clone();

                            if (transformActive) {
                                float txPixels = txNorm * processed.cols;
                                float tyPixels = tyNorm * processed.rows;
                                cv::Point2f center(processed.cols/2.0f, processed.rows/2.0f);
                                cv::Mat M = cv::getRotationMatrix2D(center, rotDeg, scl);
                                M.at<double>(0,2) += txPixels;
                                M.at<double>(1,2) -= tyPixels;
                                cv::Mat warped;
                                cv::warpAffine(processed, warped, M, processed.size());
                                processed = std::move(warped);
                            }

                            cv::flip(processed, processed, 0);
                            videoTexture->update(processed.data, processed.cols, processed.rows, true);

                            // CPU uses default shader and identity quad transform so image shows as-warped
                            quad->setShader(defaultShader);
                            quad->setTranslate(glm::vec3(0.0f,0.0f,0.0f));
                            quad->setRotate(0.0f);
                            quad->setScale(1.0f);
                        }

                        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                        scene->render(cam);
                        glFinish();

                        glfwSwapBuffers(window);
                        glfwPollEvents();

                        frames++;
                        auto frameEnd = chrono::high_resolution_clock::now();
                        double frameMs = chrono::duration<double, milli>(frameEnd - frameStart).count();
                        totalFrameMs += frameMs;

                        if (glfwWindowShouldClose(window)) break;
                    } // per-config loop

                    // compute results
                    double avgFps = frames > 0 ? double(frames) / double(runSeconds) : 0.0;
                    double avgFrameMs = frames > 0 ? totalFrameMs / double(frames) : 0.0;

                    csv << w << "," << h << "," << (localUseGPU ? "GPU" : "CPU") << ","
                        << (f==FILTER_NONE ? "NONE" : (f==FILTER_PIXELATE ? "PIXELATE" : "SINCITY")) << ","
                        << (transformActive ? "ON" : "OFF") << ","
                        << fixed << setprecision(3) << avgFps << ","
                        << runSeconds << "," << build_type << ","
                        << fixed << setprecision(3) << avgFrameMs << "\n";
                    csv.flush();

                    cout << "[BATCH] result -> " << w << "x" << h << " "
                         << (localUseGPU ? "GPU" : "CPU") << " "
                         << (f==FILTER_NONE ? "NONE" : (f==FILTER_PIXELATE ? "PIXELATE" : "SINCITY"))
                         << " transform=" << (transformActive ? "ON" : "OFF")
                         << " avg_fps=" << avgFps << " avg_frame_ms=" << avgFrameMs << "\n";

                    std::this_thread::sleep_for(std::chrono::milliseconds(120));
                    if (glfwWindowShouldClose(window)) break;
                } // transform flags
                if (glfwWindowShouldClose(window)) break;
            } // filters
            if (glfwWindowShouldClose(window)) break;
        } // backends
        if (glfwWindowShouldClose(window)) break;
    } // resolutions

    // restore camera original resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, (int)origW);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, (int)origH);

    csv.close();
    cout << "[BATCH] Finished automatic experiments. Results appended to " << csvName << "\n";
    batchRunning = false;
}

// ---------------------- main ----------------------
int main() {
    // open camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: could not open camera\n";
        return -1;
    }

    // 
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    if (!warmupCamera(cap, 80, 15)) {
        cerr << "[WARN] Camera warmup failed to get frames quickly â€” continuing anyway\n";
    }

    if (!initWindow("Video Processing")) return -1;
    if (!gladLoadGL(glfwGetProcAddress)) return -1;

    glEnable(GL_DEPTH_TEST);
    GLuint VAO; glGenVertexArrays(1, &VAO); glBindVertexArray(VAO);

    // Capture first frame
    cv::Mat frame;
    if (!grabSafeFrame(cap, frame)) {
        cerr << "Error: could not capture initial frame\n";
        cap.release();
        glfwTerminate();
        return -1;
    }

    cv::flip(frame, frame, 0);

    // Create resources
    Texture* videoTexture = new Texture(frame.data, frame.cols, frame.rows, true);

    TextureShader* defaultShader = new TextureShader("videoTextureShader.vert", "videoTextureShader.frag");
    TextureShader* pixelateShader = new TextureShader("videoTextureShader.vert", "pixelate.frag");
    TextureShader* sinCityShader  = new TextureShader("videoTextureShader.vert", "sincity.frag");
    

    defaultShader->setTexture(videoTexture);
    pixelateShader->setTexture(videoTexture);
    sinCityShader->setTexture(videoTexture);

    Scene* scene = new Scene();
    Camera* cam = new Camera();
    cam->setPosition(glm::vec3(0,0,-2.5f));

    Quad* quad = new Quad((float)frame.cols / (float)frame.rows);
    quad->setShader(defaultShader);
    scene->addObject(quad);
    Cube* cube = new Cube();
    cube->setScale(glm::vec3(0.02f));   // Scale cube to 5cm (half of 10cm marker)
    cube->setVisible(false);            // hidden until a marker is detected
    Shader* simple3DShader = new Shader("cube.vert", "cube.frag"); // or use existing shader
    cube->setShader(new Shader("cube.vert", "cube.frag"));
    scene->addObject(cube);


    
    // Interactive FPS logging CSV
    std::ofstream csv("fps_log.csv");
csv << "Frame,Backend,Filter,FPS\n";

    // ---- POSE LOGGING CSV ----
    std::ofstream poseCSV("pose_log.csv");
    poseCSV << "frame,tx,ty,tz,rx,ry,rz\n";

    int frameCount = 0;
    auto startTime = chrono::high_resolution_clock::now();

    // ---- ARUCO SETUP ----
    cv::aruco::Dictionary arucoDictObj =
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);

    cv::Ptr<cv::aruco::Dictionary> arucoDict = 
    cv::makePtr<cv::aruco::Dictionary>(arucoDictObj);




    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;

    /** STANDARD CAMERA INTRINSICS â€” REPLACE WITH CALIBRATED VALUES FOR BEST RESULTS
    cv::Mat cameraMatrix = (cv::Mat1d(3,3) <<
    1000, 0, frame.cols/2,
    0, 1000, frame.rows/2,
    0, 0, 1
    );

    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F); 


    // My calibrated camera intrinsics:
    cv::Mat cameraMatrix = (cv::Mat1d(3,3) <<
        833.77442760678105, 0.0, 311.2045296538675,
        0.0, 837.7048165040502, 186.13605911537414,
        0.0, 0.0, 1.0
    );

    cv::Mat distCoeffs = (cv::Mat1d(5,1) <<
        -0.06023616996408216,
        2.1682565145419943,
        -0.034188391219249624,
        0.0045726338185274121,
        -10.253123727898847
    ); **/


    // ---- CAMERA INTRINSICS ----
    // Default guess (will be overridden by YAML if available)
    cv::Mat cameraMatrix = (cv::Mat1d(3,3) <<
        1000, 0, frame.cols / 2.0,
        0, 1000, frame.rows / 2.0,
        0, 0, 1
    );
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);

    // Try to load calibrated intrinsics.
    // Prefer the camera_calibration.yml that sits next to this executable
    // (e.g. build\Debug\camera_calibration.yml — the exact file CalibrateCam.exe
    // writes when run from build\Debug). Fall back to the current directory so it
    // still works when launched with cwd set to the project folder.
    {
        std::string ymlPath = executableDir() + "camera_calibration.yml";
        cv::FileStorage fs(ymlPath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            ymlPath = "camera_calibration.yml";
            fs.open(ymlPath, cv::FileStorage::READ);
        }
        if (fs.isOpened()) {
            std::cout << "[INFO] Loading " << ymlPath << "\n";
            fs["camera_matrix"] >> cameraMatrix;
            fs["distortion_coefficients"] >> distCoeffs;
            fs.release();
            std::cout << "cameraMatrix =\n" << cameraMatrix << "\n";
            std::cout << "distCoeffs =\n" << distCoeffs << "\n";
            std::cerr << "[DEBUG] Focal length X: " << cameraMatrix.at<double>(0,0) << "\n";
            std::cerr << "[DEBUG] Focal length Y: " << cameraMatrix.at<double>(1,1) << "\n";
            std::cerr << "[DEBUG] Principal point: (" << cameraMatrix.at<double>(0,2) << ", " << cameraMatrix.at<double>(1,2) << ")\n";
            std::cerr << "[DEBUG] Image resolution used: " << frame.cols << "x" << frame.rows << "\n";
        } else {
            std::cout << "[WARN] Could not open camera_calibration.yml, using default intrinsics.\n";
        }
    }





    // main loop
    while (!glfwWindowShouldClose(window)) {
        processInput();

        // If user requested a batch and none is running, run it
        if (batchRequested.exchange(false) && !batchRunning.load()) {
            // run batch in-line 
            runBatchExperiments(cap, videoTexture, quad, scene, cam,
                                defaultShader, pixelateShader, sinCityShader);
            // continue;
        }

        // Grab safe frame
        if (!grabSafeFrame(cap, frame)) {
            // If no valid frame, let events happen and continue
            glfwPollEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            continue;
        }

        // ====== STAGE 1: DETECT MARKER ======
        markerIds.clear();
        markerCorners.clear();
        cv::Ptr<cv::aruco::DetectorParameters> params = cv::makePtr<cv::aruco::DetectorParameters>();
        params->cornerRefinementMethod   = cv::aruco::CORNER_REFINE_SUBPIX;
        params->adaptiveThreshWinSizeMin = 3;
        params->adaptiveThreshWinSizeMax = 23;
        params->adaptiveThreshWinSizeStep= 10;
        params->minMarkerPerimeterRate   = 0.03;
        params->maxMarkerPerimeterRate   = 4.0;
        cv::aruco::detectMarkers(frame, arucoDict, markerCorners, markerIds, params);
        cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
        if (!markerIds.empty()) {
        std::cout << "[ARUCO] Detected " << markerIds.size() << " marker(s). IDs: ";
        for (int id : markerIds) std::cout << id << " ";
        std::cout << std::endl;
    } else {
        std::cout << "[ARUCO] No markers detected\n" << std::flush;
    }

        // ====== STAGE 2: ESTIMATE POSE ======
        std::vector<cv::Vec3d> rvecs, tvecs;
        bool markerDetected = false;

        if (!markerIds.empty()) {
            cv::aruco::estimatePoseSingleMarkers(
                markerCorners,
                0.10f,         // marker size in meters â€” 10cm physical marker
                cameraMatrix,
                distCoeffs,
                rvecs,
                tvecs
            );
            markerDetected = true;
            // Draw axis on the frame for each marker
    for (size_t i = 0; i < markerIds.size(); i++) {
        cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.05f);
    }


        }
        

        // ====== STAGE 3: CONVERT POSE TO GL MODEL MATRIX ======
        glm::mat4 cubeModel(1.0f);
        std::cerr << "[STAGE3] Checking markerDetected\n" << std::flush;

        if (markerDetected) {
            std::cerr << "[STAGE3] INSIDE markerDetected block\n" << std::flush;

            // ================= pose logging =================
            static int poseFrame = 0;
            poseCSV
                << poseFrame++ << ","
                << tvecs[0][0] << ","
                << tvecs[0][1] << ","
                << tvecs[0][2] << ","
                << rvecs[0][0] << ","
                << rvecs[0][1] << ","
                << rvecs[0][2] << "\n";
            // =================================================

            // Rodrigues rotation vector → 3×3 rotation matrix (marker pose in
            // the OpenCV camera frame: X right, Y down, Z forward into scene).
            cv::Mat R;
            cv::Rodrigues(rvecs[0], R);

            // =================================================================
            //  PROPER AR RENDERING
            // -----------------------------------------------------------------
            //  Render the cube with a projection built from the calibrated
            //  intrinsics and a modelview from the SAME rvec/tvec that draw the
            //  (correct) axes, so the cube's perspective/tilt matches the video
            //  background. A fixed per-axis clip-space scale then maps the real
            //  camera's NDC onto the displayed (flipped/mirrored) background.
            // =================================================================
            const float Wf = static_cast<float>(frame.cols);
            const float Hf = static_cast<float>(frame.rows);
            const float A  = 1.777f;   // must match aspectRatio in videoTextureShader.vert

            // --- Pose [R|t] (OpenCV camera frame) as a GLM matrix ---
            glm::mat4 RT(1.0f);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c)
                    RT[c][r] = static_cast<float>(R.at<double>(r, c));
                RT[3][r] = static_cast<float>(tvecs[0][r]);
            }
            // OpenCV camera (X right, Y down, Z forward) → OpenGL camera
            // (X right, Y up, looking down −Z):  multiply by diag(1,−1,−1).
            glm::mat4 G(1.0f);  G[1][1] = -1.0f;  G[2][2] = -1.0f;
            glm::mat4 MV = G * RT;

            // --- OpenGL projection from the calibrated intrinsics ---
            const double fx = cameraMatrix.at<double>(0, 0);
            const double fy = cameraMatrix.at<double>(1, 1);
            const double cx = cameraMatrix.at<double>(0, 2);
            const double cy = cameraMatrix.at<double>(1, 2);
            const float nearP = 0.01f, farP = 100.0f;
            glm::mat4 P(0.0f);
            P[0][0] = 2.0f * static_cast<float>(fx) / Wf;
            P[1][1] = 2.0f * static_cast<float>(fy) / Hf;
            P[2][0] = 1.0f - 2.0f * static_cast<float>(cx) / Wf;
            P[2][1] = 2.0f * static_cast<float>(cy) / Hf - 1.0f;
            P[2][2] = -(farP + nearP) / (farP - nearP);
            P[2][3] = -1.0f;
            P[3][2] = -2.0f * farP * nearP / (farP - nearP);

            // --- Fixed clip-space remap onto the displayed background ---
            // The real camera projects a 3D point to image NDC (nx, ny). The
            // displayed background maps image pixel (px,py) to the world plane
            // point (A*nx, ny, 0) (this is the SAME mapping that already centred
            // the cube), which the scene camera then projects to screen. Because
            // the plane z = 0 has constant view depth, that screen projection is
            // an exact per-axis scale with no offset:
            //     screen_ndc.x = ax * nx ,   screen_ndc.y = ay * ny
            // ax, ay depend only on the (fixed) scene camera + quad, so we compute
            // them once per frame from camVP — robust, no fragile least-squares.
            const glm::mat4 camVP = cam->getViewProjectionMatrix();
            const glm::vec4 ex = camVP * glm::vec4(A, 0.0f, 0.0f, 1.0f);
            const glm::vec4 ey = camVP * glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
            const float ax = ex.x / ex.w;   // nx → screen ndc x
            const float ay = ey.y / ey.w;   // ny → screen ndc y

            // Apply the scale in clip space (perspective preserved); keep the real
            // depth (z) so the cube self-occludes correctly.
            glm::mat4 S(1.0f);
            S[0][0] = ax;
            S[1][1] = ay;
            glm::mat4 P_screen = S * P;

            // --- Cube model in the marker frame: centred, resting on the plane ---
            const float Lm = 0.10f;          // marker side length (metres)
            const float e  = Lm * 0.5f;      // cube edge = half the marker
            glm::mat4 cubeLocal =
                glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, e * 0.5f)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(e));

            cube->setVisible(true);
            cube->setScale(glm::vec3(1.0f));   // scale is baked into cubeLocal
            cube->setAROverride(cubeLocal, MV, P_screen);
        } 
        else {
            cube->setVisible(false);
        }



        if (useGPU) {
            cv::flip(frame, frame, 0);
            videoTexture->update(frame.data, frame.cols, frame.rows, true);

           
            quad->setTranslate(glm::vec3(translateX, translateY, 0.0f));
            quad->setRotate(rotateAngle);
            quad->setScale(scaleFactor);

            
            if (activeFilter == FILTER_PIXELATE) quad->setShader(pixelateShader);
            else if (activeFilter == FILTER_SINCITY) quad->setShader(sinCityShader);
            else quad->setShader(defaultShader);

        } else {
            // CPU path: apply filter then warpAffine transforms
            cv::Mat processed;
            if (activeFilter == FILTER_PIXELATE) CPUFilters::pixelate(frame, processed, 10);
            else if (activeFilter == FILTER_SINCITY) CPUFilters::sinCity(frame, processed);
            else processed = frame.clone();

            cv::Mat rotated;
            cv::Point2f center(processed.cols/2.0f, processed.rows/2.0f);
            cv::Mat M = cv::getRotationMatrix2D(center, rotateAngle, scaleFactor);
            M.at<double>(0,2) += translateX * processed.cols;
            M.at<double>(1,2) -= translateY * processed.rows;
            cv::warpAffine(processed, rotated, M, processed.size());
            cv::flip(rotated, rotated, 0);

            videoTexture->update(rotated.data, rotated.cols, rotated.rows, true);

            // CPU output uses default shader; show transformed image as-is
            quad->setShader(defaultShader);
            // ensure quad identity transform so warped image maps directly
            quad->setTranslate(glm::vec3(0.0f,0.0f,0.0f));
            quad->setRotate(0.0f);
            quad->setScale(1.0f);
        }

        // Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        scene->render(cam);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // FPS logging
        ++frameCount;
        auto now = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(now - startTime).count();
        if (elapsed >= 1.0) {
            double fps = frameCount / elapsed;
            csv << frameCount << "," << (useGPU ? "GPU" : "CPU") << "," << activeFilter << "," << fps << "\n";
            frameCount = 0;
            startTime = now;
            // also print to console for convenience
            cout << "[MAIN] FPS: " << fixed << setprecision(2) << fps
                 << " | Mode: " << (useGPU ? "GPU" : "CPU")
                 << " | Filter: " << (activeFilter==FILTER_NONE ? "NONE" : (activeFilter==FILTER_PIXELATE ? "PIXELATE" : "SINCITY"))
                 << "\n";
        }
    }

    // cleanup
    cap.release();

    delete videoTexture;
    delete quad;
    delete cam;
    delete scene;
    delete defaultShader;
    delete pixelateShader;
    delete sinCityShader;

    glfwTerminate();
    poseCSV.close();
    csv.close();
    return 0;
}
