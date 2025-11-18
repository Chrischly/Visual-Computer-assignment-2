#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <iomanip>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <glad/gl.h>
#define GLAD_GL_IMPLEMENTATION
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
        cerr << "[WARN] Camera warmup failed to get frames quickly — continuing anyway\n";
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

    // Interactive FPS logging CSV
    std::ofstream csv("fps_log.csv", ios::app);
    if (csv.tellp() == 0) csv << "Frame,Backend,Filter,FPS\n";

    int frameCount = 0;
    auto startTime = chrono::high_resolution_clock::now();

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
    csv.close();
    return 0;
}
