#include <iostream>
#include <fstream>
#include <chrono>
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

// --- Globals ---
GLFWwindow* window;
float rotateAngle = 0.0f;
float translateX = 0.0f, translateY = 0.0f;
float scaleFactor = 1.0f;
bool mousePressed = false;
double lastX = 0.0, lastY = 0.0;

enum FilterType { FILTER_NONE, FILTER_PIXELATE, FILTER_SINCITY };
FilterType filter = FILTER_NONE;
bool useGPU = true;
glm::mat4 MVP = glm::mat4(1.0f);

// --- Init window + input callbacks ---
bool initWindow(const std::string& name) {
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1024, 768, name.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);

    // --- Mouse input ---
    glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
            mousePressed = (action == GLFW_PRESS);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* win, double xpos, double ypos) {
        if (mousePressed) {
            float dx = static_cast<float>(xpos - lastX) / 500.0f;
            float dy = static_cast<float>(ypos - lastY) / 500.0f;
            translateX += dx;
            translateY -= dy; // invert Y for intuitive drag
        }
        lastX = xpos;
        lastY = ypos;
    });

    glfwSetScrollCallback(window, [](GLFWwindow* win, double xoffset, double yoffset) {
        scaleFactor *= (1.0f + 0.1f * static_cast<float>(yoffset));
        if (scaleFactor < 0.1f) scaleFactor = 0.1f;
    });

    return true;
}

// --- Keyboard control ---
void processInput() {
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) translateY += 0.01f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) translateY -= 0.01f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) translateX -= 0.01f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) translateX += 0.01f;

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) rotateAngle -= 1.0f;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) rotateAngle += 1.0f;

    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) scaleFactor *= 1.01f;
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) scaleFactor *= 0.99f;

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) filter = FILTER_NONE;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) filter = FILTER_PIXELATE;
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) filter = FILTER_SINCITY;

    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) useGPU = true;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) useGPU = false;
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    if (!initWindow("Video Processing")) return -1;
    if (!gladLoadGL(glfwGetProcAddress)) return -1;

    glEnable(GL_DEPTH_TEST);
    GLuint VAO; glGenVertexArrays(1, &VAO); glBindVertexArray(VAO);

    // First frame
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) return -1;
    cv::flip(frame, frame, 0);

    Texture* videoTexture = new Texture(frame.data, frame.cols, frame.rows, true);

    // Shaders
    TextureShader* defaultShader = new TextureShader("videoTextureShader.vert", "videoTextureShader.frag");
    TextureShader* pixelateShader = new TextureShader("videoTextureShader.vert", "pixelate.frag");
    TextureShader* sinCityShader  = new TextureShader("videoTextureShader.vert", "sincity.frag");

    defaultShader->setTexture(videoTexture);
    pixelateShader->setTexture(videoTexture);
    sinCityShader->setTexture(videoTexture);

    Scene* scene = new Scene();
    Camera* cam = new Camera();
    cam->setPosition(glm::vec3(0, 0, -2.5));
    Quad* quad = new Quad((float)frame.cols / frame.rows);
    quad->setShader(defaultShader);
    scene->addObject(quad);

    // Logging
    std::ofstream csv("fps_log.csv");
    csv << "Frame,Backend,Filter,FPS\n";
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    // --- Main loop ---
    while (!glfwWindowShouldClose(window)) {
        processInput();

        cap >> frame;
        if (frame.empty()) break;

        if (useGPU) {
            // --- GPU mode ---
            cv::flip(frame, frame, 0);
            videoTexture->update(frame.data, frame.cols, frame.rows, true);

            // Select shader
            if (filter == FILTER_PIXELATE) quad->setShader(pixelateShader);
            else if (filter == FILTER_SINCITY) quad->setShader(sinCityShader);
            else quad->setShader(defaultShader);

            // Apply transformations (MVP)
            MVP = glm::mat4(1.0f);
            MVP = glm::translate(MVP, glm::vec3(translateX, translateY, 0.0f));
            MVP = glm::rotate(MVP, glm::radians(rotateAngle), glm::vec3(0, 0, 1));
            MVP = glm::scale(MVP, glm::vec3(scaleFactor, scaleFactor, 1.0f));

            quad->getShader()->SetMVP(MVP);

        } else {
            // --- CPU mode ---
            cv::Mat rotated;
            cv::Point2f center(frame.cols / 2.0f, frame.rows / 2.0f);
            cv::Mat M = cv::getRotationMatrix2D(center, rotateAngle, scaleFactor);
            M.at<double>(0, 2) += translateX * frame.cols;
            M.at<double>(1, 2) -= translateY * frame.rows;
            cv::warpAffine(frame, rotated, M, frame.size());
            cv::flip(rotated, rotated, 0);
            videoTexture->update(rotated.data, rotated.cols, rotated.rows, true);
            quad->setShader(defaultShader);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        scene->render(cam);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // FPS logging
        frameCount++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - startTime).count();
        if (elapsed >= 1.0) {
            double fps = frameCount / elapsed;
            csv << frameCount << "," << (useGPU ? "GPU" : "CPU") << "," << filter << "," << fps << "\n";
            frameCount = 0;
            startTime = now;
        }
    }

    // Cleanup
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
