#include "Cube.hpp"
#include "Shader.hpp"
#include "Camera.hpp"

// Cube vertex data: position (3), normal (3), UV (2)
static float cubeVertices[] = {
    // positions           // normals        // uv
    -0.5f,-0.5f,-0.5f,     0,0,-1,          0,0,
     0.5f,-0.5f,-0.5f,     0,0,-1,          1,0,
     0.5f, 0.5f,-0.5f,     0,0,-1,          1,1,
    -0.5f, 0.5f,-0.5f,     0,0,-1,          0,1,

    -0.5f,-0.5f, 0.5f,     0,0,1,           0,0,
     0.5f,-0.5f, 0.5f,     0,0,1,           1,0,
     0.5f, 0.5f, 0.5f,     0,0,1,           1,1,
    -0.5f, 0.5f, 0.5f,     0,0,1,           0,1,
};

static unsigned int cubeIndices[] = {
    // back
    0,1,2,  2,3,0,
    // front
    4,5,6,  6,7,4,
    // left
    0,3,7,  7,4,0,
    // right
    1,5,6,  6,2,1,
    // bottom
    0,1,5,  5,4,0,
    // top
    3,2,6,  6,7,3
};

Cube::Cube() {

    hasOverride = false;
    visible = true;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // Vertices
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

    // Indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIndices), cubeIndices, GL_STATIC_DRAW);

    // layout(location = 0) position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // layout(location = 1) normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    // layout(location = 2) texcoords (unused but keeps shader-compatible layout)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
}

Cube::~Cube() {
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO);
}

void Cube::setModelMatrix(const glm::mat4& m) {
    baseModelMatrix = m;
    updateTransformFromModel();  // if you want to apply scaling or other transforms
}


void Cube::render(Camera* camera) {
    if (!visible) return;
    if (!shader) return;

    shader->bind();

    // Model matrix: override if pose comes from AR marker
    glm::mat4 model = modelMatrix;

    // Camera matrices
    glm::mat4 view  = camera->getViewMatrix();
    glm::mat4 proj  = camera->getProjectionMatrix();

    // Send all matrices to your shader
    shader->setMat4("model", model);
    shader->setMat4("view",  view);
    shader->setMat4("projection", proj);

    // Draw cube
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // Your Shader class has no unBind(), so we don’t call it.
}

void Cube::updateTransformFromModel() {
    glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), scaleVec);
    modelMatrix = baseModelMatrix * scaleMat;
}
