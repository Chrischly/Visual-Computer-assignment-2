#include "Quad.hpp"

// Default constructor: creates a 1:1 aspect ratio quad
Quad::Quad(){
    init(1.0f); // Default to a square
};

// Overloaded constructor that takes an aspect ratio
Quad::Quad(float aspectRatio){
    init(aspectRatio);
};

Quad::~Quad(){
    // Cleanup VBO + VAO
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteVertexArrays(1, &vao);
};

// init takes an aspect ratio to define the quad's shape
void Quad::init(float aspectRatio){
    float width = aspectRatio;
    float height = 1.0f;

    // Define 6 vertices for the two triangles that make up the quad
    g_vertex_buffer_data[0] = -width; g_vertex_buffer_data[1] = -height; g_vertex_buffer_data[2] = 0.0f;
    g_vertex_buffer_data[3] =  width; g_vertex_buffer_data[4] = -height; g_vertex_buffer_data[5] = 0.0f;
    g_vertex_buffer_data[6] = -width; g_vertex_buffer_data[7] =  height; g_vertex_buffer_data[8] = 0.0f;

    g_vertex_buffer_data[9]  = -width; g_vertex_buffer_data[10] =  height; g_vertex_buffer_data[11] = 0.0f;
    g_vertex_buffer_data[12] =  width; g_vertex_buffer_data[13] = -height; g_vertex_buffer_data[14] = 0.0f;
    g_vertex_buffer_data[15] =  width; g_vertex_buffer_data[16] =  height; g_vertex_buffer_data[17] = 0.0f;

    // Create VBO
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    // Create VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    // unbind VAO
    glBindVertexArray(0);
}

// render() uses the VAO now
void Quad::render(Camera* camera){
    bindShaders();
    glm::mat4 ModelMatrix = this->getTransform();
    glm::mat4 MVP = camera->getViewProjectionMatrix() * ModelMatrix;
    shader->updateMVP(MVP);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

// directRender() also uses the VAO
void Quad::directRender(){
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}
