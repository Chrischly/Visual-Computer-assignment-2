#ifndef CUBE_HPP
#define CUBE_HPP

#include "Object.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glad/gl.h>
#include <GLFW/glfw3.h>


class Cube : public Object {
public:
    Cube();
    virtual ~Cube();

    // Rendering
    void render(Camera* camera) override;

    // AR pose override
    void setModelMatrix(const glm::mat4& m);

    // Full AR override: render the cube with its own model/view/projection,
    // built from the real camera intrinsics + marker pose, instead of the
    // scene's generic virtual camera. This is what makes the cube perspective
    // match the perspective baked into the video background.
    void setAROverride(const glm::mat4& model, const glm::mat4& view, const glm::mat4& proj) {
        baseModelMatrix = model;
        overrideView = view;
        overrideProjection = proj;
        hasOverride = true;
        updateTransformFromModel();
    }
    void clearAROverride() { hasOverride = false; }
    
    // Visibility
    void setVisible(bool v) { visible = v; }
    bool isVisible() const { return visible; }

    // Scaling convenience
    void setScale(const glm::vec3& s) { scaleVec = s; updateTransformFromModel(); }
    void setScale(float s) { scaleVec = glm::vec3(s); updateTransformFromModel(); }

private:
    // GL buffers
    GLuint VAO = 0, VBO = 0, EBO = 0;

    // State
    bool visible = true;

    // AR override
    bool hasOverride = false;
    glm::mat4 overrideModel = glm::mat4(1.0f);
    glm::mat4 overrideView = glm::mat4(1.0f);
    glm::mat4 overrideProjection = glm::mat4(1.0f);

    // Extra from old header
    glm::mat4 baseModelMatrix = glm::mat4(1.0f);
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    glm::vec3 scaleVec = glm::vec3(1.0f);

    void updateTransformFromModel();
    void initGL();
};

#endif
