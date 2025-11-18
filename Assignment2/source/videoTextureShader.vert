#version 330 core
layout (location = 0) in vec3 vertexPosition_modelspace;

out vec2 UV;

// MVP here will be used to simulate translation/rotation/scale in clip space
uniform mat4 MVP;
const float aspectRatio = 1.777;

void main() {
    // Apply MVP directly — this lets GPU do transformations
    gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0);

    vec2 normalized_pos = vec2(vertexPosition_modelspace.x / aspectRatio, vertexPosition_modelspace.y);
    UV = normalized_pos * 0.5 + 0.5;

}
