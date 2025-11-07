#version 330 core
layout (location = 0) in vec3 vertexPosition_modelspace;

out vec2 UV;

uniform mat4 MVP;
const float aspectRatio=1.777; // <-- NEW: The video's aspect ratio

void main() {
    gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0);

    // Create a temporary vec2 with the x position "normalized"
    // by dividing it by the aspect ratio. This maps the
    // stretched x-coordinate back to a -1.0 to 1.0 range.
    vec2 normalized_pos = vec2(vertexPosition_modelspace.x / aspectRatio, vertexPosition_modelspace.y);

    // Now, calculate the UVs using this corrected position.
    UV = normalized_pos * 0.5 + 0.5;
}