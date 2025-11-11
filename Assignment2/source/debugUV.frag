#version 330 core
in vec2 UV;
out vec4 FragColor;
void main() {
    // visualize UVs: x->r, y->g
    FragColor = vec4(UV.xy, 0.0, 1.0);
}
