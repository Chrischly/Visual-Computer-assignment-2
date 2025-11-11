#version 330 core
in vec2 UV;
out vec3 color;

uniform sampler2D myTextureSampler;
uniform float pixelSize = 0.02; // adjust for pixelation strength

void main() {
    vec2 uv = floor(UV / pixelSize) * pixelSize;
    color = texture(myTextureSampler, uv).rgb;
}
