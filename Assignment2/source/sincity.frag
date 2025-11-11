#version 330 core
in vec2 UV;
out vec4 color;
uniform sampler2D videoTexture;
uniform mat4 MVP;

void main() {
    vec4 texColor = texture(videoTexture, UV);
    float gray = dot(texColor.rgb, vec3(0.299, 0.587, 0.114));
    vec3 result = vec3(gray);
    if (texColor.r > 0.6 && texColor.r > texColor.g * 1.3 && texColor.r > texColor.b * 1.3)
        result = texColor.rgb;
    color = vec4(result, 1.0);
}
