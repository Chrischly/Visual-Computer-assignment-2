#version 330 core

in vec3 vNormal;
out vec4 fragColor;

void main() {
    float lighting = max(dot(normalize(vNormal), normalize(vec3(0,0,1))), 0.2);
    fragColor = vec4(lighting, lighting*0.5, lighting*0.2, 1.0);
}
