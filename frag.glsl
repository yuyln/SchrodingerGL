#version 330 core
out vec4 Fcolor;

in vec2 f_uv;
uniform sampler2D tex;

void main() {
    Fcolor = texture(tex, f_uv);
}
