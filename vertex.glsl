#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_uv;

out vec2 f_uv;

void main()
{
    gl_Position = vec4(a_pos.xyz, 1.0);
	f_uv = a_uv;
}