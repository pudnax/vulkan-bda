#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_colot;

layout(location = 0) out vec4 out_color;

void main() {
  out_color = vec4(in_colot, 1.0);
  // out_color = vec4(vec3(1., 0., 0.), 1.);
}
