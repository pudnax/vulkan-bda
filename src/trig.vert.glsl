#version 460
#extension GL_EXT_buffer_reference : require
// #extension GL_EXT_buffer_reference_uvec2 : require

layout(std430, buffer_reference,
       buffer_reference_align = 8) readonly buffer Transform {
  mat2 transform;
};

layout(push_constant) uniform _ { Transform tr_ptr; }
pc;

layout(std430, set = 0, binding = 0) buffer Colors { vec4[3] col; }
colors;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_color;

const vec3 gcolors[3] = vec3[3](vec3(1.0f, 0.0f, 0.0f), // red
                                vec3(0.0f, 1.0f, 0.0f), // green
                                vec3(00.f, 0.0f, 1.0f)  // blue
);

void main() {
  const vec2 positions[3] =
      vec2[3](vec2(0.25, 0.25), vec2(-0.25, 0.25), vec2(0., -0.25));

  mat2 trans = pc.tr_ptr.transform;
  gl_Position = vec4(trans * positions[gl_VertexIndex], 0., 1.0);
  out_color = colors.col[gl_VertexIndex].rgb;
}
