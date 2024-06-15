#version 460
#extension GL_EXT_buffer_reference : require
// #extension GL_EXT_buffer_reference_uvec2 : require

layout(std430, buffer_reference,
       buffer_reference_align = 8) readonly buffer Transform {
  mat2 transform;
};

layout(push_constant) uniform _ { Transform tr_ptr; }
pc;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_color;

const vec3 colors[3] = vec3[3](vec3(1.0f, 0.0f, 0.0f), // red
                               vec3(0.0f, 1.0f, 0.0f), // green
                               vec3(00.f, 0.0f, 1.0f)  // blue
);

void main() {
  const vec2 positions[3] =
      vec2[3](vec2(0.25, 0.25), vec2(-0.25, 0.25), vec2(0., -0.25));

  mat2 trans = pc.tr_ptr.transform;
  gl_Position = vec4(trans * positions[gl_VertexIndex], 0., 1.0);
  out_color = colors[gl_VertexIndex];
}

// void junk() {
//   // const vec3 positions[3] = vec3[3](vec3(1.f, 1.f, 0.0f), vec3(-1.f, 1.f,
//   // 0.0f),
//   //                                   vec3(0.f, -1.f, 0.0f));
//   const vec3 positions[3] =
//       vec3[3](vec3(0.25f, 0.25f, 0.0f), vec3(-0.25f, 0.25f, 0.0f),
//               vec3(0.f, -0.25f, 0.0f));
//
//   const vec3 colors[3] = vec3[3](vec3(1.0f, 0.0f, 0.0f), // red
//                                  vec3(0.0f, 1.0f, 0.0f), // green
//                                  vec3(00.f, 0.0f, 1.0f)  // blue
//   );
//
//   mat2 trans = pc.transform.tr;
//   vec3 pos = positions[gl_VertexIndex];
//   pos.xy *= trans;
//
//   // output the position of each vertex
//   gl_Position = vec4(pos, 1.0f);
//   out_color = colors[gl_VertexIndex];
//
//   // out_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
//   // gl_Position = vec4(out_uv * 2.0f + -1.0f, 0.0, 1.0);
// }
