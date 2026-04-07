
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec3f,
    @location(3) bone_indices: vec4i,
    @location(4) bone_weights: vec4f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
}
struct SkinnedUniforms {
    mvp: mat4x4f,
    bone_matrices: array<mat4x4f, 256>,
    num_blendshapes: u32,
    num_vertices: u32,
    blendshape_weights: array<vec4f, 32>,
}
struct BlendshapeOffset {
    position_offset: vec3f,
    normal_offset: vec3f,
    tangent_offset: vec3f,
}
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
@vertex
fn vs_main(
    in: VertexInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position;
    var norm = in.normal;
    var tang = in.tangent;
    for (var i = 0u; i < uniforms.num_blendshapes; i++) {
        let q = i / 4u;
        let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset_idx = i * uniforms.num_vertices + vertex_index;
            let offset = blendshape_offsets[offset_idx];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_normal = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_tangent = vec4f(0.0, 0.0, 0.0, 0.0);
    let total_weight = in.bone_weights[0] + in.bone_weights[1] + in.bone_weights[2] + in.bone_weights[3];
    let inv_total = select(1.0, 1.0 / total_weight, total_weight > 1e-6);
    for (var i = 0; i < 4; i++) {
        let idx = clamp(in.bone_indices[i], 0, 255);
        let w = in.bone_weights[i] * inv_total;
        if w > 0.0 {
            let bone = uniforms.bone_matrices[idx];
            world_pos += w * bone * vec4f(pos, 1.0);
            world_normal += w * bone * vec4f(norm, 0.0);
            world_tangent += w * bone * vec4f(tang, 0.0);
        }
    }
    _ = world_tangent;
    out.clip_position = uniforms.mvp * world_pos;
    let n = world_normal.xyz;
    let len = length(n);
    out.world_normal = select(vec3f(0.0, 1.0, 0.0), n / len, len > 1e-6);
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let n = normalize(in.world_normal);
    return vec4f(n * 0.5 + 0.5, 1.0);
}
