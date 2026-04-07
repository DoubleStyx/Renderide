// Linear blend skinning (compute). Bind buffers expected to match layout produced by mesh preprocess.
// Bone palette entries are world_bone * unity_bindpose (inverse bind matrix per bone), built on CPU each frame.

@group(0) @binding(0) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read> src_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bone_idx: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bone_weights: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> dst_pos: array<vec4<f32>>;

@compute @workgroup_size(64)
fn skin_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&src_pos);
    if (i >= n) {
        return;
    }
    let p = src_pos[i];
    let idx = bone_idx[i];
    let w = bone_weights[i];
    let p4 = vec4<f32>(p.xyz, 1.0);
    var acc = vec4<f32>(0.0);
    acc += w.x * (bone_matrices[idx.x] * p4);
    acc += w.y * (bone_matrices[idx.y] * p4);
    acc += w.z * (bone_matrices[idx.z] * p4);
    acc += w.w * (bone_matrices[idx.w] * p4);
    let ws = w.x + w.y + w.z + w.w;
    if (ws > 1e-6) {
        dst_pos[i] = vec4<f32>((acc / ws).xyz, p.w);
    } else {
        dst_pos[i] = p;
    }
}
