//! Shared clustered-forward math used by compute light assignment and fragment lookup.

#define_import_path renderide::cluster_math

const TILE_SIZE: u32 = 32u;
const MAX_LIGHTS_PER_TILE: u32 = 64u;
const CLUSTER_NEAR_CLIP_MIN: f32 = 0.0001;
const CLUSTER_FAR_CLIP_MIN_SPAN: f32 = 0.0001;
const CLUSTER_BOUNDARY_EPSILON: f32 = 0.00001;

struct ClusterClipPlanes {
    near_clip: f32,
    far_clip: f32,
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    return select(fallback, value, abs(value) <= 3.402823e38);
}

fn sanitize_cluster_clip_planes(near_clip: f32, far_clip: f32) -> ClusterClipPlanes {
    let near_input = finite_or(near_clip, CLUSTER_NEAR_CLIP_MIN);
    let near_safe = max(near_input, CLUSTER_NEAR_CLIP_MIN);
    let far_input = finite_or(far_clip, near_safe + CLUSTER_FAR_CLIP_MIN_SPAN);
    let far_safe = max(far_input, near_safe + CLUSTER_FAR_CLIP_MIN_SPAN);
    return ClusterClipPlanes(near_safe, far_safe);
}

fn cluster_z_from_view_z(view_z: f32, near_clip: f32, far_clip: f32, cluster_count_z: u32) -> u32 {
    let z_count = max(cluster_count_z, 1u);
    let clip = sanitize_cluster_clip_planes(near_clip, far_clip);
    let d = clamp(-view_z, clip.near_clip, clip.far_clip);
    let z = log(d / clip.near_clip) / log(clip.far_clip / clip.near_clip) * f32(z_count);
    return u32(clamp(z, 0.0, f32(z_count - 1u)));
}

fn cluster_z_depth_bounds(
    cluster_z: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
) -> vec2<f32> {
    let z_count = max(cluster_count_z, 1u);
    let z = min(cluster_z, z_count - 1u);
    let clip = sanitize_cluster_clip_planes(near_clip, far_clip);
    let num_z = f32(z_count);
    let zf = f32(z);
    let near_depth = clip.near_clip * pow(clip.far_clip / clip.near_clip, zf / num_z);
    let far_depth = clip.near_clip * pow(clip.far_clip / clip.near_clip, (zf + 1.0) / num_z);
    return vec2<f32>(near_depth, far_depth);
}
