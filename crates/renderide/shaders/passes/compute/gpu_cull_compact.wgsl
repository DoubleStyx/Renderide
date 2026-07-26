// Conservative arena draw culling and indexed-indirect command compaction.

const CANDIDATE_HIZ_ELIGIBLE: u32 = 1u;
const CANDIDATE_ALWAYS_VISIBLE: u32 = 2u;
const PARAM_FIXED_SLOTS: u32 = 1u;
const PARAM_HIZ_ENABLED: u32 = 2u;
const CLIP_W_EPSILON: f32 = 0.00001;

struct IndexedIndirectCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct GpuCullCandidate {
    command: IndexedIndirectCommand,
    run_index: u32,
    matrix_index: u32,
    flags: u32,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}

struct GpuCullRun {
    candidate_start: u32,
    candidate_count: u32,
    output_start: u32,
    output_capacity: u32,
    count_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuCullMatrix {
    current_frustum_planes: array<array<vec4<f32>, 6>, 2>,
    previous_view_proj: array<mat4x4<f32>, 2>,
    eye_count: u32,
    previous_valid_mask: u32,
    _pad0: u32,
    _pad1: u32,
}

struct GpuCullParams {
    candidate_count: u32,
    run_count: u32,
    matrix_count: u32,
    output_capacity: u32,
    count_capacity: u32,
    flags: u32,
    hiz_eye_mask: u32,
    hiz_depth_bias: f32,
}

struct ProjectedBounds {
    valid: u32,
    uv_min: vec2<f32>,
    uv_max: vec2<f32>,
    nearest_depth: f32,
}

@group(0) @binding(0)
var<uniform> params: GpuCullParams;
@group(0) @binding(1)
var<storage, read> candidates: array<GpuCullCandidate>;
@group(0) @binding(2)
var<storage, read> runs: array<GpuCullRun>;
@group(0) @binding(3)
var<storage, read> matrices: array<GpuCullMatrix>;
@group(0) @binding(4)
var<storage, read_write> output_commands: array<IndexedIndirectCommand>;
@group(0) @binding(5)
var<storage, read_write> visible_counts: array<atomic<u32>>;
@group(0) @binding(6)
var previous_hiz_left: texture_2d<f32>;
@group(0) @binding(7)
var previous_hiz_right: texture_2d<f32>;

fn aabb_intersects_frustum(
    candidate: GpuCullCandidate,
    matrix: GpuCullMatrix,
    eye: u32,
) -> bool {
    let center = (candidate.bounds_min.xyz + candidate.bounds_max.xyz) * 0.5;
    let extents = (candidate.bounds_max.xyz - candidate.bounds_min.xyz) * 0.5;
    for (var plane_index = 0u; plane_index < 6u; plane_index += 1u) {
        let plane = matrix.current_frustum_planes[eye][plane_index];
        let distance = dot(plane.xyz, center) + plane.w;
        let radius = dot(abs(plane.xyz), extents);
        if distance + radius < 0.0 {
            return false;
        }
    }
    return true;
}

fn project_previous_bounds(
    candidate: GpuCullCandidate,
    previous_local_to_clip: mat4x4<f32>,
) -> ProjectedBounds {
    var uv_min = vec2<f32>(1.0);
    var uv_max = vec2<f32>(0.0);
    var nearest_depth = 0.0;

    for (var corner_index = 0u; corner_index < 8u; corner_index += 1u) {
        let local_position = vec3<f32>(
            select(candidate.bounds_min.x, candidate.bounds_max.x, (corner_index & 1u) != 0u),
            select(candidate.bounds_min.y, candidate.bounds_max.y, (corner_index & 2u) != 0u),
            select(candidate.bounds_min.z, candidate.bounds_max.z, (corner_index & 4u) != 0u),
        );
        let clip = previous_local_to_clip * vec4<f32>(local_position, 1.0);
        if clip.w <= CLIP_W_EPSILON {
            return ProjectedBounds(0u, vec2<f32>(0.0), vec2<f32>(0.0), 0.0);
        }
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        uv_min = min(uv_min, uv);
        uv_max = max(uv_max, uv);
        nearest_depth = max(nearest_depth, ndc.z);
    }

    // Off-screen or depth-clipped history is a disocclusion risk, so keep the draw.
    if any(uv_min < vec2<f32>(0.0))
        || any(uv_max > vec2<f32>(1.0))
        || nearest_depth < 0.0
        || nearest_depth > 1.0
    {
        return ProjectedBounds(0u, vec2<f32>(0.0), vec2<f32>(0.0), 0.0);
    }
    return ProjectedBounds(1u, uv_min, uv_max, nearest_depth);
}

fn select_hiz_mip(projected: ProjectedBounds, dimensions: vec2<u32>, levels: u32) -> u32 {
    let pixel_extent =
        (projected.uv_max - projected.uv_min) * vec2<f32>(vec2<u32>(dimensions));
    let largest_extent = max(max(pixel_extent.x, pixel_extent.y), 1.0);
    // Round up so the four endpoint taps cover the complete footprint.
    let requested = u32(ceil(log2(largest_extent)));
    return min(requested, levels - 1u);
}

fn uv_to_texel(uv: vec2<f32>, dimensions: vec2<u32>) -> vec2<i32> {
    let dimensions_i = vec2<i32>(dimensions);
    return clamp(
        vec2<i32>(floor(uv * vec2<f32>(dimensions))),
        vec2<i32>(0),
        dimensions_i - vec2<i32>(1),
    );
}

fn occluded_left(projected: ProjectedBounds) -> bool {
    if projected.valid == 0u {
        return false;
    }
    let base_dimensions = textureDimensions(previous_hiz_left, 0);
    let levels = textureNumLevels(previous_hiz_left);
    let mip = select_hiz_mip(projected, base_dimensions, levels);
    let dimensions = textureDimensions(previous_hiz_left, i32(mip));
    let p00 = uv_to_texel(projected.uv_min, dimensions);
    let p11 = uv_to_texel(projected.uv_max, dimensions);
    let p10 = vec2<i32>(p11.x, p00.y);
    let p01 = vec2<i32>(p00.x, p11.y);
    let conservative_depth = min(
        min(
            textureLoad(previous_hiz_left, p00, i32(mip)).x,
            textureLoad(previous_hiz_left, p10, i32(mip)).x,
        ),
        min(
            textureLoad(previous_hiz_left, p01, i32(mip)).x,
            textureLoad(previous_hiz_left, p11, i32(mip)).x,
        ),
    );
    return projected.nearest_depth <= conservative_depth - params.hiz_depth_bias;
}

fn occluded_right(projected: ProjectedBounds) -> bool {
    if projected.valid == 0u {
        return false;
    }
    let base_dimensions = textureDimensions(previous_hiz_right, 0);
    let levels = textureNumLevels(previous_hiz_right);
    let mip = select_hiz_mip(projected, base_dimensions, levels);
    let dimensions = textureDimensions(previous_hiz_right, i32(mip));
    let p00 = uv_to_texel(projected.uv_min, dimensions);
    let p11 = uv_to_texel(projected.uv_max, dimensions);
    let p10 = vec2<i32>(p11.x, p00.y);
    let p01 = vec2<i32>(p00.x, p11.y);
    let conservative_depth = min(
        min(
            textureLoad(previous_hiz_right, p00, i32(mip)).x,
            textureLoad(previous_hiz_right, p10, i32(mip)).x,
        ),
        min(
            textureLoad(previous_hiz_right, p01, i32(mip)).x,
            textureLoad(previous_hiz_right, p11, i32(mip)).x,
        ),
    );
    return projected.nearest_depth <= conservative_depth - params.hiz_depth_bias;
}

fn candidate_visible(candidate: GpuCullCandidate, matrix: GpuCullMatrix) -> bool {
    if (candidate.flags & CANDIDATE_ALWAYS_VISIBLE) != 0u {
        return true;
    }

    for (var eye = 0u; eye < min(matrix.eye_count, 2u); eye += 1u) {
        if !aabb_intersects_frustum(candidate, matrix, eye) {
            continue;
        }

        let eye_bit = 1u << eye;
        let history_valid =
            (params.flags & PARAM_HIZ_ENABLED) != 0u
            && (candidate.flags & CANDIDATE_HIZ_ELIGIBLE) != 0u
            && (matrix.previous_valid_mask & eye_bit) != 0u
            && (params.hiz_eye_mask & eye_bit) != 0u;
        if !history_valid {
            return true;
        }

        let projected =
            project_previous_bounds(candidate, matrix.previous_view_proj[eye]);
        var occluded = false;
        if eye == 0u {
            occluded = occluded_left(projected);
        } else {
            occluded = occluded_right(projected);
        }
        if !occluded {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let candidate_index = global_id.x;
    if candidate_index >= params.candidate_count {
        return;
    }

    let candidate = candidates[candidate_index];
    if candidate.run_index >= params.run_count || candidate.matrix_index >= params.matrix_count {
        return;
    }
    let run = runs[candidate.run_index];
    if candidate_index < run.candidate_start
        || candidate_index >= run.candidate_start + run.candidate_count
    {
        return;
    }

    let visible = candidate_visible(candidate, matrices[candidate.matrix_index]);
    if (params.flags & PARAM_FIXED_SLOTS) != 0u {
        let run_slot = candidate_index - run.candidate_start;
        let output_index = run.output_start + run_slot;
        if output_index >= params.output_capacity || run_slot >= run.output_capacity {
            return;
        }
        var command = candidate.command;
        if !visible {
            command.instance_count = 0u;
        }
        output_commands[output_index] = command;
        return;
    }

    if !visible || run.count_index >= params.count_capacity {
        return;
    }
    let run_slot = atomicAdd(&visible_counts[run.count_index], 1u);
    let output_index = run.output_start + run_slot;
    if run_slot < run.output_capacity && output_index < params.output_capacity {
        output_commands[output_index] = candidate.command;
    }
}
