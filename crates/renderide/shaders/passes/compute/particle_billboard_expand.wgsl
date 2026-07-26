// Expands one point per invocation into four billboard vertices and six indices.
// Unused capacity slots write degenerate geometry to keep the draw range stable.

struct PointInstance {
    // xyz = center position, w = roll
    pos_roll: vec4<f32>,
    // Linear color.
    color: vec4<f32>,
    // xy = size, z = frame index, w = frame-index flag.
    size_frame: vec4<f32>,
    // xyz = forward, w unused.
    forward: vec4<f32>,
    // xyz = up, w unused.
    up: vec4<f32>,
}

struct Params {
    live_count: u32,
    capacity: u32,
    frame_cols: u32,
    frame_rows: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> instances: array<PointInstance>;
// Interleaved vertex stream: 12 f32 per vertex (pos.xyz, point_data.xyz, uv.xy, color.rgba).
@group(0) @binding(2) var<storage, read_write> interleaved: array<f32>;
@group(0) @binding(3) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> uv0: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> colors: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> tangents: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> raw_tangents: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> uv1: array<vec2<f32>>;
@group(0) @binding(10) var<storage, read_write> indices: array<u32>;

const CORNERS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

struct FrameUvState {
    enabled: bool,
    cols: u32,
    rows: u32,
    column: u32,
    row: u32,
}

fn frame_uv_state(frame_index: u32, has_frame: bool) -> FrameUvState {
    let cols = params.frame_cols;
    let rows = params.frame_rows;
    if (!has_frame || cols == 0u || rows == 0u) {
        return FrameUvState(false, cols, rows, 0u, 0u);
    }
    let frame_count = max(cols * rows, 1u);
    let frame = min(frame_index, frame_count - 1u);
    return FrameUvState(
        true,
        cols,
        rows,
        frame % cols,
        rows - 1u - frame / cols,
    );
}

fn frame_uv(corner: vec2<f32>, state: FrameUvState) -> vec2<f32> {
    if (!state.enabled) {
        return corner;
    }
    return vec2<f32>(
        (f32(state.column) + corner.x) / f32(state.cols),
        (f32(state.row) + corner.y) / f32(state.rows),
    );
}

fn write_interleaved(vertex: u32, position: vec3<f32>, point_data: vec3<f32>, uv: vec2<f32>, color: vec4<f32>) {
    let base = vertex * 12u;
    interleaved[base + 0u] = position.x;
    interleaved[base + 1u] = position.y;
    interleaved[base + 2u] = position.z;
    interleaved[base + 3u] = point_data.x;
    interleaved[base + 4u] = point_data.y;
    interleaved[base + 5u] = point_data.z;
    interleaved[base + 6u] = uv.x;
    interleaved[base + 7u] = uv.y;
    interleaved[base + 8u] = color.x;
    interleaved[base + 9u] = color.y;
    interleaved[base + 10u] = color.z;
    interleaved[base + 11u] = color.w;
}

fn write_vertex(
    vertex: u32,
    position: vec3<f32>,
    point_data: vec3<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    forward: vec3<f32>,
    up: vec3<f32>,
) {
    write_interleaved(vertex, position, point_data, uv, color);
    positions[vertex] = vec4<f32>(position, 1.0);
    normals[vertex] = vec4<f32>(point_data, 0.0);
    uv0[vertex] = uv;
    colors[vertex] = color;
    let tangent = vec4<f32>(forward, up.z);
    tangents[vertex] = tangent;
    raw_tangents[vertex] = tangent;
    uv1[vertex] = vec2<f32>(up.x, up.y);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.capacity) {
        return;
    }
    let base_vertex = slot * 4u;
    let base_index = slot * 6u;

    if (slot >= params.live_count) {
        // Degenerate dead slot: zero vertices and a zero-area triangle pair.
        for (var corner = 0u; corner < 4u; corner = corner + 1u) {
            write_vertex(
                base_vertex + corner,
                vec3<f32>(0.0),
                vec3<f32>(0.0),
                vec2<f32>(0.0),
                vec4<f32>(0.0),
                vec3<f32>(0.0, 0.0, 1.0),
                vec3<f32>(0.0, 1.0, 0.0),
            );
        }
        for (var i = 0u; i < 6u; i = i + 1u) {
            indices[base_index + i] = base_vertex;
        }
        return;
    }

    let instance = instances[slot];
    let position = instance.pos_roll.xyz;
    let roll = instance.pos_roll.w;
    let color = instance.color;
    let size = instance.size_frame.xy;
    let frame_index = u32(max(instance.size_frame.z, 0.0));
    let has_frame = instance.size_frame.w > 0.5;
    let uv_state = frame_uv_state(frame_index, has_frame);
    let forward = instance.forward.xyz;
    let up = instance.up.xyz;

    for (var corner = 0u; corner < 4u; corner = corner + 1u) {
        let corner_uv = CORNERS[corner];
        let corner_sign = corner_uv * 2.0 - vec2<f32>(1.0);
        let point_data = vec3<f32>(size.x * 0.5 * corner_sign.x, size.y * 0.5 * corner_sign.y, roll);
        let uv = frame_uv(corner_uv, uv_state);
        write_vertex(base_vertex + corner, position, point_data, uv, color, forward, up);
    }

    indices[base_index + 0u] = base_vertex;
    indices[base_index + 1u] = base_vertex + 1u;
    indices[base_index + 2u] = base_vertex + 2u;
    indices[base_index + 3u] = base_vertex + 2u;
    indices[base_index + 4u] = base_vertex + 1u;
    indices[base_index + 5u] = base_vertex + 3u;
}
