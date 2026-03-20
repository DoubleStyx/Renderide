//! WGSL shader sources for debug visualization pipelines (normal debug, UV debug, MRT variants).

/// Normal debug shader: colors surfaces by smooth world normal (single color target).
pub(crate) const NORMAL_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.world_normal = (u.model * vec4f(in.normal, 0.0)).xyz;
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let n = normalize(in.world_normal);
    return vec4f(n * 0.5 + 0.5, 1.0);
}
"#;

/// UV debug shader: colors surfaces using HSV hue-saturation from UV coordinates (single color target).
pub(crate) const UV_DEBUG_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    return vec4f(rgb, 1.0);
}
"#;

/// Normal debug MRT shader: outputs color, camera-relative world position, and world normal.
///
/// MRT layout: `@location(0)` = color, `@location(1)` = position, `@location(2)` = normal.
/// Position is stored as `world - view_position` (camera-relative) for `Rgba16Float` precision.
pub(crate) const NORMAL_DEBUG_MRT_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
struct MrtGbufferFrame {
    view_position: vec3f,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> mrt_frame: MrtGbufferFrame;
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    let world_pos = u.model * vec4f(in.position, 1.0);
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.world_normal = (u.model * vec4f(in.normal, 0.0)).xyz;
    out.world_position = world_pos.xyz;
    return out;
}
struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let n = normalize(in.world_normal);
    let rel = in.world_position - mrt_frame.view_position;
    return FragmentOutput(
        vec4f(n * 0.5 + 0.5, 1.0),
        vec4f(rel, 1.0),
        vec4f(n, 0.0),
    );
}
"#;

/// UV debug MRT shader: outputs color (HSV from UV), camera-relative world position, world normal.
///
/// UV meshes lack per-vertex normals; uses model +Y as fallback normal.
/// MRT layout matches [`NORMAL_DEBUG_MRT_SHADER_SRC`].
pub(crate) const UV_DEBUG_MRT_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) world_position: vec3f,
    @location(2) world_normal: vec3f,
}
struct UniformsSlot {
    mvp: mat4x4f,
    model: mat4x4f,
    _pad: array<vec4f, 8>,
}
struct MrtGbufferFrame {
    view_position: vec3f,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> uniforms: array<UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> mrt_frame: MrtGbufferFrame;
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    let world_pos = u.model * vec4f(in.position, 1.0);
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    out.world_position = world_pos.xyz;
    out.world_normal = (u.model * vec4f(0.0, 1.0, 0.0, 0.0)).xyz;
    return out;
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
struct UvFragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> UvFragmentOutput {
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    let n = normalize(in.world_normal);
    let rel = in.world_position - mrt_frame.view_position;
    return UvFragmentOutput(
        vec4f(rgb, 1.0),
        vec4f(rel, 1.0),
        vec4f(n, 0.0),
    );
}
"#;
