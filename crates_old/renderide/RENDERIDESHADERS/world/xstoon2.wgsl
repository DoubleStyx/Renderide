// WGSL equivalent for third_party/Resonite.UnityShaders/.../Xiexes/Main/Shaders/XSToon2.0.shader
// Implements cel-shading / toon rendering: albedo * toon-ramp diffuse + rim light + specular.
//
// Property-name parity target:
// _MainTex _Color _Cutoff _Ramp _RimColor _RimIntensity _RimRange
// _Glossiness _SpecularIntensity _Saturation _BumpMap

#import renderide_uniform_ring

struct XsToon2MaterialUniform {
    /// _Color tint applied to _MainTex
    _Color: vec4f,
    /// _MainTex tiling + offset (scale.xy, offset.zw)
    _MainTex_ST: vec4f,
    /// _RimColor (rgb) – rim light colour
    _RimColor: vec4f,
    /// _Cutoff, _RimIntensity, _RimRange, _Glossiness
    _Cutoff: f32,
    _RimIntensity: f32,
    _RimRange: f32,
    _Glossiness: f32,
    /// _SpecularIntensity, _Saturation, padding
    _SpecularIntensity: f32,
    _Saturation: f32,
    _Pad: vec2f,
}

@group(0) @binding(0) var<uniform> uniforms: array<renderide_uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var<uniform> material: XsToon2MaterialUniform;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
// Toon ramp: horizontal axis = N·L remapped to 0..1, vertical = centre row.
// White → fully lit; dark → shadowed.  Use a gradient for smooth toon or a stepped
// image for hard cel shading.
@group(1) @binding(3) var _Ramp: texture_2d<f32>;
@group(1) @binding(4) var _Ramp_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) world_normal: vec3f,
    @location(2) world_position: vec3f,
    @location(3) @interpolate(flat) uniform_slot: u32,
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    let world_pos = u.model * vec4f(in.position, 1.0);
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv * material._MainTex_ST.xy + material._MainTex_ST.zw;
    out.world_normal = normalize((u.model * vec4f(in.normal, 0.0)).xyz);
    out.world_position = world_pos.xyz;
    out.uniform_slot = instance_index;
    return out;
}

// --- Lighting helpers --------------------------------------------------------

/// Primary diffuse direction: blend of a sky (Y-up) and a fixed key light.
/// This is a lightweight approximation: ~0.5 from above, ~0.5 from the front.
fn primary_light_direction() -> vec3f {
    return normalize(vec3f(0.5, 1.0, 0.5));
}

/// Fake view direction approximating a camera at a large distance on the Z axis.
/// Sufficient for rim computation when camera position is unavailable.
fn fake_view_dir(world_pos: vec3f) -> vec3f {
    // Slight top-down angle so the rim is visible on avatars.
    return normalize(vec3f(0.0, 0.2, 1.0));
}

// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Sample base texture and apply colour tint.
    let tex_col = textureSample(_MainTex, _MainTex_sampler, in.uv);
    let albedo = tex_col * material._Color;

    // Alpha cutout.
    if (albedo.a < material._Cutoff) {
        discard;
    }

    let n = normalize(in.world_normal);

    // --- Saturation ----------------------------------------------------------
    let gray_weight = vec3f(0.2126, 0.7152, 0.0722);
    let luminance = dot(albedo.rgb, gray_weight);
    let saturated = mix(vec3f(luminance), albedo.rgb, material._Saturation);

    // --- Toon / Ramp diffuse -------------------------------------------------
    let light_dir = primary_light_direction();
    let ndl = dot(n, light_dir);                       // -1..1
    let ramp_u = clamp(ndl * 0.5 + 0.5, 0.001, 0.999); // 0..1, avoid edge clamping
    let ramp = textureSample(_Ramp, _Ramp_sampler, vec2f(ramp_u, 0.5)).rgb;

    // Toon diffuse = albedo * ramp.
    var col = saturated * ramp;

    // --- Specular (Blinn-Phong) ----------------------------------------------
    let view_dir = fake_view_dir(in.world_position);
    let h = normalize(light_dir + view_dir);
    let ndh = max(dot(n, h), 0.0);
    let shininess = exp2(material._Glossiness * 10.0 + 1.0);
    let spec = pow(ndh, shininess) * material._SpecularIntensity;
    col += vec3f(spec);

    // --- Rim lighting --------------------------------------------------------
    let vdn = max(dot(view_dir, n), 0.0);
    // rim_range controls the spread: high value = thin rim, low value = wide.
    let safe_range = max(material._RimRange, 0.001);
    let rim_factor = pow(clamp(1.0 - vdn, 0.0, 1.0), 1.0 / safe_range);
    col += material._RimColor.rgb * rim_factor * material._RimIntensity;

    // --- Ambient floor -------------------------------------------------------
    // Prevent fully black shadows.
    let ambient = vec3f(0.04) * saturated;
    col = max(col, ambient);

    return vec4f(col, albedo.a);
}
