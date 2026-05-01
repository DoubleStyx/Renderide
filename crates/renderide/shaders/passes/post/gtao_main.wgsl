//! Fullscreen pass: Ground-Truth Ambient Occlusion (Jimenez et al. 2016) — production stage.
//!
//! Reads the imported scene depth, reconstructs view-space positions and normals on the fly
//! from depth derivatives, evaluates the analytic cosine-weighted GTAO integral, applies the
//! multi-bounce fit, and writes:
//!
//! - `@location(0)` — `saturate(visibility / OCCLUSION_TERM_SCALE)` to an `R8Unorm` ping-pong
//!   target. Storing scaled-down AO matches XeGTAO's `XeGTAO_OutputWorkingTerm` and gives the
//!   bilateral kernel headroom: a weighted sum of neighbour visibilities can transiently
//!   exceed `1.0`, but stays representable in `[0, 1]` after the divide. The final-apply pass
//!   multiplies by `OCCLUSION_TERM_SCALE` to recover the true visibility before modulating
//!   HDR.
//! - `@location(1)` — packed `LRTB` depth-edge weights produced by `gtao_pack_edges` to an
//!   `R8Unorm` ping-pong target. The bilateral denoise / apply stages unpack these and use
//!   them as kernel weights so silhouette detail survives the blur.
//!
//! The horizon search mirrors XeGTAO's `XeGTAO_MainPass` inner loop with one slice per pixel
//! (paper §4.1 spatiotemporal distribution; temporal accumulation is a follow-up). The
//! direction is orthogonalised against the view vector, the slice-plane normal `axis_vec` is
//! explicitly normalised before projecting the surface normal, horizon cosines are
//! initialised at the tangent-plane bound `cos(n ± π/2)`, each candidate is smoothly faded
//! toward that bound by a distance falloff, and the running horizon is a pure `max`. The
//! analytic inner integral uses `cos(n)` directly (from the projected-normal dot product)
//! rather than recomputing a trigonometric `cos(gamma)`.
//!
//! Build script composes this into `gtao_main_default` (mono; depth as `texture_depth_2d`)
//! and `gtao_main_multiview` (stereo; `@builtin(view_index)` selects the eye and depth is
//! `texture_depth_2d_array`) via naga-oil's `#ifdef MULTIVIEW`.
//!
//! Bind group (`@group(0)`):
//! - `@binding(0)` scene depth (`texture_depth_2d` mono, `texture_depth_2d_array` multiview).
//! - `@binding(1)` `FrameGlobals` uniform (per-eye proj coefficients + near/far + frame index).
//! - `@binding(2)` `GtaoParams` uniform (radius / intensity / steps / denoise tunables).

#import renderide::fullscreen as fs
#import renderide::gtao_filter as gf

#ifdef MULTIVIEW
@group(0) @binding(0) var scene_depth: texture_depth_2d_array;
#else
@group(0) @binding(0) var scene_depth: texture_depth_2d;
#endif

/// Matches the GTAO-visible prefix of [`crate::gpu::frame_globals::FrameGpuUniforms`] through
/// `frame_tail` exactly (144 bytes). Duplicated rather than imported because post-processing
/// shaders run with a bespoke `@group(0)` layout (no lights / cluster storage), so the full
/// globals module would introduce unreferenced bindings that naga-oil would drop.
struct FrameGlobals {
    camera_world_pos: vec4<f32>,
    camera_world_pos_right: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    view_space_z_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
    proj_params_left: vec4<f32>,
    proj_params_right: vec4<f32>,
    frame_tail: vec4<u32>,
}

@group(0) @binding(1) var<uniform> frame: FrameGlobals;

/// User-tunable GTAO parameters. Updated every record from the live
/// [`crate::config::GtaoSettings`] via the `GtaoSettingsSlot` blackboard slot.
///
/// `denoise_blur_beta` and `final_apply` are written stage-specifically (see the Rust pass
/// records); they're declared here so the AO production shader can share one `GtaoParams`
/// layout with the denoise / apply shaders without padding mismatch.
struct GtaoParams {
    radius_world: f32,
    max_pixel_radius: f32,
    intensity: f32,
    step_count: u32,
    falloff_range: f32,
    albedo_multibounce: f32,
    denoise_blur_beta: f32,
    final_apply: u32,
}

@group(0) @binding(2) var<uniform> gtao: GtaoParams;

struct GtaoMainOutput {
    @location(0) ao_term: vec4<f32>,
    @location(1) edges: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vid);
}

/// Raw depth at integer pixel coords for the current view layer.
fn load_depth(pix: vec2<i32>, view_layer: u32) -> f32 {
#ifdef MULTIVIEW
    return textureLoad(scene_depth, pix, i32(view_layer), 0);
#else
    return textureLoad(scene_depth, pix, 0);
#endif
}

/// Reverse-Z NDC depth → positive view-space Z.
///
/// Renderide's perspective matrix (see `reverse_z_perspective_from_scales`) is −Z forward in
/// view space; this helper returns `|view_z|` so the rest of the shader can treat all depth
/// and delta arithmetic as magnitudes (matches XeGTAO's internal convention).
fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    let denom = d * (far - near) + near;
    return (near * far) / max(denom, 1e-6);
}

/// Selects per-eye `(P[0][0], P[1][1], P[0][2], P[1][2])` for the active view layer.
fn proj_params_for_view(view_layer: u32) -> vec4<f32> {
    if (view_layer == 0u) {
        return frame.proj_params_left;
    }
    return frame.proj_params_right;
}

/// Screen UV (`[0, 1]`) → view-space position, given the linearized positive view-space Z for
/// that pixel.
///
/// Renderide's reverse-Z perspective matrix has column 2 = `(skew_x, skew_y, z2, -1)`. With
/// `clip_w = -view_z` and `linearize_depth` returning `|view_z|`, the unprojection reduces to
/// `view_x = (ndc_x + skew_x) * |view_z| / x_scale` (and similarly for y). The `+skew` sign
/// is load-bearing for asymmetric VR frustums; desktop (skew ≈ 0) is a no-op.
fn view_pos_from_uv(uv: vec2<f32>, view_z: f32, proj_params: vec4<f32>) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let view_x = (ndc_xy.x + proj_params.z) * view_z / proj_params.x;
    let view_y = (ndc_xy.y + proj_params.w) * view_z / proj_params.y;
    return vec3<f32>(view_x, view_y, view_z);
}

/// Reconstructs a view-space normal from screen-space derivatives of the view-space position.
///
/// Uses the 4-neighbor min-depth-delta trick: pick the closer neighbor on each axis so creases
/// at silhouettes pick the continuous surface instead of averaging across the depth gap.
///
/// Coordinate convention (positive-depth mirrored view space, matching `linearize_depth`):
/// `+X` = screen-right, `+Y` = view-up (= **pixel-down**; WebGPU's NDC Y is flipped from
/// pixel Y), `+Z` = away from camera. `dx` points in `+view_x`, `dy` points in `-view_y`
/// (screen-down-in-pixel = view-down-in-Y). `cross(dx, dy)` then yields a vector toward the
/// camera — the correct outward-facing surface normal. **Do not flip the cross order**;
/// `cross(dy, dx)` produces a normal pointing away from the camera, and downstream
/// `cos_n = dot(normal, view_dir)` comes out negative, collapsing the horizon integral to 0
/// on camera-facing surfaces (fresnel-like inversion).
fn reconstruct_view_normal(
    center_pix: vec2<i32>,
    center_view: vec3<f32>,
    proj_params: vec4<f32>,
    viewport: vec2<f32>,
    z_left: f32,
    z_right: f32,
    z_top: f32,
    z_bottom: f32,
) -> vec3<f32> {
    let inv_viewport = 1.0 / viewport;
    let uv_center = (vec2<f32>(center_pix) + vec2<f32>(0.5)) * inv_viewport;

    let dx_pos = abs(z_right - center_view.z) < abs(z_left - center_view.z);
    let dy_pos = abs(z_bottom - center_view.z) < abs(z_top - center_view.z);

    let uv_x = select(
        uv_center - vec2<f32>(inv_viewport.x, 0.0),
        uv_center + vec2<f32>(inv_viewport.x, 0.0),
        dx_pos,
    );
    let uv_y = select(
        uv_center - vec2<f32>(0.0, inv_viewport.y),
        uv_center + vec2<f32>(0.0, inv_viewport.y),
        dy_pos,
    );
    let z_x = select(z_left, z_right, dx_pos);
    let z_y = select(z_top, z_bottom, dy_pos);

    let p_x = view_pos_from_uv(uv_x, z_x, proj_params);
    let p_y = view_pos_from_uv(uv_y, z_y, proj_params);

    let dx = select(center_view - p_x, p_x - center_view, dx_pos);
    let dy = select(center_view - p_y, p_y - center_view, dy_pos);

    return normalize(cross(dx, dy));
}

/// Interleaved-gradient spatial noise (Jiménez) so adjacent pixels cover different slice
/// angles.
fn spatial_phase(pix: vec2<i32>) -> f32 {
    let p = vec2<f32>(pix);
    let jitter = 52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y);
    return fract(jitter);
}

/// Per-frame phase rotation across 6 directions — feeds the temporal-accumulation follow-up.
fn temporal_phase(frame_index: u32) -> f32 {
    let k = f32(frame_index % 6u);
    return k * (1.0 / 6.0);
}

/// Paper Eq. 10 cubic fit: recovers near-field multi-bounce indirect illumination lost when
/// applying ambient occlusion alone. Uses a gray-albedo proxy since we don't sample per-pixel
/// albedo.
fn multi_bounce_fit(ao: f32, albedo: f32) -> f32 {
    let a = 2.0404 * albedo - 0.3324;
    let b = 4.7951 * albedo - 0.6417;
    let c = 2.7552 * albedo + 0.6903;
    return max(ao, ((a * ao - b) * ao + c) * ao);
}

struct GtaoSampleOutput {
    visibility: f32,
    edges_lrtb: vec4<f32>,
}

/// Runs the full GTAO computation for a pixel and returns the visibility factor in `[0, 1]`
/// together with the four cardinal-neighbour edge weights consumed by the depth-aware
/// denoise stage.
///
/// Structure mirrors XeGTAO's `XeGTAO_MainPass` inner loop with one slice per pixel.
fn compute_gtao(pix: vec2<i32>, uv: vec2<f32>, view_layer: u32) -> GtaoSampleOutput {
    let viewport = vec2<f32>(f32(frame.viewport_width), f32(frame.viewport_height));
    let proj_params = proj_params_for_view(view_layer);

    let raw_depth = load_depth(pix, view_layer);
    let near = frame.near_clip;
    let far = frame.far_clip;

    let z_center = linearize_depth(raw_depth, near, far);
    let z_left = linearize_depth(load_depth(pix - vec2<i32>(1, 0), view_layer), near, far);
    let z_right = linearize_depth(load_depth(pix + vec2<i32>(1, 0), view_layer), near, far);
    let z_top = linearize_depth(load_depth(pix - vec2<i32>(0, 1), view_layer), near, far);
    let z_bottom = linearize_depth(load_depth(pix + vec2<i32>(0, 1), view_layer), near, far);

    let edges_lrtb = gf::gtao_calculate_edges(z_center, z_left, z_right, z_top, z_bottom);

    if (raw_depth <= 0.0) {
        return GtaoSampleOutput(1.0, edges_lrtb);
    }

    let view_pos = view_pos_from_uv(uv, z_center, proj_params);
    let view_normal = reconstruct_view_normal(
        pix,
        view_pos,
        proj_params,
        viewport,
        z_left,
        z_right,
        z_top,
        z_bottom,
    );
    let view_dir = normalize(-view_pos);

    let pixel_radius_raw = gtao.radius_world * proj_params.x * viewport.x * 0.5
        / max(z_center, 1e-3);
    let pixel_radius = min(gtao.max_pixel_radius, pixel_radius_raw);
    let step_count = max(gtao.step_count, 1u);
    let step_pixels = max(pixel_radius / f32(step_count), 1.0);

    let phase = spatial_phase(pix) + temporal_phase(frame.frame_tail.x);
    let angle = phase * 3.14159265359;
    let dir_ss = vec2<f32>(cos(angle), sin(angle));

    let direction_vec = vec3<f32>(dir_ss.x, dir_ss.y, 0.0);
    let ortho_direction_vec = direction_vec - view_dir * dot(direction_vec, view_dir);
    let axis_raw = cross(ortho_direction_vec, view_dir);
    let axis_len = length(axis_raw);
    if (axis_len < 1e-6) {
        return GtaoSampleOutput(1.0, edges_lrtb);
    }
    let axis_vec = axis_raw / axis_len;

    let projected_normal_vec = view_normal - axis_vec * dot(view_normal, axis_vec);
    let projected_normal_vec_length = length(projected_normal_vec);
    if (projected_normal_vec_length < 1e-6) {
        return GtaoSampleOutput(1.0, edges_lrtb);
    }
    let sign_n = sign(dot(ortho_direction_vec, projected_normal_vec));
    // `saturate` (not `clamp(-1, 1)`) mirrors XeGTAO: a negative projected dot means the
    // normal is on the wrong side of the view plane (e.g. a silhouette with an
    // ill-conditioned depth derivative), which would otherwise drive the integral into the
    // mirrored hemisphere and flip the sign of the AO contribution. Clamping to 0 collapses
    // that case to `n = π/2` (fully grazing) — still wrong for the affected pixel, but not
    // inverted.
    let cos_n = clamp(
        dot(projected_normal_vec, view_dir) / projected_normal_vec_length,
        0.0,
        1.0,
    );
    let n = sign_n * acos(cos_n);
    let sin_n = sin(n);

    let low_horizon_cos0 = -sin_n; // cos(n + π/2)
    let low_horizon_cos1 = sin_n;  // cos(n - π/2)
    var horizon_cos0 = low_horizon_cos0;
    var horizon_cos1 = low_horizon_cos1;

    let falloff_range_world = max(gtao.falloff_range, 1e-4) * gtao.radius_world;
    let falloff_mul = -1.0 / max(falloff_range_world, 1e-4);
    let falloff_add = gtao.radius_world / max(falloff_range_world, 1e-4);
    let inv_viewport = 1.0 / viewport;

    for (var s: u32 = 1u; s <= step_count; s = s + 1u) {
        let step_len = f32(s) * step_pixels;

        let pix_pos_offset = vec2<f32>(pix) + dir_ss * step_len;
        let pix_neg_offset = vec2<f32>(pix) - dir_ss * step_len;

        let ipp = vec2<i32>(pix_pos_offset);
        let inn = vec2<i32>(pix_neg_offset);
        let pp_in = ipp.x >= 0 && ipp.y >= 0
            && ipp.x < i32(viewport.x) && ipp.y < i32(viewport.y);
        let nn_in = inn.x >= 0 && inn.y >= 0
            && inn.x < i32(viewport.x) && inn.y < i32(viewport.y);

        if (pp_in) {
            let uv_pp = (vec2<f32>(ipp) + vec2<f32>(0.5)) * inv_viewport;
            let z_pp = linearize_depth(load_depth(ipp, view_layer), near, far);
            let sp_pos = view_pos_from_uv(uv_pp, z_pp, proj_params);
            let d_pos = sp_pos - view_pos;
            let d_pos_len = length(d_pos);
            if (d_pos_len > 1e-4) {
                let candidate = dot(d_pos / d_pos_len, view_dir);
                let weight = clamp(d_pos_len * falloff_mul + falloff_add, 0.0, 1.0);
                let shc = mix(low_horizon_cos1, candidate, weight);
                horizon_cos1 = max(horizon_cos1, shc);
            }
        }
        if (nn_in) {
            let uv_nn = (vec2<f32>(inn) + vec2<f32>(0.5)) * inv_viewport;
            let z_nn = linearize_depth(load_depth(inn, view_layer), near, far);
            let sp_neg = view_pos_from_uv(uv_nn, z_nn, proj_params);
            let d_neg = sp_neg - view_pos;
            let d_neg_len = length(d_neg);
            if (d_neg_len > 1e-4) {
                let candidate = dot(d_neg / d_neg_len, view_dir);
                let weight = clamp(d_neg_len * falloff_mul + falloff_add, 0.0, 1.0);
                let shc = mix(low_horizon_cos0, candidate, weight);
                horizon_cos0 = max(horizon_cos0, shc);
            }
        }
    }

    let h0 = -acos(clamp(horizon_cos1, -1.0, 1.0));
    let h1 = acos(clamp(horizon_cos0, -1.0, 1.0));

    let iarc0 = (cos_n + 2.0 * h0 * sin_n - cos(2.0 * h0 - n)) * 0.25;
    let iarc1 = (cos_n + 2.0 * h1 * sin_n - cos(2.0 * h1 - n)) * 0.25;
    let local_visibility = projected_normal_vec_length * (iarc0 + iarc1);

    let ao = clamp(local_visibility, 0.0, 1.0);
    let boosted = clamp(pow(ao, max(gtao.intensity, 0.0)), 0.0, 1.0);
    return GtaoSampleOutput(
        multi_bounce_fit(boosted, gtao.albedo_multibounce),
        edges_lrtb,
    );
}

/// Wraps the production shader's two outputs: AO scaled by `1 / OCCLUSION_TERM_SCALE` (the
/// XeGTAO headroom convention) and the packed `LRTB` edges.
fn pack_main_output(s: GtaoSampleOutput) -> GtaoMainOutput {
    let scaled = clamp(s.visibility / gf::GTAO_OCCLUSION_TERM_SCALE, 0.0, 1.0);
    let packed = gf::gtao_pack_edges(s.edges_lrtb);
    return GtaoMainOutput(
        vec4<f32>(scaled, 0.0, 0.0, 1.0),
        vec4<f32>(packed, 0.0, 0.0, 1.0),
    );
}

#ifdef MULTIVIEW
@fragment
fn fs_main(in: fs::FullscreenVertexOutput, @builtin(view_index) view: u32) -> GtaoMainOutput {
    let pix = vec2<i32>(in.clip_pos.xy);
    return pack_main_output(compute_gtao(pix, in.uv, view));
}
#else
@fragment
fn fs_main(in: fs::FullscreenVertexOutput) -> GtaoMainOutput {
    let pix = vec2<i32>(in.clip_pos.xy);
    return pack_main_output(compute_gtao(pix, in.uv, 0u));
}
#endif
