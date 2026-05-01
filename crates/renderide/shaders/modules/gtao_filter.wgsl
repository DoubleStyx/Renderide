//! XeGTAO depth-aware-filter helpers shared by `gtao_main`, `gtao_denoise`, and `gtao_apply`.
//!
//! The math is a direct port of `XeGTAO_CalculateEdges` / `XeGTAO_PackEdges` /
//! `XeGTAO_UnpackEdges` from `XeGTAO.hlsli` (Intel reference, 2021–22). Constants match the
//! reference exactly:
//!
//! - `OCCLUSION_TERM_SCALE = 1.5` — the AO production pass stores `saturate(visibility / 1.5)`
//!   so the bilateral kernel has headroom when summing weighted neighbours; the final-apply
//!   pass multiplies by 1.5 to recover the true visibility before modulating HDR.
//! - `DIAG_WEIGHT = 0.85 * 0.5` — diagonal-neighbour scaling so the 3×3 kernel's diagonal
//!   energy is comparable to its cardinal energy.
//! - `LEAK_THRESHOLD = 2.5`, `LEAK_STRENGTH = 0.5` — small bilateral leak past strong edge
//!   clusters, which reduces both spatial aliasing and TAA shimmer at silhouettes.
//!
//! All helpers are pure functions on view-space (positive) depths and unit-interval edge
//! values; they do not reach for any global bindings, so the same module imports cleanly into
//! every GTAO stage without dragging unused resources in.

#define_import_path renderide::gtao_filter

/// Headroom factor applied at production / removed at final apply. Matches XeGTAO's
/// `XE_GTAO_OCCLUSION_TERM_SCALE`.
const GTAO_OCCLUSION_TERM_SCALE: f32 = 1.5;

/// Diagonal-neighbour weight in the 3×3 bilateral kernel. Matches XeGTAO's `0.85 * 0.5` literal.
const GTAO_DIAG_WEIGHT: f32 = 0.425;

/// Edge sum at which the per-pixel "edge leak" begins to take effect. XeGTAO references 2.5
/// in `XeGTAO_Denoise` (commented "this allows some small amount of AO leaking from neighbours
/// if there are 3 or 4 edges").
const GTAO_LEAK_THRESHOLD: f32 = 2.5;

/// Strength of the per-pixel "edge leak" added on top of the cardinal edge weights.
const GTAO_LEAK_STRENGTH: f32 = 0.5;

/// Direct port of `XeGTAO_CalculateEdges`. Returns four edge weights in `LRTB` order; `1.0`
/// means "no edge" (full bilateral connectivity), `0.0` means "strong edge" (kernel weight
/// drops to zero across the boundary).
///
/// `depth_*` are positive view-space depths. The slope-correction terms (`slope_lr`,
/// `slope_tb`) suppress false silhouettes on slanted surfaces by predicting the
/// depth-difference that pure perspective foreshortening would already produce. XeGTAO then
/// takes the `min(abs(raw), abs(slope_adjusted))` (not just the slope-adjusted edges) so a
/// flat slope still detects true geometric edges that happen to align with the slope vector.
fn gtao_calculate_edges(
    depth_center: f32,
    depth_left: f32,
    depth_right: f32,
    depth_top: f32,
    depth_bottom: f32,
) -> vec4<f32> {
    let edges_raw = vec4<f32>(
        depth_left,
        depth_right,
        depth_top,
        depth_bottom,
    ) - depth_center;

    let slope_lr = (edges_raw.y - edges_raw.x) * 0.5;
    let slope_tb = (edges_raw.w - edges_raw.z) * 0.5;
    let edges_slope = edges_raw + vec4<f32>(slope_lr, -slope_lr, slope_tb, -slope_tb);

    let edges = min(abs(edges_raw), abs(edges_slope));
    let denom = max(depth_center * 0.011, 1e-6);
    return clamp(
        vec4<f32>(1.25) - edges / vec4<f32>(denom),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
}

/// Direct port of `XeGTAO_PackEdges`. Quantises four `LRTB` edges to four levels each
/// (`0`, `1/3`, `2/3`, `1`) and packs them into a single `R8Unorm` value.
///
/// XeGTAO uses `round(saturate(*) * 2.9)` rather than `* 3.0` to bias slightly toward the
/// "strong edge" buckets — this matches the reference exactly so any tuning done against the
/// reference numbers carries over.
fn gtao_pack_edges(edges_lrtb: vec4<f32>) -> f32 {
    let q = round(clamp(edges_lrtb, vec4<f32>(0.0), vec4<f32>(1.0)) * 2.9);
    return dot(
        q,
        vec4<f32>(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0),
    );
}

/// Direct port of `XeGTAO_UnpackEdges`. Inverse of `gtao_pack_edges`.
///
/// Note the `* 255.5` (not `255.0`) inside the reference — the extra `0.5` rounds the unorm
/// sample into the correct integer bucket without an explicit `round`. We keep the literal
/// since later code does its own `saturate`.
fn gtao_unpack_edges(packed: f32) -> vec4<f32> {
    let p = u32(clamp(packed, 0.0, 1.0) * 255.5);
    return clamp(
        vec4<f32>(
            f32((p >> 6u) & 3u),
            f32((p >> 4u) & 3u),
            f32((p >> 2u) & 3u),
            f32(p & 3u),
        ) * (1.0 / 3.0),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
}

/// XeGTAO's per-pixel "edge leak" (`XeGTAO_Denoise` lines 772–776). When the four cardinal
/// edges sum below `4 - LEAK_THRESHOLD = 1.5` (i.e. three or four directions are strong
/// edges), allow up to `LEAK_STRENGTH = 0.5` of bilateral leakage so neighbour AO can flow
/// past the edge cluster. This prevents both spatial aliasing and TAA shimmer at silhouette
/// pixels surrounded on multiple sides by depth discontinuities.
fn gtao_apply_edge_leak(edges_c_lrtb: vec4<f32>) -> vec4<f32> {
    let edginess = (clamp(
        4.0 - GTAO_LEAK_THRESHOLD - dot(edges_c_lrtb, vec4<f32>(1.0)),
        0.0,
        1.0,
    ) / (4.0 - GTAO_LEAK_THRESHOLD)) * GTAO_LEAK_STRENGTH;
    return clamp(edges_c_lrtb + vec4<f32>(edginess), vec4<f32>(0.0), vec4<f32>(1.0));
}

/// `LRTB` cardinal edge weights, with the per-direction symmetricity correction from
/// `XeGTAO_Denoise` line 770: a center-to-neighbour weight is multiplied by the neighbour's
/// edge weight pointing back at the center (`L`'s right edge gates `C`'s left direction,
/// etc.). XeGTAO calls this out specifically: "edges aren't perfectly symmetrical… this line
/// further enforces the symmetricity, creating a slightly sharper blur. Works real nice with
/// TAA."
fn gtao_symmetricise_edges(
    edges_c_lrtb: vec4<f32>,
    edges_l_lrtb: vec4<f32>,
    edges_r_lrtb: vec4<f32>,
    edges_t_lrtb: vec4<f32>,
    edges_b_lrtb: vec4<f32>,
) -> vec4<f32> {
    return edges_c_lrtb * vec4<f32>(
        edges_l_lrtb.y,
        edges_r_lrtb.x,
        edges_t_lrtb.w,
        edges_b_lrtb.z,
    );
}

/// Diagonal weights from `XeGTAO_Denoise` lines 785–788. Each diagonal uses the two cardinal
/// edges that straddle it, contributed by both the center pixel and the relevant immediate
/// neighbour, summed and scaled by `DIAG_WEIGHT`. Indices (`LRTB`): `x = L`, `y = R`, `z = T`,
/// `w = B`.
struct GtaoDiagonalWeights {
    tl: f32,
    tr: f32,
    bl: f32,
    br: f32,
}

fn gtao_diagonal_weights(
    edges_c_lrtb: vec4<f32>,
    edges_l_lrtb: vec4<f32>,
    edges_r_lrtb: vec4<f32>,
    edges_t_lrtb: vec4<f32>,
    edges_b_lrtb: vec4<f32>,
) -> GtaoDiagonalWeights {
    let tl = GTAO_DIAG_WEIGHT * (edges_c_lrtb.x * edges_l_lrtb.z + edges_c_lrtb.z * edges_t_lrtb.x);
    let tr = GTAO_DIAG_WEIGHT * (edges_c_lrtb.z * edges_t_lrtb.y + edges_c_lrtb.y * edges_r_lrtb.z);
    let bl = GTAO_DIAG_WEIGHT * (edges_c_lrtb.w * edges_b_lrtb.x + edges_c_lrtb.x * edges_l_lrtb.w);
    let br = GTAO_DIAG_WEIGHT * (edges_c_lrtb.y * edges_r_lrtb.w + edges_c_lrtb.w * edges_b_lrtb.y);
    return GtaoDiagonalWeights(tl, tr, bl, br);
}

/// AO-term inputs for the 3×3 bilateral kernel. Indices match the kernel diagram:
///
/// ```text
/// tl  t  tr
///  l  c  r
/// bl  b  br
/// ```
struct GtaoKernelAo {
    c: f32,
    l: f32,
    r: f32,
    t: f32,
    b: f32,
    tl: f32,
    tr: f32,
    bl: f32,
    br: f32,
}

/// XeGTAO's bilateral-kernel core (`XeGTAO_Denoise` lines 801–814). `blur_amount` is the
/// caller-supplied `XeGTAO_AddSample` seed weight — `denoise_blur_beta` for the final-apply
/// pass, `denoise_blur_beta / 5.0` for the intermediate denoise pass.
///
/// Returns the weighted-average AO term in the same `[0, 1]` scale as the inputs (the
/// production pass stores `visibility / OCCLUSION_TERM_SCALE`, and this kernel preserves
/// that scale; the apply caller multiplies by `OCCLUSION_TERM_SCALE` before modulating HDR).
fn gtao_denoise_kernel(
    edges_c_sym: vec4<f32>,
    diag: GtaoDiagonalWeights,
    ao: GtaoKernelAo,
    blur_amount: f32,
) -> f32 {
    var sum_w = blur_amount;
    var sum = ao.c * sum_w;

    sum = sum + ao.l * edges_c_sym.x;
    sum = sum + ao.r * edges_c_sym.y;
    sum = sum + ao.t * edges_c_sym.z;
    sum = sum + ao.b * edges_c_sym.w;
    sum_w = sum_w + edges_c_sym.x + edges_c_sym.y + edges_c_sym.z + edges_c_sym.w;

    sum = sum + ao.tl * diag.tl;
    sum = sum + ao.tr * diag.tr;
    sum = sum + ao.bl * diag.bl;
    sum = sum + ao.br * diag.br;
    sum_w = sum_w + diag.tl + diag.tr + diag.bl + diag.br;

    return sum / max(sum_w, 1e-4);
}
