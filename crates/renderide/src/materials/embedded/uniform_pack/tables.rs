//! Name-driven keyword inference and scalar default tables for embedded uniform packing.

use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};

use super::super::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};
use super::helpers::{
    first_float_by_pids, is_keyword_like_field, keyword_float_enabled_any_pids,
    shader_writer_unescaped_field_name, texture_property_present_pids,
};

/// Tolerance (in radians) for treating a `Projection360` `_FOV.xy` value as the default
/// full-sphere `(TAU, π)`. Tighter than any FOV the host realistically writes (the host
/// converts whole-degree values through `* π / 180`, so the residual is below `1e-6`).
const PROJECTION360_FULL_SPHERE_EPSILON: f32 = 1e-3;

/// Infers a scalar keyword uniform from host-visible material state.
pub(super) fn inferred_keyword_float_f32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if let Some(probes) = ids.keyword_field_probe_ids.get(field_name)
        && keyword_float_enabled_any_pids(store, lookup, probes)
    {
        return Some(1.0);
    }

    let kw = ids.shared.as_ref();
    if let Some(value) = blend_keyword_inferred(field_name, store, lookup, kw) {
        return Some(value);
    }
    if is_projection360_keyword(field_name)
        && let Some(value) = projection360_keyword_inferred(field_name, store, lookup, ids)
    {
        return Some(value);
    }

    let inferred = match texture_keyword_pids(field_name, kw) {
        Some(pids) => texture_property_present_pids(store, lookup, &pids),
        None if is_keyword_like_field(field_name) => false,
        None => return None,
    };
    Some(if inferred { 1.0 } else { 0.0 })
}

/// True for any keyword name that participates in `Projection360` keyword inference.
fn is_projection360_keyword(field_name: &str) -> bool {
    matches!(
        field_name,
        "OUTSIDE_CLIP"
            | "OUTSIDE_COLOR"
            | "OUTSIDE_CLAMP"
            | "_PERSPECTIVE"
            | "CUBEMAP"
            | "CUBEMAP_LOD"
            | "SECOND_TEXTURE"
            | "_OFFSET"
            | "_CLAMP_INTENSITY"
            | "TINT_TEX_DIRECT"
            | "TINT_TEX_LERP"
    )
}

/// Resolves alpha-test/alpha-blend/alpha-premultiply/`_MUL_RGB_BY_ALPHA` keywords from host blend
/// state. Returns `None` for unrelated keyword names.
fn blend_keyword_inferred(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> Option<f32> {
    let value = match field_name {
        "_ALPHATEST_ON" | "_ALPHATEST" | "_ALPHACLIP" => alpha_test_on_inferred(store, lookup, kw),
        "_ALPHABLEND_ON" => alpha_blend_on_inferred(store, lookup, kw),
        "_ALPHAPREMULTIPLY_ON" => alpha_premultiply_on_inferred(store, lookup, kw),
        "_MUL_RGB_BY_ALPHA" => mul_rgb_by_alpha_inferred(store, lookup, kw),
        _ => return None,
    };
    Some(if value { 1.0 } else { 0.0 })
}

/// Returns the host property ids whose presence drives the texture-presence keyword for
/// `field_name`, or `None` for keywords not driven by texture presence.
fn texture_keyword_pids(field_name: &str, kw: &EmbeddedSharedKeywordIds) -> Option<Vec<i32>> {
    Some(match field_name {
        "_LERPTEX" => vec![kw.lerp_tex],
        "_TEXTURE" => vec![
            kw.far_tex,
            kw.near_tex,
            kw.far_tex0,
            kw.near_tex0,
            kw.far_tex1,
            kw.near_tex1,
        ],
        "GRADIENT" => vec![kw.gradient],
        "_ALBEDOTEX" => vec![kw.main_tex, kw.main_tex1],
        "_EMISSION" | "_EMISSIONTEX" => vec![kw.emission_map, kw.emission_map1],
        "_NORMALMAP" => vec![kw.normal_map, kw.normal_map0, kw.normal_map1, kw.bump_map],
        "_SPECULARMAP" => vec![kw.specular_map, kw.specular_map1, kw.spec_gloss_map],
        "_METALLICGLOSSMAP" => vec![kw.metallic_gloss_map],
        "_METALLICMAP" => vec![kw.metallic_map, kw.metallic_map1, kw.metallic_gloss_map],
        "_DETAIL_MULX2" => vec![kw.detail_albedo_map, kw.detail_normal_map, kw.detail_mask],
        "_PARALLAXMAP" => vec![kw.parallax_map],
        "_OCCLUSION" => vec![kw.occlusion, kw.occlusion1, kw.occlusion_map],
        _ => return None,
    })
}

/// Discriminant of [`crate::shared::MaterialRenderType::TransparentCutout`] on the wire.
/// Captured under the synthetic `_RenderType` property by
/// [`crate::materials::host_data::parse_materials_update_batch_into_store`].
const RENDER_TYPE_TRANSPARENT_CUTOUT: i32 = 1;
/// Discriminant of [`crate::shared::MaterialRenderType::Transparent`] on the wire.
const RENDER_TYPE_TRANSPARENT: i32 = 2;
/// FrooxEngine `BlendMode.Cutout` discriminant (matches Unity Standard `_Mode = 1`).
const BLEND_MODE_CUTOUT: i32 = 1;
/// FrooxEngine `BlendMode.Alpha` discriminant — Unity Standard `_Mode = 2` (alpha-blend / fade).
const BLEND_MODE_ALPHA: i32 = 2;
/// FrooxEngine `BlendMode.Transparent` discriminant — Unity Standard `_Mode = 3` (premultiplied).
const BLEND_MODE_TRANSPARENT_PREMULTIPLY: i32 = 3;
/// `UnityEngine.Rendering.BlendMode.One`.
const UNITY_BLEND_FACTOR_ONE: i32 = 1;
/// `UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha`.
const UNITY_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: i32 = 10;
/// Inclusive lower bound of Unity's AlphaTest queue range (FrooxEngine writes 2450 for
/// `AlphaHandling.AlphaClip` / `BlendMode.Cutout`).
const RENDER_QUEUE_ALPHA_TEST_MIN: i32 = 2450;
/// Inclusive lower bound of Unity's Transparent queue range (FrooxEngine writes 3000 for
/// `AlphaHandling.AlphaBlend` / `BlendMode.Alpha` / `BlendMode.Transparent`). Also the
/// exclusive upper bound of the AlphaTest range.
const RENDER_QUEUE_TRANSPARENT_MIN: i32 = 3000;

/// Reads a float-valued material property as the integer enum/discriminant it represents.
fn read_int_property(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw_pid: i32,
) -> Option<i32> {
    first_float_by_pids(store, lookup, &[kw_pid]).map(|v| v.round() as i32)
}

/// Returns whether either render-type or older mode properties match the requested values.
fn render_type_or_legacy_mode_is(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
    render_type_value: i32,
    legacy_mode_value: i32,
) -> bool {
    if read_int_property(store, lookup, kw.render_type) == Some(render_type_value) {
        return true;
    }
    let legacy_mode = read_int_property(store, lookup, kw.mode);
    let legacy_blend = read_int_property(store, lookup, kw.blend_mode);
    legacy_mode == Some(legacy_mode_value) || legacy_blend == Some(legacy_mode_value)
}

/// Returns whether the host blend factors match `src_factor` and `dst_factor`.
fn blend_factors_are(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
    src_factor: i32,
    dst_factor: i32,
) -> bool {
    let src = read_int_property(store, lookup, kw.src_blend);
    let dst = read_int_property(store, lookup, kw.dst_blend);
    src == Some(src_factor) && dst == Some(dst_factor)
}

/// Returns whether blend factors describe Unity/FrooxEngine premultiplied alpha blending.
fn premultiplied_blend_factors(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    blend_factors_are(
        store,
        lookup,
        kw,
        UNITY_BLEND_FACTOR_ONE,
        UNITY_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    )
}

/// Returns whether blend factors describe Unity/FrooxEngine additive blending.
fn additive_blend_factors(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    blend_factors_are(
        store,
        lookup,
        kw,
        UNITY_BLEND_FACTOR_ONE,
        UNITY_BLEND_FACTOR_ONE,
    )
}

/// Classification of an inferred render queue value.
///
/// Mirrors Unity's standard queue ranges and the values FrooxEngine writes from both
/// `MaterialProvider.SetBlendMode` (Opaque=2000/2550, Cutout=2450/2750, Transparent=3000)
/// and the PBS `AlphaHandling` family (Opaque=2000, AlphaClip=2450, AlphaBlend=3000).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InferredQueueRange {
    /// Below the AlphaTest threshold (Background / Geometry).
    Opaque,
    /// `[2450, 3000)` — Unity AlphaTest range.
    AlphaTest,
    /// `>= 3000` — Unity Transparent range and beyond.
    Transparent,
}

/// Classifies the host render queue into the alpha range implied by Unity's queue constants.
fn render_queue_range(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> Option<InferredQueueRange> {
    let queue = read_int_property(store, lookup, kw.render_queue)?;
    if queue >= RENDER_QUEUE_TRANSPARENT_MIN {
        Some(InferredQueueRange::Transparent)
    } else if queue >= RENDER_QUEUE_ALPHA_TEST_MIN {
        Some(InferredQueueRange::AlphaTest)
    } else {
        Some(InferredQueueRange::Opaque)
    }
}

/// Returns whether host-visible state implies an alpha-test/cutout shader keyword.
fn alpha_test_on_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    if render_queue_range(store, lookup, kw) == Some(InferredQueueRange::AlphaTest) {
        return true;
    }
    render_type_or_legacy_mode_is(
        store,
        lookup,
        kw,
        RENDER_TYPE_TRANSPARENT_CUTOUT,
        BLEND_MODE_CUTOUT,
    )
}

/// Returns whether host-visible state implies straight alpha blending.
fn alpha_blend_on_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    let render_type = read_int_property(store, lookup, kw.render_type);
    if render_type == Some(RENDER_TYPE_TRANSPARENT) {
        return !premultiplied_blend_factors(store, lookup, kw);
    }
    if render_queue_range(store, lookup, kw) == Some(InferredQueueRange::Transparent) {
        return !premultiplied_blend_factors(store, lookup, kw);
    }
    let legacy_mode = read_int_property(store, lookup, kw.mode);
    let legacy_blend = read_int_property(store, lookup, kw.blend_mode);
    legacy_mode == Some(BLEND_MODE_ALPHA) || legacy_blend == Some(BLEND_MODE_ALPHA)
}

/// Returns whether host-visible state implies premultiplied alpha blending.
fn alpha_premultiply_on_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    let render_type = read_int_property(store, lookup, kw.render_type);
    if render_type == Some(RENDER_TYPE_TRANSPARENT)
        && premultiplied_blend_factors(store, lookup, kw)
    {
        return true;
    }
    if render_queue_range(store, lookup, kw) == Some(InferredQueueRange::Transparent)
        && premultiplied_blend_factors(store, lookup, kw)
    {
        return true;
    }
    let legacy_mode = read_int_property(store, lookup, kw.mode);
    let legacy_blend = read_int_property(store, lookup, kw.blend_mode);
    legacy_mode == Some(BLEND_MODE_TRANSPARENT_PREMULTIPLY)
        || legacy_blend == Some(BLEND_MODE_TRANSPARENT_PREMULTIPLY)
}

/// Inferred values for the `Projection360` material family's multi_compile keyword fields,
/// gated on the stem owning a `_FOV` uniform field.
///
/// FrooxEngine sets every keyword via `keywords.SetKeyword(...)`, but `ShaderKeywords.Variant`
/// is never serialized over IPC (`MaterialUpdateWriter` exposes no `SetKeyword`;
/// `MaterialPropertyUpdateType` has no keyword opcode). The renderer reconstructs each
/// keyword from a property the host *does* send, mirroring exactly the predicate FrooxEngine
/// uses host-side in `Projection360Material.UpdateKeywords`:
///
/// | Keyword           | Host predicate                                         | Renderer probe                                        |
/// |-------------------|--------------------------------------------------------|-------------------------------------------------------|
/// | `_PERSPECTIVE`    | `Projection.Value == Mode.Perspective`                 | `_PerspectiveFOV` written (only sent in Perspective)  |
/// | `OUTSIDE_CLAMP`   | `OutsideMode.Value == Outside.Clamp` (no wire signal)  | partial `_FOV` (full-sphere `(TAU, π)` keeps default) |
/// | `CUBEMAP_LOD`     | cubemap target + `CubemapLOD.Value.HasValue`           | `_MainCube`/`_SecondCube` texture + `_CubeLOD`        |
/// | `CUBEMAP`         | cubemap target + no LOD                                | `_MainCube`/`_SecondCube` texture, no `_CubeLOD`      |
/// | `SECOND_TEXTURE`  | `SecondaryTexture/Cubemap` set or `TextureLerp != 0`   | `_SecondTex`/`_SecondCube` texture                    |
/// | `_OFFSET`         | `OffsetTexture.Asset != null`                          | `_OffsetTex` texture                                  |
/// | `_CLAMP_INTENSITY`| `MaxIntensity.HasValue || HDR texture`                 | `_MaxIntensity` written (only sent when enabled)      |
/// | `TINT_TEX_LERP`   | `TintTexture` + `TintTextureMode == Lerp`              | `_TintTex` texture + `_Tint0` written (Lerp-only send)|
/// | `TINT_TEX_DIRECT` | `TintTexture` + `TintTextureMode == Direct`            | `_TintTex` texture, no `_Tint0`                       |
///
/// `_VIEW`/`_NORMAL`/`_WORLD_VIEW`, `OUTSIDE_COLOR`, `_RIGHT_EYE_ST`, and `RECTCLIP` have no
/// property-stream signal (they map to `bool`/`enum` fields the host never writes as
/// properties). These default to `0`; the shader's existing fallthrough renders such
/// materials in the most common configuration (`_VIEW` + `OUTSIDE_CLIP` + non-stereo +
/// non-rect-clip), and they would only become observable if a host change starts sending
/// the discriminator.
///
/// Returns `None` when the stem has no `_FOV` uniform field (i.e., not the `Projection360`
/// family) so the generic keyword-like fallthrough handles those identically to the current
/// behavior.
fn projection360_keyword_inferred(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    let fov_pid = *ids.uniform_field_ids.get("_FOV")?;
    if !uniform_property_present(store, lookup, fov_pid) {
        return None;
    }

    let kw = ids.shared.as_ref();
    let uniform_pid = |name: &str| ids.uniform_field_ids.get(name).copied();
    let uniform_written = |name: &str| {
        uniform_pid(name).is_some_and(|pid| uniform_property_present(store, lookup, pid))
    };
    let texture_lerp_nonzero = uniform_pid("_TextureLerp")
        .and_then(|pid| match store.get_merged(lookup, pid) {
            Some(MaterialPropertyValue::Float(f)) => Some(*f),
            _ => None,
        })
        .is_some_and(|v| v != 0.0);

    let cubemap_present =
        texture_property_present_pids(store, lookup, &[kw.main_cube, kw.second_cube]);
    let cube_lod_written = uniform_written("_CubeLOD");
    let secondary_texture_present =
        texture_property_present_pids(store, lookup, &[kw.second_tex, kw.second_cube]);
    let tint_tex_present = texture_property_present_pids(store, lookup, &[kw.tint_tex]);
    let tint0_written = uniform_written("_Tint0");

    let enabled = match field_name {
        "OUTSIDE_CLAMP" => {
            let fov_xy = read_float4_xy(store, lookup, fov_pid)?;
            let eps = PROJECTION360_FULL_SPHERE_EPSILON;
            let full_sphere = (fov_xy[0] - std::f32::consts::TAU).abs() <= eps
                && (fov_xy[1] - std::f32::consts::PI).abs() <= eps;
            !full_sphere
        }
        "OUTSIDE_CLIP" | "OUTSIDE_COLOR" => false,
        "_PERSPECTIVE" => uniform_written("_PerspectiveFOV"),
        "CUBEMAP_LOD" => cubemap_present && cube_lod_written,
        "CUBEMAP" => cubemap_present && !cube_lod_written,
        "SECOND_TEXTURE" => secondary_texture_present || texture_lerp_nonzero,
        "_OFFSET" => texture_property_present_pids(store, lookup, &[kw.offset_tex]),
        "_CLAMP_INTENSITY" => uniform_written("_MaxIntensity"),
        "TINT_TEX_LERP" => tint_tex_present && tint0_written,
        "TINT_TEX_DIRECT" => tint_tex_present && !tint0_written,
        _ => return None,
    };
    Some(if enabled { 1.0 } else { 0.0 })
}

/// Reads the `.xy` of a `Float4` property, ignoring scalar `Float` writes.
///
/// `_FOV` is always packed as a `float4` on the host (`Projection360Material.cs:445` writes
/// `SetFloat4(_FOV, new float4(FieldOfView, AngleOffset) * (π/180))`); a scalar write would
/// be a host-side bug, so we don't paper over it by accepting a bare `Float`.
fn read_float4_xy(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> Option<[f32; 2]> {
    match store.get_merged(lookup, property_id) {
        Some(MaterialPropertyValue::Float4(v)) => Some([v[0], v[1]]),
        _ => None,
    }
}

/// `true` when the host has written *any* value to `property_id` — used to mirror the
/// FrooxEngine predicate that gates whether a property is sent at all (e.g., `_PerspectiveFOV`
/// only travels the wire when the material is in `Mode.Perspective`).
fn uniform_property_present(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> bool {
    store.get_merged(lookup, property_id).is_some()
}

/// Returns whether host-visible state implies Unlit's additive RGB-by-alpha multiplication.
fn mul_rgb_by_alpha_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    let render_type = read_int_property(store, lookup, kw.render_type);
    if render_type == Some(RENDER_TYPE_TRANSPARENT) && additive_blend_factors(store, lookup, kw) {
        return true;
    }
    if render_queue_range(store, lookup, kw) == Some(InferredQueueRange::Transparent)
        && additive_blend_factors(store, lookup, kw)
    {
        return true;
    }
    false
}

// Every uniform field reaching `build_embedded_uniform_bytes` is one of:
//   1. A host-declared property — `MaterialPropertyStore` always has a value by the time the
//      renderer reads (first material batch pushes every `Sync<X>` via `MaterialUpdateWriter` per
//      `MaterialProviderBase.cs:48-51`).
//   2. A multi-compile keyword field (`_NORMALMAP`, `_ALPHATEST_ON`, etc.) — inferred by
//      [`inferred_keyword_float_f32`] from texture presence / blend factor reconstruction.
//   3. `_TextMode` / `_RectClip` / `_OVERLAY` explicit-zero defaults and `_Cutoff` — handled
//      by special-case probes in the caller.
//
// Previously-held Unity-Properties{} fallback values are irrelevant: FrooxEngine supplies its own
// initial values (from each `MaterialProvider.OnAwake()`), not Unity's. See the audit for detail.
