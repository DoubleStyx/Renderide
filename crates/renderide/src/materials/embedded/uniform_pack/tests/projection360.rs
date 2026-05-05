//! Projection360 keyword inference tests.

use super::super::tables::inferred_keyword_float_f32;
use super::super::*;
use super::common::*;

use crate::materials::host_data::{
    MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};

/// Builds a `StemEmbeddedPropertyIds` that mirrors the projection360 family -- every
/// uniform-field probe used by the keyword inference is registered. Texture-binding
/// pids live on `EmbeddedSharedKeywordIds` so they don't need per-stem registration.
fn projection360_ids(reg: &PropertyIdRegistry) -> StemEmbeddedPropertyIds {
    let mut ids = StemEmbeddedPropertyIds::minimal_for_tests(reg);
    for field_name in [
        "_FOV",
        "_PerspectiveFOV",
        "_TextureLerp",
        "_CubeLOD",
        "_MaxIntensity",
        "_Tint0",
    ] {
        ids.uniform_field_ids
            .insert(field_name.to_string(), reg.intern(field_name));
    }
    ids
}

/// Inserts the default full-sphere `_FOV` value `Projection360Material` writes after
/// `OnAwake()` (`FieldOfView = (360 deg, 180 deg)` converted to radians).
fn set_full_sphere_fov(store: &mut MaterialPropertyStore, reg: &PropertyIdRegistry, mat: i32) {
    store.set_material(
        mat,
        reg.intern("_FOV"),
        MaterialPropertyValue::Float4([std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0]),
    );
}

/// Packs a host texture id with an explicit kind tag, matching the shared
/// `IdPacker<TextureAssetType>` layout `unpack_host_texture_packed` decodes.
fn packed_texture(asset_id: i32, kind: crate::assets::texture::HostTextureAssetKind) -> i32 {
    let type_bits = 3u32;
    let pack_type_shift = 32u32.saturating_sub(type_bits);
    asset_id | ((kind as i32) << pack_type_shift)
}

/// Default `Projection360` materials send `_FOV = (TAU, PI, 0, 0)` -- the OUTSIDE-mode
/// inference must leave every keyword field at zero so the fragment shader's existing
/// fallthrough behaves like Unity's default `OUTSIDE_CLIP` (and the choice is moot
/// since every direction is in-FOV anyway).
#[test]
fn projection360_full_sphere_fov_keeps_outside_keywords_zero() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    store.set_material(
        30,
        reg.intern("_FOV"),
        MaterialPropertyValue::Float4([std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0]),
    );
    for field_name in ["OUTSIDE_CLIP", "OUTSIDE_COLOR", "OUTSIDE_CLAMP"] {
        assert_eq!(
            inferred_keyword_float_f32(field_name, &store, lookup(30), &ids),
            Some(0.0),
            "{field_name} should be 0 for full-sphere FOV"
        );
    }
}

/// Narrow FOV is what the video player writes -- the renderer must enable
/// `OUTSIDE_CLAMP` so the partial-FOV pixels render edge-clamped instead of being
/// discarded by the default-clip fallthrough.
#[test]
fn projection360_narrow_fov_enables_outside_clamp() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    let one_deg = std::f32::consts::PI / 180.0;
    store.set_material(
        31,
        reg.intern("_FOV"),
        MaterialPropertyValue::Float4([one_deg, one_deg, 0.0, 0.0]),
    );
    assert_eq!(
        inferred_keyword_float_f32("OUTSIDE_CLAMP", &store, lookup(31), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("OUTSIDE_CLIP", &store, lookup(31), &ids),
        Some(0.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("OUTSIDE_COLOR", &store, lookup(31), &ids),
        Some(0.0)
    );
}

/// Hemispherical FOVs (`180 deg x 180 deg`) are partial in X -- must enable `OUTSIDE_CLAMP`.
#[test]
fn projection360_hemispherical_fov_enables_outside_clamp() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    store.set_material(
        32,
        reg.intern("_FOV"),
        MaterialPropertyValue::Float4([std::f32::consts::PI, std::f32::consts::PI, 0.0, 0.0]),
    );
    assert_eq!(
        inferred_keyword_float_f32("OUTSIDE_CLAMP", &store, lookup(32), &ids),
        Some(1.0)
    );
}

/// Full-azimuth, half-elevation FOV (`360 deg x 90 deg`) is partial in Y -- must enable
/// `OUTSIDE_CLAMP`.
#[test]
fn projection360_full_azimuth_half_elevation_enables_outside_clamp() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    store.set_material(
        33,
        reg.intern("_FOV"),
        MaterialPropertyValue::Float4([
            std::f32::consts::TAU,
            std::f32::consts::FRAC_PI_2,
            0.0,
            0.0,
        ]),
    );
    assert_eq!(
        inferred_keyword_float_f32("OUTSIDE_CLAMP", &store, lookup(33), &ids),
        Some(1.0)
    );
}

/// Float-precision residuals from the host's degrees->radians conversion
/// (`* PI / 180`) must still classify whole-sphere defaults as full-sphere.
#[test]
fn projection360_full_sphere_tolerates_float_residual() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    let fov_x = 360.0_f32 * std::f32::consts::PI / 180.0;
    let fov_y = 180.0_f32 * std::f32::consts::PI / 180.0;
    store.set_material(
        34,
        reg.intern("_FOV"),
        MaterialPropertyValue::Float4([fov_x, fov_y, 0.0, 0.0]),
    );
    assert_eq!(
        inferred_keyword_float_f32("OUTSIDE_CLAMP", &store, lookup(34), &ids),
        Some(0.0),
        "host-computed (TAU, PI) within tolerance must remain full-sphere"
    );
}

/// `Mode.Perspective` is the video player's mode -- `_PerspectiveFOV` is sent only when
/// `Projection.Value == Mode.Perspective`, so its presence drives `_PERSPECTIVE = 1`.
/// Without this inference the shader silently downgrades to `_VIEW`, producing a
/// vertical-line-stretched-vertically render because object-space view-direction's
/// `atan2(view_dir.x, view_dir.z)` is near-constant across a flat quad.
#[test]
fn projection360_perspective_fov_present_enables_perspective_keyword() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 40);
    store.set_material(
        40,
        reg.intern("_PerspectiveFOV"),
        MaterialPropertyValue::Float4([
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
            0.0,
        ]),
    );
    assert_eq!(
        inferred_keyword_float_f32("_PERSPECTIVE", &store, lookup(40), &ids),
        Some(1.0)
    );
}

/// `Mode.View` (default) never sends `_PerspectiveFOV` -- the keyword stays off and the
/// shader falls through to the object-space view direction.
#[test]
fn projection360_no_perspective_fov_keeps_perspective_keyword_off() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 41);
    assert_eq!(
        inferred_keyword_float_f32("_PERSPECTIVE", &store, lookup(41), &ids),
        Some(0.0)
    );
}

/// Cubemap textures bound on `_MainCube` or `_SecondCube` enable the `CUBEMAP` keyword
/// (no `_CubeLOD` written -> fixed-mip cubemap path).
#[test]
fn projection360_main_cube_present_enables_cubemap_keyword() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 42);
    store.set_material(
        42,
        reg.intern("_MainCube"),
        MaterialPropertyValue::Texture(packed_texture(
            3,
            crate::assets::texture::HostTextureAssetKind::Cubemap,
        )),
    );
    assert_eq!(
        inferred_keyword_float_f32("CUBEMAP", &store, lookup(42), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("CUBEMAP_LOD", &store, lookup(42), &ids),
        Some(0.0)
    );
}

/// `_CubeLOD` written alongside a cubemap routes to `CUBEMAP_LOD` (mirrors host's
/// `CubemapLOD.Value.HasValue` predicate).
#[test]
fn projection360_cubemap_with_cube_lod_enables_cubemap_lod_keyword() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 43);
    store.set_material(
        43,
        reg.intern("_MainCube"),
        MaterialPropertyValue::Texture(packed_texture(
            3,
            crate::assets::texture::HostTextureAssetKind::Cubemap,
        )),
    );
    store.set_material(
        43,
        reg.intern("_CubeLOD"),
        MaterialPropertyValue::Float(2.0),
    );
    assert_eq!(
        inferred_keyword_float_f32("CUBEMAP_LOD", &store, lookup(43), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("CUBEMAP", &store, lookup(43), &ids),
        Some(0.0)
    );
}

/// Secondary 2D texture bound on `_SecondTex` enables `SECOND_TEXTURE`.
#[test]
fn projection360_secondary_texture_enables_second_texture_keyword() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 44);
    store.set_material(
        44,
        reg.intern("_SecondTex"),
        MaterialPropertyValue::Texture(packed_texture(
            7,
            crate::assets::texture::HostTextureAssetKind::Texture2D,
        )),
    );
    assert_eq!(
        inferred_keyword_float_f32("SECOND_TEXTURE", &store, lookup(44), &ids),
        Some(1.0)
    );
}

/// `TextureLerp != 0` on its own enables `SECOND_TEXTURE` (mirrors the host's
/// `state2 = ... || TextureLerp.Value != 0f` predicate). Even without a secondary
/// asset, the host turns on the keyword so the shader's lerp branch runs.
#[test]
fn projection360_nonzero_texture_lerp_enables_second_texture_keyword() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 45);
    store.set_material(
        45,
        reg.intern("_TextureLerp"),
        MaterialPropertyValue::Float(0.5),
    );
    assert_eq!(
        inferred_keyword_float_f32("SECOND_TEXTURE", &store, lookup(45), &ids),
        Some(1.0)
    );
}

/// `_OffsetTex` texture binding enables the `_OFFSET` direction-perturbation path.
#[test]
fn projection360_offset_texture_enables_offset_keyword() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 46);
    store.set_material(
        46,
        reg.intern("_OffsetTex"),
        MaterialPropertyValue::Texture(packed_texture(
            11,
            crate::assets::texture::HostTextureAssetKind::Texture2D,
        )),
    );
    assert_eq!(
        inferred_keyword_float_f32("_OFFSET", &store, lookup(46), &ids),
        Some(1.0)
    );
}

/// `_MaxIntensity` is sent only when `MaxIntensity.HasValue || HDR texture` -- its
/// presence drives `_CLAMP_INTENSITY`.
#[test]
fn projection360_max_intensity_present_enables_clamp_intensity_keyword() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 47);
    store.set_material(
        47,
        reg.intern("_MaxIntensity"),
        MaterialPropertyValue::Float(8.0),
    );
    assert_eq!(
        inferred_keyword_float_f32("_CLAMP_INTENSITY", &store, lookup(47), &ids),
        Some(1.0)
    );
}

/// `TintTexture` set with `TintTextureMode == Direct` sends `_TintTex` but not
/// `_Tint0`/`_Tint1` -- `TINT_TEX_DIRECT` enables, `TINT_TEX_LERP` stays off.
#[test]
fn projection360_tint_texture_without_tint0_routes_to_direct() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 48);
    store.set_material(
        48,
        reg.intern("_TintTex"),
        MaterialPropertyValue::Texture(packed_texture(
            13,
            crate::assets::texture::HostTextureAssetKind::Texture2D,
        )),
    );
    assert_eq!(
        inferred_keyword_float_f32("TINT_TEX_DIRECT", &store, lookup(48), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("TINT_TEX_LERP", &store, lookup(48), &ids),
        Some(0.0)
    );
}

/// `TintTextureMode == Lerp` sends `_Tint0`/`_Tint1` alongside `_TintTex` --
/// `TINT_TEX_LERP` enables, `TINT_TEX_DIRECT` stays off.
#[test]
fn projection360_tint_texture_with_tint0_routes_to_lerp() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = projection360_ids(&reg);
    set_full_sphere_fov(&mut store, &reg, 49);
    store.set_material(
        49,
        reg.intern("_TintTex"),
        MaterialPropertyValue::Texture(packed_texture(
            13,
            crate::assets::texture::HostTextureAssetKind::Texture2D,
        )),
    );
    store.set_material(
        49,
        reg.intern("_Tint0"),
        MaterialPropertyValue::Float4([1.0, 0.0, 0.0, 1.0]),
    );
    assert_eq!(
        inferred_keyword_float_f32("TINT_TEX_LERP", &store, lookup(49), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("TINT_TEX_DIRECT", &store, lookup(49), &ids),
        Some(0.0)
    );
}

/// Stems without an `_FOV` uniform field (i.e., not `Projection360`) must not
/// participate in the inference -- the call falls through to the generic
/// keyword-like-field default of `0`.
#[test]
fn outside_mode_inference_does_not_fire_for_non_projection360_stems() {
    let store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    for field_name in [
        "OUTSIDE_CLIP",
        "OUTSIDE_COLOR",
        "OUTSIDE_CLAMP",
        "_PERSPECTIVE",
        "CUBEMAP",
        "CUBEMAP_LOD",
        "SECOND_TEXTURE",
        "_OFFSET",
        "_CLAMP_INTENSITY",
        "TINT_TEX_DIRECT",
        "TINT_TEX_LERP",
    ] {
        assert_eq!(
            inferred_keyword_float_f32(field_name, &store, lookup(35), &ids),
            Some(0.0),
            "{field_name} should default to 0 when stem has no _FOV"
        );
    }
}
