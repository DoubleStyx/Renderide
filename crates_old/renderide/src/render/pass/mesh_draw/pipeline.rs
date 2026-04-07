//! Pipeline variant resolution for mesh draw recording (MRT, PBR, orthographic overlay).

use super::types::MeshDrawParams;
use crate::gpu::PipelineVariant;

/// Resolves the pipeline variant for a draw group, applying MRT/PBR and orthographic overrides.
///
/// Native UI variants ([`PipelineVariant::NativeUiUnlit`] / text / stencil) are returned unchanged
/// even when `params.use_mrt` is true, so `UI_Unlit` / `UI_TextUnlit` WGSL continues to render in
/// MRT sessions.
///
/// [`PipelineVariant::Material`] (host-unlit pilot) is only kept when **not** using MRT. When MRT is
/// active (RTAO G-buffer), the variant must be remapped—see [`mesh_pipeline_variant_for_mrt`]—or
/// wgpu would bind a single-target host-unlit pipeline to a three-target render pass.
pub fn resolve_pipeline_for_group(
    variant: &PipelineVariant,
    params: &MeshDrawParams,
    is_overlay_group: bool,
) -> PipelineVariant {
    let pbr_ray_query = params
        .pbr_scene
        .as_ref()
        .is_some_and(|p| p.use_ray_tracing_scene);
    resolve_pipeline_variant_core(
        *variant,
        params.use_mrt,
        params.use_pbr,
        params.pbr_scene.is_some(),
        pbr_ray_query,
        params.overlay_orthographic,
        is_overlay_group,
    )
}

/// Pure routing for [`resolve_pipeline_for_group`]; unit-tested without a full [`MeshDrawParams`].
pub(crate) fn resolve_pipeline_variant_core(
    variant: PipelineVariant,
    use_mrt: bool,
    use_pbr: bool,
    has_pbr_scene: bool,
    pbr_ray_query: bool,
    overlay_orthographic: bool,
    is_overlay_group: bool,
) -> PipelineVariant {
    if matches!(
        variant,
        PipelineVariant::NativeUiUnlit { .. }
            | PipelineVariant::NativeUiTextUnlit { .. }
            | PipelineVariant::NativeUiUnlitStencil { .. }
            | PipelineVariant::NativeUiTextUnlitStencil { .. }
    ) {
        return variant;
    }
    if matches!(variant, PipelineVariant::Material { .. }) && !use_mrt {
        return variant;
    }
    overlay_pipeline_variant_for_orthographic(
        &mesh_pipeline_variant_for_mrt(&variant, use_mrt, use_pbr, has_pbr_scene, pbr_ray_query),
        overlay_orthographic && is_overlay_group,
    )
}

/// Maps overlay pipeline variant to no-depth variant when orthographic overlay is used.
/// Orthographic screen-space UI should not be occluded by scene depth.
/// MaskWrite/MaskClear variants are not mapped to no-depth since they need stencil.
pub fn overlay_pipeline_variant_for_orthographic(
    variant: &PipelineVariant,
    overlay_orthographic: bool,
) -> PipelineVariant {
    if !overlay_orthographic {
        return *variant;
    }
    match variant {
        PipelineVariant::NormalDebug => PipelineVariant::OverlayNoDepthNormalDebug,
        PipelineVariant::UvDebug => PipelineVariant::OverlayNoDepthUvDebug,
        PipelineVariant::Skinned => PipelineVariant::OverlayNoDepthSkinned,
        PipelineVariant::OverlayStencilMaskWrite
        | PipelineVariant::OverlayStencilMaskClear
        | PipelineVariant::OverlayStencilMaskWriteSkinned
        | PipelineVariant::OverlayStencilMaskClearSkinned => *variant,
        PipelineVariant::Pbr
        | PipelineVariant::PbrHostAlbedo
        | PipelineVariant::PbrMRT
        | PipelineVariant::PbrRayQuery
        | PipelineVariant::PbrMRTRayQuery
        | PipelineVariant::SkinnedPbr
        | PipelineVariant::SkinnedPbrMRT
        | PipelineVariant::SkinnedPbrRayQuery
        | PipelineVariant::SkinnedPbrMRTRayQuery => *variant,
        _ => *variant,
    }
}

/// Maps non-overlay pipeline variant to MRT or PBR variant.
/// When use_mrt, outputs color/position/normal for RTAO. When use_pbr && !use_mrt, uses PBR.
/// Falls back to debug variants when cluster buffers are unavailable.
///
/// For [`PipelineVariant::Material`], when `use_mrt` is true this returns
/// [`PipelineVariant::NormalDebugMRT`] (no dedicated host-unlit MRT shader). Native UI variants map
/// the same way here; [`resolve_pipeline_for_group`] keeps native UI unchanged at record time and
/// relies on native UI WGSL for MRT compatibility.
pub fn mesh_pipeline_variant_for_mrt(
    variant: &PipelineVariant,
    use_mrt: bool,
    use_pbr: bool,
    has_pbr_scene: bool,
    pbr_ray_query: bool,
) -> PipelineVariant {
    if matches!(
        variant,
        PipelineVariant::Material { .. }
            | PipelineVariant::NativeUiUnlit { .. }
            | PipelineVariant::NativeUiTextUnlit { .. }
            | PipelineVariant::NativeUiUnlitStencil { .. }
            | PipelineVariant::NativeUiTextUnlitStencil { .. }
    ) {
        if use_mrt {
            return PipelineVariant::NormalDebugMRT;
        }
        return *variant;
    }
    if !has_pbr_scene {
        return match variant {
            PipelineVariant::Pbr
            | PipelineVariant::PbrHostAlbedo
            | PipelineVariant::PbrRayQuery => PipelineVariant::NormalDebug,
            PipelineVariant::SkinnedPbr | PipelineVariant::SkinnedPbrRayQuery => {
                PipelineVariant::Skinned
            }
            PipelineVariant::PbrMRT | PipelineVariant::PbrMRTRayQuery => {
                PipelineVariant::NormalDebugMRT
            }
            PipelineVariant::SkinnedPbrMRT | PipelineVariant::SkinnedPbrMRTRayQuery => {
                PipelineVariant::SkinnedMRT
            }
            _ => *variant,
        };
    }
    if use_mrt && use_pbr && has_pbr_scene {
        return match variant {
            PipelineVariant::NormalDebug => {
                if pbr_ray_query {
                    PipelineVariant::PbrMRTRayQuery
                } else {
                    PipelineVariant::PbrMRT
                }
            }
            PipelineVariant::UvDebug => PipelineVariant::UvDebugMRT,
            PipelineVariant::Skinned => {
                if pbr_ray_query {
                    PipelineVariant::SkinnedPbrMRTRayQuery
                } else {
                    PipelineVariant::SkinnedPbrMRT
                }
            }
            PipelineVariant::Pbr
            | PipelineVariant::PbrHostAlbedo
            | PipelineVariant::PbrRayQuery => {
                if pbr_ray_query {
                    PipelineVariant::PbrMRTRayQuery
                } else {
                    PipelineVariant::PbrMRT
                }
            }
            PipelineVariant::SkinnedPbr | PipelineVariant::SkinnedPbrRayQuery => {
                if pbr_ray_query {
                    PipelineVariant::SkinnedPbrMRTRayQuery
                } else {
                    PipelineVariant::SkinnedPbrMRT
                }
            }
            _ => *variant,
        };
    }
    if use_mrt {
        return match variant {
            PipelineVariant::NormalDebug => PipelineVariant::NormalDebugMRT,
            PipelineVariant::UvDebug => PipelineVariant::UvDebugMRT,
            PipelineVariant::Skinned => PipelineVariant::SkinnedMRT,
            _ => *variant,
        };
    }
    if use_pbr && has_pbr_scene {
        return match variant {
            PipelineVariant::NormalDebug => {
                if pbr_ray_query {
                    PipelineVariant::PbrRayQuery
                } else {
                    PipelineVariant::Pbr
                }
            }
            PipelineVariant::Skinned => {
                if pbr_ray_query {
                    PipelineVariant::SkinnedPbrRayQuery
                } else {
                    PipelineVariant::SkinnedPbr
                }
            }
            PipelineVariant::Pbr
            | PipelineVariant::PbrHostAlbedo
            | PipelineVariant::PbrRayQuery => {
                if pbr_ray_query {
                    PipelineVariant::PbrRayQuery
                } else {
                    PipelineVariant::Pbr
                }
            }
            PipelineVariant::SkinnedPbr | PipelineVariant::SkinnedPbrRayQuery => {
                if pbr_ray_query {
                    PipelineVariant::SkinnedPbrRayQuery
                } else {
                    PipelineVariant::SkinnedPbr
                }
            }
            _ => *variant,
        };
    }
    *variant
}

#[cfg(test)]
mod mesh_pipeline_variant_mrt_tests {
    use crate::gpu::PipelineVariant;

    use super::{mesh_pipeline_variant_for_mrt, resolve_pipeline_variant_core};

    #[test]
    fn resolve_core_material_maps_to_normal_debug_mrt_when_use_mrt() {
        let v = PipelineVariant::Material { material_id: 7 };
        let out = resolve_pipeline_variant_core(v, true, false, true, false, false, false);
        assert_eq!(out, PipelineVariant::NormalDebugMRT);
    }

    #[test]
    fn resolve_core_material_unchanged_when_no_mrt() {
        let v = PipelineVariant::Material { material_id: 7 };
        let out = resolve_pipeline_variant_core(v, false, false, true, false, false, false);
        assert_eq!(out, v);
    }

    #[test]
    fn material_variant_maps_to_normal_debug_mrt_when_use_mrt() {
        let v = PipelineVariant::Material { material_id: 1 };
        let out = mesh_pipeline_variant_for_mrt(&v, true, false, true, false);
        assert_eq!(out, PipelineVariant::NormalDebugMRT);
    }

    #[test]
    fn material_variant_unchanged_when_no_mrt() {
        let v = PipelineVariant::Material { material_id: 1 };
        let out = mesh_pipeline_variant_for_mrt(&v, false, false, true, false);
        assert_eq!(out, v);
    }

    #[test]
    fn native_ui_unlit_unchanged_for_mrt_in_direct_helper() {
        let v = PipelineVariant::NativeUiUnlit { material_id: 2 };
        let out = mesh_pipeline_variant_for_mrt(&v, true, false, true, false);
        assert_eq!(out, PipelineVariant::NormalDebugMRT);
    }

    #[test]
    fn pbr_upgrades_to_pbr_mrt_when_scene_ready() {
        let v = PipelineVariant::Pbr;
        let out = mesh_pipeline_variant_for_mrt(&v, true, true, true, false);
        assert_eq!(out, PipelineVariant::PbrMRT);
    }
}
