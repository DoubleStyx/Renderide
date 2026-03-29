//! [`ShaderKey`] describes how a drawable's shader was resolved: optional host shader asset id
//! plus the builtin [`PipelineVariant`](super::PipelineVariant) used when no native Renderide
//! shader route applies.

use crate::assets::NativeShaderRoute;

use super::PipelineVariant;

/// Host shader identity and fallback variant from the pre-host-resolution path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ShaderKey {
    /// Shader asset id from a `MaterialPropertyUpdateType::set_shader` batch for this drawable's
    /// material property block, when present.
    pub host_shader_asset_id: Option<i32>,
    /// Variant that would apply without host shader selection (debug UV, PBR, skinned, stencil, etc.).
    pub fallback_variant: PipelineVariant,
}

impl ShaderKey {
    /// Builds a key with no host shader override.
    pub const fn builtin_only(fallback_variant: PipelineVariant) -> Self {
        Self {
            host_shader_asset_id: None,
            fallback_variant,
        }
    }

    /// Effective pipeline variant for batching and GPU pipeline lookup.
    ///
    /// When [`Self::host_shader_asset_id`] is set, non-MRT non-skinned non-overlay draws use
    /// [`PipelineVariant::Material`] for world-unlit and unsupported host shaders so they bind a
    /// concrete Renderide WGSL shader instead of inheriting the global PBR fallback.
    /// Native PBS / UI routes stay on their own variants.
    #[allow(clippy::too_many_arguments)]
    pub fn effective_variant(
        self,
        shader_debug_override_force_legacy: bool,
        material_block_id: i32,
        use_mrt: bool,
        is_skinned: bool,
        is_overlay: bool,
        native_shader_route: NativeShaderRoute,
    ) -> PipelineVariant {
        if shader_debug_override_force_legacy {
            return self.fallback_variant;
        }
        if self.host_shader_asset_id.is_none()
            || material_block_id < 0
            || use_mrt
            || is_skinned
            || is_overlay
        {
            return self.fallback_variant;
        }
        match native_shader_route {
            NativeShaderRoute::WorldUnlit | NativeShaderRoute::Unsupported => {
                PipelineVariant::Material {
                    material_id: material_block_id,
                }
            }
            NativeShaderRoute::PbsMetallic
            | NativeShaderRoute::UiUnlit
            | NativeShaderRoute::UiTextUnlit => self.fallback_variant,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ShaderKey;
    use crate::assets::NativeShaderRoute;
    use crate::gpu::PipelineVariant;

    #[test]
    fn effective_variant_uses_material_when_world_unlit() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::Pbr,
        };
        let v = k.effective_variant(false, 7, false, false, false, NativeShaderRoute::WorldUnlit);
        assert_eq!(v, PipelineVariant::Material { material_id: 7 });
    }

    #[test]
    fn effective_variant_unsupported_uses_material() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::Pbr,
        };
        let v = k.effective_variant(
            false,
            7,
            false,
            false,
            false,
            NativeShaderRoute::Unsupported,
        );
        assert_eq!(v, PipelineVariant::Material { material_id: 7 });
    }

    #[test]
    fn effective_variant_pbs_metallic_stays_on_fallback_pbr() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::Pbr,
        };
        let v = k.effective_variant(
            false,
            7,
            false,
            false,
            false,
            NativeShaderRoute::PbsMetallic,
        );
        assert_eq!(v, PipelineVariant::Pbr);
    }

    #[test]
    fn effective_variant_respects_legacy_override() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::NormalDebug,
        };
        let v = k.effective_variant(true, 7, false, false, false, NativeShaderRoute::Unsupported);
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    #[test]
    fn effective_variant_skips_overlay_and_skinned() {
        let k = ShaderKey {
            host_shader_asset_id: Some(1),
            fallback_variant: PipelineVariant::Skinned,
        };
        assert_eq!(
            k.effective_variant(false, 3, false, true, false, NativeShaderRoute::Unsupported,),
            PipelineVariant::Skinned
        );
        let k2 = ShaderKey {
            host_shader_asset_id: Some(1),
            fallback_variant: PipelineVariant::NormalDebug,
        };
        assert_eq!(
            k2.effective_variant(false, 3, false, false, true, NativeShaderRoute::Unsupported,),
            PipelineVariant::NormalDebug
        );
    }
}
