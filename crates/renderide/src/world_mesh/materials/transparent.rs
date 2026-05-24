//! Transparent material compatibility classes derived from renderer-local material state.

use crate::materials::{MaterialBlendMode, MaterialRenderState};

use super::key::render_queue_uses_transparent_sorting;

/// Renderer-local transparent behavior bucket inferred from existing material and shader state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TransparentMaterialClass {
    /// Opaque or alpha-tested material that does not need transparent ordering.
    #[default]
    Opaque,
    /// Normal order-dependent alpha blending.
    OrderedAlpha,
    /// Transparent material whose effective pass writes depth.
    DepthWritingTransparent,
    /// Scene-color snapshot filter or grab-pass material.
    GrabPassFilter,
    /// Order-independent additive or multiplicative blend that can relax batching within a bucket.
    CommutativeBlend,
    /// Known two-sided transparent material that relies on authored front/back pass order.
    KnownTwoSidedTransparent,
    /// Transparent state shape not safe to optimize beyond stable compatibility ordering.
    CompatibilityFallback,
}

impl TransparentMaterialClass {
    /// Returns whether this class participates in transparent-style rendering behavior.
    #[inline]
    pub fn is_transparent(self) -> bool {
        !matches!(self, Self::Opaque)
    }

    /// Returns whether same-key draws in this class may group despite transparent sorting depth.
    #[inline]
    pub fn allows_relaxed_batching(self) -> bool {
        matches!(self, Self::CommutativeBlend)
    }

    /// Short label used by diagnostics and HUD text.
    pub fn label(self) -> &'static str {
        match self {
            Self::Opaque => "opaque",
            Self::OrderedAlpha => "ordered",
            Self::DepthWritingTransparent => "zwrite",
            Self::GrabPassFilter => "grab",
            Self::CommutativeBlend => "commutative",
            Self::KnownTwoSidedTransparent => "two-sided",
            Self::CompatibilityFallback => "fallback",
        }
    }
}

/// Inputs used to classify transparent material behavior from resolved renderer-local state.
#[derive(Clone, Copy, Debug)]
pub(super) struct TransparentMaterialClassInput {
    /// Effective Unity render queue after material override and fallback resolution.
    pub(super) render_queue: i32,
    /// Runtime depth, stencil, color, and cull state resolved from host properties.
    pub(super) render_state: MaterialRenderState,
    /// Material blend mode reconstructed from host `_SrcBlend` and `_DstBlend` values.
    pub(super) blend_mode: MaterialBlendMode,
    /// Whether the material is known to use alpha blending from shader metadata or host state.
    pub(super) alpha_blended: bool,
    /// Whether the shader samples a scene-color snapshot.
    pub(super) uses_scene_color_snapshot: bool,
    /// Whether the shader metadata declares a blended pass that writes depth by default.
    pub(super) uses_blended_depth_write: bool,
    /// Whether shader metadata identifies authored front/back transparent passes.
    pub(super) uses_two_sided_transparency: bool,
}

/// Classifies one resolved material into the most conservative transparent behavior bucket.
pub(super) fn transparent_class_for_material(
    input: TransparentMaterialClassInput,
) -> TransparentMaterialClass {
    let transparent_like = input.alpha_blended
        || render_queue_uses_transparent_sorting(
            input.render_queue,
            input.alpha_blended || input.blend_mode.is_transparent(),
        )
        || input.uses_scene_color_snapshot
        || input.render_state.depth_write == Some(false);
    if !transparent_like {
        return TransparentMaterialClass::Opaque;
    }

    if input.uses_scene_color_snapshot {
        return TransparentMaterialClass::GrabPassFilter;
    }

    if input.render_state.depth_write == Some(true)
        || (input.uses_blended_depth_write && input.render_state.depth_write != Some(false))
    {
        return TransparentMaterialClass::DepthWritingTransparent;
    }

    if input.uses_two_sided_transparency {
        return TransparentMaterialClass::KnownTwoSidedTransparent;
    }

    match input.blend_mode {
        MaterialBlendMode::UnityBlend { src: 1, dst: 1 }
        | MaterialBlendMode::UnityBlend { src: 2, dst: 0 } => {
            TransparentMaterialClass::CommutativeBlend
        }
        MaterialBlendMode::UnityBlend {
            src: 5 | 1,
            dst: 10,
        } => TransparentMaterialClass::OrderedAlpha,
        MaterialBlendMode::UnityBlend { .. } => TransparentMaterialClass::CompatibilityFallback,
        MaterialBlendMode::StemDefault if input.alpha_blended => {
            TransparentMaterialClass::OrderedAlpha
        }
        MaterialBlendMode::StemDefault | MaterialBlendMode::Opaque => {
            TransparentMaterialClass::CompatibilityFallback
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::materials::{MaterialBlendMode, MaterialRenderState};

    use super::{
        TransparentMaterialClass, TransparentMaterialClassInput, transparent_class_for_material,
    };

    /// Builds classification input with opaque defaults.
    fn input() -> TransparentMaterialClassInput {
        TransparentMaterialClassInput {
            render_queue: crate::materials::UNITY_RENDER_QUEUE_GEOMETRY,
            render_state: MaterialRenderState::default(),
            blend_mode: MaterialBlendMode::StemDefault,
            alpha_blended: false,
            uses_scene_color_snapshot: false,
            uses_blended_depth_write: false,
            uses_two_sided_transparency: false,
        }
    }

    #[test]
    fn plain_opaque_material_classifies_as_opaque() {
        assert_eq!(
            transparent_class_for_material(input()),
            TransparentMaterialClass::Opaque
        );
    }

    #[test]
    fn stem_alpha_blending_classifies_as_ordered_alpha() {
        let mut value = input();
        value.alpha_blended = true;

        assert_eq!(
            transparent_class_for_material(value),
            TransparentMaterialClass::OrderedAlpha
        );
    }

    #[test]
    fn additive_and_multiply_blends_classify_as_commutative() {
        for blend_mode in [
            MaterialBlendMode::UnityBlend { src: 1, dst: 1 },
            MaterialBlendMode::UnityBlend { src: 2, dst: 0 },
        ] {
            let mut value = input();
            value.blend_mode = blend_mode;
            value.render_queue = 2600;

            assert_eq!(
                transparent_class_for_material(value),
                TransparentMaterialClass::CommutativeBlend
            );
        }
    }

    #[test]
    fn late_opaque_queue_classifies_as_opaque_until_transparent_queue() {
        let mut value = input();
        value.render_queue = crate::materials::UNITY_RENDER_QUEUE_TRANSPARENT - 1;
        value.blend_mode = MaterialBlendMode::Opaque;

        assert_eq!(
            transparent_class_for_material(value),
            TransparentMaterialClass::Opaque
        );

        value.render_queue = crate::materials::UNITY_RENDER_QUEUE_TRANSPARENT;

        assert_eq!(
            transparent_class_for_material(value),
            TransparentMaterialClass::CompatibilityFallback
        );
    }

    #[test]
    fn scene_color_snapshot_classifies_as_grab_pass_filter() {
        let mut value = input();
        value.uses_scene_color_snapshot = true;

        assert_eq!(
            transparent_class_for_material(value),
            TransparentMaterialClass::GrabPassFilter
        );
    }

    #[test]
    fn blended_depth_write_classifies_as_depth_writing_transparent() {
        let mut value = input();
        value.alpha_blended = true;
        value.uses_blended_depth_write = true;

        assert_eq!(
            transparent_class_for_material(value),
            TransparentMaterialClass::DepthWritingTransparent
        );
    }

    #[test]
    fn zwrite_off_override_prevents_depth_writing_transparent_class() {
        let mut value = input();
        value.alpha_blended = true;
        value.uses_blended_depth_write = true;
        value.render_state.depth_write = Some(false);

        assert_eq!(
            transparent_class_for_material(value),
            TransparentMaterialClass::OrderedAlpha
        );
    }

    #[test]
    fn two_sided_transparency_classifies_before_ordered_alpha() {
        let mut value = input();
        value.alpha_blended = true;
        value.uses_two_sided_transparency = true;

        assert_eq!(
            transparent_class_for_material(value),
            TransparentMaterialClass::KnownTwoSidedTransparent
        );
    }

    #[test]
    fn unknown_transparent_queue_classifies_as_compatibility_fallback() {
        let mut value = input();
        value.render_queue = crate::materials::UNITY_RENDER_QUEUE_TRANSPARENT;

        assert_eq!(
            transparent_class_for_material(value),
            TransparentMaterialClass::CompatibilityFallback
        );
    }
}
