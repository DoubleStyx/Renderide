//! Vertex-stream queries derived from pass vertex-entry reflection on a composed embedded WGSL stem.

#[cfg(test)]
use crate::materials::ShaderPermutation;
use crate::materials::{ReflectedRasterLayout, ReflectedVertexInputFormat};

#[cfg(test)]
use super::EmbeddedStemQuery;

/// Mesh-forward UV channel count exposed to material vertex shaders.
pub const EMBEDDED_UV_STREAM_COUNT: usize = 8;

/// Shader input locations for UV0 through UV7.
pub const EMBEDDED_UV_SHADER_LOCATIONS: [u32; EMBEDDED_UV_STREAM_COUNT] =
    [2, 5, 6, 7, 8, 9, 10, 11];

/// Exact vertex streams declared by one composed embedded material vertex shader.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EmbeddedVertexStreamMask {
    /// UV0 stream at `@location(2)`.
    pub uv0: bool,
    /// Vertex color stream at `@location(3)`.
    pub color: bool,
    /// Tangent stream at `@location(4)`.
    pub tangent: bool,
    /// UV1 stream at `@location(5)`.
    pub uv1: bool,
    /// UV2 stream at `@location(6)`.
    pub uv2: bool,
    /// UV3 stream at `@location(7)`.
    pub uv3: bool,
    /// Packed UV0-UV3 stream for 3D/4D low UV inputs.
    pub wide_low_uvs: bool,
    /// Packed UV4-UV7 stream for high UV inputs.
    pub wide_high_uvs: bool,
}

impl EmbeddedVertexStreamMask {
    /// `true` when any stream outside UV0/color/UV1 is needed.
    pub fn needs_extended_vertex_streams(self) -> bool {
        self.tangent || self.uv2 || self.uv3 || self.wide_low_uvs || self.wide_high_uvs
    }

    /// `true` when any wide UV page is needed.
    #[cfg(test)]
    pub fn needs_wide_uv_streams(self) -> bool {
        self.wide_low_uvs || self.wide_high_uvs
    }
}

/// Builds an [`EmbeddedVertexStreamMask`] from a reflected WGSL layout (empty if reflection failed).
pub(super) fn derive_vertex_stream_mask(
    reflected: Option<&ReflectedRasterLayout>,
) -> EmbeddedVertexStreamMask {
    let mut mask = EmbeddedVertexStreamMask::default();
    let Some(reflected) = reflected else {
        return mask;
    };
    for input in &reflected.vs_vertex_inputs {
        match (input.location, input.format) {
            (2, ReflectedVertexInputFormat::Float32x2) => mask.uv0 = true,
            (3, ReflectedVertexInputFormat::Float32x4) => mask.color = true,
            (4, ReflectedVertexInputFormat::Float32x4) => mask.tangent = true,
            (5, ReflectedVertexInputFormat::Float32x2) => mask.uv1 = true,
            (6, ReflectedVertexInputFormat::Float32x2) => mask.uv2 = true,
            (7, ReflectedVertexInputFormat::Float32x2) => mask.uv3 = true,
            (location, format) if uv_channel_from_location(location).is_some() => {
                apply_uv_requirement(&mut mask, location, format);
            }
            _ => {}
        }
    }
    mask
}

fn apply_uv_requirement(
    mask: &mut EmbeddedVertexStreamMask,
    location: u32,
    format: ReflectedVertexInputFormat,
) {
    let Some(channel) = uv_channel_from_location(location) else {
        return;
    };
    let supported = matches!(
        format,
        ReflectedVertexInputFormat::Float32x2
            | ReflectedVertexInputFormat::Float32x3
            | ReflectedVertexInputFormat::Float32x4
    );
    if !supported {
        return;
    }
    match channel {
        0 => mask.uv0 = true,
        1 => mask.uv1 = true,
        2 => mask.uv2 = true,
        3 => mask.uv3 = true,
        _ => {}
    }
    if channel >= 4 {
        mask.wide_high_uvs = true;
    } else if format != ReflectedVertexInputFormat::Float32x2 {
        mask.wide_low_uvs = true;
    }
}

fn uv_channel_from_location(location: u32) -> Option<usize> {
    EMBEDDED_UV_SHADER_LOCATIONS
        .iter()
        .position(|candidate| *candidate == location)
}

/// `true` when composed embedded WGSL's reflected pass vertex entries use UV0.
#[cfg(test)]
pub fn embedded_stem_needs_uv0_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_uv0_stream()
}

/// `true` when composed embedded WGSL's reflected pass vertex entries use vertex color.
#[cfg(test)]
pub fn embedded_stem_needs_color_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_color_stream()
}

/// `true` when composed embedded WGSL's reflected pass vertex entries use tangent.
#[cfg(test)]
pub fn embedded_stem_needs_tangent_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_tangent_stream()
}

/// `true` when composed embedded WGSL's reflected pass vertex entries use UV1.
#[cfg(test)]
pub fn embedded_stem_needs_uv1_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_uv1_stream()
}

/// `true` when composed embedded WGSL's reflected pass vertex entries use UV2.
#[cfg(test)]
pub fn embedded_stem_needs_uv2_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_uv2_stream()
}

/// `true` when composed embedded WGSL's reflected pass vertex entries use UV3.
#[cfg(test)]
pub fn embedded_stem_needs_uv3_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_uv3_stream()
}

/// `true` when composed embedded WGSL's reflected pass vertex entries need any packed wide UV page.
#[cfg(test)]
pub fn embedded_stem_needs_wide_uv_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_wide_uv_stream()
}

/// `true` when composed embedded WGSL's reflected pass vertex entries use tangent/UV2/UV3 or wide UVs.
#[cfg(test)]
pub fn embedded_stem_needs_extended_vertex_streams(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_extended_vertex_streams()
}

/// `true` when `@location(4)` carries raw shader payload rather than a geometric tangent.
pub(super) fn stem_uses_raw_tangent_payload(base_stem: &str) -> bool {
    matches!(
        canonical_stem_name(base_stem),
        "billboardunlit" | "ui_circlesegment" | "ui_unlit"
    )
}

/// `true` when `@location(4)` carries raw shader payload rather than a geometric tangent.
#[cfg(test)]
pub fn embedded_stem_uses_raw_tangent_payload(base_stem: &str) -> bool {
    stem_uses_raw_tangent_payload(base_stem)
}

/// `true` when `@location(1)` carries raw shader payload rather than a lighting normal.
pub(super) fn stem_uses_raw_normal_payload(base_stem: &str) -> bool {
    matches!(
        canonical_stem_name(base_stem),
        "billboardunlit" | "textunlit" | "ui_textunlit"
    )
}

/// `true` when `@location(1)` carries raw shader payload rather than a lighting normal.
#[cfg(test)]
pub fn embedded_stem_uses_raw_normal_payload(base_stem: &str) -> bool {
    stem_uses_raw_normal_payload(base_stem)
}

/// `true` when the stem should fall back to transparent UI state until host state arrives.
pub(super) fn stem_uses_ui_transparent_fallback(base_stem: &str) -> bool {
    matches!(
        canonical_stem_name(base_stem),
        "ui_circlesegment" | "ui_textunlit" | "ui_unlit"
    )
}

/// `true` when the stem should fall back to transparent UI state until host state arrives.
#[cfg(test)]
pub fn embedded_stem_uses_ui_transparent_fallback(base_stem: &str) -> bool {
    stem_uses_ui_transparent_fallback(base_stem)
}

fn canonical_stem_name(base_stem: &str) -> &str {
    base_stem
        .strip_suffix("_default")
        .or_else(|| base_stem.strip_suffix("_multiview"))
        .unwrap_or(base_stem)
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use crate::materials::SHADER_PERM_MULTIVIEW_STEREO;
    use crate::materials::{
        ReflectedRasterLayout, ReflectedVertexInput, ReflectedVertexInputFormat, ShaderPermutation,
    };

    use super::{
        derive_vertex_stream_mask, embedded_stem_needs_color_stream,
        embedded_stem_needs_extended_vertex_streams, embedded_stem_needs_tangent_stream,
        embedded_stem_needs_uv0_stream, embedded_stem_needs_uv1_stream,
        embedded_stem_needs_uv2_stream, embedded_stem_needs_uv3_stream,
        embedded_stem_needs_wide_uv_stream, embedded_stem_uses_raw_normal_payload,
        embedded_stem_uses_raw_tangent_payload, embedded_stem_uses_ui_transparent_fallback,
    };

    fn reflected_with_inputs(inputs: Vec<ReflectedVertexInput>) -> ReflectedRasterLayout {
        ReflectedRasterLayout {
            layout_fingerprint: 0,
            material_entries: Vec::new(),
            per_draw_entries: Vec::new(),
            material_uniform: None,
            material_group1_names: HashMap::new(),
            vs_vertex_inputs: inputs,
            vs_max_vertex_location: None,
            uses_scene_depth_snapshot: false,
            uses_scene_color_snapshot: false,
            requires_intersection_pass: false,
        }
    }

    #[test]
    fn null_no_uv0_stream() {
        assert!(!embedded_stem_needs_uv0_stream(
            "null_default",
            ShaderPermutation(0)
        ));
        assert!(!embedded_stem_needs_uv0_stream(
            "null_default",
            SHADER_PERM_MULTIVIEW_STEREO
        ));
    }

    #[test]
    fn unlit_pass_vertex_entries_need_color_stream() {
        assert!(embedded_stem_needs_color_stream(
            "unlit_default",
            ShaderPermutation(0),
        ));
        assert!(embedded_stem_needs_color_stream(
            "unlit_default",
            SHADER_PERM_MULTIVIEW_STEREO,
        ));
    }

    /// Regression guard: the compiled-render-graph per-view pre-warm uploads a mesh's
    /// tangent / UV1..3 streams only when its material stem is flagged as needing extended
    /// vertex streams. If this ever flips for `ui_circlesegment` (the context-menu material,
    /// whose vertex shader declares `@location(0..=7)`), VR draws will start silently skipping
    /// again because the per-view record path uses an immutable `MeshPool` and cannot upload
    /// the streams on demand.
    #[test]
    fn ui_circlesegment_needs_extended_vertex_streams_both_permutations() {
        assert!(embedded_stem_needs_extended_vertex_streams(
            "ui_circlesegment_default",
            ShaderPermutation(0),
        ));
        assert!(embedded_stem_needs_extended_vertex_streams(
            "ui_circlesegment_default",
            SHADER_PERM_MULTIVIEW_STEREO,
        ));
    }

    /// Counterpart to `ui_circlesegment_needs_extended_vertex_streams_both_permutations`: the
    /// text material fits in `@location(0..=3)`, so it must never be flagged as needing
    /// extended streams. If this flips, the VR pre-warm would try to upload empty tangent /
    /// UV1..3 buffers for every text draw.
    #[test]
    fn ui_textunlit_does_not_need_extended_vertex_streams() {
        assert!(!embedded_stem_needs_extended_vertex_streams(
            "ui_textunlit_default",
            ShaderPermutation(0),
        ));
        assert!(!embedded_stem_needs_extended_vertex_streams(
            "ui_textunlit_default",
            SHADER_PERM_MULTIVIEW_STEREO,
        ));
    }

    #[test]
    fn furfx_pass_vertex_entries_need_uv0_stream() {
        assert!(embedded_stem_needs_uv0_stream(
            "furfx-basic-10layer_default",
            ShaderPermutation(0),
        ));
        assert!(embedded_stem_needs_uv0_stream(
            "furfx-basic-10layer_default",
            SHADER_PERM_MULTIVIEW_STEREO,
        ));
    }

    #[test]
    fn furfx_modern_pass_vertex_entries_need_tangent_stream() {
        assert!(embedded_stem_needs_tangent_stream(
            "furfx-3.0-shell-10layer_default",
            ShaderPermutation(0),
        ));
        assert!(embedded_stem_needs_extended_vertex_streams(
            "furfx-3.0-shell-10layer_default",
            ShaderPermutation(0),
        ));
        assert!(embedded_stem_needs_tangent_stream(
            "furfx-3.0-shell-10layer_default",
            SHADER_PERM_MULTIVIEW_STEREO,
        ));
        assert!(embedded_stem_needs_extended_vertex_streams(
            "furfx-3.0-shell-10layer_default",
            SHADER_PERM_MULTIVIEW_STEREO,
        ));
    }

    #[test]
    fn debug_pass_vertex_entries_need_wide_uv_streams() {
        assert!(embedded_stem_needs_uv1_stream(
            "debug_default",
            ShaderPermutation(0),
        ));
        assert!(embedded_stem_needs_uv2_stream(
            "debug_default",
            ShaderPermutation(0),
        ));
        assert!(embedded_stem_needs_uv3_stream(
            "debug_default",
            ShaderPermutation(0),
        ));
        assert!(embedded_stem_needs_wide_uv_stream(
            "debug_default",
            ShaderPermutation(0),
        ));
    }

    #[test]
    fn reflected_vec4_uv0_requests_compact_uv0_and_wide_low_uvs() {
        let reflected = reflected_with_inputs(vec![ReflectedVertexInput {
            location: 2,
            format: ReflectedVertexInputFormat::Float32x4,
        }]);

        let mask = derive_vertex_stream_mask(Some(&reflected));

        assert!(mask.uv0);
        assert!(mask.wide_low_uvs);
        assert!(!mask.wide_high_uvs);
        assert!(mask.needs_extended_vertex_streams());
    }

    #[test]
    fn reflected_uv7_requests_wide_high_uvs_without_compact_uv_alias() {
        let reflected = reflected_with_inputs(vec![ReflectedVertexInput {
            location: 11,
            format: ReflectedVertexInputFormat::Float32x2,
        }]);

        let mask = derive_vertex_stream_mask(Some(&reflected));

        assert!(!mask.uv0);
        assert!(!mask.uv1);
        assert!(!mask.uv2);
        assert!(!mask.uv3);
        assert!(!mask.wide_low_uvs);
        assert!(mask.wide_high_uvs);
        assert!(mask.needs_extended_vertex_streams());
    }

    #[test]
    fn ui_payload_stems_mark_raw_semantics() {
        for stem in [
            "billboardunlit_default",
            "ui_circlesegment_default",
            "ui_unlit_default",
        ] {
            assert!(embedded_stem_uses_raw_tangent_payload(stem), "{stem}");
        }
        assert!(!embedded_stem_uses_raw_tangent_payload(
            "pbsmetallic_default"
        ));

        for stem in [
            "billboardunlit_default",
            "textunlit_default",
            "ui_textunlit_default",
        ] {
            assert!(embedded_stem_uses_raw_normal_payload(stem), "{stem}");
        }
        assert!(!embedded_stem_uses_raw_normal_payload("unlit_default"));
    }

    #[test]
    fn ui_stems_use_transparent_fallback_defaults() {
        for stem in [
            "ui_circlesegment_default",
            "ui_textunlit_default",
            "ui_unlit_default",
            "ui_unlit_multiview",
        ] {
            assert!(embedded_stem_uses_ui_transparent_fallback(stem), "{stem}");
        }
        assert!(!embedded_stem_uses_ui_transparent_fallback("unlit_default"));
    }
}
