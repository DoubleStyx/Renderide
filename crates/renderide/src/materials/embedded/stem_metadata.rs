//! Reflection-derived metadata for composed embedded WGSL stems and the pipeline construction
//! that builds raster pipelines from them.
//!
//! The composed targets live under `shaders/target/` (built into the binary by `build.rs`).
//! [`EmbeddedStemQuery`] is the central cached query type; the cluster submodules expose
//! topic-specific free function shims (vertex streams, tangent fallback, pass counts,
//! blending / snapshot flags) on top of it.

mod blending;
mod passes;
mod tangent_fallback;
mod vertex_streams;

pub use blending::{
    embedded_stem_uses_alpha_blending, embedded_stem_uses_scene_color_snapshot,
    embedded_stem_uses_scene_depth_snapshot,
};
pub use passes::{
    embedded_stem_depth_prepass_pass, embedded_stem_pipeline_pass_count,
    embedded_stem_requires_intersection_pass,
};
pub use tangent_fallback::{EmbeddedTangentFallbackMode, embedded_stem_tangent_fallback_mode};
pub use vertex_streams::{
    EmbeddedVertexStreamMask, embedded_stem_needs_color_stream,
    embedded_stem_needs_extended_vertex_streams, embedded_stem_needs_tangent_stream,
    embedded_stem_needs_uv0_stream, embedded_stem_needs_uv1_stream, embedded_stem_needs_uv2_stream,
    embedded_stem_needs_uv3_stream, embedded_stem_needs_wide_uv_stream,
    embedded_stem_uses_raw_normal_payload, embedded_stem_uses_raw_tangent_payload,
    embedded_stem_uses_ui_transparent_fallback,
};

use hashbrown::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use crate::embedded_shaders;
use crate::materials::SHADER_PERM_MULTIVIEW_STEREO;
use crate::materials::ShaderPermutation;
use crate::materials::pipeline_build_error::PipelineBuildError;
use crate::materials::raster_pipeline::{
    ShaderModuleBuildRefs, VertexStreamToggles, create_reflective_raster_mesh_forward_pipelines,
};
use crate::materials::{
    MaterialBlendMode, MaterialPassDesc, MaterialRenderState, RasterFrontFace,
    RasterPrimitiveTopology, ReflectedRasterLayout, SnapshotRequirements,
    materialized_embedded_pass_for_blend_mode,
};

use self::tangent_fallback::tangent_fallback_mode_for_stem;
use self::vertex_streams::derive_vertex_stream_mask;

/// Host material identity and blend/render state for embedded raster pipeline creation (separate from WGSL build inputs).
pub(in crate::materials) struct EmbeddedRasterPipelineSource {
    /// Embedded shader stem (e.g. cache key).
    pub stem: Arc<str>,
    /// Stereo vs mono composed target.
    pub permutation: ShaderPermutation,
    /// Blend mode from the host material.
    pub blend_mode: MaterialBlendMode,
    /// Runtime depth/stencil/color overrides.
    pub render_state: MaterialRenderState,
    /// Front-face winding selected from draw transform handedness.
    pub front_face: RasterFrontFace,
    /// Primitive topology selected from the mesh's per-submesh topology.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// Cache key for reflection-derived metadata on a composed embedded target.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct EmbeddedStemMetadataKey {
    /// Base material stem before permutation composition.
    base_stem: String,
    /// Shader permutation used to select the composed target.
    permutation: ShaderPermutation,
}

/// Reflection-derived metadata used by draw collection, pre-warm, and pipeline setup.
#[derive(Clone, Debug)]
struct EmbeddedStemMetadata {
    /// Reflected WGSL layout when the composed target exists and validates.
    reflected: Option<ReflectedRasterLayout>,
    /// Tangent fallback policy for lazy mesh tangent upload.
    tangent_fallback_mode: EmbeddedTangentFallbackMode,
    /// Number of declared material passes submitted for this target.
    pass_count: usize,
    /// Whether any declared pass has a blend state.
    uses_alpha_blending: bool,
    /// Single forward pass that is safe to mirror with the generic depth prepass, if any.
    depth_prepass_pass: Option<MaterialPassDesc>,
}

impl EmbeddedStemMetadata {
    /// Exact mesh-forward vertex streams declared by reflected material pass vertex entries.
    fn vertex_stream_mask(&self) -> EmbeddedVertexStreamMask {
        derive_vertex_stream_mask(self.reflected.as_ref())
    }

    /// Whether reflected material pass vertex entries need any stream beyond UV0/color/UV1.
    fn needs_extended_vertex_streams(&self) -> bool {
        self.vertex_stream_mask().needs_extended_vertex_streams()
    }

    /// Tangent fallback policy for lazy mesh tangent upload.
    fn tangent_fallback_mode(&self) -> EmbeddedTangentFallbackMode {
        self.tangent_fallback_mode
    }
}

/// Reflection-derived metadata snapshot for one composed embedded material target.
///
/// Hot paths (draw collection, pre-warm, pipeline setup) call [`Self::for_stem`] once and then
/// query as many flags as they need without re-running naga reflection or hashing through the
/// metadata cache. The free `embedded_stem_*` / `embedded_wgsl_*` functions are thin shims over
/// this type.
#[derive(Clone, Debug)]
pub(in crate::materials) struct EmbeddedStemQuery {
    metadata: EmbeddedStemMetadata,
}

impl EmbeddedStemQuery {
    /// Builds a query for the composed target of `(base_stem, permutation)`.
    pub fn for_stem(base_stem: &str, permutation: ShaderPermutation) -> Self {
        Self {
            metadata: embedded_stem_metadata(base_stem, permutation),
        }
    }

    /// `true` when reflected material pass vertex entries use `@location(2)` as a UV0 stream.
    pub fn needs_uv0_stream(&self) -> bool {
        self.vertex_stream_mask().uv0
    }

    /// `true` when reflected material pass vertex entries use `@location(3)` as a `vec4<f32>` color stream.
    pub fn needs_color_stream(&self) -> bool {
        self.vertex_stream_mask().color
    }

    /// `true` when reflected material pass vertex entries use `@location(4)` as a `vec4<f32>` tangent stream.
    pub fn needs_tangent_stream(&self) -> bool {
        self.vertex_stream_mask().tangent
    }

    /// `true` when reflected material pass vertex entries use `@location(5)` as a `vec2<f32>` UV1 stream.
    pub fn needs_uv1_stream(&self) -> bool {
        self.vertex_stream_mask().uv1
    }

    /// `true` when reflected material pass vertex entries use `@location(6)` as a `vec2<f32>` UV2 stream.
    pub fn needs_uv2_stream(&self) -> bool {
        self.vertex_stream_mask().uv2
    }

    /// `true` when reflected material pass vertex entries use `@location(7)` as a `vec2<f32>` UV3 stream.
    pub fn needs_uv3_stream(&self) -> bool {
        self.vertex_stream_mask().uv3
    }

    /// `true` when reflected material pass vertex entries need the packed UV0-UV7 stream.
    pub fn needs_wide_uv_stream(&self) -> bool {
        self.vertex_stream_mask().wide_uvs
    }

    /// Exact mesh-forward vertex streams declared by reflected material pass vertex entries.
    pub fn vertex_stream_mask(&self) -> EmbeddedVertexStreamMask {
        self.metadata.vertex_stream_mask()
    }

    /// `true` when reflected material pass vertex entries need any stream beyond UV0/color/UV1.
    pub fn needs_extended_vertex_streams(&self) -> bool {
        self.metadata.needs_extended_vertex_streams()
    }

    /// Tangent fallback policy for lazy mesh tangent upload.
    pub fn tangent_fallback_mode(&self) -> EmbeddedTangentFallbackMode {
        self.metadata.tangent_fallback_mode()
    }

    /// Number of raster passes that will be submitted for one embedded draw batch.
    pub fn pipeline_pass_count(&self) -> usize {
        self.metadata.pass_count
    }

    /// `true` when any declared pass has a blend state (transparent material).
    pub fn uses_alpha_blending(&self) -> bool {
        self.metadata.uses_alpha_blending
    }

    /// Unified scene-snapshot requirement flags, or [`SnapshotRequirements::default`] when the
    /// stem failed to reflect.
    pub fn snapshot_requirements(&self) -> SnapshotRequirements {
        self.metadata
            .reflected
            .as_ref()
            .map_or(SnapshotRequirements::default(), |r| {
                r.snapshot_requirements()
            })
    }

    /// Returns the single forward pass that can be mirrored by the generic depth prepass.
    pub fn depth_prepass_pass(&self) -> Option<MaterialPassDesc> {
        self.metadata.depth_prepass_pass
    }
}

fn embedded_stem_metadata_cache()
-> &'static Mutex<HashMap<EmbeddedStemMetadataKey, EmbeddedStemMetadata>> {
    static CACHE: LazyLock<Mutex<HashMap<EmbeddedStemMetadataKey, EmbeddedStemMetadata>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    &CACHE
}

/// Returns cached metadata for an embedded material stem and permutation.
fn embedded_stem_metadata(base_stem: &str, permutation: ShaderPermutation) -> EmbeddedStemMetadata {
    let key = EmbeddedStemMetadataKey {
        base_stem: base_stem.to_string(),
        permutation,
    };
    let mut guard = embedded_stem_metadata_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(metadata) = guard.get(&key) {
        return metadata.clone();
    }

    let composed = embedded_composed_stem_for_permutation(base_stem, permutation);
    let wgsl = embedded_shaders::embedded_target_wgsl(&composed);
    let passes = embedded_shaders::embedded_target_passes(&composed);
    let vertex_entries = passes
        .iter()
        .map(|pass| pass.vertex_entry)
        .collect::<Vec<_>>();
    let reflected = wgsl.and_then(|wgsl| {
        crate::materials::wgsl_reflect::reflect_raster_material_wgsl_with_vertex_entries(
            wgsl,
            &vertex_entries,
        )
        .ok()
    });
    let depth_prepass_pass = depth_prepass_pass_for_target(wgsl, reflected.as_ref(), passes);
    let metadata = EmbeddedStemMetadata {
        reflected,
        tangent_fallback_mode: tangent_fallback_mode_for_stem(base_stem),
        pass_count: passes.len().max(1),
        uses_alpha_blending: passes.iter().any(|p| p.blend.is_some()),
        depth_prepass_pass,
    };
    guard.insert(key, metadata.clone());
    metadata
}

fn depth_prepass_pass_for_target(
    wgsl: Option<&str>,
    reflected: Option<&ReflectedRasterLayout>,
    passes: &[MaterialPassDesc],
) -> Option<MaterialPassDesc> {
    let wgsl = wgsl?;
    let reflected = reflected?;
    let [pass] = passes else {
        return None;
    };
    let snapshots = reflected.snapshot_requirements();
    let pass_is_opaque_forward = matches!(pass.name, "forward" | "forward_two_sided");
    (pass_is_opaque_forward
        && pass.blend.is_none()
        && pass.depth_write
        && !pass.alpha_to_coverage
        && !wgsl.contains("discard")
        && snapshots == SnapshotRequirements::default())
    .then_some(*pass)
}

/// Composed target stem for an embedded base stem (e.g. `unlit_default` -> `unlit_multiview`).
pub(in crate::materials) fn embedded_composed_stem_for_permutation(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> String {
    if permutation.0 == SHADER_PERM_MULTIVIEW_STEREO.0 {
        if base_stem.ends_with("_default") {
            return format!("{}_multiview", base_stem.trim_end_matches("_default"));
        }
        return base_stem.to_string();
    }
    if base_stem.ends_with("_multiview") {
        return format!("{}_default", base_stem.trim_end_matches("_multiview"));
    }
    base_stem.to_string()
}

pub(in crate::materials) fn build_embedded_wgsl(
    stem: &Arc<str>,
    permutation: ShaderPermutation,
) -> Result<String, PipelineBuildError> {
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), permutation);
    let wgsl = embedded_shaders::embedded_target_wgsl(&composed)
        .ok_or_else(|| PipelineBuildError::MissingEmbeddedShader(composed.clone()))?;
    Ok(wgsl.to_string())
}

pub(in crate::materials) fn create_embedded_render_pipelines(
    source: EmbeddedRasterPipelineSource,
    refs: ShaderModuleBuildRefs<'_>,
) -> Result<Vec<wgpu::RenderPipeline>, PipelineBuildError> {
    let EmbeddedRasterPipelineSource {
        stem,
        permutation,
        blend_mode,
        render_state,
        front_face,
        primitive_topology,
    } = source;
    let streams = VertexStreamToggles {
        include_uv_vertex_buffer: true,
        include_color_vertex_buffer: true,
        include_uv1_vertex_buffer: true,
    };
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), permutation);
    let shader = refs.with_label(format!("embedded_raster_material__{composed}"));
    let declared_passes = embedded_shaders::embedded_target_passes(&composed);
    if declared_passes.is_empty() {
        // Build script enforces that every material WGSL declares at least one `//#pass`.
        return Err(PipelineBuildError::MissingEmbeddedShader(format!(
            "{composed}: embedded material stem has no declared passes"
        )));
    }
    let materialized_passes = declared_passes
        .iter()
        .map(|p| {
            materialized_embedded_pass_for_blend_mode(
                stem.as_ref(),
                p,
                effective_blend_mode_for_stem(stem.as_ref(), blend_mode),
            )
        })
        .collect::<Vec<_>>();
    create_reflective_raster_mesh_forward_pipelines(
        shader,
        streams,
        &materialized_passes,
        render_state,
        front_face,
        primitive_topology,
    )
}

fn effective_blend_mode_for_stem(stem: &str, blend_mode: MaterialBlendMode) -> MaterialBlendMode {
    if blend_mode == MaterialBlendMode::StemDefault
        && embedded_stem_uses_ui_transparent_fallback(stem)
    {
        MaterialBlendMode::UnityBlend { src: 5, dst: 10 }
    } else {
        blend_mode
    }
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use super::*;
    use crate::materials::ShaderPermutation;
    use crate::materials::{ReflectedVertexInput, ReflectedVertexInputFormat};

    pub(super) fn query_with_vertex_inputs(inputs: Vec<ReflectedVertexInput>) -> EmbeddedStemQuery {
        let max_location = inputs.iter().map(|input| input.location).max();
        EmbeddedStemQuery {
            metadata: EmbeddedStemMetadata {
                reflected: Some(ReflectedRasterLayout {
                    layout_fingerprint: 0,
                    material_entries: Vec::new(),
                    per_draw_entries: Vec::new(),
                    material_uniform: None,
                    material_group1_names: HashMap::new(),
                    vs_vertex_inputs: inputs,
                    vs_max_vertex_location: max_location,
                    uses_scene_depth_snapshot: false,
                    uses_scene_color_snapshot: false,
                    requires_intersection_pass: false,
                }),
                tangent_fallback_mode: EmbeddedTangentFallbackMode::default(),
                pass_count: 1,
                uses_alpha_blending: false,
                depth_prepass_pass: None,
            },
        }
    }

    #[test]
    fn metadata_flags_distinguish_uv1_only_from_color_and_extended_streams() {
        let uv1_only = query_with_vertex_inputs(vec![
            ReflectedVertexInput {
                location: 0,
                format: ReflectedVertexInputFormat::Float32x4,
            },
            ReflectedVertexInput {
                location: 1,
                format: ReflectedVertexInputFormat::Float32x4,
            },
            ReflectedVertexInput {
                location: 2,
                format: ReflectedVertexInputFormat::Float32x2,
            },
            ReflectedVertexInput {
                location: 5,
                format: ReflectedVertexInputFormat::Float32x2,
            },
        ]);
        assert!(uv1_only.needs_uv0_stream());
        assert!(uv1_only.needs_uv1_stream());
        assert!(!uv1_only.needs_color_stream());
        assert!(!uv1_only.needs_extended_vertex_streams());

        let color = query_with_vertex_inputs(vec![ReflectedVertexInput {
            location: 3,
            format: ReflectedVertexInputFormat::Float32x4,
        }]);
        assert!(color.needs_color_stream());
        assert!(!color.needs_uv1_stream());
        assert!(!color.needs_extended_vertex_streams());

        let tangent = query_with_vertex_inputs(vec![ReflectedVertexInput {
            location: 4,
            format: ReflectedVertexInputFormat::Float32x4,
        }]);
        assert!(tangent.needs_extended_vertex_streams());
    }

    #[test]
    fn metadata_flags_cover_common_material_classes() {
        let mono = ShaderPermutation(0);

        assert_eq!(embedded_stem_pipeline_pass_count("null_default", mono), 1);
        assert!(!embedded_stem_uses_scene_color_snapshot(
            "null_default",
            mono
        ));
        assert!(!embedded_stem_requires_intersection_pass(
            "null_default",
            mono
        ));
        assert!(!embedded_stem_uses_scene_depth_snapshot(
            "null_default",
            mono
        ));
        assert!(!embedded_stem_needs_color_stream("null_default", mono));

        assert!(embedded_stem_needs_color_stream(
            "ui_textunlit_default",
            mono
        ));
        assert!(embedded_stem_needs_color_stream("unlit_default", mono));
        assert!(embedded_stem_needs_color_stream(
            "unlit_default",
            SHADER_PERM_MULTIVIEW_STEREO
        ));
        assert!(!embedded_stem_needs_extended_vertex_streams(
            "ui_textunlit_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "ui_textunlit_default",
            mono
        ));

        assert!(embedded_stem_needs_extended_vertex_streams(
            "ui_circlesegment_default",
            mono
        ));
        assert!(embedded_stem_needs_extended_vertex_streams(
            "ui_circlesegment_default",
            SHADER_PERM_MULTIVIEW_STEREO
        ));

        assert!(embedded_stem_uses_alpha_blending("circle_default"));
    }

    #[test]
    fn ui_stem_default_blend_uses_alpha_until_host_state_arrives() {
        assert_eq!(
            effective_blend_mode_for_stem("ui_unlit_default", MaterialBlendMode::StemDefault),
            MaterialBlendMode::UnityBlend { src: 5, dst: 10 }
        );
        assert_eq!(
            effective_blend_mode_for_stem("ui_unlit_default", MaterialBlendMode::Opaque),
            MaterialBlendMode::Opaque
        );
        assert_eq!(
            effective_blend_mode_for_stem("unlit_default", MaterialBlendMode::StemDefault),
            MaterialBlendMode::StemDefault
        );
    }

    #[test]
    fn metadata_flags_cover_xstoon_material_class() {
        let mono = ShaderPermutation(0);

        assert_eq!(
            embedded_stem_pipeline_pass_count("xstoon2.0_default", mono),
            1
        );
        assert!(embedded_stem_needs_extended_vertex_streams(
            "xstoon2.0_default",
            mono
        ));
        assert!(!embedded_stem_uses_scene_color_snapshot(
            "xstoon2.0_default",
            mono
        ));
        assert!(!embedded_stem_requires_intersection_pass(
            "xstoon2.0_default",
            mono
        ));
        assert!(!embedded_stem_uses_scene_depth_snapshot(
            "xstoon2.0_default",
            mono
        ));
    }
}
