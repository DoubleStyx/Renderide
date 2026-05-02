//! Embedded mesh raster materials: composed WGSL stems under `shaders/target/` (see crate `build.rs`).

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
    MaterialBlendMode, MaterialRenderState, RasterFrontFace, ReflectedRasterLayout,
    ReflectedVertexInputFormat, SnapshotRequirements, materialized_pass_for_blend_mode,
};

/// Host material identity and blend/render state for embedded raster pipeline creation (separate from WGSL build inputs).
pub(crate) struct EmbeddedRasterPipelineSource {
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
    /// Number of declared material passes submitted for this target.
    pass_count: usize,
    /// Whether any declared pass has a blend state.
    uses_alpha_blending: bool,
}

impl EmbeddedStemMetadata {
    /// Whether `vs_main` declares a vertex input with this exact location and format.
    fn needs_vertex_input(&self, location: u32, format: ReflectedVertexInputFormat) -> bool {
        self.reflected.as_ref().is_some_and(|r| {
            r.vs_vertex_inputs
                .iter()
                .any(|input| input.location == location && input.format == format)
        })
    }

    /// Whether `vs_main` needs tangent/UV2/UV3 or an unsupported UV1 shape.
    fn needs_extended_vertex_streams(&self) -> bool {
        self.reflected.as_ref().is_some_and(|r| {
            r.vs_vertex_inputs.iter().any(|input| {
                matches!(input.location, 4 | 6 | 7)
                    || (input.location == 5
                        && input.format != ReflectedVertexInputFormat::Float32x2)
            })
        })
    }
}

/// Reflection-derived metadata snapshot for one composed embedded material target.
///
/// Hot paths (draw collection, pre-warm, pipeline setup) call [`Self::for_stem`] once and then
/// query as many flags as they need without re-running naga reflection or hashing through the
/// metadata cache. The free `embedded_stem_*` / `embedded_wgsl_*` functions are thin shims over
/// this type.
#[derive(Clone, Debug)]
pub struct EmbeddedStemQuery {
    metadata: EmbeddedStemMetadata,
}

impl EmbeddedStemQuery {
    /// Builds a query for the composed target of `(base_stem, permutation)`.
    pub fn for_stem(base_stem: &str, permutation: ShaderPermutation) -> Self {
        Self {
            metadata: embedded_stem_metadata(base_stem, permutation),
        }
    }

    /// `true` when `vs_main` uses `@location(2)` or higher (UV0 stream).
    pub fn needs_uv0_stream(&self) -> bool {
        self.metadata
            .needs_vertex_input(2, ReflectedVertexInputFormat::Float32x2)
    }

    /// `true` when `vs_main` uses `@location(3)` as a `vec4<f32>` color stream.
    pub fn needs_color_stream(&self) -> bool {
        self.metadata
            .needs_vertex_input(3, ReflectedVertexInputFormat::Float32x4)
    }

    /// `true` when `vs_main` uses `@location(5)` as a `vec2<f32>` UV1 stream.
    pub fn needs_uv1_stream(&self) -> bool {
        self.metadata
            .needs_vertex_input(5, ReflectedVertexInputFormat::Float32x2)
    }

    /// `true` when `vs_main` needs the full tangent/UV1-UV3 extended stream set.
    pub fn needs_extended_vertex_streams(&self) -> bool {
        self.metadata.needs_extended_vertex_streams()
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
    let reflected = embedded_shaders::embedded_target_wgsl(&composed)
        .and_then(|wgsl| crate::materials::wgsl_reflect::reflect_raster_material_wgsl(wgsl).ok());
    let passes = embedded_shaders::embedded_target_passes(&composed);
    let metadata = EmbeddedStemMetadata {
        reflected,
        pass_count: passes.len().max(1),
        uses_alpha_blending: passes.iter().any(|p| p.blend.is_some()),
    };
    guard.insert(key, metadata.clone());
    metadata
}

/// `true` when composed embedded WGSL's `vs_main` uses `@location(2)` as a UV0 vertex stream.
pub fn embedded_stem_needs_uv0_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_uv0_stream()
}

/// `true` when `vs_main` reflection reports a UV0 input at `@location(2)`.
pub fn embedded_wgsl_needs_uv0_stream(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_vertex_shader_needs_uv0_stream(wgsl_source)
}

/// `true` when composed embedded WGSL's `vs_main` uses `@location(3)` as vertex color.
pub fn embedded_stem_needs_color_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_color_stream()
}

/// `true` when `vs_main` reflection reports a color input at `@location(3)`.
pub fn embedded_wgsl_needs_color_stream(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_vertex_shader_needs_color_stream(wgsl_source)
}

/// `true` when composed embedded WGSL's `vs_main` uses `@location(5)` as UV1.
pub fn embedded_stem_needs_uv1_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_uv1_stream()
}

/// `true` when `vs_main` reflection reports a UV1 input at `@location(5)`.
pub fn embedded_wgsl_needs_uv1_stream(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_vertex_shader_needs_uv1_stream(wgsl_source)
}

/// `true` when composed embedded WGSL's `vs_main` uses `@location(4)` or higher.
pub fn embedded_stem_needs_extended_vertex_streams(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).needs_extended_vertex_streams()
}

/// `true` when `vs_main` reflection requires the full extended tangent/UV stream set.
pub fn embedded_wgsl_needs_extended_vertex_streams(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_vertex_shader_needs_extended_vertex_streams(wgsl_source)
}

/// Number of raster passes that will be submitted for one embedded draw batch.
pub fn embedded_stem_pipeline_pass_count(base_stem: &str, permutation: ShaderPermutation) -> usize {
    EmbeddedStemQuery::for_stem(base_stem, permutation).pipeline_pass_count()
}

/// `true` when reflection reports `_IntersectColor` in the material uniform (intersection forward subpass).
pub fn embedded_wgsl_requires_intersection_pass(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_raster_material_requires_intersection_pass(wgsl_source)
}

/// `true` when the composed embedded target uses an intersection subpass.
pub fn embedded_stem_requires_intersection_pass(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .requires_intersection_pass
}

/// `true` when reflection reports that the WGSL declares a scene-depth snapshot binding.
pub fn embedded_wgsl_uses_scene_depth_snapshot(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_raster_material_uses_scene_depth_snapshot(wgsl_source)
}

/// `true` when the composed embedded target declares a scene-depth snapshot binding.
pub fn embedded_stem_uses_scene_depth_snapshot(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .uses_scene_depth
}

/// `true` when reflection reports that the WGSL declares a scene-color snapshot binding.
pub fn embedded_wgsl_uses_scene_color_snapshot(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_raster_material_uses_scene_color_snapshot(wgsl_source)
}

/// `true` when the composed embedded target declares a scene-color snapshot binding.
pub fn embedded_stem_uses_scene_color_snapshot(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .uses_scene_color
}

/// Composed target stem for an embedded base stem (e.g. `unlit_default` -> `unlit_multiview`).
pub fn embedded_composed_stem_for_permutation(
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

pub(crate) fn build_embedded_wgsl(
    stem: &Arc<str>,
    permutation: ShaderPermutation,
) -> Result<String, PipelineBuildError> {
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), permutation);
    let wgsl = embedded_shaders::embedded_target_wgsl(&composed)
        .ok_or_else(|| PipelineBuildError::MissingEmbeddedShader(composed.clone()))?;
    Ok(wgsl.to_string())
}

pub(crate) fn create_embedded_render_pipelines(
    source: EmbeddedRasterPipelineSource,
    refs: ShaderModuleBuildRefs<'_>,
) -> Result<Vec<wgpu::RenderPipeline>, PipelineBuildError> {
    let EmbeddedRasterPipelineSource {
        stem,
        permutation,
        blend_mode,
        render_state,
        front_face,
    } = source;
    let shader = refs.with_label("embedded_raster_material");
    let streams = VertexStreamToggles {
        include_uv_vertex_buffer: true,
        include_color_vertex_buffer: true,
        include_uv1_vertex_buffer: true,
    };
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), permutation);
    let declared_passes = embedded_shaders::embedded_target_passes(&composed);
    if declared_passes.is_empty() {
        // Build script enforces that every material WGSL declares at least one `//#pass`.
        return Err(PipelineBuildError::MissingEmbeddedShader(format!(
            "{composed}: embedded material stem has no declared passes"
        )));
    }
    let materialized_passes = declared_passes
        .iter()
        .map(|p| materialized_pass_for_blend_mode(p, blend_mode))
        .collect::<Vec<_>>();
    create_reflective_raster_mesh_forward_pipelines(
        shader,
        streams,
        &materialized_passes,
        render_state,
        front_face,
    )
}

/// Returns whether the embedded material stem declares alpha blending (any `//#pass` directive
/// with non-None blend state). Memoized per base stem.
pub fn embedded_stem_uses_alpha_blending(base_stem: &str) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, ShaderPermutation(0)).uses_alpha_blending()
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use super::*;
    use crate::materials::MaterialPassState;
    use crate::materials::ShaderPermutation;
    use crate::materials::{ReflectedVertexInput, ReflectedVertexInputFormat};

    fn query_with_vertex_inputs(inputs: Vec<ReflectedVertexInput>) -> EmbeddedStemQuery {
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
                pass_count: 1,
                uses_alpha_blending: false,
            },
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
    fn metadata_flags_cover_snapshot_and_intersection_material_classes() {
        let mono = ShaderPermutation(0);

        assert!(embedded_stem_uses_scene_color_snapshot(
            "blur_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_color_snapshot(
            "blur_default",
            SHADER_PERM_MULTIVIEW_STEREO
        ));
        assert!(!embedded_stem_requires_intersection_pass(
            "blur_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "blur_default",
            mono
        ));

        assert!(embedded_stem_uses_scene_color_snapshot(
            "refract_default",
            mono
        ));
        assert!(!embedded_stem_requires_intersection_pass(
            "refract_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "refract_default",
            mono
        ));

        assert!(embedded_stem_requires_intersection_pass(
            "pbsintersect_default",
            mono
        ));
        assert!(!embedded_stem_uses_scene_color_snapshot(
            "pbsintersect_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "pbsintersect_default",
            mono
        ));
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

    #[test]
    fn first_shader_batch_fixed_state_stems_keep_expected_passes() {
        let circle = embedded_shaders::embedded_target_passes("circle_default");
        assert_eq!(circle.len(), 1);
        assert_eq!(circle[0].name, "transparent_rgb");
        assert_eq!(circle[0].material_state, MaterialPassState::Static);
        assert_eq!(circle[0].write_mask, wgpu::ColorWrites::COLOR);
        assert!(!circle[0].depth_write);
        assert_eq!(circle[0].cull_mode, None);
        assert!(circle[0].blend.is_some());

        let depth_projection = embedded_shaders::embedded_target_passes("depthprojection_default");
        assert_eq!(depth_projection.len(), 1);
        assert_eq!(depth_projection[0].name, "forward_two_sided");
        assert_eq!(
            depth_projection[0].material_state,
            MaterialPassState::Forward
        );
        assert_eq!(depth_projection[0].cull_mode, None);
    }
}
