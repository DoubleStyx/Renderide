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

pub use passes::{embedded_stem_depth_prepass_pass, embedded_stem_pipeline_pass_count};
pub use tangent_fallback::EmbeddedTangentFallbackMode;
pub use vertex_streams::EmbeddedVertexStreamMask;

use hashbrown::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use crate::embedded_shaders;
use crate::materials::SHADER_PERM_MULTIVIEW_STEREO;
use crate::materials::ShaderPermutation;
use crate::materials::pipeline_build_error::PipelineBuildError;
use crate::materials::raster_pipeline::{
    ShaderModuleBuildRefs, VertexStreamToggles, create_reflective_raster_mesh_forward_pipelines,
};
#[cfg(test)]
use crate::materials::render_queue::UNITY_RENDER_QUEUE_GEOMETRY;
use crate::materials::{
    MaterialBlendMode, MaterialPassDesc, MaterialRenderState, RasterFrontFace,
    RasterPrimitiveTopology, ReflectedRasterLayout, SceneColorSnapshotMode, SnapshotRequirements,
    materialized_embedded_pass_for_blend_mode,
};

use self::tangent_fallback::tangent_fallback_mode_for_stem;
use self::vertex_streams::{
    derive_vertex_stream_mask, stem_uses_raw_normal_payload, stem_uses_raw_tangent_payload,
    stem_uses_ui_transparent_fallback,
};

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
#[derive(Clone, Copy, Debug)]
struct EmbeddedStemMetadata {
    /// Exact mesh-forward vertex streams declared by reflected material pass vertex entries.
    vertex_stream_mask: EmbeddedVertexStreamMask,
    /// Scene-snapshot resources required by the reflected material target.
    snapshot_requirements: SnapshotRequirements,
    /// Tangent fallback policy for lazy mesh tangent upload.
    tangent_fallback_mode: EmbeddedTangentFallbackMode,
    /// Number of declared material passes submitted for this target.
    pass_count: usize,
    /// Whether `@location(4)` carries raw shader payload rather than a geometric tangent.
    uses_raw_tangent_payload: bool,
    /// Whether `@location(1)` carries raw shader payload rather than a lighting normal.
    uses_raw_normal_payload: bool,
    /// Whether this UI stem should fall back to transparent state until host state arrives.
    uses_ui_transparent_fallback: bool,
    /// Whether any declared pass has a blend state.
    uses_alpha_blending: bool,
    /// Whether any declared blended pass writes depth by default.
    uses_blended_depth_write: bool,
    /// Whether declared blended passes include authored front/back cull ordering.
    uses_two_sided_transparency: bool,
    /// How this material expects scene-color snapshots to be refreshed.
    scene_color_snapshot_mode: SceneColorSnapshotMode,
    /// Shader default render queue used when material `_RenderQueue` is negative or absent.
    default_render_queue: i32,
    /// Single forward pass that is safe to mirror with the generic depth prepass, if any.
    depth_prepass_pass: Option<MaterialPassDesc>,
}

impl EmbeddedStemMetadata {
    /// Exact mesh-forward vertex streams declared by reflected material pass vertex entries.
    fn vertex_stream_mask(&self) -> EmbeddedVertexStreamMask {
        self.vertex_stream_mask
    }

    /// Whether reflected material pass vertex entries need any stream beyond UV0/color/UV1.
    #[cfg(test)]
    fn needs_extended_vertex_streams(&self) -> bool {
        self.vertex_stream_mask.needs_extended_vertex_streams()
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
/// metadata cache.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbeddedStemQuery {
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
    #[cfg(test)]
    pub fn needs_uv0_stream(&self) -> bool {
        self.vertex_stream_mask().uv0
    }

    /// `true` when reflected material pass vertex entries use `@location(3)` as a `vec4<f32>` color stream.
    #[cfg(test)]
    pub fn needs_color_stream(&self) -> bool {
        self.vertex_stream_mask().color
    }

    /// `true` when reflected material pass vertex entries use `@location(4)` as a `vec4<f32>` tangent stream.
    #[cfg(test)]
    pub fn needs_tangent_stream(&self) -> bool {
        self.vertex_stream_mask().tangent
    }

    /// `true` when reflected material pass vertex entries use `@location(5)` as a `vec2<f32>` UV1 stream.
    #[cfg(test)]
    pub fn needs_uv1_stream(&self) -> bool {
        self.vertex_stream_mask().uv1
    }

    /// `true` when reflected material pass vertex entries use `@location(6)` as a `vec2<f32>` UV2 stream.
    #[cfg(test)]
    pub fn needs_uv2_stream(&self) -> bool {
        self.vertex_stream_mask().uv2
    }

    /// `true` when reflected material pass vertex entries use `@location(7)` as a `vec2<f32>` UV3 stream.
    #[cfg(test)]
    pub fn needs_uv3_stream(&self) -> bool {
        self.vertex_stream_mask().uv3
    }

    /// `true` when reflected material pass vertex entries need any packed wide UV page.
    #[cfg(test)]
    pub fn needs_wide_uv_stream(&self) -> bool {
        self.vertex_stream_mask().needs_wide_uv_streams()
    }

    /// Exact mesh-forward vertex streams declared by reflected material pass vertex entries.
    pub fn vertex_stream_mask(&self) -> EmbeddedVertexStreamMask {
        self.metadata.vertex_stream_mask()
    }

    /// `true` when reflected material pass vertex entries need any stream beyond UV0/color/UV1.
    #[cfg(test)]
    pub fn needs_extended_vertex_streams(&self) -> bool {
        self.metadata.needs_extended_vertex_streams()
    }

    /// Tangent fallback policy for lazy mesh tangent upload.
    pub fn tangent_fallback_mode(&self) -> EmbeddedTangentFallbackMode {
        self.metadata.tangent_fallback_mode()
    }

    /// `true` when `@location(4)` carries raw shader payload rather than a geometric tangent.
    pub fn uses_raw_tangent_payload(&self) -> bool {
        self.metadata.uses_raw_tangent_payload
    }

    /// `true` when `@location(1)` carries raw shader payload rather than a lighting normal.
    pub fn uses_raw_normal_payload(&self) -> bool {
        self.metadata.uses_raw_normal_payload
    }

    /// `true` when this UI stem should fall back to transparent state until host state arrives.
    pub fn uses_ui_transparent_fallback(&self) -> bool {
        self.metadata.uses_ui_transparent_fallback
    }

    /// Number of raster passes that will be submitted for one embedded draw batch.
    pub fn pipeline_pass_count(&self) -> usize {
        self.metadata.pass_count
    }

    /// `true` when any declared pass has a blend state (transparent material).
    pub fn uses_alpha_blending(&self) -> bool {
        self.metadata.uses_alpha_blending
    }

    /// `true` when any declared blended pass writes depth by default.
    pub fn uses_blended_depth_write(&self) -> bool {
        self.metadata.uses_blended_depth_write
    }

    /// `true` when declared blended passes include authored front/back cull ordering.
    pub fn uses_two_sided_transparency(&self) -> bool {
        self.metadata.uses_two_sided_transparency
    }

    /// Unified scene-snapshot requirement flags, or [`SnapshotRequirements::default`] when the
    /// stem failed to reflect.
    pub fn snapshot_requirements(&self) -> SnapshotRequirements {
        self.metadata.snapshot_requirements
    }

    /// How this material expects scene-color snapshots to be refreshed.
    pub fn scene_color_snapshot_mode(&self) -> SceneColorSnapshotMode {
        self.metadata.scene_color_snapshot_mode
    }

    /// Shader default render queue used when material `_RenderQueue` is negative or absent.
    pub fn default_render_queue(&self) -> i32 {
        self.metadata.default_render_queue
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
        return *metadata;
    }

    let composed = embedded_composed_stem_for_permutation(base_stem, permutation);
    let wgsl = embedded_shaders::embedded_target_wgsl(&composed);
    let passes = embedded_shaders::embedded_target_passes(&composed);
    let default_render_queue = embedded_shaders::embedded_target_default_render_queue(&composed);
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
    let snapshot_requirements = reflected
        .as_ref()
        .map_or_else(SnapshotRequirements::default, |r| r.snapshot_requirements());
    let metadata = EmbeddedStemMetadata {
        vertex_stream_mask: derive_vertex_stream_mask(reflected.as_ref()),
        snapshot_requirements,
        tangent_fallback_mode: tangent_fallback_mode_for_stem(base_stem),
        pass_count: passes.len().max(1),
        uses_raw_tangent_payload: stem_uses_raw_tangent_payload(base_stem),
        uses_raw_normal_payload: stem_uses_raw_normal_payload(base_stem),
        uses_ui_transparent_fallback: stem_uses_ui_transparent_fallback(base_stem),
        uses_alpha_blending: passes.iter().any(|p| p.blend.is_some()),
        uses_blended_depth_write: passes.iter().any(|p| p.blend.is_some() && p.depth_write),
        uses_two_sided_transparency: passes_use_two_sided_transparency(passes),
        scene_color_snapshot_mode: scene_color_snapshot_mode_for_stem(
            base_stem,
            snapshot_requirements,
        ),
        default_render_queue,
        depth_prepass_pass,
    };
    guard.insert(key, metadata);
    metadata
}

/// Derives the refresh policy for scene-color snapshots from reflection and shader-family stem.
fn scene_color_snapshot_mode_for_stem(
    base_stem: &str,
    requirements: SnapshotRequirements,
) -> SceneColorSnapshotMode {
    if !requirements.uses_scene_color {
        return SceneColorSnapshotMode::None;
    }
    if stem_uses_named_background_grab(base_stem) {
        SceneColorSnapshotMode::NamedBackgroundGrab
    } else {
        SceneColorSnapshotMode::PerObjectGrab
    }
}

/// Returns true for filter stems that use ShaderLab's named `_BackgroundTexture` grab.
fn stem_uses_named_background_grab(base_stem: &str) -> bool {
    let stem = base_stem
        .strip_suffix("_default")
        .or_else(|| base_stem.strip_suffix("_multiview"))
        .unwrap_or(base_stem);
    matches!(stem, "blur" | "pixelate" | "refract")
}

/// Returns whether a target declares blended front- and back-culled passes.
fn passes_use_two_sided_transparency(passes: &[MaterialPassDesc]) -> bool {
    let has_front = passes
        .iter()
        .any(|pass| pass.blend.is_some() && pass.cull_mode == Some(wgpu::Face::Front));
    let has_back = passes
        .iter()
        .any(|pass| pass.blend.is_some() && pass.cull_mode == Some(wgpu::Face::Back));
    has_front && has_back
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
    let pass_is_opaque_forward = pass.pass_type == crate::materials::PassType::Forward;
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

/// Returns device features required by the composed embedded target for `stem` and `permutation`.
pub(in crate::materials) fn embedded_required_features_for_permutation(
    stem: &Arc<str>,
    permutation: ShaderPermutation,
) -> wgpu::Features {
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), permutation);
    embedded_shaders::embedded_target_required_features(&composed)
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
    if blend_mode == MaterialBlendMode::StemDefault && stem_uses_ui_transparent_fallback(stem) {
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
                vertex_stream_mask: derive_vertex_stream_mask(Some(&ReflectedRasterLayout {
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
                })),
                snapshot_requirements: SnapshotRequirements::default(),
                tangent_fallback_mode: EmbeddedTangentFallbackMode::default(),
                pass_count: 1,
                uses_raw_tangent_payload: false,
                uses_raw_normal_payload: false,
                uses_ui_transparent_fallback: false,
                uses_alpha_blending: false,
                uses_blended_depth_write: false,
                uses_two_sided_transparency: false,
                scene_color_snapshot_mode: SceneColorSnapshotMode::None,
                default_render_queue: UNITY_RENDER_QUEUE_GEOMETRY,
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
        let null = EmbeddedStemQuery::for_stem("null_default", mono);
        let null_snapshots = null.snapshot_requirements();
        assert!(!null_snapshots.uses_scene_color);
        assert!(!null_snapshots.requires_intersection_pass);
        assert!(!null_snapshots.uses_scene_depth);
        assert!(!null.needs_color_stream());

        let ui_text = EmbeddedStemQuery::for_stem("ui_textunlit_default", mono);
        assert!(ui_text.needs_color_stream());
        assert!(!ui_text.needs_extended_vertex_streams());
        assert!(ui_text.snapshot_requirements().uses_scene_depth);

        assert!(EmbeddedStemQuery::for_stem("unlit_default", mono).needs_color_stream());
        assert!(
            EmbeddedStemQuery::for_stem("unlit_default", SHADER_PERM_MULTIVIEW_STEREO)
                .needs_color_stream()
        );
        assert!(
            EmbeddedStemQuery::for_stem("ui_circlesegment_default", mono)
                .needs_extended_vertex_streams()
        );
        assert!(
            EmbeddedStemQuery::for_stem("ui_circlesegment_default", SHADER_PERM_MULTIVIEW_STEREO)
                .needs_extended_vertex_streams()
        );
        assert!(EmbeddedStemQuery::for_stem("circle_default", mono).uses_alpha_blending());
    }

    #[test]
    fn query_covers_cached_metadata_fields() {
        let mono = ShaderPermutation(0);

        let debug = EmbeddedStemQuery::for_stem("debug_default", mono);
        assert!(debug.needs_uv1_stream());
        assert!(debug.needs_uv2_stream());
        assert!(debug.needs_uv3_stream());
        assert!(debug.needs_wide_uv_stream());
        assert_eq!(
            EmbeddedStemQuery::for_stem("pbsmetallic_default", mono).tangent_fallback_mode(),
            EmbeddedTangentFallbackMode::GenerateMissing
        );
        assert_eq!(
            EmbeddedStemQuery::for_stem("blur_default", mono).scene_color_snapshot_mode(),
            SceneColorSnapshotMode::NamedBackgroundGrab
        );
        assert!(
            EmbeddedStemQuery::for_stem("furfx-basic-10layer_default", mono)
                .uses_blended_depth_write()
        );
        assert!(
            EmbeddedStemQuery::for_stem("pbsdualsidedtransparent_default", mono)
                .uses_two_sided_transparency()
        );
        assert_eq!(
            EmbeddedStemQuery::for_stem("unlit_default", mono).default_render_queue(),
            2650
        );
        assert_eq!(
            EmbeddedStemQuery::for_stem("pixelate_perobject_default", mono).default_render_queue(),
            3500
        );
        assert_eq!(
            EmbeddedStemQuery::for_stem("pbsdistancelerptransparent_default", mono)
                .default_render_queue(),
            UNITY_RENDER_QUEUE_GEOMETRY
        );
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
        let xstoon = EmbeddedStemQuery::for_stem("xstoon2.0_default", mono);
        assert!(xstoon.needs_extended_vertex_streams());
        let snapshots = xstoon.snapshot_requirements();
        assert!(!snapshots.uses_scene_color);
        assert!(!snapshots.requires_intersection_pass);
        assert!(!snapshots.uses_scene_depth);
    }
}
