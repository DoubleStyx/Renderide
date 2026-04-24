//! Embedded mesh raster materials: composed WGSL stems under `shaders/target/` (see crate `build.rs`).

use hashbrown::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::embedded_shaders;
use crate::materials::pipeline_build_error::PipelineBuildError;
use crate::materials::raster_pipeline::{
    create_reflective_raster_mesh_forward_pipelines, ShaderModuleBuildRefs, VertexStreamToggles,
};
use crate::materials::{
    default_pass_for_blend_mode, materialized_pass_for_blend_mode, MaterialBlendMode,
    MaterialRenderState,
};
use crate::pipelines::raster::SHADER_PERM_MULTIVIEW_STEREO;
use crate::pipelines::ShaderPermutation;

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
}

fn embedded_uv0_stream_cache() -> &'static Mutex<HashMap<String, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn embedded_color_stream_cache() -> &'static Mutex<HashMap<String, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn embedded_extended_vertex_stream_cache() -> &'static Mutex<HashMap<String, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn embedded_intersection_pass_cache() -> &'static Mutex<HashMap<String, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn embedded_grab_pass_cache() -> &'static Mutex<HashMap<String, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn embedded_alpha_blending_cache() -> &'static Mutex<HashMap<String, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `true` when composed embedded WGSL's `vs_main` uses `@location(2)` or higher (UV0 vertex stream).
///
/// Uses the same embedded source and reflection as the embedded raster pipeline for the given
/// [`ShaderPermutation`], independent of [`crate::backend::EmbeddedMaterialBindResources`].
///
/// Results are memoized per `(base_stem, permutation)` so draw collection and other hot paths do not
/// re-run naga reflection once per mesh draw.
pub fn embedded_stem_needs_uv0_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    let key = format!("{base_stem}:{}", permutation.0);
    let mut guard = embedded_uv0_stream_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(v) = guard.get(&key) {
        return *v;
    }
    let composed = embedded_composed_stem_for_permutation(base_stem, permutation);
    let v = embedded_shaders::embedded_target_wgsl(&composed)
        .map(embedded_wgsl_needs_uv0_stream)
        .unwrap_or(false);
    guard.insert(key, v);
    v
}

/// `true` when `vs_main` reflection reports a highest vertex `@location` index ≥ 2 (UV at `location(2)`).
pub fn embedded_wgsl_needs_uv0_stream(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_vertex_shader_needs_uv0_stream(wgsl_source)
}

/// `true` when composed embedded WGSL's `vs_main` uses `@location(3)` or higher (vertex color stream).
pub fn embedded_stem_needs_color_stream(base_stem: &str, permutation: ShaderPermutation) -> bool {
    let key = format!("{base_stem}:{}", permutation.0);
    let mut guard = embedded_color_stream_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(v) = guard.get(&key) {
        return *v;
    }
    let composed = embedded_composed_stem_for_permutation(base_stem, permutation);
    let v = embedded_shaders::embedded_target_wgsl(&composed)
        .map(embedded_wgsl_needs_color_stream)
        .unwrap_or(false);
    guard.insert(key, v);
    v
}

/// `true` when `vs_main` reflection reports a highest vertex `@location` index >= 3 (color at `location(3)`).
pub fn embedded_wgsl_needs_color_stream(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_vertex_shader_needs_color_stream(wgsl_source)
}

/// `true` when composed embedded WGSL's `vs_main` uses `@location(4)` or higher.
pub fn embedded_stem_needs_extended_vertex_streams(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    let key = format!("{base_stem}:{}", permutation.0);
    let mut guard = embedded_extended_vertex_stream_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(v) = guard.get(&key) {
        return *v;
    }
    let composed = embedded_composed_stem_for_permutation(base_stem, permutation);
    let v = embedded_shaders::embedded_target_wgsl(&composed)
        .map(embedded_wgsl_needs_extended_vertex_streams)
        .unwrap_or(false);
    guard.insert(key, v);
    v
}

/// `true` when `vs_main` reflection reports a highest vertex `@location` index >= 4.
pub fn embedded_wgsl_needs_extended_vertex_streams(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_raster_material_wgsl(wgsl_source)
        .ok()
        .and_then(|r| r.vs_max_vertex_location)
        .is_some_and(|m| m >= 4)
}

/// Number of raster passes that will be submitted for one embedded draw batch.
pub fn embedded_stem_pipeline_pass_count(base_stem: &str, permutation: ShaderPermutation) -> usize {
    let composed = embedded_composed_stem_for_permutation(base_stem, permutation);
    embedded_shaders::embedded_target_passes(&composed)
        .len()
        .max(1)
}

/// `true` when reflection reports a grab-pass material (uniform field `_GrabPass`).
pub fn embedded_wgsl_requires_grab_pass(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_raster_material_requires_grab_pass(wgsl_source)
}

/// `true` when the composed embedded target uses a grab pass (reflection of `_GrabPass`).
///
/// Memoized per `(base_stem, permutation)` like [`embedded_stem_needs_uv0_stream`].
pub fn embedded_stem_requires_grab_pass(base_stem: &str, permutation: ShaderPermutation) -> bool {
    let key = format!("{base_stem}:{}", permutation.0);
    let mut guard = embedded_grab_pass_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(v) = guard.get(&key) {
        return *v;
    }
    let composed = embedded_composed_stem_for_permutation(base_stem, permutation);
    let v = embedded_shaders::embedded_target_wgsl(&composed)
        .map(embedded_wgsl_requires_grab_pass)
        .unwrap_or(false);
    guard.insert(key, v);
    v
}

/// `true` when reflection reports `_IntersectColor` in the material uniform (intersection forward subpass).
pub fn embedded_wgsl_requires_intersection_pass(wgsl_source: &str) -> bool {
    crate::materials::wgsl_reflect::reflect_raster_material_requires_intersection_pass(wgsl_source)
}

/// `true` when the composed embedded target uses an intersection subpass (reflection of `_IntersectColor`).
///
/// Memoized per `(base_stem, permutation)` like [`embedded_stem_needs_uv0_stream`].
pub fn embedded_stem_requires_intersection_pass(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    let key = format!("{base_stem}:{}", permutation.0);
    let mut guard = embedded_intersection_pass_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(v) = guard.get(&key) {
        return *v;
    }
    let composed = embedded_composed_stem_for_permutation(base_stem, permutation);
    let v = embedded_shaders::embedded_target_wgsl(&composed)
        .map(embedded_wgsl_requires_intersection_pass)
        .unwrap_or(false);
    guard.insert(key, v);
    v
}

/// Composed target stem for an embedded base stem (e.g. `unlit_default` → `unlit_multiview`).
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
    } = source;
    let shader = refs.with_label("embedded_raster_material");
    let streams = VertexStreamToggles {
        include_uv_vertex_buffer: true,
        include_color_vertex_buffer: true,
    };
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), permutation);
    let declared_passes = embedded_shaders::embedded_target_passes(&composed);
    if !declared_passes.is_empty() {
        let materialized_passes = declared_passes
            .iter()
            .map(|p| materialized_pass_for_blend_mode(p, blend_mode))
            .collect::<Vec<_>>();
        return create_reflective_raster_mesh_forward_pipelines(
            shader,
            streams,
            &materialized_passes,
            render_state,
        );
    }

    let pass = default_pass_for_blend_mode(blend_mode);
    create_reflective_raster_mesh_forward_pipelines(
        shader,
        streams,
        std::slice::from_ref(&pass),
        render_state,
    )
}

/// Returns whether the embedded material stem declares alpha blending (any `//#pass` directive
/// with non-None blend state). Memoized per base stem.
pub fn embedded_stem_uses_alpha_blending(base_stem: &str) -> bool {
    let key = base_stem.to_string();
    let mut guard = embedded_alpha_blending_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(v) = guard.get(&key) {
        return *v;
    }
    let composed = embedded_composed_stem_for_permutation(base_stem, ShaderPermutation(0));
    let v = embedded_shaders::embedded_target_passes(&composed)
        .iter()
        .any(|p| p.blend.is_some());
    guard.insert(key, v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::ShaderPermutation;

    #[test]
    fn debug_world_normals_no_uv0_stream() {
        assert!(!embedded_stem_needs_uv0_stream(
            "debug_world_normals_default",
            ShaderPermutation(0)
        ));
        assert!(!embedded_stem_needs_uv0_stream(
            "debug_world_normals_default",
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
}
