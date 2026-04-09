//! Embedded mesh raster materials: composed WGSL stems under `shaders/target/` (see crate `build.rs`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::embedded_shaders;
use crate::materials::raster_pipeline::create_reflective_raster_mesh_forward_pipeline;
use crate::materials::MaterialPipelineDesc;
use crate::pipelines::raster::SHADER_PERM_MULTIVIEW_STEREO;
use crate::pipelines::ShaderPermutation;

fn embedded_uv0_stream_cache() -> &'static Mutex<HashMap<String, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn embedded_color_stream_cache() -> &'static Mutex<HashMap<String, bool>> {
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

pub(crate) fn build_embedded_wgsl(stem: &Arc<str>, permutation: ShaderPermutation) -> String {
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), permutation);
    embedded_shaders::embedded_target_wgsl(&composed)
        .unwrap_or_else(|| {
            panic!("composed shader missing for stem {composed} (run build with shaders/source)")
        })
        .to_string()
}

pub(crate) fn create_embedded_render_pipeline(
    stem: &Arc<str>,
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    desc: &MaterialPipelineDesc,
    wgsl_source: &str,
) -> wgpu::RenderPipeline {
    let alpha_ui = embedded_stem_uses_alpha_blending(stem.as_ref());
    create_reflective_raster_mesh_forward_pipeline(
        device,
        module,
        desc,
        wgsl_source,
        "embedded_raster_material",
        true,
        true,
        alpha_ui,
        !alpha_ui,
    )
}

pub fn embedded_stem_uses_alpha_blending(stem: &str) -> bool {
    let stem = stem
        .trim_end_matches("_default")
        .trim_end_matches("_multiview");
    stem.starts_with("ui_")
        || stem == "overlayunlit"
        || stem == "textunlit"
        || stem == "textunit"
        || stem == "text_unlit"
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
}
