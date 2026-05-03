//! Material batch-key resolution for world-mesh draw prep.

use crate::materials::ShaderPermutation;
use crate::materials::host_data::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::materials::{
    MaterialBlendMode, MaterialPipelinePropertyIds, MaterialRenderState, MaterialRouter,
    RasterFrontFace, RasterPipelineKind, RasterPrimitiveTopology, embedded_stem_needs_color_stream,
    embedded_stem_needs_extended_vertex_streams, embedded_stem_needs_uv0_stream,
    embedded_stem_needs_uv1_stream, embedded_stem_requires_intersection_pass,
    embedded_stem_uses_alpha_blending, embedded_stem_uses_scene_color_snapshot,
    embedded_stem_uses_scene_depth_snapshot, fallback_render_queue_for_material,
    material_blend_mode_from_maps, material_render_queue_from_maps,
    material_render_state_from_maps, resolve_raster_pipeline,
};

use super::FrameMaterialBatchCache;
use super::key::MaterialDrawBatchKey;

/// Read-only material-resolution context threaded through the cache refresh walker and the cached
/// batch-key lookup.
#[derive(Copy, Clone)]
pub(crate) struct MaterialResolveCtx<'a> {
    /// Material property dictionary for batch keys.
    pub dict: &'a MaterialDictionary<'a>,
    /// Shader stem / pipeline routing.
    pub router: &'a MaterialRouter,
    /// Interned material property ids that affect pipeline state.
    pub pipeline_property_ids: &'a MaterialPipelinePropertyIds,
    /// Default vs multiview permutation for embedded materials.
    pub shader_perm: ShaderPermutation,
}

/// Batch key fields derived from one `(material_asset_id, property_block_id)` pair.
#[derive(Clone)]
pub(crate) struct ResolvedMaterialBatch {
    /// Host shader asset id from material `set_shader` (`-1` when unknown).
    pub shader_asset_id: i32,
    /// Resolved raster pipeline kind for this material's shader.
    pub pipeline: RasterPipelineKind,
    /// Whether the active shader permutation requires a UV0 vertex stream.
    pub embedded_needs_uv0: bool,
    /// Whether the active shader permutation requires a color vertex stream.
    pub embedded_needs_color: bool,
    /// Whether the active shader permutation requires a UV1 vertex stream.
    pub embedded_needs_uv1: bool,
    /// Whether the active shader permutation requires extended vertex streams.
    pub embedded_needs_extended_vertex_streams: bool,
    /// Whether the material requires a second forward subpass with a depth snapshot.
    pub embedded_requires_intersection_pass: bool,
    /// Whether the active shader permutation declares a scene-depth snapshot binding.
    pub embedded_uses_scene_depth_snapshot: bool,
    /// Whether the active shader permutation declares a scene-color snapshot binding.
    pub embedded_uses_scene_color_snapshot: bool,
    /// Resolved material blend mode.
    pub blend_mode: MaterialBlendMode,
    /// Effective Unity render queue for draw ordering.
    pub render_queue: i32,
    /// Runtime color, stencil, and depth state for this material/property-block pair.
    pub render_state: MaterialRenderState,
    /// Whether draws using this material should be sorted back-to-front.
    pub alpha_blended: bool,
}

/// Builds a [`MaterialDrawBatchKey`] for one material slot from dictionary + router state.
///
/// This is the full per-draw computation path. Used for cache warm-up and as a fallback for
/// materials not present in [`FrameMaterialBatchCache`] (e.g. render-context override materials).
pub(crate) fn batch_key_for_slot(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    front_face: RasterFrontFace,
    primitive_topology: RasterPrimitiveTopology,
    ctx: MaterialResolveCtx<'_>,
) -> MaterialDrawBatchKey {
    let shader_asset_id = ctx
        .dict
        .shader_asset_for_material(material_asset_id)
        .unwrap_or(-1);
    let pipeline = resolve_raster_pipeline(shader_asset_id, ctx.router);
    let embedded_needs_uv0 = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_uv0_stream(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_needs_color = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_color_stream(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_needs_uv1 = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_uv1_stream(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_needs_extended_vertex_streams = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_extended_vertex_streams(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_requires_intersection_pass = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_requires_intersection_pass(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_uses_scene_depth_snapshot = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_uses_scene_depth_snapshot(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let embedded_uses_scene_color_snapshot = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_uses_scene_color_snapshot(stem.as_ref(), ctx.shader_perm)
        }
        RasterPipelineKind::Null => false,
    };
    let lookup_ids = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: property_block_id,
    };
    let (mat_map, pb_map) = ctx.dict.fetch_property_maps(lookup_ids);
    let material_blend_mode =
        material_blend_mode_from_maps(mat_map, pb_map, ctx.pipeline_property_ids);
    let render_state = material_render_state_from_maps(mat_map, pb_map, ctx.pipeline_property_ids);
    let alpha_blended = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => embedded_stem_uses_alpha_blending(stem.as_ref()),
        RasterPipelineKind::Null => false,
    } || material_blend_mode.is_transparent()
        || embedded_uses_scene_color_snapshot;
    let render_queue = material_render_queue_from_maps(
        mat_map,
        pb_map,
        ctx.pipeline_property_ids,
        fallback_render_queue_for_material(alpha_blended),
    );
    MaterialDrawBatchKey {
        pipeline,
        shader_asset_id,
        material_asset_id,
        property_block_slot0: property_block_id,
        skinned,
        front_face,
        primitive_topology,
        embedded_needs_uv0,
        embedded_needs_color,
        embedded_needs_uv1,
        embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass,
        embedded_uses_scene_depth_snapshot,
        embedded_uses_scene_color_snapshot,
        render_queue,
        render_state,
        blend_mode: material_blend_mode,
        alpha_blended,
    }
}

/// Builds a [`MaterialDrawBatchKey`] using a pre-built [`FrameMaterialBatchCache`].
///
/// Falls back to the full dictionary / router lookup path when the material is not cached.
pub(crate) fn batch_key_for_slot_cached(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    front_face: RasterFrontFace,
    primitive_topology: RasterPrimitiveTopology,
    cache: &FrameMaterialBatchCache,
    ctx: MaterialResolveCtx<'_>,
) -> MaterialDrawBatchKey {
    if let Some(resolved) = cache.get(material_asset_id, property_block_id) {
        batch_key_from_resolved(
            material_asset_id,
            property_block_id,
            skinned,
            front_face,
            primitive_topology,
            resolved,
        )
    } else {
        batch_key_for_slot(
            material_asset_id,
            property_block_id,
            skinned,
            front_face,
            primitive_topology,
            ctx,
        )
    }
}

/// Computes all batch key fields for one `(material_asset_id, property_block_id)` pair.
pub(crate) fn resolve_material_batch(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
    shader_perm: ShaderPermutation,
) -> ResolvedMaterialBatch {
    let shader_asset_id = dict
        .shader_asset_for_material(material_asset_id)
        .unwrap_or(-1);
    let pipeline = resolve_raster_pipeline(shader_asset_id, router);
    let (
        embedded_needs_uv0,
        embedded_needs_color,
        embedded_needs_uv1,
        embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass,
        embedded_uses_scene_depth_snapshot,
        embedded_uses_scene_color_snapshot,
        embedded_uses_alpha_blending,
    ) = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            let s = stem.as_ref();
            (
                embedded_stem_needs_uv0_stream(s, shader_perm),
                embedded_stem_needs_color_stream(s, shader_perm),
                embedded_stem_needs_uv1_stream(s, shader_perm),
                embedded_stem_needs_extended_vertex_streams(s, shader_perm),
                embedded_stem_requires_intersection_pass(s, shader_perm),
                embedded_stem_uses_scene_depth_snapshot(s, shader_perm),
                embedded_stem_uses_scene_color_snapshot(s, shader_perm),
                embedded_stem_uses_alpha_blending(s),
            )
        }
        RasterPipelineKind::Null => (false, false, false, false, false, false, false, false),
    };
    let lookup_ids = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: property_block_id,
    };
    let (mat_map, pb_map) = dict.fetch_property_maps(lookup_ids);
    let blend_mode = material_blend_mode_from_maps(mat_map, pb_map, pipeline_property_ids);
    let render_state = material_render_state_from_maps(mat_map, pb_map, pipeline_property_ids);
    let alpha_blended = embedded_uses_alpha_blending
        || blend_mode.is_transparent()
        || embedded_uses_scene_color_snapshot;
    let render_queue = material_render_queue_from_maps(
        mat_map,
        pb_map,
        pipeline_property_ids,
        fallback_render_queue_for_material(alpha_blended),
    );
    ResolvedMaterialBatch {
        shader_asset_id,
        pipeline,
        embedded_needs_uv0,
        embedded_needs_color,
        embedded_needs_uv1,
        embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass,
        embedded_uses_scene_depth_snapshot,
        embedded_uses_scene_color_snapshot,
        blend_mode,
        render_queue,
        render_state,
        alpha_blended,
    }
}

/// Assembles a [`MaterialDrawBatchKey`] from a pre-resolved [`ResolvedMaterialBatch`] entry.
#[inline]
fn batch_key_from_resolved(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    front_face: RasterFrontFace,
    primitive_topology: RasterPrimitiveTopology,
    r: &ResolvedMaterialBatch,
) -> MaterialDrawBatchKey {
    MaterialDrawBatchKey {
        pipeline: r.pipeline.clone(),
        shader_asset_id: r.shader_asset_id,
        material_asset_id,
        property_block_slot0: property_block_id,
        skinned,
        front_face,
        primitive_topology,
        embedded_needs_uv0: r.embedded_needs_uv0,
        embedded_needs_color: r.embedded_needs_color,
        embedded_needs_uv1: r.embedded_needs_uv1,
        embedded_needs_extended_vertex_streams: r.embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass: r.embedded_requires_intersection_pass,
        embedded_uses_scene_depth_snapshot: r.embedded_uses_scene_depth_snapshot,
        embedded_uses_scene_color_snapshot: r.embedded_uses_scene_color_snapshot,
        render_queue: r.render_queue,
        render_state: r.render_state,
        blend_mode: r.blend_mode,
        alpha_blended: r.alpha_blended,
    }
}
