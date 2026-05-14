//! Material batch-key resolution for world-mesh draw prep.

use crate::materials::ShaderPermutation;
use crate::materials::host_data::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::materials::{
    EmbeddedTangentFallbackMode, MaterialBlendMode, MaterialPipelinePropertyIds,
    MaterialRenderState, MaterialRouter, PropertyMapRef, RasterFrontFace, RasterPipelineKind,
    RasterPrimitiveTopology, embedded_stem_needs_color_stream,
    embedded_stem_needs_extended_vertex_streams, embedded_stem_needs_tangent_stream,
    embedded_stem_needs_uv0_stream, embedded_stem_needs_uv1_stream, embedded_stem_needs_uv2_stream,
    embedded_stem_needs_uv3_stream, embedded_stem_needs_wide_uv_stream,
    embedded_stem_requires_intersection_pass, embedded_stem_tangent_fallback_mode,
    embedded_stem_uses_alpha_blending, embedded_stem_uses_raw_normal_payload,
    embedded_stem_uses_raw_tangent_payload, embedded_stem_uses_scene_color_snapshot,
    embedded_stem_uses_scene_depth_snapshot, embedded_stem_uses_ui_transparent_fallback,
    fallback_render_queue_for_material, first_float_from_maps, first_vec4_from_maps,
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
    /// Whether the active shader permutation requires a tangent vertex stream.
    pub embedded_needs_tangent: bool,
    /// Tangent fallback policy for lazy tangent upload.
    pub embedded_tangent_fallback_mode: EmbeddedTangentFallbackMode,
    /// Whether the tangent stream carries raw shader payload instead of a geometric tangent.
    pub embedded_raw_tangent_payload: bool,
    /// Whether the normal stream carries raw shader payload instead of a lighting normal.
    pub embedded_raw_normal_payload: bool,
    /// Whether the active shader permutation requires a UV2 vertex stream.
    pub embedded_needs_uv2: bool,
    /// Whether the active shader permutation requires a UV3 vertex stream.
    pub embedded_needs_uv3: bool,
    /// Whether the active shader permutation requires the packed UV0-UV7 stream.
    pub embedded_needs_wide_uvs: bool,
    /// Whether the active shader permutation requires any stream outside UV0/color/UV1.
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
    /// Object-local UI rect clip. `Some` only when `_RectClip > 0.5` and the rect has area.
    pub ui_rect_clip_local: Option<glam::Vec4>,
}

/// Resolves the object-local UI rect clip from pre-fetched property maps.
///
/// Returns `Some(rect)` only when `_RectClip >= 0.5` and `_Rect` has area
/// (matches the `rect_has_area` predicate the `rect_clip.wgsl` UI module uses to decide
/// whether to discard fragments).
fn ui_rect_clip_local_from_maps(
    material_map: PropertyMapRef<'_>,
    property_block_map: PropertyMapRef<'_>,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
) -> Option<glam::Vec4> {
    let rect_clip = first_float_from_maps(
        material_map,
        property_block_map,
        &pipeline_property_ids.rect_clip,
    )?;
    if rect_clip < 0.5 {
        return None;
    }
    let rect = first_vec4_from_maps(
        material_map,
        property_block_map,
        &pipeline_property_ids.rect,
    )?;
    let v = glam::Vec4::from_array(rect);
    // `_Rect` is `(xMin, yMin, xMax, yMax)` -- same predicate `rect_clip.wgsl` uses to gate the
    // fragment-shader discard. A zero-area or inverted rect would clip everything; treat that
    // as "no rect cull active" so we don't accidentally cull legitimate non-clipped UI draws
    // on degenerate input.
    (v.z > v.x && v.w > v.y).then_some(v)
}

#[derive(Copy, Clone, Default)]
struct EmbeddedMaterialFeatures {
    needs_uv0: bool,
    needs_color: bool,
    needs_uv1: bool,
    needs_tangent: bool,
    tangent_fallback_mode: EmbeddedTangentFallbackMode,
    raw_tangent_payload: bool,
    raw_normal_payload: bool,
    needs_uv2: bool,
    needs_uv3: bool,
    needs_wide_uvs: bool,
    needs_extended_vertex_streams: bool,
    requires_intersection_pass: bool,
    uses_scene_depth_snapshot: bool,
    uses_scene_color_snapshot: bool,
    uses_alpha_blending: bool,
    uses_ui_transparent_fallback: bool,
}

fn embedded_material_features(
    pipeline: &RasterPipelineKind,
    shader_perm: ShaderPermutation,
) -> EmbeddedMaterialFeatures {
    let RasterPipelineKind::EmbeddedStem(stem) = pipeline else {
        return EmbeddedMaterialFeatures::default();
    };
    let stem = stem.as_ref();
    EmbeddedMaterialFeatures {
        needs_uv0: embedded_stem_needs_uv0_stream(stem, shader_perm),
        needs_color: embedded_stem_needs_color_stream(stem, shader_perm),
        needs_uv1: embedded_stem_needs_uv1_stream(stem, shader_perm),
        needs_tangent: embedded_stem_needs_tangent_stream(stem, shader_perm),
        tangent_fallback_mode: embedded_stem_tangent_fallback_mode(stem, shader_perm),
        raw_tangent_payload: embedded_stem_uses_raw_tangent_payload(stem),
        raw_normal_payload: embedded_stem_uses_raw_normal_payload(stem),
        needs_uv2: embedded_stem_needs_uv2_stream(stem, shader_perm),
        needs_uv3: embedded_stem_needs_uv3_stream(stem, shader_perm),
        needs_wide_uvs: embedded_stem_needs_wide_uv_stream(stem, shader_perm),
        needs_extended_vertex_streams: embedded_stem_needs_extended_vertex_streams(
            stem,
            shader_perm,
        ),
        requires_intersection_pass: embedded_stem_requires_intersection_pass(stem, shader_perm),
        uses_scene_depth_snapshot: embedded_stem_uses_scene_depth_snapshot(stem, shader_perm),
        uses_scene_color_snapshot: embedded_stem_uses_scene_color_snapshot(stem, shader_perm),
        uses_alpha_blending: embedded_stem_uses_alpha_blending(stem),
        uses_ui_transparent_fallback: embedded_stem_uses_ui_transparent_fallback(stem),
    }
}

/// Builds a [`MaterialDrawBatchKey`] for one material slot from dictionary + router state.
///
/// This is the full per-draw computation path. Used for cache warm-up and as a fallback for
/// materials not present in [`FrameMaterialBatchCache`] (e.g. render-context override materials).
///
/// Also returns the optional object-local UI rect clip (`Some` only when `_RectClip > 0.5`
/// and `_Rect` has area), used for overlay UI CPU rect-cull and per-draw scissor.
pub(crate) fn batch_key_for_slot(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    front_face: RasterFrontFace,
    primitive_topology: RasterPrimitiveTopology,
    ctx: MaterialResolveCtx<'_>,
) -> (MaterialDrawBatchKey, Option<glam::Vec4>) {
    let resolved = resolve_material_batch(
        material_asset_id,
        property_block_id,
        ctx.dict,
        ctx.router,
        ctx.pipeline_property_ids,
        ctx.shader_perm,
    );
    let key = batch_key_from_resolved(
        material_asset_id,
        property_block_id,
        skinned,
        front_face,
        primitive_topology,
        &resolved,
    );
    (key, resolved.ui_rect_clip_local)
}

/// Builds a [`MaterialDrawBatchKey`] using a pre-built [`FrameMaterialBatchCache`].
///
/// Falls back to the full dictionary / router lookup path when the material is not cached.
/// The second tuple element is the optional object-local UI rect clip (`Some` only when
/// `_RectClip > 0.5` and `_Rect` has area).
pub(crate) fn batch_key_for_slot_cached(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    skinned: bool,
    front_face: RasterFrontFace,
    primitive_topology: RasterPrimitiveTopology,
    cache: &FrameMaterialBatchCache,
    ctx: MaterialResolveCtx<'_>,
) -> (MaterialDrawBatchKey, Option<glam::Vec4>) {
    if let Some(resolved) = cache.get(material_asset_id, property_block_id) {
        let key = batch_key_from_resolved(
            material_asset_id,
            property_block_id,
            skinned,
            front_face,
            primitive_topology,
            resolved,
        );
        (key, resolved.ui_rect_clip_local)
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
    let embedded = embedded_material_features(&pipeline, shader_perm);
    let lookup_ids = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: property_block_id,
        mesh_renderer_property_block_id: None,
    };
    let (mat_map, pb_map) = dict.fetch_property_maps(lookup_ids);
    let blend_mode = material_blend_mode_from_maps(mat_map, pb_map, pipeline_property_ids);
    let render_state = material_render_state_from_maps(mat_map, pb_map, pipeline_property_ids);
    let ui_stem_default_transparent =
        embedded.uses_ui_transparent_fallback && blend_mode == MaterialBlendMode::StemDefault;
    let alpha_blended = ui_stem_default_transparent
        || embedded.uses_alpha_blending
        || blend_mode.is_transparent()
        || embedded.uses_scene_color_snapshot;
    let render_queue = material_render_queue_from_maps(
        mat_map,
        pb_map,
        pipeline_property_ids,
        fallback_render_queue_for_material(alpha_blended),
    );
    let ui_rect_clip_local = ui_rect_clip_local_from_maps(mat_map, pb_map, pipeline_property_ids);
    ResolvedMaterialBatch {
        shader_asset_id,
        pipeline,
        embedded_needs_uv0: embedded.needs_uv0,
        embedded_needs_color: embedded.needs_color,
        embedded_needs_uv1: embedded.needs_uv1,
        embedded_needs_tangent: embedded.needs_tangent,
        embedded_tangent_fallback_mode: embedded.tangent_fallback_mode,
        embedded_raw_tangent_payload: embedded.raw_tangent_payload,
        embedded_raw_normal_payload: embedded.raw_normal_payload,
        embedded_needs_uv2: embedded.needs_uv2,
        embedded_needs_uv3: embedded.needs_uv3,
        embedded_needs_wide_uvs: embedded.needs_wide_uvs,
        embedded_needs_extended_vertex_streams: embedded.needs_extended_vertex_streams,
        embedded_requires_intersection_pass: embedded.requires_intersection_pass,
        embedded_uses_scene_depth_snapshot: embedded.uses_scene_depth_snapshot,
        embedded_uses_scene_color_snapshot: embedded.uses_scene_color_snapshot,
        blend_mode,
        render_queue,
        render_state,
        alpha_blended,
        ui_rect_clip_local,
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
        embedded_needs_tangent: r.embedded_needs_tangent,
        embedded_tangent_fallback_mode: r.embedded_tangent_fallback_mode,
        embedded_raw_tangent_payload: r.embedded_raw_tangent_payload,
        embedded_raw_normal_payload: r.embedded_raw_normal_payload,
        embedded_needs_uv2: r.embedded_needs_uv2,
        embedded_needs_uv3: r.embedded_needs_uv3,
        embedded_needs_wide_uvs: r.embedded_needs_wide_uvs,
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

#[cfg(test)]
mod ui_rect_clip_tests {
    //! Resolution of `ResolvedMaterialBatch::ui_rect_clip_local` from `_RectClip` + `_Rect`.

    use super::*;
    use crate::materials::ShaderPermutation;
    use crate::materials::host_data::{
        MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
    };
    use crate::materials::{
        MaterialRouter, RasterPipelineKind, UNITY_RENDER_QUEUE_GEOMETRY,
        UNITY_RENDER_QUEUE_TRANSPARENT,
    };

    struct Fixture {
        registry: PropertyIdRegistry,
        store: MaterialPropertyStore,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                registry: PropertyIdRegistry::new(),
                store: MaterialPropertyStore::new(),
            }
        }

        fn set(&mut self, mat: i32, name: &str, value: MaterialPropertyValue) {
            let pid = self.registry.intern(name);
            self.store.set_material(mat, pid, value);
        }

        fn resolve(&self, mat: i32) -> ResolvedMaterialBatch {
            let dict = MaterialDictionary::new(&self.store);
            let router = MaterialRouter::new(RasterPipelineKind::Null);
            let ids = MaterialPipelinePropertyIds::new(&self.registry);
            resolve_material_batch(
                mat,
                None,
                &dict,
                &router,
                &ids,
                ShaderPermutation::default(),
            )
        }
    }

    #[test]
    fn ui_rect_clip_local_is_none_when_rect_clip_missing() {
        let mut fx = Fixture::new();
        fx.set(
            7,
            "_Rect",
            MaterialPropertyValue::Float4([0.0, 0.0, 1.0, 1.0]),
        );
        assert!(fx.resolve(7).ui_rect_clip_local.is_none());
    }

    #[test]
    fn ui_rect_clip_local_is_none_when_rect_clip_zero() {
        let mut fx = Fixture::new();
        fx.set(7, "_RectClip", MaterialPropertyValue::Float(0.0));
        fx.set(
            7,
            "_Rect",
            MaterialPropertyValue::Float4([0.0, 0.0, 1.0, 1.0]),
        );
        assert!(fx.resolve(7).ui_rect_clip_local.is_none());
    }

    #[test]
    fn ui_rect_clip_local_is_none_when_rect_has_zero_area() {
        let mut fx = Fixture::new();
        fx.set(7, "_RectClip", MaterialPropertyValue::Float(1.0));
        fx.set(
            7,
            "_Rect",
            MaterialPropertyValue::Float4([0.5, 0.5, 0.5, 0.5]),
        );
        assert!(fx.resolve(7).ui_rect_clip_local.is_none());
    }

    #[test]
    fn ui_rect_clip_local_is_some_when_clip_enabled_and_rect_has_area() {
        let mut fx = Fixture::new();
        fx.set(7, "_RectClip", MaterialPropertyValue::Float(1.0));
        fx.set(
            7,
            "_Rect",
            MaterialPropertyValue::Float4([0.1, 0.2, 0.7, 0.9]),
        );
        assert_eq!(
            fx.resolve(7).ui_rect_clip_local,
            Some(glam::Vec4::new(0.1, 0.2, 0.7, 0.9))
        );
    }

    #[test]
    fn pbsvoronoicrystal_tangent_policy_reaches_uncached_batch_key() {
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        let dict = MaterialDictionary::new(&store);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(std::sync::Arc::from("pbsvoronoicrystal_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&PropertyIdRegistry::new());
        let ctx = MaterialResolveCtx {
            dict: &dict,
            router: &router,
            pipeline_property_ids: &ids,
            shader_perm: ShaderPermutation::default(),
        };

        let (key, _) = batch_key_for_slot(
            7,
            None,
            false,
            RasterFrontFace::Clockwise,
            RasterPrimitiveTopology::TriangleList,
            ctx,
        );

        assert!(key.embedded_needs_tangent);
        assert_eq!(
            key.embedded_tangent_fallback_mode,
            EmbeddedTangentFallbackMode::GenerateMissing
        );
    }

    #[test]
    fn ui_stem_default_state_falls_back_to_transparent_queue() {
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        let dict = MaterialDictionary::new(&store);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(std::sync::Arc::from("ui_unlit_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&PropertyIdRegistry::new());

        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());

        assert!(resolved.alpha_blended);
        assert_eq!(resolved.render_queue, UNITY_RENDER_QUEUE_TRANSPARENT);
        assert!(resolved.embedded_raw_tangent_payload);
    }

    #[test]
    fn explicit_opaque_ui_blend_keeps_opaque_queue_fallback() {
        let registry = PropertyIdRegistry::new();
        let src = registry.intern("_SrcBlend");
        let dst = registry.intern("_DstBlend");
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        store.set_material(7, src, MaterialPropertyValue::Float(1.0));
        store.set_material(7, dst, MaterialPropertyValue::Float(0.0));
        let dict = MaterialDictionary::new(&store);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(std::sync::Arc::from("ui_unlit_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&registry);

        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());

        assert!(!resolved.alpha_blended);
        assert_eq!(resolved.render_queue, UNITY_RENDER_QUEUE_GEOMETRY);
    }
}
