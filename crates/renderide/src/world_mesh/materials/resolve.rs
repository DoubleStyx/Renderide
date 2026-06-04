//! Material batch-key resolution for world-mesh draw prep.

use std::sync::Arc;

use crate::materials::ShaderPermutation;
use crate::materials::host_data::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::materials::{
    EmbeddedStemQuery, EmbeddedTangentFallbackMode, MaterialBlendMode, MaterialPipelinePropertyIds,
    MaterialRenderState, MaterialRouter, MaterialShaderSpecializationKey, PropertyMapRef,
    RasterFrontFace, RasterPipelineKind, RasterPrimitiveTopology, SceneColorSnapshotMode,
    fallback_render_queue_for_material, first_float_from_maps, first_vec4_from_maps,
    material_blend_mode_from_maps, material_render_queue_from_maps,
    material_render_state_from_maps, resolve_raster_pipeline,
};

use super::FrameMaterialBatchCache;
use super::key::MaterialDrawBatchKey;
use super::transparent::{
    TransparentMaterialClass, TransparentMaterialClassInput, transparent_class_for_material,
};

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
    /// Renderer-local shader specialization constants for material keyword branches.
    pub shader_specialization: MaterialShaderSpecializationKey,
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
    /// Whether the active shader permutation requires the packed UV0-UV3 stream.
    pub embedded_needs_wide_low_uvs: bool,
    /// Whether the active shader permutation requires the packed UV4-UV7 stream.
    pub embedded_needs_wide_high_uvs: bool,
    /// Whether the active shader permutation requires any stream outside UV0/color/UV1.
    pub embedded_needs_extended_vertex_streams: bool,
    /// Whether the material declares intersection-depth behavior and needs the depth snapshot.
    pub embedded_requires_intersection_pass: bool,
    /// Whether the active shader permutation declares a scene-depth snapshot binding.
    pub embedded_uses_scene_depth_snapshot: bool,
    /// Whether the active shader permutation declares a scene-color snapshot binding.
    pub embedded_uses_scene_color_snapshot: bool,
    /// How the active shader permutation expects scene-color snapshots to be refreshed.
    pub scene_color_snapshot_mode: SceneColorSnapshotMode,
    /// Renderer-local transparent behavior class inferred from resolved material state.
    pub transparent_class: TransparentMaterialClass,
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
    needs_wide_low_uvs: bool,
    needs_wide_high_uvs: bool,
    needs_extended_vertex_streams: bool,
    requires_intersection_pass: bool,
    uses_scene_depth_snapshot: bool,
    uses_scene_color_snapshot: bool,
    scene_color_snapshot_mode: SceneColorSnapshotMode,
    uses_alpha_blending: bool,
    uses_ui_transparent_fallback: bool,
    uses_blended_depth_write: bool,
    uses_two_sided_transparency: bool,
    uses_renderide_variant_bits: bool,
    default_render_queue: Option<i32>,
}

fn embedded_material_features(
    pipeline: &RasterPipelineKind,
    shader_perm: ShaderPermutation,
) -> EmbeddedMaterialFeatures {
    let RasterPipelineKind::EmbeddedStem(stem) = pipeline else {
        return EmbeddedMaterialFeatures::default();
    };
    let stem = stem.as_ref();
    let query = EmbeddedStemQuery::for_stem(stem, shader_perm);
    let vertex_streams = query.vertex_stream_mask();
    let snapshots = query.snapshot_requirements();
    EmbeddedMaterialFeatures {
        needs_uv0: vertex_streams.uv0,
        needs_color: vertex_streams.color,
        needs_uv1: vertex_streams.uv1,
        needs_tangent: vertex_streams.tangent,
        tangent_fallback_mode: query.tangent_fallback_mode(),
        raw_tangent_payload: query.uses_raw_tangent_payload(),
        raw_normal_payload: query.uses_raw_normal_payload(),
        needs_uv2: vertex_streams.uv2,
        needs_uv3: vertex_streams.uv3,
        needs_wide_low_uvs: vertex_streams.wide_low_uvs,
        needs_wide_high_uvs: vertex_streams.wide_high_uvs,
        needs_extended_vertex_streams: vertex_streams.needs_extended_vertex_streams(),
        requires_intersection_pass: snapshots.requires_intersection_pass,
        uses_scene_depth_snapshot: snapshots.uses_scene_depth,
        uses_scene_color_snapshot: snapshots.uses_scene_color,
        scene_color_snapshot_mode: query.scene_color_snapshot_mode(),
        uses_alpha_blending: query.uses_alpha_blending(),
        uses_ui_transparent_fallback: query.uses_ui_transparent_fallback(),
        uses_blended_depth_write: query.uses_blended_depth_write(),
        uses_two_sided_transparency: query.uses_two_sided_transparency(),
        uses_renderide_variant_bits: query.uses_renderide_variant_bits(),
        default_render_queue: Some(query.default_render_queue()),
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
    let shader_specialization = if embedded.uses_renderide_variant_bits {
        MaterialShaderSpecializationKey::from_optional_variant_bits(
            router.variant_bits_for_shader_asset(shader_asset_id),
        )
    } else {
        MaterialShaderSpecializationKey::disabled()
    };
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
        embedded
            .default_render_queue
            .unwrap_or_else(|| fallback_render_queue_for_material(alpha_blended)),
    );
    let transparent_class = transparent_class_for_material(TransparentMaterialClassInput {
        render_queue,
        render_state,
        blend_mode,
        alpha_blended,
        uses_scene_color_snapshot: embedded.uses_scene_color_snapshot,
        uses_blended_depth_write: embedded.uses_blended_depth_write,
        uses_two_sided_transparency: embedded.uses_two_sided_transparency,
    });
    let ui_rect_clip_local = ui_rect_clip_local_from_maps(mat_map, pb_map, pipeline_property_ids);
    ResolvedMaterialBatch {
        shader_asset_id,
        pipeline,
        shader_specialization,
        embedded_needs_uv0: embedded.needs_uv0,
        embedded_needs_color: embedded.needs_color,
        embedded_needs_uv1: embedded.needs_uv1,
        embedded_needs_tangent: embedded.needs_tangent,
        embedded_tangent_fallback_mode: embedded.tangent_fallback_mode,
        embedded_raw_tangent_payload: embedded.raw_tangent_payload,
        embedded_raw_normal_payload: embedded.raw_normal_payload,
        embedded_needs_uv2: embedded.needs_uv2,
        embedded_needs_uv3: embedded.needs_uv3,
        embedded_needs_wide_low_uvs: embedded.needs_wide_low_uvs,
        embedded_needs_wide_high_uvs: embedded.needs_wide_high_uvs,
        embedded_needs_extended_vertex_streams: embedded.needs_extended_vertex_streams,
        embedded_requires_intersection_pass: embedded.requires_intersection_pass,
        embedded_uses_scene_depth_snapshot: embedded.uses_scene_depth_snapshot,
        embedded_uses_scene_color_snapshot: embedded.uses_scene_color_snapshot,
        scene_color_snapshot_mode: embedded.scene_color_snapshot_mode,
        transparent_class,
        blend_mode,
        render_queue,
        render_state,
        alpha_blended,
        ui_rect_clip_local,
    }
}

/// WGSL stem used for generated PhotonDust billboard mesh draws.
const RENDER_BUFFER_BILLBOARD_STEM: &str = "billboardunlit_default";

const RENDER_BUFFER_ORDERED_ALPHA_FALLBACK_BLEND: MaterialBlendMode =
    MaterialBlendMode::UnityBlend { src: 5, dst: 10 };

/// Routes generated PhotonDust billboard meshes through Billboard/Unlit.
///
/// Point render-buffer uploads expand each particle into four co-located quad vertices. Ordinary
/// mesh shaders see those vertices as degenerate geometry, while Billboard/Unlit expands them in
/// the vertex stage using the normal stream as point data.
pub(crate) fn apply_render_buffer_mesh_pipeline_override(
    batch_key: &mut MaterialDrawBatchKey,
    mesh_asset_id: i32,
    shader_perm: ShaderPermutation,
) {
    if !crate::particles::is_generated_billboard_mesh_asset_id(mesh_asset_id) {
        return;
    }
    if let RasterPipelineKind::EmbeddedStem(stem) = &batch_key.pipeline
        && stem.as_ref().starts_with("billboardunlit")
    {
        batch_key.shader_specialization = MaterialShaderSpecializationKey::disabled();
        return;
    }
    let pipeline = RasterPipelineKind::EmbeddedStem(Arc::from(RENDER_BUFFER_BILLBOARD_STEM));
    let features = embedded_material_features(&pipeline, shader_perm);
    batch_key.pipeline = pipeline;
    batch_key.shader_specialization = MaterialShaderSpecializationKey::disabled();
    batch_key.embedded_needs_uv0 = features.needs_uv0;
    batch_key.embedded_needs_color = features.needs_color;
    batch_key.embedded_needs_uv1 = features.needs_uv1;
    batch_key.embedded_needs_tangent = features.needs_tangent;
    batch_key.embedded_tangent_fallback_mode = features.tangent_fallback_mode;
    batch_key.embedded_raw_tangent_payload = features.raw_tangent_payload;
    batch_key.embedded_raw_normal_payload = features.raw_normal_payload;
    batch_key.embedded_needs_uv2 = features.needs_uv2;
    batch_key.embedded_needs_uv3 = features.needs_uv3;
    batch_key.embedded_needs_wide_low_uvs = features.needs_wide_low_uvs;
    batch_key.embedded_needs_wide_high_uvs = features.needs_wide_high_uvs;
    batch_key.embedded_needs_extended_vertex_streams = features.needs_extended_vertex_streams;
    batch_key.embedded_requires_intersection_pass = features.requires_intersection_pass;
    batch_key.embedded_uses_scene_depth_snapshot = features.uses_scene_depth_snapshot;
    batch_key.embedded_uses_scene_color_snapshot = features.uses_scene_color_snapshot;
    batch_key.scene_color_snapshot_mode = features.scene_color_snapshot_mode;
    batch_key.alpha_blended = batch_key.alpha_blended
        || features.uses_alpha_blending
        || features.uses_scene_color_snapshot;
    batch_key.transparent_class = transparent_class_for_material(TransparentMaterialClassInput {
        render_queue: batch_key.render_queue,
        render_state: batch_key.render_state,
        blend_mode: batch_key.blend_mode,
        alpha_blended: batch_key.alpha_blended,
        uses_scene_color_snapshot: features.uses_scene_color_snapshot,
        uses_blended_depth_write: features.uses_blended_depth_write,
        uses_two_sided_transparency: features.uses_two_sided_transparency,
    });
    batch_key.blend_mode = render_buffer_billboard_blend_mode_for_source(
        batch_key.blend_mode,
        batch_key.transparent_class,
    );
}

fn render_buffer_billboard_blend_mode_for_source(
    blend_mode: MaterialBlendMode,
    transparent_class: TransparentMaterialClass,
) -> MaterialBlendMode {
    if blend_mode == MaterialBlendMode::StemDefault
        && transparent_class == TransparentMaterialClass::OrderedAlpha
    {
        RENDER_BUFFER_ORDERED_ALPHA_FALLBACK_BLEND
    } else {
        blend_mode
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
        shader_specialization: r.shader_specialization,
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
        embedded_needs_wide_low_uvs: r.embedded_needs_wide_low_uvs,
        embedded_needs_wide_high_uvs: r.embedded_needs_wide_high_uvs,
        embedded_needs_extended_vertex_streams: r.embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass: r.embedded_requires_intersection_pass,
        embedded_uses_scene_depth_snapshot: r.embedded_uses_scene_depth_snapshot,
        embedded_uses_scene_color_snapshot: r.embedded_uses_scene_color_snapshot,
        scene_color_snapshot_mode: r.scene_color_snapshot_mode,
        render_queue: r.render_queue,
        render_state: r.render_state,
        blend_mode: r.blend_mode,
        alpha_blended: r.alpha_blended,
        transparent_class: r.transparent_class,
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
        MaterialRouter, RasterPipelineKind, SceneColorSnapshotMode, UNITY_RENDER_QUEUE_ALPHA_TEST,
        UNITY_RENDER_QUEUE_GEOMETRY, UNITY_RENDER_QUEUE_OVERLAY, UNITY_RENDER_QUEUE_TRANSPARENT,
    };
    use crate::shared::TrailTextureMode;

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
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsvoronoicrystal_default")),
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
            RasterPipelineKind::EmbeddedStem(Arc::from("ui_unlit_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&PropertyIdRegistry::new());

        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());

        assert!(resolved.alpha_blended);
        assert_eq!(resolved.render_queue, UNITY_RENDER_QUEUE_TRANSPARENT);
        assert!(resolved.embedded_raw_tangent_payload);
    }

    #[test]
    fn scene_color_snapshot_mode_reaches_resolved_material_batch() {
        let registry = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 70);
        store.set_shader_asset_for_material(8, 80);

        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            70,
            RasterPipelineKind::EmbeddedStem(Arc::from("pixelate_default")),
        );
        router.set_shader_pipeline(
            80,
            RasterPipelineKind::EmbeddedStem(Arc::from("pixelate_perobject_default")),
        );

        let dict = MaterialDictionary::new(&store);
        let named =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());
        let per_object =
            resolve_material_batch(8, None, &dict, &router, &ids, ShaderPermutation::default());

        assert!(named.embedded_uses_scene_color_snapshot);
        assert_eq!(
            named.scene_color_snapshot_mode,
            SceneColorSnapshotMode::NamedBackgroundGrab
        );
        assert!(per_object.embedded_uses_scene_color_snapshot);
        assert_eq!(
            per_object.scene_color_snapshot_mode,
            SceneColorSnapshotMode::PerObjectGrab
        );
    }

    #[test]
    fn explicit_opaque_ui_blend_keeps_shader_default_queue() {
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
            RasterPipelineKind::EmbeddedStem(Arc::from("ui_unlit_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&registry);

        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());

        assert!(!resolved.alpha_blended);
        assert_eq!(resolved.render_queue, UNITY_RENDER_QUEUE_TRANSPARENT);
    }

    #[test]
    fn embedded_shader_default_render_queue_handles_negative_material_queue() {
        let registry = PropertyIdRegistry::new();
        let render_queue = registry.intern("_RenderQueue");
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        store.set_material(7, render_queue, MaterialPropertyValue::Float(-1.0));
        let dict = MaterialDictionary::new(&store);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsintersect_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&registry);

        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());

        assert_eq!(resolved.render_queue, UNITY_RENDER_QUEUE_TRANSPARENT);
    }

    #[test]
    fn embedded_shader_default_render_queue_handles_offsets_and_aliases() {
        let registry = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 70);
        store.set_shader_asset_for_material(8, 80);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            70,
            RasterPipelineKind::EmbeddedStem(Arc::from("unlit_default")),
        );
        router.set_shader_pipeline(
            80,
            RasterPipelineKind::EmbeddedStem(Arc::from("pixelate_perobject_default")),
        );
        let dict = MaterialDictionary::new(&store);

        let unlit =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());
        let pixelate =
            resolve_material_batch(8, None, &dict, &router, &ids, ShaderPermutation::default());

        assert_eq!(unlit.render_queue, UNITY_RENDER_QUEUE_ALPHA_TEST + 200);
        assert_eq!(pixelate.render_queue, UNITY_RENDER_QUEUE_TRANSPARENT + 500);
    }

    #[test]
    fn blended_geometry_shader_keeps_geometry_default_queue() {
        let registry = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsdistancelerptransparent_default")),
        );
        let dict = MaterialDictionary::new(&store);

        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());

        assert!(resolved.alpha_blended);
        assert_eq!(resolved.render_queue, UNITY_RENDER_QUEUE_GEOMETRY);
    }

    #[test]
    fn explicit_material_render_queue_override_still_wins() {
        let registry = PropertyIdRegistry::new();
        let render_queue = registry.intern("_RenderQueue");
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        store.set_material(7, render_queue, MaterialPropertyValue::Float(4000.0));
        let dict = MaterialDictionary::new(&store);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsintersect_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&registry);

        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());

        assert_eq!(resolved.render_queue, UNITY_RENDER_QUEUE_OVERLAY);
    }

    #[test]
    fn generated_billboard_mesh_overrides_to_billboard_pipeline() {
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        let dict = MaterialDictionary::new(&store);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(Arc::from("unlit_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&PropertyIdRegistry::new());
        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());
        let mut key = batch_key_from_resolved(
            7,
            None,
            false,
            RasterFrontFace::Clockwise,
            RasterPrimitiveTopology::TriangleList,
            &resolved,
        );
        let mesh_asset_id = crate::particles::billboard_render_buffer_mesh_asset_id(3).unwrap();

        apply_render_buffer_mesh_pipeline_override(
            &mut key,
            mesh_asset_id,
            ShaderPermutation::default(),
        );

        let RasterPipelineKind::EmbeddedStem(stem) = &key.pipeline else {
            panic!("expected embedded billboard pipeline");
        };
        assert_eq!(stem.as_ref(), "billboardunlit_default");
        assert!(key.embedded_needs_uv0);
        assert!(key.embedded_needs_color);
        assert!(key.embedded_needs_tangent);
        assert!(key.embedded_raw_tangent_payload);
        assert!(key.embedded_raw_normal_payload);
    }

    #[test]
    fn generated_billboard_mesh_preserves_transparent_material_class() {
        let registry = PropertyIdRegistry::new();
        let src = registry.intern("_SrcBlend");
        let dst = registry.intern("_DstBlend");
        let render_queue = registry.intern("_RenderQueue");
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        store.set_material(7, src, MaterialPropertyValue::Float(5.0));
        store.set_material(7, dst, MaterialPropertyValue::Float(10.0));
        store.set_material(
            7,
            render_queue,
            MaterialPropertyValue::Float(UNITY_RENDER_QUEUE_TRANSPARENT as f32),
        );
        let dict = MaterialDictionary::new(&store);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(Arc::from("unlit_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());
        let mut key = batch_key_from_resolved(
            7,
            None,
            false,
            RasterFrontFace::Clockwise,
            RasterPrimitiveTopology::TriangleList,
            &resolved,
        );
        let mesh_asset_id = crate::particles::billboard_render_buffer_mesh_asset_id(3).unwrap();

        apply_render_buffer_mesh_pipeline_override(
            &mut key,
            mesh_asset_id,
            ShaderPermutation::default(),
        );

        assert!(key.alpha_blended);
        assert_eq!(key.render_queue, UNITY_RENDER_QUEUE_TRANSPARENT);
        assert!(key.transparent_class.is_transparent());
    }

    #[test]
    fn generated_billboard_mesh_uses_alpha_blend_for_stem_default_ordered_transparency() {
        let registry = PropertyIdRegistry::new();
        let render_queue = registry.intern("_RenderQueue");
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        store.set_material(
            7,
            render_queue,
            MaterialPropertyValue::Float(UNITY_RENDER_QUEUE_TRANSPARENT as f32),
        );
        let dict = MaterialDictionary::new(&store);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsvertexcolortransparent_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&registry);
        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());
        let mut key = batch_key_from_resolved(
            7,
            None,
            false,
            RasterFrontFace::Clockwise,
            RasterPrimitiveTopology::TriangleList,
            &resolved,
        );
        let mesh_asset_id = crate::particles::billboard_render_buffer_mesh_asset_id(3).unwrap();

        assert_eq!(key.blend_mode, MaterialBlendMode::StemDefault);
        assert_eq!(
            key.transparent_class,
            TransparentMaterialClass::OrderedAlpha
        );

        apply_render_buffer_mesh_pipeline_override(
            &mut key,
            mesh_asset_id,
            ShaderPermutation::default(),
        );

        assert_eq!(key.blend_mode, RENDER_BUFFER_ORDERED_ALPHA_FALLBACK_BLEND);
        assert_eq!(
            key.transparent_class,
            TransparentMaterialClass::OrderedAlpha
        );
    }

    #[test]
    fn render_buffer_billboard_blend_fallback_keeps_explicit_and_non_ordered_modes() {
        assert_eq!(
            render_buffer_billboard_blend_mode_for_source(
                MaterialBlendMode::UnityBlend { src: 1, dst: 1 },
                TransparentMaterialClass::OrderedAlpha,
            ),
            MaterialBlendMode::UnityBlend { src: 1, dst: 1 }
        );
        assert_eq!(
            render_buffer_billboard_blend_mode_for_source(
                MaterialBlendMode::Opaque,
                TransparentMaterialClass::OrderedAlpha,
            ),
            MaterialBlendMode::Opaque
        );
        assert_eq!(
            render_buffer_billboard_blend_mode_for_source(
                MaterialBlendMode::StemDefault,
                TransparentMaterialClass::DepthWritingTransparent,
            ),
            MaterialBlendMode::StemDefault
        );
    }

    #[test]
    fn generated_trail_mesh_does_not_get_billboard_pipeline_override() {
        let mut store = MaterialPropertyStore::new();
        store.set_shader_asset_for_material(7, 99);
        let dict = MaterialDictionary::new(&store);
        let mut router = MaterialRouter::new(RasterPipelineKind::Null);
        router.set_shader_pipeline(
            99,
            RasterPipelineKind::EmbeddedStem(Arc::from("unlit_default")),
        );
        let ids = MaterialPipelinePropertyIds::new(&PropertyIdRegistry::new());
        let resolved =
            resolve_material_batch(7, None, &dict, &router, &ids, ShaderPermutation::default());
        let mut key = batch_key_from_resolved(
            7,
            None,
            false,
            RasterFrontFace::Clockwise,
            RasterPrimitiveTopology::TriangleList,
            &resolved,
        );
        let original = key.pipeline.clone();
        let mesh_asset_id =
            crate::particles::trail_render_buffer_mesh_asset_id(3, TrailTextureMode::Stretch)
                .unwrap();

        apply_render_buffer_mesh_pipeline_override(
            &mut key,
            mesh_asset_id,
            ShaderPermutation::default(),
        );

        assert_eq!(key.pipeline, original);
    }
}
