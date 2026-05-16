//! Material batch packet resolution for world-mesh forward draws.
//!
//! The resolver is the single boundary between sorted CPU draw runs and concrete raster state.
//! Backend frame planning builds [`PipelineVariantKey`] once per batch so raster recording cannot
//! drift on MSAA, front-face, blend, render-state, or shader permutations.

use std::sync::Arc;

use rayon::prelude::*;

use crate::diagnostics::log_throttle::LogThrottle;
use crate::materials::ShaderPermutation;
use crate::materials::embedded::EmbeddedMaterialBindError;
use crate::materials::{
    EmbeddedMaterialBindResources, EmbeddedMaterialBindShader, EmbeddedTexturePools,
};
use crate::materials::{
    MaterialBlendMode, MaterialPipelineDesc, MaterialPipelineResolution, MaterialPipelineSet,
    MaterialPipelineVariantSpec, MaterialRegistry, MaterialRenderState, RasterFrontFace,
    RasterPipelineKind, RasterPrimitiveTopology,
};
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

/// Throttles repeated embedded-bind failures so a single bad material cannot flood logs.
static EMBEDDED_MATERIAL_BIND_FAILURE_LOG: LogThrottle = LogThrottle::new();

/// Inclusive `(first_draw_idx, last_draw_idx)` span over the sorted world-mesh draw list
/// identifying one contiguous material batch run.
pub(crate) type MaterialBatchBoundary = (usize, usize);

/// Kind-only summary of the group-1 binding carried by a material packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaterialGroup1BindingKind {
    /// Empty material bind group used by the Null fallback pipeline.
    Empty,
    /// Reflected embedded material bind group used by embedded raster pipelines.
    Embedded,
}

/// Explicit group-1 binding state for one world-mesh material packet.
#[derive(Clone)]
pub(crate) enum MaterialGroup1Binding {
    /// Bind the shared empty material bind group.
    Empty,
    /// Bind an embedded material group and optional uniform dynamic offset.
    Embedded {
        /// Reflected bind group matching the selected embedded pipeline layout.
        bind_group: Arc<wgpu::BindGroup>,
        /// Dynamic offset into the material uniform arena, when the material block is dynamic.
        uniform_dynamic_offset: Option<u32>,
    },
}

impl MaterialGroup1Binding {
    /// Returns the kind-only binding identity for validation and tests.
    fn kind(&self) -> MaterialGroup1BindingKind {
        match self {
            Self::Empty => MaterialGroup1BindingKind::Empty,
            Self::Embedded { .. } => MaterialGroup1BindingKind::Embedded,
        }
    }
}

/// One resolved per-batch draw packet covering a contiguous range of sorted draws with the same
/// [`crate::world_mesh::MaterialDrawBatchKey`].
///
/// Populated by backend frame planning so the recording loop can drive pipeline and bind-group state
/// entirely from this table, without material-cache lookups inside `RenderPass`.
#[derive(Clone)]
pub(crate) struct MaterialBatchPacket {
    /// First draw index (into the sorted draw list) covered by this entry.
    pub first_draw_idx: usize,
    /// Last draw index (inclusive) covered by this entry.
    pub last_draw_idx: usize,
    /// Exact pipeline variant requested for this batch.
    pub(crate) pipeline_key: PipelineVariantKey,
    /// Actual pipeline kind selected for this packet, or [`None`] when the batch is skipped.
    pub(crate) resolved_pipeline_kind: Option<RasterPipelineKind>,
    /// Explicit `@group(1)` binding that matches [`Self::resolved_pipeline_kind`].
    pub(crate) group1_binding: MaterialGroup1Binding,
    /// Resolved pipeline set for this batch, or `None` when the pipeline is unavailable (skip draws).
    pub pipelines: Option<MaterialPipelineSet>,
}

/// Inputs needed to build a [`PipelineVariantKey`] for one material draw run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PipelineVariantKeyInput {
    /// Base pass descriptor for the owning view.
    pub pass_desc: MaterialPipelineDesc,
    /// Shader permutation selected for the owning view.
    pub shader_perm: ShaderPermutation,
    /// Host shader asset id for diagnostics and material registry lookup.
    pub shader_asset_id: i32,
    /// Resolved material blend state.
    pub blend_mode: MaterialBlendMode,
    /// Resolved material render state.
    pub render_state: MaterialRenderState,
    /// Front-face winding selected from the draw transform.
    pub front_face: RasterFrontFace,
    /// Primitive topology selected from the mesh's per-submesh topology.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// Exact material pipeline variant used by backend frame planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PipelineVariantKey {
    /// Host shader asset id for diagnostics and material registry lookup.
    pub shader_asset_id: i32,
    /// Color attachment format.
    pub surface_format: wgpu::TextureFormat,
    /// Optional depth/stencil format.
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    /// Effective sample count for the active render pass.
    pub sample_count: u32,
    /// Optional multiview mask.
    pub multiview_mask: Option<std::num::NonZeroU32>,
    /// Shader permutation selected for the view.
    pub shader_perm: ShaderPermutation,
    /// Resolved material blend state.
    pub blend_mode: MaterialBlendMode,
    /// Resolved material render state.
    pub render_state: MaterialRenderState,
    /// Front-face winding selected from the draw transform.
    pub front_face: RasterFrontFace,
    /// Primitive topology selected from the mesh's per-submesh topology.
    pub primitive_topology: RasterPrimitiveTopology,
}

impl PipelineVariantKey {
    /// Builds the key used for material packet resolution.
    pub(crate) fn new(input: PipelineVariantKeyInput) -> Self {
        let PipelineVariantKeyInput {
            pass_desc,
            shader_perm,
            shader_asset_id,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        } = input;
        Self {
            shader_asset_id,
            surface_format: pass_desc.surface_format,
            depth_stencil_format: pass_desc.depth_stencil_format,
            sample_count: pass_desc.sample_count,
            multiview_mask: pass_desc.multiview_mask,
            shader_perm,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        }
    }

    /// Rehydrates the material pipeline descriptor used by [`MaterialRegistry`].
    pub(crate) fn pass_desc(self) -> MaterialPipelineDesc {
        MaterialPipelineDesc {
            surface_format: self.surface_format,
            depth_stencil_format: self.depth_stencil_format,
            sample_count: self.sample_count,
            multiview_mask: self.multiview_mask,
        }
    }

    /// Rehydrates the material pipeline variant selectors used by [`MaterialRegistry`].
    pub(crate) fn variant_spec(self) -> MaterialPipelineVariantSpec {
        MaterialPipelineVariantSpec {
            permutation: self.shader_perm,
            blend_mode: self.blend_mode,
            render_state: self.render_state,
            front_face: self.front_face,
            primitive_topology: self.primitive_topology,
        }
    }

    /// Builds a key directly from a sorted draw item and view-level pipeline state.
    pub(crate) fn for_draw_item(
        item: &WorldMeshDrawItem,
        pass_desc: MaterialPipelineDesc,
        shader_perm: ShaderPermutation,
    ) -> Self {
        let batch_key = &item.batch_key;
        Self::new(PipelineVariantKeyInput {
            pass_desc,
            shader_perm,
            shader_asset_id: batch_key.shader_asset_id,
            blend_mode: batch_key.blend_mode,
            render_state: batch_key.render_state,
            front_face: batch_key.front_face,
            primitive_topology: batch_key.primitive_topology,
        })
    }
}

/// Material pipeline and embedded-bind resolver for one world-mesh forward view plan.
pub(crate) struct MaterialDrawResolver<'a> {
    /// Material registry used for pipeline lookup.
    registry: Option<&'a MaterialRegistry>,
    /// Embedded material bind resources used for `@group(1)` lookup.
    embedded_bind: Option<&'a EmbeddedMaterialBindResources>,
    /// Material property store used by embedded bind resolution.
    store: &'a crate::materials::host_data::MaterialPropertyStore,
    /// Texture pools used by embedded bind resolution.
    pools: EmbeddedTexturePools<'a>,
    /// Upload sink used by embedded uniform updates.
    uploads: GraphUploadSink<'a>,
    /// View-level material pipeline descriptor before per-material overrides.
    pass_desc: MaterialPipelineDesc,
    /// Shader permutation for this view.
    shader_perm: ShaderPermutation,
    /// Offscreen render texture being written by this view, if any.
    offscreen_write_render_texture_asset_id: Option<i32>,
}

impl<'a> MaterialDrawResolver<'a> {
    /// Builds a resolver from the forward encode references for this view.
    pub(crate) fn new(
        encode: &'a WorldMeshForwardEncodeRefs<'_>,
        uploads: GraphUploadSink<'a>,
        pass_desc: MaterialPipelineDesc,
        shader_perm: ShaderPermutation,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Self {
        Self {
            registry: encode.materials.material_registry(),
            embedded_bind: encode.materials.embedded_material_bind(),
            store: encode.materials.material_property_store(),
            pools: encode.embedded_texture_pools(),
            uploads,
            pass_desc,
            shader_perm,
            offscreen_write_render_texture_asset_id,
        }
    }

    /// Resolves every contiguous material run in `draws` into record-ready packets.
    ///
    /// `boundaries_scratch` is cleared and refilled with the material-batch boundary spans; the
    /// caller owns the buffer so its capacity survives across frames and reallocates only on
    /// growth past the previous high-water mark.
    pub(crate) fn resolve_batches(
        &self,
        draws: &[WorldMeshDrawItem],
        boundaries_scratch: &mut Vec<MaterialBatchBoundary>,
    ) -> Vec<MaterialBatchPacket> {
        profiling::scope!("world_mesh_forward::resolve_material_packets");
        boundaries_scratch.clear();
        if draws.is_empty() {
            return Vec::new();
        }

        collect_material_batch_boundaries_into(draws, boundaries_scratch);
        if boundaries_scratch.len() < 2 {
            let mut packets = Vec::with_capacity(boundaries_scratch.len());
            for &(first, last) in boundaries_scratch.iter() {
                packets.push(self.resolve_one_batch(draws, first, last));
            }
            packets
        } else {
            boundaries_scratch
                .par_iter()
                .copied()
                .map(|(first, last)| self.resolve_one_batch(draws, first, last))
                .collect()
        }
    }

    /// Resolves one material run into a record-ready packet.
    fn resolve_one_batch(
        &self,
        draws: &[WorldMeshDrawItem],
        first: usize,
        last: usize,
    ) -> MaterialBatchPacket {
        let item = &draws[first];
        let mut pipeline_key =
            PipelineVariantKey::for_draw_item(item, self.pass_desc, self.shader_perm);
        if self.offscreen_write_render_texture_asset_id.is_some() {
            // View-projection matrices for offscreen-RT views are pre-multiplied by a clip-space
            // Y flip so the resulting render-texture lands in Unity (V=0 bottom) orientation.
            // That mirrors triangle winding, so the pipeline needs the inverted `front_face` to
            // keep back-face culling correct.
            pipeline_key.front_face = pipeline_key.front_face.flipped();
        }

        let resolved = self.resolve_pipeline_and_group1(item, pipeline_key);

        if let Some((resolution, group1_binding)) = resolved {
            debug_assert!(material_group1_binding_matches_pipeline(
                group1_binding.kind(),
                &resolution.kind
            ));
            return MaterialBatchPacket {
                first_draw_idx: first,
                last_draw_idx: last,
                pipeline_key,
                resolved_pipeline_kind: Some(resolution.kind),
                group1_binding,
                pipelines: Some(resolution.pipelines),
            };
        }

        MaterialBatchPacket {
            first_draw_idx: first,
            last_draw_idx: last,
            pipeline_key,
            resolved_pipeline_kind: None,
            group1_binding: MaterialGroup1Binding::Empty,
            pipelines: None,
        }
    }

    /// Resolves the material pipeline and matching group-1 binding for one batch.
    fn resolve_pipeline_and_group1(
        &self,
        item: &WorldMeshDrawItem,
        pipeline_key: PipelineVariantKey,
    ) -> Option<(MaterialPipelineResolution, MaterialGroup1Binding)> {
        let resolution =
            self.resolve_pipeline_resolution(&item.batch_key.pipeline, pipeline_key)?;
        match &resolution.kind {
            RasterPipelineKind::Null => Some((resolution, MaterialGroup1Binding::Empty)),
            RasterPipelineKind::EmbeddedStem(stem) => {
                match self.resolve_embedded_group1_binding(item, stem.as_ref()) {
                    Ok(group1_binding) => Some((resolution, group1_binding)),
                    Err(error) => {
                        let fallback = self.resolve_null_fallback_pipeline(pipeline_key);
                        self.log_embedded_bind_failure(
                            item,
                            stem.as_ref(),
                            &error,
                            fallback.is_some(),
                        );
                        fallback.map(|fallback_resolution| {
                            (fallback_resolution, MaterialGroup1Binding::Empty)
                        })
                    }
                }
            }
        }
    }

    /// Resolves the material pipeline set and concrete raster kind for one batch.
    fn resolve_pipeline_resolution(
        &self,
        pipeline_kind: &RasterPipelineKind,
        pipeline_key: PipelineVariantKey,
    ) -> Option<MaterialPipelineResolution> {
        let registry = self.registry?;

        let pass_desc = pipeline_key.pass_desc();
        let resolution = registry.pipeline_for_resolved_kind(
            pipeline_key.shader_asset_id,
            pipeline_kind,
            &pass_desc,
            pipeline_key.variant_spec(),
        );

        match resolution {
            Some(resolution) if !resolution.pipelines.is_empty() => Some(resolution),
            Some(resolution) => {
                logger::trace!(
                    "WorldMeshForward: empty pipeline for shader {:?} requested_kind {:?} resolved_kind {:?}, skipping batch",
                    pipeline_key.shader_asset_id,
                    pipeline_kind,
                    resolution.kind
                );
                None
            }
            None => {
                logger::trace!(
                    "WorldMeshForward: no pipeline for shader {:?} kind {:?}, skipping batch",
                    pipeline_key.shader_asset_id,
                    pipeline_kind
                );
                None
            }
        }
    }

    /// Resolves a ready Null fallback pipeline for a batch.
    fn resolve_null_fallback_pipeline(
        &self,
        pipeline_key: PipelineVariantKey,
    ) -> Option<MaterialPipelineResolution> {
        let registry = self.registry?;
        let pass_desc = pipeline_key.pass_desc();
        let resolution =
            registry.null_pipeline_for_variant(&pass_desc, pipeline_key.variant_spec());
        match resolution {
            Some(resolution) if !resolution.pipelines.is_empty() => Some(resolution),
            Some(_) => {
                logger::trace!(
                    "WorldMeshForward: empty Null fallback pipeline for shader {:?}, skipping batch",
                    pipeline_key.shader_asset_id
                );
                None
            }
            None => {
                logger::trace!(
                    "WorldMeshForward: Null fallback pipeline unavailable for shader {:?}, skipping batch",
                    pipeline_key.shader_asset_id
                );
                None
            }
        }
    }

    /// Resolves the embedded material bind group for an embedded pipeline stem.
    fn resolve_embedded_group1_binding(
        &self,
        item: &WorldMeshDrawItem,
        stem: &str,
    ) -> Result<MaterialGroup1Binding, EmbeddedMaterialBindError> {
        let batch_key = &item.batch_key;
        let Some(bind) = self.embedded_bind else {
            return Err(EmbeddedMaterialBindError::from(
                "embedded material bind resources unavailable",
            ));
        };

        let shader_variant_bits = self
            .registry
            .and_then(|registry| registry.variant_bits_for_shader_asset(batch_key.shader_asset_id));
        let (_, bind_group) = bind.embedded_material_bind_group_with_cache_key(
            EmbeddedMaterialBindShader {
                stem,
                shader_variant_bits,
            },
            self.uploads,
            self.store,
            &self.pools,
            item.lookup_ids,
            self.offscreen_write_render_texture_asset_id,
        )?;
        Ok(MaterialGroup1Binding::Embedded {
            bind_group: bind_group.bind_group,
            uniform_dynamic_offset: bind_group.uniform_dynamic_offset,
        })
    }

    /// Emits a throttled diagnostic for embedded bind failures and the selected fallback action.
    fn log_embedded_bind_failure(
        &self,
        item: &WorldMeshDrawItem,
        stem: &str,
        error: &EmbeddedMaterialBindError,
        fallback_ready: bool,
    ) {
        let Some(occurrence) = EMBEDDED_MATERIAL_BIND_FAILURE_LOG.should_log(8, 128) else {
            return;
        };
        let action = if fallback_ready {
            "using Null fallback"
        } else {
            "skipping batch until fallback is ready"
        };
        logger::warn!(
            "WorldMeshForward: embedded material bind group failed \
             (shader_asset_id={}, material_asset_id={}, slot_property_block={:?}, \
             renderer_property_block={:?}, stem={}, occurrence={}); {}: {}",
            item.batch_key.shader_asset_id,
            item.lookup_ids.material_asset_id,
            item.lookup_ids.mesh_property_block_slot0,
            item.lookup_ids.mesh_renderer_property_block_id,
            stem,
            occurrence,
            action,
            error
        );
    }
}

/// Returns the group-1 binding kind required by a concrete raster pipeline kind.
fn required_group1_binding_kind(kind: &RasterPipelineKind) -> MaterialGroup1BindingKind {
    match kind {
        RasterPipelineKind::Null => MaterialGroup1BindingKind::Empty,
        RasterPipelineKind::EmbeddedStem(_) => MaterialGroup1BindingKind::Embedded,
    }
}

/// Returns whether a group-1 binding kind is layout-compatible with a raster pipeline kind.
fn material_group1_binding_matches_pipeline(
    binding_kind: MaterialGroup1BindingKind,
    pipeline_kind: &RasterPipelineKind,
) -> bool {
    binding_kind == required_group1_binding_kind(pipeline_kind)
}

/// Walks `draws` once and writes `(first_idx, last_idx)` runs of identical material batch keys
/// into the caller-supplied `out` buffer. `out` is cleared before filling.
fn collect_material_batch_boundaries_into(
    draws: &[WorldMeshDrawItem],
    out: &mut Vec<MaterialBatchBoundary>,
) {
    out.clear();
    let mut current_start = 0usize;
    let mut last_key = &draws[0].batch_key;
    for (idx, item) in draws.iter().enumerate().skip(1) {
        if &item.batch_key != last_key {
            out.push((current_start, idx - 1));
            current_start = idx;
            last_key = &item.batch_key;
        }
    }
    out.push((current_start, draws.len() - 1));
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::sync::Arc;

    use super::*;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn base_desc() -> MaterialPipelineDesc {
        MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Rgba16Float,
            depth_stencil_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
            sample_count: 4,
            multiview_mask: NonZeroU32::new(3),
        }
    }

    fn key_for() -> PipelineVariantKey {
        PipelineVariantKey::new(PipelineVariantKeyInput {
            pass_desc: base_desc(),
            shader_perm: ShaderPermutation(1),
            shader_asset_id: 42,
            blend_mode: MaterialBlendMode::Opaque,
            render_state: MaterialRenderState::default(),
            front_face: RasterFrontFace::CounterClockwise,
            primitive_topology: RasterPrimitiveTopology::TriangleList,
        })
    }

    /// Builds an embedded pipeline kind for packet-selection tests.
    fn embedded_pipeline(stem: &'static str) -> RasterPipelineKind {
        RasterPipelineKind::EmbeddedStem(Arc::from(stem))
    }

    #[test]
    fn pipeline_key_preserves_regular_sample_count() {
        let key = key_for();
        assert_eq!(key.sample_count, 4);
        assert_eq!(key.pass_desc().sample_count, 4);
    }

    #[test]
    fn pipeline_key_preserves_grab_pass_sample_count() {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 42,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 7,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.shader_asset_id = 42;
        item.batch_key.blend_mode = MaterialBlendMode::Opaque;
        item.batch_key.front_face = RasterFrontFace::CounterClockwise;
        item.batch_key.embedded_uses_scene_color_snapshot = true;

        let key = PipelineVariantKey::for_draw_item(&item, base_desc(), ShaderPermutation(1));
        assert_eq!(key.sample_count, 4);
        assert_eq!(key.pass_desc().sample_count, 4);
        assert_eq!(key.surface_format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(
            key.depth_stencil_format,
            Some(wgpu::TextureFormat::Depth24PlusStencil8)
        );
        assert_eq!(key.multiview_mask, NonZeroU32::new(3));
    }

    #[test]
    fn pipeline_key_changes_when_front_face_changes() {
        let mut a = key_for();
        let mut b = key_for();
        a.front_face = RasterFrontFace::Clockwise;
        b.front_face = RasterFrontFace::CounterClockwise;
        assert_ne!(a, b);
    }

    /// Null pipelines require the shared empty material bind group.
    #[test]
    fn null_pipeline_requires_empty_group1_binding() {
        assert_eq!(
            required_group1_binding_kind(&RasterPipelineKind::Null),
            MaterialGroup1BindingKind::Empty
        );
        assert!(material_group1_binding_matches_pipeline(
            MaterialGroup1BindingKind::Empty,
            &RasterPipelineKind::Null
        ));
    }

    /// Embedded pipelines require the reflected embedded material bind group.
    #[test]
    fn embedded_pipeline_requires_embedded_group1_binding() {
        let kind = embedded_pipeline("xstoon2.0_default");
        assert_eq!(
            required_group1_binding_kind(&kind),
            MaterialGroup1BindingKind::Embedded
        );
        assert!(material_group1_binding_matches_pipeline(
            MaterialGroup1BindingKind::Embedded,
            &kind
        ));
    }

    /// The empty material bind group is never layout-compatible with embedded pipelines.
    #[test]
    fn empty_group1_binding_does_not_match_embedded_pipeline() {
        let kind = embedded_pipeline("xstoon2.0_default");
        assert!(!material_group1_binding_matches_pipeline(
            MaterialGroup1BindingKind::Empty,
            &kind
        ));
    }

    /// A draw batch snapshot that stayed Null still requires empty group 1 even if routing changes.
    #[test]
    fn stale_draw_batch_pipeline_requires_empty_group1_even_if_current_route_is_embedded() {
        let current_router_route = embedded_pipeline("xstoon2.0_default");
        let stale_draw_batch_pipeline = RasterPipelineKind::Null;

        assert_eq!(
            required_group1_binding_kind(&stale_draw_batch_pipeline),
            MaterialGroup1BindingKind::Empty
        );
        assert_ne!(stale_draw_batch_pipeline, current_router_route);
    }
}
