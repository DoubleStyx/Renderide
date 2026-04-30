//! Material batch packet resolution for world-mesh forward draws.
//!
//! The resolver is the single boundary between sorted CPU draw runs and concrete raster state.
//! Both pre-warm and record-time preparation build [`PipelineVariantKey`] with the same helper so
//! they cannot drift on MSAA, front-face, blend, render-state, or shader permutations.

use std::sync::Arc;

use rayon::prelude::*;

use crate::backend::WorldMeshForwardEncodeRefs;
use crate::materials::ShaderPermutation;
use crate::materials::{EmbeddedMaterialBindResources, EmbeddedTexturePools};
use crate::materials::{
    MaterialBlendMode, MaterialPipelineDesc, MaterialPipelineSet, MaterialRegistry,
    MaterialRenderState, RasterFrontFace, RasterPipelineKind,
};
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

/// One resolved per-batch draw packet covering a contiguous range of sorted draws with the same
/// [`crate::world_mesh::MaterialDrawBatchKey`].
///
/// Populated by the prepare pass so the recording loop can drive pipeline and bind-group state
/// entirely from this table, without material-cache lookups inside `RenderPass`.
#[derive(Clone)]
pub(crate) struct MaterialBatchPacket {
    /// First draw index (into the sorted draw list) covered by this entry.
    pub first_draw_idx: usize,
    /// Last draw index (inclusive) covered by this entry.
    pub last_draw_idx: usize,
    /// Exact pipeline variant requested for this batch.
    pub(crate) pipeline_key: PipelineVariantKey,
    /// Resolved `@group(1)` bind group for this batch's material, or `None` for the empty fallback.
    pub bind_group: Option<Arc<wgpu::BindGroup>>,
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
}

/// Exact material pipeline variant used by both pipeline pre-warm and record-time resolution.
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
}

impl PipelineVariantKey {
    /// Builds the key used for both pre-warm and record-time material resolution.
    pub(crate) fn new(input: PipelineVariantKeyInput) -> Self {
        let PipelineVariantKeyInput {
            pass_desc,
            shader_perm,
            shader_asset_id,
            blend_mode,
            render_state,
            front_face,
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
        })
    }
}

/// Material pipeline and embedded-bind resolver for one world-mesh forward prepare pass.
pub(crate) struct MaterialDrawResolver<'a> {
    /// Material registry used for pipeline lookup.
    registry: Option<&'a MaterialRegistry>,
    /// Embedded material bind resources used for `@group(1)` lookup.
    embedded_bind: Option<&'a EmbeddedMaterialBindResources>,
    /// Material property store used by embedded bind resolution.
    store: &'a crate::materials::host_data::MaterialPropertyStore,
    /// Texture pools used by embedded bind resolution.
    pools: EmbeddedTexturePools<'a>,
    /// Queue used by embedded uniform updates.
    queue: &'a wgpu::Queue,
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
        queue: &'a wgpu::Queue,
        pass_desc: MaterialPipelineDesc,
        shader_perm: ShaderPermutation,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Self {
        Self {
            registry: encode.materials.material_registry(),
            embedded_bind: encode.materials.embedded_material_bind(),
            store: encode.materials.material_property_store(),
            pools: encode.embedded_texture_pools(),
            queue,
            pass_desc,
            shader_perm,
            offscreen_write_render_texture_asset_id,
        }
    }

    /// Resolves every contiguous material run in `draws` into record-ready packets.
    pub(crate) fn resolve_batches(&self, draws: &[WorldMeshDrawItem]) -> Vec<MaterialBatchPacket> {
        profiling::scope!("world_mesh_forward::resolve_material_packets");
        if draws.is_empty() {
            return Vec::new();
        }

        collect_material_batch_boundaries(draws)
            .into_par_iter()
            .map(|(first, last)| self.resolve_one_batch(draws, first, last))
            .collect()
    }

    /// Resolves one material run into a record-ready packet.
    fn resolve_one_batch(
        &self,
        draws: &[WorldMeshDrawItem],
        first: usize,
        last: usize,
    ) -> MaterialBatchPacket {
        let item = &draws[first];
        let pipeline_key =
            PipelineVariantKey::for_draw_item(item, self.pass_desc, self.shader_perm);

        let pipelines = self.resolve_pipelines(pipeline_key);
        let bind_group = self.resolve_embedded_bind_group(item);

        MaterialBatchPacket {
            first_draw_idx: first,
            last_draw_idx: last,
            pipeline_key,
            bind_group,
            pipelines,
        }
    }

    /// Resolves the material pipeline set for one batch.
    fn resolve_pipelines(&self, pipeline_key: PipelineVariantKey) -> Option<MaterialPipelineSet> {
        let Some(registry) = self.registry else {
            return None;
        };

        let pass_desc = pipeline_key.pass_desc();
        let pipelines = registry.pipeline_for_shader_asset(
            pipeline_key.shader_asset_id,
            &pass_desc,
            pipeline_key.shader_perm,
            pipeline_key.blend_mode,
            pipeline_key.render_state,
            pipeline_key.front_face,
        );

        match pipelines {
            Some(p) if !p.is_empty() => Some(p),
            Some(_) => {
                logger::trace!(
                    "WorldMeshForward: empty pipeline for shader {:?}, skipping batch",
                    pipeline_key.shader_asset_id
                );
                None
            }
            None => {
                logger::trace!(
                    "WorldMeshForward: no pipeline for shader {:?}, skipping batch",
                    pipeline_key.shader_asset_id
                );
                None
            }
        }
    }

    /// Resolves the embedded material bind group for one batch when the pipeline is embedded.
    fn resolve_embedded_bind_group(
        &self,
        item: &WorldMeshDrawItem,
    ) -> Option<Arc<wgpu::BindGroup>> {
        let batch_key = &item.batch_key;
        if !matches!(&batch_key.pipeline, RasterPipelineKind::EmbeddedStem(_)) {
            return None;
        }

        let (Some(bind), Some(registry)) = (self.embedded_bind, self.registry) else {
            if self.embedded_bind.is_none() {
                logger::warn!(
                    "WorldMeshForward: embedded material bind resources unavailable; \
                     @group(1) uses empty bind group for embedded raster draws"
                );
            }
            return None;
        };

        registry
            .stem_for_shader_asset(batch_key.shader_asset_id)
            .and_then(|stem| {
                bind.embedded_material_bind_group_with_cache_key(
                    stem,
                    self.queue,
                    self.store,
                    &self.pools,
                    item.lookup_ids,
                    self.offscreen_write_render_texture_asset_id,
                )
                .ok()
                .map(|(_, bg)| bg)
            })
    }
}

/// Walks `draws` once and emits `(first_idx, last_idx)` runs of identical material batch keys.
fn collect_material_batch_boundaries(draws: &[WorldMeshDrawItem]) -> Vec<(usize, usize)> {
    let mut boundaries: Vec<(usize, usize)> = Vec::new();
    let mut current_start = 0usize;
    let mut last_key = &draws[0].batch_key;
    for (idx, item) in draws.iter().enumerate().skip(1) {
        if &item.batch_key != last_key {
            boundaries.push((current_start, idx - 1));
            current_start = idx;
            last_key = &item.batch_key;
        }
    }
    boundaries.push((current_start, draws.len() - 1));
    boundaries
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::render_graph::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

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
        })
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
}
