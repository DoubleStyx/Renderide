//! Warmup helpers for graph-seeded backend assets.

use std::num::NonZeroU32;
use std::sync::Arc;

use hashbrown::{HashMap, HashSet};

use crate::assets::mesh::{MeshDerivedStreamDemand, MeshDerivedStreamMask};
use crate::graph_inputs::PreRecordViewResourceLayout;
use crate::materials::{
    EmbeddedTangentFallbackMode, MaterialPipelineDesc, MaterialPipelineVariantSpec,
    RasterPipelineKind, SHADER_PERM_MULTIVIEW_STEREO, ShaderPermutation,
};
use crate::passes::{
    WorldMeshForwardDepthPrepassPipelineKey, WorldMeshForwardNormalPipelineKey,
    WorldMeshForwardPipelineState, depth_prepass_pipeline_key_for_draw,
    normal_pipeline_key_for_draw, pre_warm_depth_prepass_pipeline, pre_warm_normal_pipeline,
};
use crate::render_graph::compiled::{FrameView, FrameViewTarget};
use crate::world_mesh::{WorldMeshDrawItem, WorldMeshPhase};

use super::super::super::{WorldMeshDrawPlanSlot, WorldMeshOverlayDrawPlanSlot};
use super::BackendGraphAccess;

#[derive(Default)]
struct ViewAssetPrewarmRequests {
    uv1_stream_meshes: HashSet<i32>,
    tangent_stream_meshes: HashSet<i32>,
    tangent_fallback_modes: HashMap<i32, EmbeddedTangentFallbackMode>,
    uv2_stream_meshes: HashSet<i32>,
    uv3_stream_meshes: HashSet<i32>,
    wide_low_uv_stream_meshes: HashSet<i32>,
    wide_high_uv_stream_meshes: HashSet<i32>,
    billboard_stream_meshes: HashSet<i32>,
    derived_stream_demands: HashMap<i32, MeshDerivedStreamDemand>,
}

impl ViewAssetPrewarmRequests {
    fn record_item(&mut self, item: &WorldMeshDrawItem) {
        if item.mesh_asset_id < 0 {
            return;
        }
        let mut demand = MeshDerivedStreamDemand {
            mask: MeshDerivedStreamMask::DRAWABLE_PRIMARY,
            tangent_fallback_mode: item.batch_key.embedded_tangent_fallback_mode,
        };
        if item.batch_key.embedded_needs_uv1 {
            self.uv1_stream_meshes.insert(item.mesh_asset_id);
            demand.mask |= MeshDerivedStreamMask::UV1;
        }
        if item.batch_key.embedded_needs_uv0 {
            demand.mask |= MeshDerivedStreamMask::UV0;
        }
        if item.batch_key.embedded_needs_color {
            demand.mask |= MeshDerivedStreamMask::COLOR;
        }
        if item.batch_key.embedded_needs_tangent {
            self.tangent_stream_meshes.insert(item.mesh_asset_id);
            demand.mask |= MeshDerivedStreamMask::TANGENT;
            let mode = self
                .tangent_fallback_modes
                .entry(item.mesh_asset_id)
                .or_default();
            *mode = (*mode).max(item.batch_key.embedded_tangent_fallback_mode);
        }
        if item.batch_key.embedded_needs_uv2 {
            self.uv2_stream_meshes.insert(item.mesh_asset_id);
            demand.mask |= MeshDerivedStreamMask::UV2;
        }
        if item.batch_key.embedded_needs_uv3 {
            self.uv3_stream_meshes.insert(item.mesh_asset_id);
            demand.mask |= MeshDerivedStreamMask::UV3;
        }
        if item.batch_key.embedded_needs_wide_low_uvs {
            self.wide_low_uv_stream_meshes.insert(item.mesh_asset_id);
            demand.mask |= MeshDerivedStreamMask::WIDE_UV_LOW;
        }
        if item.batch_key.embedded_needs_wide_high_uvs {
            self.wide_high_uv_stream_meshes.insert(item.mesh_asset_id);
            demand.mask |= MeshDerivedStreamMask::WIDE_UV_HIGH;
        }
        if item.batch_key.embedded_uses_billboard_geometry {
            self.billboard_stream_meshes.insert(item.mesh_asset_id);
            demand.mask |= MeshDerivedStreamMask::BILLBOARD;
        }
        self.derived_stream_demands
            .entry(item.mesh_asset_id)
            .or_default()
            .merge(demand);
    }

    fn generated_tangent_mesh_count(&self) -> usize {
        self.tangent_fallback_modes
            .values()
            .filter(|mode| **mode == EmbeddedTangentFallbackMode::GenerateMissing)
            .count()
    }

    fn all_extended_stream_meshes(&self) -> HashSet<i32> {
        self.tangent_stream_meshes
            .iter()
            .filter(|mesh_asset_id| {
                self.uv1_stream_meshes.contains(*mesh_asset_id)
                    && self.uv2_stream_meshes.contains(*mesh_asset_id)
                    && self.uv3_stream_meshes.contains(*mesh_asset_id)
            })
            .copied()
            .collect()
    }

    fn tangent_fallback_mode(&self, mesh_asset_id: i32) -> EmbeddedTangentFallbackMode {
        self.tangent_fallback_modes
            .get(&mesh_asset_id)
            .copied()
            .unwrap_or_default()
    }
}

fn collect_view_asset_prewarm_requests(views: &[FrameView<'_>]) -> ViewAssetPrewarmRequests {
    let mut requests = ViewAssetPrewarmRequests::default();
    for view in views {
        record_plan_asset_prewarm_requests(
            view.initial_blackboard.get::<WorldMeshDrawPlanSlot>(),
            &mut requests,
        );
        record_plan_asset_prewarm_requests(
            view.initial_blackboard
                .get::<WorldMeshOverlayDrawPlanSlot>(),
            &mut requests,
        );
    }
    requests
}

fn record_plan_asset_prewarm_requests(
    draw_plan: Option<&crate::world_mesh::WorldMeshDrawPlan>,
    requests: &mut ViewAssetPrewarmRequests,
) {
    let Some(collection) = draw_plan.and_then(crate::world_mesh::WorldMeshDrawPlan::as_prefetched)
    else {
        return;
    };
    for item in &collection.items {
        requests.record_item(item);
    }
}

fn world_mesh_item_mirrors_to_normal_prepass(item: &WorldMeshDrawItem) -> bool {
    matches!(
        crate::world_mesh::phase_classification::classify_world_mesh_batch(&item.batch_key).phase,
        WorldMeshPhase::ForwardOpaque | WorldMeshPhase::ForwardAlphaTest
    )
}

fn next_material_warmup_run_start(items: &[WorldMeshDrawItem], start: usize) -> usize {
    let Some(first) = items.get(start) else {
        return start;
    };
    let mut next = start + 1;
    while items
        .get(next)
        .is_some_and(|item| item.batch_key == first.batch_key)
    {
        next += 1;
    }
    next
}

fn material_pass_desc_for_layout(
    layout: PreRecordViewResourceLayout,
    supports_multiview: bool,
) -> (MaterialPipelineDesc, ShaderPermutation) {
    let use_multiview = layout.stereo && supports_multiview;
    let shader_perm = if use_multiview {
        SHADER_PERM_MULTIVIEW_STEREO
    } else {
        ShaderPermutation::default()
    };
    (
        MaterialPipelineDesc {
            surface_format: layout.color_format,
            depth_stencil_format: Some(layout.depth_format),
            sample_count: layout.sample_count.max(1),
            multiview_mask: use_multiview.then(|| NonZeroU32::new(3)).flatten(),
        },
        shader_perm,
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MaterialPipelineWarmupKey {
    kind: RasterPipelineKind,
    desc: MaterialPipelineDesc,
    variant: MaterialPipelineVariantSpec,
}

#[derive(Default)]
struct WorldMeshPrepassPipelineWarmupKeys {
    depth: HashSet<WorldMeshForwardDepthPrepassPipelineKey>,
    normal: HashSet<WorldMeshForwardNormalPipelineKey>,
}

impl WorldMeshPrepassPipelineWarmupKeys {
    fn record_item(
        &mut self,
        item: &WorldMeshDrawItem,
        pipeline: &WorldMeshForwardPipelineState,
        offscreen: bool,
    ) {
        if let Some(key) = depth_prepass_pipeline_key_for_draw(item, pipeline) {
            self.depth.insert(key);
        }
        if world_mesh_item_mirrors_to_normal_prepass(item)
            && let Some(key) = normal_pipeline_key_for_draw(item, pipeline, offscreen)
        {
            self.normal.insert(key);
        }
    }

    fn warm(self, device: &wgpu::Device) -> (usize, usize) {
        let depth_count = self.depth.len();
        let normal_count = self.normal.len();
        for key in self.depth {
            pre_warm_depth_prepass_pipeline(device, key);
        }
        for key in self.normal {
            pre_warm_normal_pipeline(device, key);
        }
        (depth_count, normal_count)
    }
}

impl<'a> BackendGraphAccess<'a> {
    /// Warms backend-owned assets required by caller-seeded per-view blackboards.
    pub(crate) fn pre_warm_view_assets_from_blackboards(
        &mut self,
        device: &wgpu::Device,
        views: &[FrameView<'_>],
        view_layouts: &[Option<PreRecordViewResourceLayout>],
        resource_layouts: &[PreRecordViewResourceLayout],
    ) {
        profiling::scope!("graph::pre_warm_view_assets");
        let requests = collect_view_asset_prewarm_requests(views);
        logger::trace!(
            "graph pre-warm view assets: views={} uv1_stream_meshes={} tangent_stream_meshes={} generated_tangent_meshes={} uv2_stream_meshes={} uv3_stream_meshes={} wide_low_uv_stream_meshes={} wide_high_uv_stream_meshes={} billboard_stream_meshes={}",
            views.len(),
            requests.uv1_stream_meshes.len(),
            requests.tangent_stream_meshes.len(),
            requests.generated_tangent_mesh_count(),
            requests.uv2_stream_meshes.len(),
            requests.uv3_stream_meshes.len(),
            requests.wide_low_uv_stream_meshes.len(),
            requests.wide_high_uv_stream_meshes.len(),
            requests.billboard_stream_meshes.len(),
        );
        let mesh_ids_needing_all_extended_streams = requests.all_extended_stream_meshes();
        self.ensure_view_asset_prewarm_requests(
            device,
            &requests,
            &mesh_ids_needing_all_extended_streams,
        );
        self.pre_warm_material_assets_from_blackboards(views, view_layouts, resource_layouts);
        self.pre_warm_world_mesh_prepass_pipelines_from_blackboards(device, views, view_layouts);
    }

    fn pre_warm_material_assets_from_blackboards(
        &self,
        views: &[FrameView<'_>],
        view_layouts: &[Option<PreRecordViewResourceLayout>],
        resource_layouts: &[PreRecordViewResourceLayout],
    ) {
        profiling::scope!("graph::pre_warm_material_assets");
        let mut warmed_pipelines = HashSet::new();
        let mut warmed_embedded_stems = HashSet::new();
        for (view, layout) in views.iter().zip(view_layouts.iter()) {
            let Some(layout) = *layout else {
                continue;
            };
            let supports_multiview = self
                .gpu_limits
                .as_ref()
                .is_some_and(|limits| limits.supports_multiview);
            let offscreen = matches!(view.target, FrameViewTarget::OffscreenRt(_));
            self.pre_warm_material_draw_plan(
                view.initial_blackboard.get::<WorldMeshDrawPlanSlot>(),
                layout,
                supports_multiview,
                offscreen,
                &mut warmed_pipelines,
                &mut warmed_embedded_stems,
            );
            if let Some(overlay_layout) = view
                .desktop_overlay_resource_view_id()
                .and_then(|view_id| {
                    resource_layouts
                        .iter()
                        .find(|layout| layout.view_id == view_id)
                })
                .copied()
            {
                self.pre_warm_material_draw_plan(
                    view.initial_blackboard
                        .get::<WorldMeshOverlayDrawPlanSlot>(),
                    overlay_layout,
                    supports_multiview,
                    offscreen,
                    &mut warmed_pipelines,
                    &mut warmed_embedded_stems,
                );
            }
        }
        logger::trace!(
            "graph pre-warm material assets: pipelines={} embedded_layouts={}",
            warmed_pipelines.len(),
            warmed_embedded_stems.len()
        );
    }

    fn pre_warm_material_draw_plan(
        &self,
        draw_plan: Option<&crate::world_mesh::WorldMeshDrawPlan>,
        layout: PreRecordViewResourceLayout,
        supports_multiview: bool,
        offscreen: bool,
        warmed_pipelines: &mut HashSet<MaterialPipelineWarmupKey>,
        warmed_embedded_stems: &mut HashSet<Arc<str>>,
    ) {
        let Some(collection) =
            draw_plan.and_then(crate::world_mesh::WorldMeshDrawPlan::as_prefetched)
        else {
            return;
        };
        let (pass_desc, shader_perm) = material_pass_desc_for_layout(layout, supports_multiview);
        let mut item_index = 0usize;
        while let Some(item) = collection.items.get(item_index) {
            let mut front_face = item.batch_key.front_face;
            if offscreen {
                front_face = front_face.flipped();
            }
            let variant = MaterialPipelineVariantSpec {
                permutation: shader_perm,
                shader_specialization: item.batch_key.shader_specialization,
                blend_mode: item.batch_key.blend_mode,
                render_state: item.batch_key.render_state,
                front_face,
                primitive_topology: item.batch_key.primitive_topology,
            };
            let warmup_key = MaterialPipelineWarmupKey {
                kind: item.batch_key.pipeline.clone(),
                desc: pass_desc,
                variant,
            };
            if warmed_pipelines.insert(warmup_key.clone()) {
                self.materials.queue_material_pipeline_warmup(
                    &warmup_key.kind,
                    &warmup_key.desc,
                    warmup_key.variant,
                );
            }
            if let RasterPipelineKind::EmbeddedStem(stem) = &item.batch_key.pipeline
                && warmed_embedded_stems.insert(Arc::clone(stem))
            {
                self.materials
                    .pre_warm_embedded_material_layout(stem.as_ref());
            }
            item_index = next_material_warmup_run_start(&collection.items, item_index);
        }
    }

    fn pre_warm_world_mesh_prepass_pipelines_from_blackboards(
        &self,
        device: &wgpu::Device,
        views: &[FrameView<'_>],
        view_layouts: &[Option<PreRecordViewResourceLayout>],
    ) {
        profiling::scope!("graph::pre_warm_world_mesh_prepass_pipelines");
        let mut keys = WorldMeshPrepassPipelineWarmupKeys::default();
        for (view, layout) in views.iter().zip(view_layouts.iter()) {
            let Some(layout) = *layout else {
                continue;
            };
            let Some(draw_plan) = view.initial_blackboard.get::<WorldMeshDrawPlanSlot>() else {
                continue;
            };
            let Some(collection) = draw_plan.as_prefetched() else {
                continue;
            };
            let supports_multiview = self
                .gpu_limits
                .as_ref()
                .is_some_and(|limits| limits.supports_multiview);
            let (pass_desc, shader_perm) =
                material_pass_desc_for_layout(layout, supports_multiview);
            let pipeline = WorldMeshForwardPipelineState {
                use_multiview: pass_desc.multiview_mask.is_some(),
                pass_desc,
                shader_perm,
            };
            let offscreen = matches!(view.target, FrameViewTarget::OffscreenRt(_));
            for item in &collection.items {
                keys.record_item(item, &pipeline, offscreen);
            }
        }
        let (depth_prepass_requests, normal_prepass_requests) = keys.warm(device);
        logger::trace!(
            "graph pre-warm world mesh prepass pipelines: depth_requests={} normal_requests={}",
            depth_prepass_requests,
            normal_prepass_requests
        );
    }

    fn ensure_view_asset_prewarm_requests(
        &mut self,
        device: &wgpu::Device,
        requests: &ViewAssetPrewarmRequests,
        mesh_ids_needing_all_extended_streams: &HashSet<i32>,
    ) {
        {
            let mesh_pool = self.asset_transfers.mesh_pool_mut();
            for (&mesh_asset_id, &demand) in &requests.derived_stream_demands {
                mesh_pool.record_derived_stream_demand(mesh_asset_id, demand);
            }
        }
        for (&mesh_asset_id, demand) in &requests.derived_stream_demands {
            self.ensure_demands_streams_prewarm_requests(device, mesh_asset_id, demand);
        }
        for &mesh_asset_id in mesh_ids_needing_all_extended_streams {
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_extended_vertex_streams(
                    device,
                    mesh_asset_id,
                    requests.tangent_fallback_mode(mesh_asset_id),
                );
        }
        for &mesh_asset_id in &requests.uv1_stream_meshes {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv1_vertex_stream(device, mesh_asset_id);
        }
        for &mesh_asset_id in &requests.tangent_stream_meshes {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_tangent_vertex_stream(
                    device,
                    mesh_asset_id,
                    requests.tangent_fallback_mode(mesh_asset_id),
                );
        }
        for &mesh_asset_id in &requests.billboard_stream_meshes {
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_billboard_vertex_streams(
                    device,
                    mesh_asset_id,
                    requests.tangent_fallback_mode(mesh_asset_id),
                );
        }
        for &mesh_asset_id in &requests.uv2_stream_meshes {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv2_vertex_stream(device, mesh_asset_id);
        }
        for &mesh_asset_id in &requests.uv3_stream_meshes {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv3_vertex_stream(device, mesh_asset_id);
        }
        for &mesh_asset_id in &requests.wide_low_uv_stream_meshes {
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_wide_low_uv_vertex_stream(device, mesh_asset_id);
        }
        for &mesh_asset_id in &requests.wide_high_uv_stream_meshes {
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_wide_high_uv_vertex_stream(device, mesh_asset_id);
        }
    }

    fn ensure_demands_streams_prewarm_requests(
        &mut self,
        device: &wgpu::Device,
        mesh_asset_id: i32,
        demand: &MeshDerivedStreamDemand,
    ) {
        if demand
            .mask
            .intersects(MeshDerivedStreamMask::POSITION | MeshDerivedStreamMask::NORMAL)
        {
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_position_normal_vertex_streams(device, mesh_asset_id);
        }
        if demand.mask.contains(MeshDerivedStreamMask::UV0) {
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv0_vertex_stream(device, mesh_asset_id);
        }
        if demand.mask.contains(MeshDerivedStreamMask::COLOR) {
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_color_vertex_stream(device, mesh_asset_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::MaterialPipelineDesc;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn pipeline_state() -> WorldMeshForwardPipelineState {
        WorldMeshForwardPipelineState {
            use_multiview: false,
            pass_desc: MaterialPipelineDesc {
                surface_format: wgpu::TextureFormat::Rgba16Float,
                depth_stencil_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                sample_count: 1,
                multiview_mask: None,
            },
            shader_perm: ShaderPermutation(0),
        }
    }

    fn draw(node_id: i32) -> WorldMeshDrawItem {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        })
    }

    #[test]
    fn prepass_warmup_keys_dedupe_identical_draws() {
        let pipeline = pipeline_state();
        let item = draw(1);
        let mut keys = WorldMeshPrepassPipelineWarmupKeys::default();

        keys.record_item(&item, &pipeline, false);
        keys.record_item(&item, &pipeline, false);

        assert_eq!(keys.depth.len(), 1);
        assert_eq!(keys.normal.len(), 1);
    }

    #[test]
    fn prepass_warmup_keys_keep_offscreen_normal_front_face_distinct() {
        let pipeline = pipeline_state();
        let item = draw(1);
        let mut keys = WorldMeshPrepassPipelineWarmupKeys::default();

        keys.record_item(&item, &pipeline, false);
        keys.record_item(&item, &pipeline, true);

        assert_eq!(keys.depth.len(), 1);
        assert_eq!(keys.normal.len(), 2);
    }
}
