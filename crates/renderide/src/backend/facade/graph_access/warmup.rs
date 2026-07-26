//! Warmup helpers for graph-seeded backend assets.

use std::num::NonZeroU32;
use std::sync::Arc;

use hashbrown::{HashMap, HashSet};

use crate::assets::mesh::{MeshDerivedStreamDemand, MeshDerivedStreamMask};
use crate::graph_inputs::{OffscreenWriteTarget, PreRecordViewResourceLayout, ViewWinding};
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

/// Last completed graph warmup input with retained draw-list identities.
#[derive(Default)]
pub(in crate::backend::facade) struct BackendGraphWarmupCache {
    completed: Option<CompletedGraphWarmup>,
}

impl BackendGraphWarmupCache {
    /// Invalidates device-owned pipeline and asset readiness after a GPU attach.
    pub(in crate::backend::facade) fn clear(&mut self) {
        self.completed = None;
    }

    fn is_exact_hit(
        &self,
        views: &[FrameView<'_>],
        view_layouts: &[Option<PreRecordViewResourceLayout>],
        resource_layouts: &[PreRecordViewResourceLayout],
        mesh_pool_generation: u64,
        material_generation: u64,
        supports_multiview: bool,
    ) -> bool {
        self.completed.as_ref().is_some_and(|completed| {
            completed.matches(
                views,
                view_layouts,
                resource_layouts,
                mesh_pool_generation,
                material_generation,
                supports_multiview,
            )
        })
    }

    fn commit_if_ready(&mut self, ready: bool, completed: impl FnOnce() -> CompletedGraphWarmup) {
        self.completed = ready.then(completed);
    }
}

#[derive(Clone, Debug)]
struct CompletedGraphWarmup {
    views: Vec<CompletedViewGraphWarmup>,
    mesh_pool_generation: u64,
    material_generation: u64,
    supports_multiview: bool,
}

impl CompletedGraphWarmup {
    fn capture(
        views: &[FrameView<'_>],
        view_layouts: &[Option<PreRecordViewResourceLayout>],
        resource_layouts: &[PreRecordViewResourceLayout],
        mesh_pool_generation: u64,
        material_generation: u64,
        supports_multiview: bool,
    ) -> Self {
        Self {
            views: views
                .iter()
                .enumerate()
                .map(|(view_index, view)| {
                    CompletedViewGraphWarmup::capture_inputs(
                        view.view_id,
                        view.initial_blackboard.get::<WorldMeshDrawPlanSlot>(),
                        view.initial_blackboard
                            .get::<WorldMeshOverlayDrawPlanSlot>(),
                        view_layouts.get(view_index).copied().flatten(),
                        overlay_resource_layout(view, resource_layouts),
                        matches!(view.target, FrameViewTarget::OffscreenRt(_)),
                        view.view_winding,
                    )
                })
                .collect(),
            mesh_pool_generation,
            material_generation,
            supports_multiview,
        }
    }

    fn matches(
        &self,
        views: &[FrameView<'_>],
        view_layouts: &[Option<PreRecordViewResourceLayout>],
        resource_layouts: &[PreRecordViewResourceLayout],
        mesh_pool_generation: u64,
        material_generation: u64,
        supports_multiview: bool,
    ) -> bool {
        self.mesh_pool_generation == mesh_pool_generation
            && self.material_generation == material_generation
            && self.supports_multiview == supports_multiview
            && self.views.len() == views.len()
            && self
                .views
                .iter()
                .zip(views)
                .enumerate()
                .all(|(view_index, (completed, view))| {
                    completed.matches_inputs(
                        view.view_id,
                        view.initial_blackboard.get::<WorldMeshDrawPlanSlot>(),
                        view.initial_blackboard
                            .get::<WorldMeshOverlayDrawPlanSlot>(),
                        view_layouts.get(view_index).copied().flatten(),
                        overlay_resource_layout(view, resource_layouts),
                        matches!(view.target, FrameViewTarget::OffscreenRt(_)),
                        view.view_winding,
                    )
                })
    }
}

#[derive(Clone, Debug)]
struct CompletedViewGraphWarmup {
    view_id: crate::camera::ViewId,
    world_draws: Option<Arc<[WorldMeshDrawItem]>>,
    overlay_draws: Option<Arc<[WorldMeshDrawItem]>>,
    layout: Option<PreRecordViewResourceLayout>,
    overlay_layout: Option<PreRecordViewResourceLayout>,
    active_offscreen: bool,
    view_winding: ViewWinding,
}

impl CompletedViewGraphWarmup {
    fn capture_inputs(
        view_id: crate::camera::ViewId,
        world_plan: Option<&crate::world_mesh::WorldMeshDrawPlan>,
        overlay_plan: Option<&crate::world_mesh::WorldMeshDrawPlan>,
        layout: Option<PreRecordViewResourceLayout>,
        overlay_layout: Option<PreRecordViewResourceLayout>,
        active_offscreen: bool,
        view_winding: ViewWinding,
    ) -> Self {
        Self {
            view_id,
            world_draws: draw_list_identity(world_plan).cloned(),
            overlay_draws: draw_list_identity(overlay_plan).cloned(),
            layout,
            overlay_layout,
            active_offscreen,
            view_winding,
        }
    }

    fn matches_inputs(
        &self,
        view_id: crate::camera::ViewId,
        world_plan: Option<&crate::world_mesh::WorldMeshDrawPlan>,
        overlay_plan: Option<&crate::world_mesh::WorldMeshDrawPlan>,
        layout: Option<PreRecordViewResourceLayout>,
        overlay_layout: Option<PreRecordViewResourceLayout>,
        active_offscreen: bool,
        view_winding: ViewWinding,
    ) -> bool {
        self.view_id == view_id
            && same_draw_list_identity(self.world_draws.as_ref(), draw_list_identity(world_plan))
            && same_draw_list_identity(
                self.overlay_draws.as_ref(),
                draw_list_identity(overlay_plan),
            )
            && self.layout == layout
            && self.overlay_layout == overlay_layout
            && self.active_offscreen == active_offscreen
            && self.view_winding == view_winding
    }
}

fn draw_list_identity(
    draw_plan: Option<&crate::world_mesh::WorldMeshDrawPlan>,
) -> Option<&Arc<[WorldMeshDrawItem]>> {
    draw_plan
        .and_then(crate::world_mesh::WorldMeshDrawPlan::as_prefetched)
        .map(|collection| &collection.items)
}

fn same_draw_list_identity(
    completed: Option<&Arc<[WorldMeshDrawItem]>>,
    current: Option<&Arc<[WorldMeshDrawItem]>>,
) -> bool {
    match (completed, current) {
        (Some(completed), Some(current)) => Arc::ptr_eq(completed, current),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    }
}

fn overlay_resource_layout(
    view: &FrameView<'_>,
    resource_layouts: &[PreRecordViewResourceLayout],
) -> Option<PreRecordViewResourceLayout> {
    view.desktop_overlay_resource_view_id()
        .and_then(|view_id| {
            resource_layouts
                .iter()
                .find(|layout| layout.view_id == view_id)
        })
        .copied()
}

#[derive(Default)]
struct ViewAssetPrewarmRequests {
    uv1_stream_meshes: HashSet<i32>,
    tangent_stream_meshes: HashSet<i32>,
    raw_tangent_stream_meshes: HashSet<i32>,
    tangent_fallback_modes: HashMap<i32, EmbeddedTangentFallbackMode>,
    uv2_stream_meshes: HashSet<i32>,
    uv3_stream_meshes: HashSet<i32>,
    wide_low_uv_stream_meshes: HashSet<i32>,
    wide_high_uv_stream_meshes: HashSet<i32>,
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
        if item.batch_key.embedded_needs_tangent && item.batch_key.embedded_raw_tangent_payload {
            self.raw_tangent_stream_meshes.insert(item.mesh_asset_id);
            demand.mask |= MeshDerivedStreamMask::RAW_TANGENT;
        } else if item.batch_key.embedded_needs_tangent {
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
    for item in collection.items.iter() {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineWarmupTarget {
    pass_desc: MaterialPipelineDesc,
    shader_perm: ShaderPermutation,
    offscreen: bool,
}

#[derive(Clone, Copy, Debug)]
struct PipelineWarmupTargets {
    targets: [Option<PipelineWarmupTarget>; 3],
}

impl PipelineWarmupTargets {
    fn new(active: PipelineWarmupTarget) -> Self {
        let mut targets = Self {
            targets: [None, None, None],
        };
        targets.push(active);
        targets
    }

    fn push(&mut self, target: PipelineWarmupTarget) {
        if self.targets.iter().flatten().any(|&item| item == target) {
            return;
        }
        if let Some(slot) = self.targets.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(target);
        }
    }

    fn iter(&self) -> impl Iterator<Item = PipelineWarmupTarget> + '_ {
        self.targets.iter().flatten().copied()
    }
}

fn warmup_write_target_for_pipeline(target: PipelineWarmupTarget) -> OffscreenWriteTarget {
    if target.offscreen {
        OffscreenWriteTarget::Untracked
    } else {
        OffscreenWriteTarget::None
    }
}

fn front_face_flip_for_warmup_target(
    view_winding: ViewWinding,
    target: PipelineWarmupTarget,
) -> bool {
    view_winding.flips_front_face_for(warmup_write_target_for_pipeline(target))
}

fn material_warmup_targets_for_layout(
    layout: PreRecordViewResourceLayout,
    supports_multiview: bool,
    active_offscreen: bool,
) -> PipelineWarmupTargets {
    let (pass_desc, shader_perm) = material_pass_desc_for_layout(layout, supports_multiview);
    let mut targets = PipelineWarmupTargets::new(PipelineWarmupTarget {
        pass_desc,
        shader_perm,
        offscreen: active_offscreen,
    });
    if pass_desc.sample_count > 1 || pass_desc.multiview_mask.is_some() {
        targets.push(PipelineWarmupTarget {
            pass_desc: MaterialPipelineDesc {
                sample_count: 1,
                multiview_mask: None,
                ..pass_desc
            },
            shader_perm: ShaderPermutation::default(),
            offscreen: true,
        });
    }
    if pass_desc.sample_count > 1 && pass_desc.multiview_mask.is_some() {
        targets.push(PipelineWarmupTarget {
            pass_desc: MaterialPipelineDesc {
                multiview_mask: None,
                ..pass_desc
            },
            shader_perm: ShaderPermutation::default(),
            offscreen: true,
        });
    }
    targets
}

#[derive(Clone, Copy, Debug)]
struct ViewPipelineWarmupContext {
    layout: PreRecordViewResourceLayout,
    supports_multiview: bool,
    active_offscreen: bool,
    view_winding: ViewWinding,
}

impl ViewPipelineWarmupContext {
    fn targets(self) -> PipelineWarmupTargets {
        material_warmup_targets_for_layout(
            self.layout,
            self.supports_multiview,
            self.active_offscreen,
        )
    }
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
    fn record_item(&mut self, item: &WorldMeshDrawItem, pipeline: &WorldMeshForwardPipelineState) {
        if let Some(key) = depth_prepass_pipeline_key_for_draw(item, pipeline) {
            self.depth.insert(key);
        }
        if world_mesh_item_mirrors_to_normal_prepass(item)
            && let Some(key) = normal_pipeline_key_for_draw(item, pipeline)
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
        let supports_multiview = self
            .gpu_limits
            .as_ref()
            .is_some_and(|limits| limits.supports_multiview);
        let mesh_pool_generation = self.asset_transfers.mesh_pool().mutation_generation();
        let material_generation = self
            .materials
            .material_registry
            .as_ref()
            .map_or(0, |registry| registry.router().generation());
        if self.graph_warmup_cache.is_exact_hit(
            views,
            view_layouts,
            resource_layouts,
            mesh_pool_generation,
            material_generation,
            supports_multiview,
        ) {
            profiling::scope!("graph::pre_warm_view_assets_cache_hit");
            #[cfg(feature = "tracy")]
            tracy_client::plot!("world_mesh::graph_warmup_cache_hit", 1.0);
            return;
        }
        #[cfg(feature = "tracy")]
        tracy_client::plot!("world_mesh::graph_warmup_cache_hit", 0.0);

        let requests = collect_view_asset_prewarm_requests(views);
        logger::trace!(
            "graph pre-warm view assets: views={} uv1_stream_meshes={} tangent_stream_meshes={} raw_tangent_stream_meshes={} generated_tangent_meshes={} uv2_stream_meshes={} uv3_stream_meshes={} wide_low_uv_stream_meshes={} wide_high_uv_stream_meshes={}",
            views.len(),
            requests.uv1_stream_meshes.len(),
            requests.tangent_stream_meshes.len(),
            requests.raw_tangent_stream_meshes.len(),
            requests.generated_tangent_mesh_count(),
            requests.uv2_stream_meshes.len(),
            requests.uv3_stream_meshes.len(),
            requests.wide_low_uv_stream_meshes.len(),
            requests.wide_high_uv_stream_meshes.len(),
        );
        let mesh_ids_needing_all_extended_streams = requests.all_extended_stream_meshes();
        let assets_ready = self.ensure_view_asset_prewarm_requests(
            device,
            &requests,
            &mesh_ids_needing_all_extended_streams,
        );
        self.pre_warm_material_assets_from_blackboards(views, view_layouts, resource_layouts);
        self.pre_warm_world_mesh_prepass_pipelines_from_blackboards(device, views, view_layouts);
        let mesh_pool_generation = self.asset_transfers.mesh_pool().mutation_generation();
        self.graph_warmup_cache.commit_if_ready(assets_ready, || {
            CompletedGraphWarmup::capture(
                views,
                view_layouts,
                resource_layouts,
                mesh_pool_generation,
                material_generation,
                supports_multiview,
            )
        });
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
            let active_offscreen = matches!(view.target, FrameViewTarget::OffscreenRt(_));
            let context = ViewPipelineWarmupContext {
                layout,
                supports_multiview,
                active_offscreen,
                view_winding: view.view_winding,
            };
            self.pre_warm_material_draw_plan(
                view.initial_blackboard.get::<WorldMeshDrawPlanSlot>(),
                context,
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
                    ViewPipelineWarmupContext {
                        layout: overlay_layout,
                        ..context
                    },
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
        context: ViewPipelineWarmupContext,
        warmed_pipelines: &mut HashSet<MaterialPipelineWarmupKey>,
        warmed_embedded_stems: &mut HashSet<Arc<str>>,
    ) {
        let Some(collection) =
            draw_plan.and_then(crate::world_mesh::WorldMeshDrawPlan::as_prefetched)
        else {
            return;
        };
        let targets = context.targets();
        let mut item_index = 0usize;
        while let Some(item) = collection.items.get(item_index) {
            for target in targets.iter() {
                let mut front_face = item.batch_key.front_face;
                if front_face_flip_for_warmup_target(context.view_winding, target) {
                    front_face = front_face.flipped();
                }
                let variant = MaterialPipelineVariantSpec {
                    permutation: target.shader_perm,
                    shader_specialization: item.batch_key.shader_specialization,
                    blend_mode: item.batch_key.blend_mode,
                    render_state: item.batch_key.render_state,
                    pass_routing: item.batch_key.pass_routing(),
                    front_face,
                    primitive_topology: item.batch_key.primitive_topology,
                };
                let warmup_key = MaterialPipelineWarmupKey {
                    kind: item.batch_key.pipeline.clone(),
                    desc: target.pass_desc,
                    variant,
                };
                if warmed_pipelines.insert(warmup_key.clone()) {
                    self.materials.queue_material_pipeline_warmup(
                        &warmup_key.kind,
                        &warmup_key.desc,
                        warmup_key.variant,
                    );
                }
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
            let active_offscreen = matches!(view.target, FrameViewTarget::OffscreenRt(_));
            let targets =
                material_warmup_targets_for_layout(layout, supports_multiview, active_offscreen);
            for target in targets.iter() {
                let pipeline = WorldMeshForwardPipelineState {
                    use_multiview: target.pass_desc.multiview_mask.is_some(),
                    pass_desc: target.pass_desc,
                    shader_perm: target.shader_perm,
                    front_face_flip: front_face_flip_for_warmup_target(view.view_winding, target),
                };
                for item in collection.items.iter() {
                    keys.record_item(item, &pipeline);
                }
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
    ) -> bool {
        let mut all_ready = true;
        {
            let mesh_pool = self.asset_transfers.mesh_pool_mut();
            for (&mesh_asset_id, &demand) in &requests.derived_stream_demands {
                mesh_pool.record_derived_stream_demand(mesh_asset_id, demand);
            }
        }
        for (&mesh_asset_id, demand) in &requests.derived_stream_demands {
            if demand
                .mask
                .intersects(MeshDerivedStreamMask::POSITION | MeshDerivedStreamMask::NORMAL)
            {
                all_ready &= self
                    .asset_transfers
                    .mesh_pool_mut()
                    .ensure_position_normal_vertex_streams(device, mesh_asset_id);
            }
            if demand.mask.contains(MeshDerivedStreamMask::UV0) {
                all_ready &= self
                    .asset_transfers
                    .mesh_pool_mut()
                    .ensure_uv0_vertex_stream(device, mesh_asset_id);
            }
            if demand.mask.contains(MeshDerivedStreamMask::COLOR) {
                all_ready &= self
                    .asset_transfers
                    .mesh_pool_mut()
                    .ensure_color_vertex_stream(device, mesh_asset_id);
            }
        }
        for &mesh_asset_id in mesh_ids_needing_all_extended_streams {
            all_ready &= self
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
            all_ready &= self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv1_vertex_stream(device, mesh_asset_id);
        }
        for &mesh_asset_id in &requests.tangent_stream_meshes {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            all_ready &= self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_tangent_vertex_stream(
                    device,
                    mesh_asset_id,
                    requests.tangent_fallback_mode(mesh_asset_id),
                );
        }
        for &mesh_asset_id in &requests.raw_tangent_stream_meshes {
            all_ready &= self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_raw_tangent_vertex_stream(device, mesh_asset_id);
        }
        for &mesh_asset_id in &requests.uv2_stream_meshes {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            all_ready &= self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv2_vertex_stream(device, mesh_asset_id);
        }
        for &mesh_asset_id in &requests.uv3_stream_meshes {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            all_ready &= self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv3_vertex_stream(device, mesh_asset_id);
        }
        for &mesh_asset_id in &requests.wide_low_uv_stream_meshes {
            all_ready &= self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_wide_low_uv_vertex_stream(device, mesh_asset_id);
        }
        for &mesh_asset_id in &requests.wide_high_uv_stream_meshes {
            all_ready &= self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_wide_high_uv_vertex_stream(device, mesh_asset_id);
        }
        all_ready
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::camera::ViewId;
    use crate::materials::MaterialPipelineDesc;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::{
        PrefetchedWorldMeshViewDraws, WorldMeshDrawCollection, WorldMeshDrawPlan,
    };

    fn layout(sample_count: u32, stereo: bool) -> PreRecordViewResourceLayout {
        PreRecordViewResourceLayout {
            view_id: ViewId::Main,
            width: 1280,
            height: 720,
            stereo,
            sample_count,
            depth_format: wgpu::TextureFormat::Depth24PlusStencil8,
            color_format: wgpu::TextureFormat::Rgba16Float,
            needs_depth_snapshot: false,
            needs_per_object_color_snapshot: false,
            needs_named_color_snapshot: false,
        }
    }

    fn targets_for(
        layout: PreRecordViewResourceLayout,
        supports_multiview: bool,
        active_offscreen: bool,
    ) -> Vec<PipelineWarmupTarget> {
        material_warmup_targets_for_layout(layout, supports_multiview, active_offscreen)
            .iter()
            .collect()
    }

    fn pipeline_state_for_target(target: PipelineWarmupTarget) -> WorldMeshForwardPipelineState {
        WorldMeshForwardPipelineState {
            use_multiview: target.pass_desc.multiview_mask.is_some(),
            pass_desc: target.pass_desc,
            shader_perm: target.shader_perm,
            front_face_flip: front_face_flip_for_warmup_target(ViewWinding::normal(), target),
        }
    }

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
            front_face_flip: false,
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

    fn plan_with_items(items: Arc<[WorldMeshDrawItem]>) -> WorldMeshDrawPlan {
        let mut collection = WorldMeshDrawCollection::empty();
        collection.items = items;
        WorldMeshDrawPlan::Prefetched(Arc::new(PrefetchedWorldMeshViewDraws::new(
            collection, None,
        )))
    }

    #[test]
    fn warmup_identity_survives_projection_only_prefetched_wrapper_refresh() {
        let shared_items: Arc<[WorldMeshDrawItem]> = Arc::from(vec![draw(1)]);
        let original = plan_with_items(Arc::clone(&shared_items));
        let WorldMeshDrawPlan::Prefetched(original_draws) = &original else {
            panic!("test plan must be prefetched");
        };
        let refreshed = WorldMeshDrawPlan::Prefetched(Arc::new(PrefetchedWorldMeshViewDraws {
            cull_proj: None,
            hi_z_temporal: None,
            ..(**original_draws).clone()
        }));
        let completed = CompletedViewGraphWarmup::capture_inputs(
            ViewId::Main,
            Some(&original),
            None,
            Some(layout(1, false)),
            None,
            false,
            ViewWinding::normal(),
        );

        assert!(completed.matches_inputs(
            ViewId::Main,
            Some(&refreshed),
            None,
            Some(layout(1, false)),
            None,
            false,
            ViewWinding::normal(),
        ));
    }

    #[test]
    fn warmup_identity_rejects_equal_rows_in_a_rebuilt_draw_list() {
        let shared_items: Arc<[WorldMeshDrawItem]> = Arc::from(vec![draw(1)]);
        let original = plan_with_items(Arc::clone(&shared_items));
        let rebuilt_items: Arc<[WorldMeshDrawItem]> =
            Arc::from(shared_items.iter().cloned().collect::<Vec<_>>());
        let rebuilt = plan_with_items(rebuilt_items);
        let completed = CompletedViewGraphWarmup::capture_inputs(
            ViewId::Main,
            Some(&original),
            None,
            Some(layout(1, false)),
            None,
            false,
            ViewWinding::normal(),
        );

        assert!(!completed.matches_inputs(
            ViewId::Main,
            Some(&rebuilt),
            None,
            Some(layout(1, false)),
            None,
            false,
            ViewWinding::normal(),
        ));
    }

    #[test]
    fn warmup_identity_includes_relevant_layout_and_view_state() {
        let plan = plan_with_items(Arc::from(vec![draw(1)]));
        let mut overlay_layout = layout(1, false);
        overlay_layout.view_id = ViewId::MainOverlay;
        let completed = CompletedViewGraphWarmup::capture_inputs(
            ViewId::Main,
            Some(&plan),
            None,
            Some(layout(1, false)),
            Some(overlay_layout),
            false,
            ViewWinding::normal(),
        );

        assert!(!completed.matches_inputs(
            ViewId::Main,
            Some(&plan),
            None,
            Some(layout(4, false)),
            Some(overlay_layout),
            false,
            ViewWinding::normal(),
        ));
        assert!(!completed.matches_inputs(
            ViewId::Main,
            Some(&plan),
            None,
            Some(layout(1, false)),
            Some(overlay_layout),
            true,
            ViewWinding::normal(),
        ));
        assert!(!completed.matches_inputs(
            ViewId::Main,
            Some(&plan),
            None,
            Some(layout(1, false)),
            Some(overlay_layout),
            false,
            ViewWinding::mirror_reflection(),
        ));
    }

    #[test]
    fn failed_asset_ensure_does_not_commit_a_warmup_hit() {
        let builds = Cell::new(0);
        let completed = CompletedGraphWarmup {
            views: Vec::new(),
            mesh_pool_generation: 7,
            material_generation: 11,
            supports_multiview: false,
        };
        let mut cache = BackendGraphWarmupCache::default();
        cache.commit_if_ready(true, || {
            builds.set(builds.get() + 1);
            completed.clone()
        });
        assert!(cache.completed.is_some());

        cache.commit_if_ready(false, || {
            builds.set(builds.get() + 1);
            completed
        });

        assert!(cache.completed.is_none());
        assert_eq!(builds.get(), 1);
    }

    #[test]
    fn msaa_layout_adds_mono_offscreen_single_sample_companion() {
        let targets = targets_for(layout(4, false), false, false);

        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].pass_desc.sample_count, 4);
        assert_eq!(targets[0].pass_desc.multiview_mask, None);
        assert_eq!(targets[0].shader_perm, ShaderPermutation::default());
        assert!(!targets[0].offscreen);
        assert_eq!(targets[1].pass_desc.sample_count, 1);
        assert_eq!(targets[1].pass_desc.multiview_mask, None);
        assert_eq!(targets[1].shader_perm, ShaderPermutation::default());
        assert!(targets[1].offscreen);
    }

    #[test]
    fn multiview_msaa_layout_adds_camera_companions() {
        let targets = targets_for(layout(4, true), true, false);

        assert_eq!(targets.len(), 3);
        assert_eq!(targets[0].pass_desc.sample_count, 4);
        assert!(targets[0].pass_desc.multiview_mask.is_some());
        assert_eq!(targets[0].shader_perm, SHADER_PERM_MULTIVIEW_STEREO);
        assert!(!targets[0].offscreen);
        assert_eq!(targets[1].pass_desc.sample_count, 1);
        assert_eq!(targets[1].pass_desc.multiview_mask, None);
        assert_eq!(targets[1].shader_perm, ShaderPermutation::default());
        assert!(targets[1].offscreen);
        assert_eq!(targets[2].pass_desc.sample_count, 4);
        assert_eq!(targets[2].pass_desc.multiview_mask, None);
        assert_eq!(targets[2].shader_perm, ShaderPermutation::default());
        assert!(targets[2].offscreen);
    }

    #[test]
    fn single_sample_offscreen_layout_keeps_only_active_target() {
        let targets = targets_for(layout(1, false), false, true);

        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].pass_desc.sample_count, 1);
        assert_eq!(targets[0].pass_desc.multiview_mask, None);
        assert!(targets[0].offscreen);
    }

    #[test]
    fn prepass_warmup_keys_dedupe_identical_draws() {
        let pipeline = pipeline_state();
        let item = draw(1);
        let mut keys = WorldMeshPrepassPipelineWarmupKeys::default();

        keys.record_item(&item, &pipeline);
        keys.record_item(&item, &pipeline);

        assert_eq!(keys.depth.len(), 1);
        assert_eq!(keys.normal.len(), 1);
    }

    #[test]
    fn prepass_warmup_keys_keep_offscreen_normal_front_face_distinct() {
        let pipeline = pipeline_state();
        let item = draw(1);
        let mut keys = WorldMeshPrepassPipelineWarmupKeys::default();

        let mut flipped_pipeline = pipeline_state();
        flipped_pipeline.front_face_flip = true;

        keys.record_item(&item, &pipeline);
        keys.record_item(&item, &flipped_pipeline);

        assert_eq!(keys.depth.len(), 2);
        assert_eq!(keys.normal.len(), 2);
    }

    #[test]
    fn prepass_warmup_keys_dedupe_across_companion_targets() {
        let item = draw(1);
        let targets = targets_for(layout(4, false), false, true);
        let mut keys = WorldMeshPrepassPipelineWarmupKeys::default();

        for target in targets {
            let pipeline = pipeline_state_for_target(target);
            keys.record_item(&item, &pipeline);
            keys.record_item(&item, &pipeline);
        }

        assert_eq!(keys.depth.len(), 2);
        assert_eq!(keys.normal.len(), 2);
    }
}
