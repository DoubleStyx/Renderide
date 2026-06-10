//! Frame extraction packets between runtime view planning and backend graph execution.
//!
//! This module owns the immutable CPU-side hand-off for one render tick: prepared views,
//! cull snapshots, prefetched draw plans, and the final submit packet. Keeping these types out
//! of [`super::render`] makes the render entrypoint an orchestration layer instead of another
//! subsystem owner.

mod cull;
mod queue;
mod sort;
mod visible_deform;

pub(in crate::runtime) use queue::select_inner_parallelism;

use rayon::prelude::*;

use crate::backend::{
    ExtractedFrameShared, RenderBackend, WorldMeshDrawPlanSlot, WorldMeshOverlayDrawPlanSlot,
};
use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy};
use crate::gpu::GpuContext;
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::{
    FrameGlobalView, FrameView, FrameViewResourceHints, GraphExecuteError,
    ViewFamilyGraphRequirements,
};
use crate::world_mesh::{
    WorldMeshCommandCache, WorldMeshDrawArrangeParallelism, WorldMeshDrawPlan,
};

use cull::{ViewCullSnapshot, cull_snapshot_for_view};
use queue::{QueuedViewDraws, queue_view_draws};
use sort::{select_arrange_parallelism, sort_view_draws, trace_view_draw_plans};
use visible_deform::visible_mesh_deform_keys_from_draw_plans;

use super::view_plan::{FrameViewPlan, ViewFamilyPlan};

/// Prepared view plans assigned to one cull-snapshot worker.
const CULL_SNAPSHOT_PARALLEL_CHUNK_VIEWS: usize = 1;

/// Immutable runtime-owned extraction packet built before per-view draw collection starts.
///
/// Prepared views live beside the backend's read-only draw-prep view so later stages no longer
/// need to reach back into mutable runtime or backend state.
pub(in crate::runtime) struct ExtractedFrame<'views, 'backend> {
    /// Ordered per-frame view plans and aggregate graph requirements.
    prepared_views: PreparedViews<'views>,
    /// Backend-owned draw-prep view assembled once for the frame.
    shared: ExtractedFrameShared<'backend>,
    /// Mesh LOD bias multiplier for every view in this schedule.
    mesh_lod_bias: f32,
}

impl<'views, 'backend> ExtractedFrame<'views, 'backend> {
    /// Builds a frame extraction packet from prepared views and backend shared setup.
    pub(in crate::runtime) fn new(
        prepared_views: PreparedViews<'views>,
        shared: ExtractedFrameShared<'backend>,
        mesh_lod_bias: f32,
    ) -> Self {
        ExtractedFrame {
            prepared_views,
            shared,
            mesh_lod_bias,
        }
    }

    /// Queues explicit world-mesh draw candidates for each prepared view.
    pub(in crate::runtime) fn queue_draws(self) -> QueuedDraws<'views, 'backend> {
        let ExtractedFrame {
            prepared_views,
            shared,
            mesh_lod_bias,
        } = self;
        let cull_snapshots = gather_view_cull_snapshots(&shared, prepared_views.plans());
        let view_draws = queue_view_draws(
            &shared,
            prepared_views.plans(),
            cull_snapshots,
            mesh_lod_bias,
        );
        let arrange_parallelism = select_arrange_parallelism(&view_draws);
        QueuedDraws {
            prepared_views,
            view_draws,
            arrange_parallelism,
            command_cache: shared.command_cache,
        }
    }
}

fn gather_view_cull_snapshots(
    shared: &ExtractedFrameShared<'_>,
    plans: &[FrameViewPlan<'_>],
) -> Vec<Option<ViewCullSnapshot>> {
    profiling::scope!("render::gather_view_cull_snapshots");
    match plans.len() {
        0 => Vec::new(),
        1 => vec![cull_snapshot_for_view(shared, &plans[0])],
        _ if FrameParallelPolicy::for_current_thread_pool()
            .admit_independent_items(
                FrameCpuWorkload::independent_items(plans.len()),
                CULL_SNAPSHOT_PARALLEL_CHUNK_VIEWS,
            )
            .is_parallel() =>
        {
            plans
                .par_iter()
                .with_min_len(CULL_SNAPSHOT_PARALLEL_CHUNK_VIEWS)
                .map(|prep| cull_snapshot_for_view(shared, prep))
                .collect()
        }
        _ => plans
            .iter()
            .map(|prep| cull_snapshot_for_view(shared, prep))
            .collect(),
    }
}

/// Queued per-view draw candidates built after view planning and before phase sorting.
pub(in crate::runtime) struct QueuedDraws<'a, 'backend> {
    /// Ordered per-frame view plans and aggregate graph requirements.
    prepared_views: PreparedViews<'a>,
    /// Queued draw candidates for every prepared view.
    view_draws: Vec<QueuedViewDraws>,
    /// Rayon tier to use for final draw arrangement inside each queued view.
    arrange_parallelism: WorldMeshDrawArrangeParallelism,
    /// Persistent arranged draw command-list cache owned by the backend.
    command_cache: &'backend WorldMeshCommandCache,
}

impl<'a, 'backend> QueuedDraws<'a, 'backend> {
    /// Sorts queued draws and promotes them into final per-view draw plans.
    pub(in crate::runtime) fn sort_draws(self) -> PreparedDraws<'a> {
        let view_draws = sort_view_draws(
            self.view_draws,
            self.arrange_parallelism,
            self.command_cache,
        );
        {
            profiling::scope!("render::sort_view_draws::trace_plans");
            trace_view_draw_plans(self.prepared_views.plans(), &view_draws);
        }
        PreparedDraws {
            prepared_views: self.prepared_views,
            view_draws,
        }
    }
}

/// Prepared per-frame view list plus aggregate graph requirements.
pub(in crate::runtime) struct PreparedViews<'a> {
    /// Ordered view family and aggregate graph requirements for this tick.
    family: ViewFamilyPlan<'a>,
}

impl<'a> PreparedViews<'a> {
    /// Builds prepared views from the ordered plan.
    pub(in crate::runtime) fn new(family: ViewFamilyPlan<'a>) -> Self {
        Self { family }
    }

    /// Returns `true` when no view should be rendered this tick.
    pub(in crate::runtime) fn is_empty(&self) -> bool {
        self.family.is_empty()
    }

    /// Shared slice of the ordered planned views.
    pub(in crate::runtime) fn plans(&self) -> &[FrameViewPlan<'a>] {
        self.family.plans()
    }

    /// Primary-view metadata for frame-global graph passes.
    pub(in crate::runtime) fn frame_global(&self) -> &FrameGlobalView {
        self.family.frame_global()
    }

    /// Aggregate graph-shaping requirements for the ordered views.
    pub(in crate::runtime) fn graph_requirements(&self) -> ViewFamilyGraphRequirements {
        self.family.requirements()
    }

    /// Builds executable graph views from the prepared plans and collected draw plans.
    fn build_execution_views<'b>(
        &'b self,
        draw_plans: Vec<ViewWorldMeshDrawPlans>,
    ) -> Vec<FrameView<'b>>
    where
        'a: 'b,
    {
        self.family
            .plans()
            .iter()
            .zip(draw_plans)
            .map(|(prep, draws)| {
                let helper_needs = draws.world.helper_needs();
                let resource_hints = FrameViewResourceHints {
                    needs_depth_snapshot: helper_needs.depth_snapshot,
                    needs_color_snapshot: helper_needs.color_snapshot,
                };
                let mut initial_blackboard = Blackboard::new();
                initial_blackboard.insert::<WorldMeshDrawPlanSlot>(draws.world);
                if let Some(overlay) = draws.desktop_overlay {
                    initial_blackboard.insert::<WorldMeshOverlayDrawPlanSlot>(overlay);
                }
                prep.to_frame_view(resource_hints, initial_blackboard)
            })
            .collect()
    }
}

/// Immutable per-view draw packet built after culling and draw sorting.
pub(in crate::runtime) struct PreparedDraws<'a> {
    /// Ordered per-frame view plans and aggregate graph requirements.
    prepared_views: PreparedViews<'a>,
    /// Explicit draw plan for every prepared view.
    view_draws: Vec<ViewWorldMeshDrawPlans>,
}

impl<'a> PreparedDraws<'a> {
    /// Promotes prepared views plus explicit draws into the final submit packet.
    pub(in crate::runtime) fn into_submit_frame(self) -> SubmitFrame<'a> {
        SubmitFrame {
            prepared_views: self.prepared_views,
            view_draws: self.view_draws,
        }
    }
}

/// Final immutable runtime packet handed to backend execution for one frame.
pub(in crate::runtime) struct SubmitFrame<'a> {
    /// Ordered per-frame view plans and aggregate graph requirements.
    prepared_views: PreparedViews<'a>,
    /// Explicit draw plan for every prepared view.
    view_draws: Vec<ViewWorldMeshDrawPlans>,
}

impl SubmitFrame<'_> {
    /// Prepares frame resources that depend on the sorted draw list.
    pub(in crate::runtime) fn prepare_resources(
        &self,
        scene: &crate::scene::SceneCoordinator,
        backend: &mut RenderBackend,
    ) {
        backend.prepare_lights_for_views(
            scene,
            self.prepared_views.plans().iter().flat_map(|view| {
                let desc = view.light_view_desc();
                let overlay = view.desktop_overlay_resource_view_id().map(|view_id| {
                    let mut desc = desc;
                    desc.view_id = view_id;
                    desc
                });
                std::iter::once(desc).chain(overlay)
            }),
        );
        backend.prepare_shadow_frame_for_views(
            self.prepared_views
                .plans()
                .iter()
                .zip(self.view_draws.iter())
                .map(|(view, draws)| (view.view_id, &draws.shadow_casters)),
        );
        let mut visible_deform_keys = visible_mesh_deform_keys_from_draw_plans(&self.view_draws);
        visible_deform_keys.extend(backend.frame_resources().shadow_mesh_deform_keys());
        backend
            .frame_resources_mut()
            .begin_mesh_deform_submission(visible_deform_keys);
    }

    /// Executes the final submit packet after [`Self::prepare_resources`] has run.
    pub(in crate::runtime) fn execute_after_resource_prepare(
        self,
        gpu: &mut GpuContext,
        scene: &crate::scene::SceneCoordinator,
        backend: &mut RenderBackend,
    ) -> Result<(), GraphExecuteError> {
        let requirements = self.prepared_views.graph_requirements();
        let frame_global = *self.prepared_views.frame_global();
        let mut views = self.prepared_views.build_execution_views(self.view_draws);
        backend.execute_multi_view_frame(gpu, scene, &frame_global, &mut views, requirements, true)
    }
}

/// Sorted draw plans for one executable view.
pub(in crate::runtime) struct ViewWorldMeshDrawPlans {
    /// Regular camera-world draw plan consumed by the main world-mesh pass stack.
    pub(super) world: WorldMeshDrawPlan,
    /// Shadow-caster draw plan consumed by shadow-map preparation.
    pub(super) shadow_casters: WorldMeshDrawPlan,
    /// Desktop overlay draw plan consumed by the post-compose overlay pass.
    pub(super) desktop_overlay: Option<WorldMeshDrawPlan>,
}

impl ViewWorldMeshDrawPlans {
    /// Total draw count represented by all draw plans for this view.
    pub(super) fn draw_count(&self) -> usize {
        self.world.draw_count()
            + self.shadow_casters.draw_count()
            + self
                .desktop_overlay
                .as_ref()
                .map_or(0, WorldMeshDrawPlan::draw_count)
    }
}

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;

    use glam::Mat4;

    use crate::camera::{HostCameraFrame, ViewId};
    use crate::mesh_deform::{SkinCacheKey, SkinCacheRendererKind};
    use crate::occlusion::OcclusionSystem;
    use crate::render_graph::{FrameViewClear, OffscreenWriteTarget, RenderPathProfile};
    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::RenderingContext;
    use crate::world_mesh::CameraTransformDrawFilter;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::{
        PrefetchedWorldMeshViewDraws, ViewLayerPolicy, WorldMeshCullProjParams,
        WorldMeshDrawArrangeParallelism, WorldMeshDrawCollectParallelism, WorldMeshDrawCollection,
        WorldMeshDrawPlan,
    };

    use super::super::view_plan::{FrameViewPlan, FrameViewPlanParams, FrameViewPlanTarget};
    use super::cull::{build_cull_snapshot_for_view, cull_projection_for_write_target};
    use super::queue::{
        desktop_overlay_view_inputs, select_inner_parallelism_for_prepared_work_with_policy,
        shadow_caster_view_inputs, should_parallelize_view_collection_with_policy,
    };
    use super::sort::{
        OverlayTraceStats, overlay_trace_stats, select_arrange_parallelism_for_draws_with_policy,
    };
    use super::visible_deform::visible_mesh_deform_keys_from_draw_plans;
    use super::*;

    fn main_swapchain_plan() -> FrameViewPlan<'static> {
        FrameViewPlan::new(
            &HostCameraFrame::default(),
            FrameViewPlanParams {
                render_context: RenderingContext::UserView,
                frame_time_seconds: 0.0,
                view_id: ViewId::Main,
                viewport_px: (640, 480),
                clear: FrameViewClear::default(),
                profile: RenderPathProfile::desktop_main(),
                target: FrameViewPlanTarget::Swapchain,
            },
        )
    }

    #[test]
    fn suppressed_occlusion_still_builds_frustum_cull_snapshot() {
        let scene = SceneCoordinator::new();
        let occlusion = OcclusionSystem::new();
        let mut plan = main_swapchain_plan();
        plan.host_camera.suppress_occlusion_temporal = true;

        let snapshot =
            build_cull_snapshot_for_view(&scene, &occlusion, &plan).expect("frustum cull snapshot");

        assert!(snapshot.hi_z.is_none());
        assert!(snapshot.hi_z_temporal.is_none());
        assert!(snapshot.proj.vr_stereo.is_none());
    }

    fn asymmetric_cull_projection_bundle() -> WorldMeshCullProjParams {
        WorldMeshCullProjParams {
            world_proj: Mat4::from_cols_array(&[
                1.0, 0.25, 0.0, 0.0, //
                0.5, 2.0, 0.0, 0.0, //
                0.0, 0.0, 3.0, 0.75, //
                0.0, 0.0, 1.0, 1.0,
            ]),
            overlay_proj: Mat4::from_cols_array(&[
                1.5, 0.125, 0.0, 0.0, //
                0.75, 1.25, 0.0, 0.0, //
                0.0, 0.0, 2.5, 0.5, //
                0.0, 0.0, 1.0, 1.0,
            ]),
            vr_stereo: Some((
                Mat4::from_cols_array(&[
                    1.0, 0.0, 0.0, 0.0, //
                    0.1, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, 1.0,
                ]),
                Mat4::from_cols_array(&[
                    1.0, 0.0, 0.0, 0.0, //
                    -0.1, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, 1.0,
                ]),
            )),
        }
    }

    #[test]
    fn primary_cull_projection_preserves_camera_convention() {
        let raw = asymmetric_cull_projection_bundle();
        let adjusted = cull_projection_for_write_target(&raw, OffscreenWriteTarget::None);

        assert_eq!(adjusted.world_proj, raw.world_proj);
        assert_eq!(adjusted.overlay_proj, raw.overlay_proj);
        assert_eq!(adjusted.vr_stereo, raw.vr_stereo);
    }

    #[test]
    fn host_render_texture_cull_projection_uses_offscreen_convention() {
        let raw = asymmetric_cull_projection_bundle();
        let write_target = OffscreenWriteTarget::host_render_texture(77);
        let adjusted = cull_projection_for_write_target(&raw, write_target);
        let (left, right) = raw.vr_stereo.expect("stereo pair");

        assert_eq!(
            adjusted.world_proj,
            write_target.render_projection(raw.world_proj)
        );
        assert_eq!(
            adjusted.overlay_proj,
            write_target.render_projection(raw.overlay_proj)
        );
        assert_eq!(
            adjusted.vr_stereo,
            Some((
                write_target.render_projection(left),
                write_target.render_projection(right)
            ))
        );
    }

    #[test]
    fn untracked_offscreen_cull_projection_uses_offscreen_convention() {
        let raw = asymmetric_cull_projection_bundle();
        let write_target = OffscreenWriteTarget::Untracked;
        let adjusted = cull_projection_for_write_target(&raw, write_target);
        let (left, right) = raw.vr_stereo.expect("stereo pair");

        assert_eq!(
            adjusted.world_proj,
            write_target.render_projection(raw.world_proj)
        );
        assert_eq!(
            adjusted.overlay_proj,
            write_target.render_projection(raw.overlay_proj)
        );
        assert_eq!(
            adjusted.vr_stereo,
            Some((
                write_target.render_projection(left),
                write_target.render_projection(right)
            ))
        );
    }

    #[test]
    fn desktop_overlay_view_inputs_do_not_inherit_main_camera_filters_or_culling() {
        let mut plan = main_swapchain_plan();
        plan.draw_filter = Some(CameraTransformDrawFilter {
            only: Some(HashSet::from_iter([42])),
            exclude: HashSet::from_iter([7]),
        });
        plan.render_space_filter = Some(RenderSpaceId(99));

        let inputs = desktop_overlay_view_inputs(&plan, 1.25);

        assert_eq!(inputs.render_context, RenderingContext::UserView);
        assert_eq!(
            inputs.head_output_transform,
            plan.host_camera.head_output_transform
        );
        assert_eq!(inputs.view_origin_world, plan.view_origin_world());
        assert!(inputs.culling.is_none());
        assert!(inputs.transform_filter.is_none());
        assert!(inputs.render_space_filter.is_none());
        assert!(inputs.reflection_probes.is_none());
        assert_eq!(inputs.mesh_lod_bias, 1.25);
        assert_eq!(inputs.layer_policy, ViewLayerPolicy::DesktopOverlay);
    }

    #[test]
    fn shadow_caster_view_inputs_preserve_view_scope_without_camera_culling() {
        let mut plan = main_swapchain_plan();
        plan.draw_filter = Some(CameraTransformDrawFilter {
            only: Some(HashSet::from_iter([42])),
            exclude: HashSet::from_iter([7]),
        });
        plan.render_space_filter = Some(RenderSpaceId(99));
        plan.layer_policy = ViewLayerPolicy::MainView;

        let inputs = shadow_caster_view_inputs(&plan, 1.25);

        assert_eq!(inputs.render_context, RenderingContext::UserView);
        assert_eq!(
            inputs.head_output_transform,
            plan.host_camera.head_output_transform
        );
        assert_eq!(inputs.view_origin_world, plan.view_origin_world());
        assert!(inputs.culling.is_none());
        assert!(inputs.transform_filter.is_some());
        assert_eq!(inputs.render_space_filter, Some(RenderSpaceId(99)));
        assert!(inputs.reflection_probes.is_none());
        assert_eq!(inputs.mesh_lod_bias, 1.25);
        assert_eq!(inputs.layer_policy, ViewLayerPolicy::MainView);
    }

    #[test]
    fn overlay_trace_stats_reports_plan_presence_and_counters() {
        assert_eq!(overlay_trace_stats(None), OverlayTraceStats::default());

        let empty = WorldMeshDrawPlan::Empty;
        assert_eq!(
            overlay_trace_stats(Some(&empty)),
            OverlayTraceStats {
                plan_present: true,
                ..Default::default()
            }
        );

        let draw = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 10,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        let prefetched =
            WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
                WorldMeshDrawCollection {
                    items: vec![draw],
                    draws_pre_cull: 5,
                    draws_culled: 2,
                    draws_hi_z_culled: 1,
                    visibility: Default::default(),
                    arrangement: Default::default(),
                },
                None,
            )));

        assert_eq!(
            overlay_trace_stats(Some(&prefetched)),
            OverlayTraceStats {
                plan_present: true,
                draws: 1,
                draws_pre_cull: 5,
                draws_culled: 2,
                draws_hi_z_culled: 1,
            }
        );
    }

    #[test]
    fn select_inner_parallelism_uses_full_for_zero_or_one_view() {
        assert_eq!(
            select_inner_parallelism(&[]),
            WorldMeshDrawCollectParallelism::Full
        );
        assert_eq!(
            select_inner_parallelism(&[main_swapchain_plan()]),
            WorldMeshDrawCollectParallelism::Full
        );
    }

    #[test]
    fn select_inner_parallelism_disables_nested_parallelism_for_multiple_views() {
        assert_eq!(
            select_inner_parallelism(&[main_swapchain_plan(), main_swapchain_plan()]),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        );
    }

    #[test]
    fn prepared_work_selector_reenables_inner_parallelism_for_large_two_view_frames() {
        let policy = FrameParallelPolicy::new(4);
        let draws_per_view = policy.draw_heavy_threshold() / 2;
        assert_eq!(
            select_inner_parallelism_for_prepared_work_with_policy(
                policy,
                2,
                draws_per_view,
                WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch,
            ),
            WorldMeshDrawCollectParallelism::Full
        );
        assert_eq!(
            select_inner_parallelism_for_prepared_work_with_policy(
                policy,
                2,
                draws_per_view - 1,
                WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch,
            ),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        );
    }

    #[test]
    fn prepared_work_selector_keeps_three_view_frames_nested_serial() {
        let policy = FrameParallelPolicy::new(4);
        assert_eq!(
            select_inner_parallelism_for_prepared_work_with_policy(
                policy,
                3,
                policy.draw_heavy_threshold(),
                WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch,
            ),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        );
    }

    #[test]
    fn arrange_parallelism_uses_draw_heavy_threshold_independent_of_collection() {
        let policy = FrameParallelPolicy::new(4);

        assert_eq!(
            select_arrange_parallelism_for_draws_with_policy(
                policy,
                policy.draw_heavy_threshold() - 1,
            ),
            WorldMeshDrawArrangeParallelism::Serial
        );
        assert_eq!(
            select_arrange_parallelism_for_draws_with_policy(policy, policy.draw_heavy_threshold()),
            WorldMeshDrawArrangeParallelism::Full
        );
    }

    #[test]
    fn view_collection_parallelism_requires_multiple_views_and_enough_work() {
        let policy = FrameParallelPolicy::new(4);
        let draws_per_view = policy.draw_heavy_threshold() / 2;
        assert!(!should_parallelize_view_collection_with_policy(
            policy,
            1,
            policy.draw_heavy_threshold()
        ));
        assert!(!should_parallelize_view_collection_with_policy(
            policy,
            2,
            draws_per_view - 1
        ));
        assert!(should_parallelize_view_collection_with_policy(
            policy,
            2,
            draws_per_view
        ));
    }

    #[test]
    fn visible_deform_keys_include_only_visible_deformed_draws() {
        let mut rigid = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 10,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        rigid.world_space_deformed = false;
        rigid.blendshape_deformed = false;

        let mut blend = rigid.clone();
        blend.node_id = 4;
        blend.renderable_index = 4;
        blend.instance_id = crate::scene::MeshRendererInstanceId(5);
        blend.blendshape_deformed = true;

        let mut skinned = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 2,
            property_block: None,
            skinned: true,
            sorting_order: 0,
            mesh_asset_id: 11,
            node_id: 8,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: false,
        });
        skinned.world_space_deformed = true;

        let mut overlay = rigid.clone();
        overlay.node_id = 9;
        overlay.renderable_index = 9;
        overlay.instance_id = crate::scene::MeshRendererInstanceId(9);
        overlay.blendshape_deformed = true;

        let plans = [ViewWorldMeshDrawPlans {
            world: WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
                WorldMeshDrawCollection {
                    items: vec![rigid, blend.clone(), skinned.clone()],
                    draws_pre_cull: 3,
                    draws_culled: 0,
                    draws_hi_z_culled: 0,
                    visibility: Default::default(),
                    arrangement: Default::default(),
                },
                None,
            ))),
            shadow_casters: WorldMeshDrawPlan::Empty,
            desktop_overlay: Some(WorldMeshDrawPlan::Prefetched(Box::new(
                PrefetchedWorldMeshViewDraws::new(
                    WorldMeshDrawCollection {
                        items: vec![overlay.clone()],
                        draws_pre_cull: 1,
                        draws_culled: 0,
                        draws_hi_z_culled: 0,
                        visibility: Default::default(),
                        arrangement: Default::default(),
                    },
                    None,
                ),
            ))),
        }];

        let keys = visible_mesh_deform_keys_from_draw_plans(&plans);

        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&SkinCacheKey::new(
            blend.space_id,
            SkinCacheRendererKind::Static,
            blend.instance_id,
        )));
        assert!(keys.contains(&SkinCacheKey::new(
            skinned.space_id,
            SkinCacheRendererKind::Skinned,
            skinned.instance_id,
        )));
        assert!(keys.contains(&SkinCacheKey::new(
            overlay.space_id,
            SkinCacheRendererKind::Static,
            overlay.instance_id,
        )));
    }
}
