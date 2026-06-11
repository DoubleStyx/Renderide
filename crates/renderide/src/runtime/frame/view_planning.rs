//! Per-tick view collection on [`super::RendererRuntime`].
//!
//! Builds the ordered list of [`FrameViewPlan`]s that drive draw collection and graph
//! execution: HMD stereo multiview (when present), then enabled secondary render-texture
//! cameras sorted by camera depth, then the main desktop view (when included). Logic
//! sits between the render entry point in [`super::render`] and the per-view extraction
//! pipeline in [`super::extract`].

mod portal_plan;

use std::sync::{Arc, LazyLock};

use crate::camera::{
    CameraPortalMode, CameraPortalSourceView, CameraPortalSurface, CameraRenderRect, EyeView,
    ViewId, Viewport, WorldProjectionSet, camera_state_double_buffered, camera_state_enabled,
    camera_state_post_processing, camera_state_render_private_ui, camera_state_render_shadows,
    host_camera_frame_for_camera_portal, host_camera_frame_for_render_texture,
    view_matrix_for_world_mesh_render_space,
};
use crate::diagnostics::log_once::KeyedLogOnce;
use crate::gpu::GpuContext;
use crate::graph_inputs::RenderTextureSelfSampling;
use crate::render_graph::{
    FrameGlobalView, FrameViewClear, OffscreenWriteTarget, RenderPathProfile, ViewPostProcessing,
    ViewWinding,
};
use crate::scene::RenderSpaceId;
use crate::scene::{
    camera_portal_disable_per_pixel_lights, camera_portal_disable_shadows,
    camera_portal_has_camera_clear_mode, camera_portal_has_far_clip_value,
    camera_portal_portal_mode,
};
use crate::shared::{CameraClearMode, RenderingContext};
use crate::world_mesh::{ViewLayerPolicy, ViewRenderSpaceScope, draw_filter_from_camera_entry};

use super::super::RendererRuntime;
use super::render::PrimaryViewRequest;
use super::view_plan::{
    FrameViewPlan, FrameViewPlanParams, FrameViewPlanTarget, OffscreenColorCopy,
    OffscreenTargetHandles, ViewFamilyPlan,
};
#[cfg(test)]
use portal_plan::camera_portal_stereo_render_rects;
use portal_plan::{CameraPortalSourceViewPlan, CameraPortalSourceViewPlans, CameraPortalViewTask};

/// Once-only diagnostic gate for secondary render textures without a depth texture.
static SECONDARY_RT_MISSING_DEPTH_LOG: LazyLock<KeyedLogOnce<i32>> =
    LazyLock::new(KeyedLogOnce::new);

/// Once-only diagnostic gate for secondary render textures without a depth view.
static SECONDARY_RT_MISSING_DEPTH_VIEW_LOG: LazyLock<KeyedLogOnce<i32>> =
    LazyLock::new(KeyedLogOnce::new);

/// Once-only diagnostic gate for camera portals without a depth texture.
static CAMERA_PORTAL_RT_MISSING_DEPTH_LOG: LazyLock<KeyedLogOnce<i32>> =
    LazyLock::new(KeyedLogOnce::new);

/// Once-only diagnostic gate for camera portals without a depth view.
static CAMERA_PORTAL_RT_MISSING_DEPTH_VIEW_LOG: LazyLock<KeyedLogOnce<i32>> =
    LazyLock::new(KeyedLogOnce::new);

/// Once-only diagnostic gate for camera portal per-pixel-light disables.
static CAMERA_PORTAL_DISABLE_PIXEL_LIGHTS_LOG: LazyLock<KeyedLogOnce<i32>> =
    LazyLock::new(KeyedLogOnce::new);

/// Returns the stable logical identity for one secondary camera view.
pub(in crate::runtime) fn secondary_camera_view_id(
    render_space_id: RenderSpaceId,
    renderable_index: i32,
    camera_index: usize,
) -> ViewId {
    ViewId::secondary_camera(
        render_space_id,
        if renderable_index >= 0 {
            renderable_index
        } else {
            camera_index as i32
        },
    )
}

/// Returns the stable logical identity for one camera-portal view.
pub(in crate::runtime) fn camera_portal_view_id(
    render_space_id: RenderSpaceId,
    renderable_index: i32,
    portal_index: usize,
) -> ViewId {
    ViewId::camera_portal(
        render_space_id,
        camera_portal_identity_index(renderable_index, portal_index),
    )
}

fn camera_portal_eye_view_id(
    render_space_id: RenderSpaceId,
    renderable_index: i32,
    portal_index: usize,
    eye_index: u8,
) -> ViewId {
    ViewId::camera_portal_eye(
        render_space_id,
        camera_portal_identity_index(renderable_index, portal_index),
        eye_index,
    )
}

fn camera_portal_identity_index(renderable_index: i32, portal_index: usize) -> i32 {
    if renderable_index >= 0 {
        renderable_index
    } else {
        portal_index as i32
    }
}

fn sort_secondary_view_tasks(tasks: &mut [(RenderSpaceId, f32, usize)]) {
    tasks.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.2.cmp(&b.2))
    });
}

fn sort_camera_portal_view_tasks(tasks: &mut [(RenderSpaceId, usize)]) {
    tasks.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
}

fn secondary_camera_render_context() -> RenderingContext {
    RenderingContext::Camera
}

fn camera_portal_render_context(mode: CameraPortalMode) -> RenderingContext {
    match mode {
        CameraPortalMode::Mirror => RenderingContext::Mirror,
        CameraPortalMode::Portal => RenderingContext::Portal,
    }
}

fn secondary_camera_write_target(rt_id: i32, flags: u16) -> OffscreenWriteTarget {
    if camera_state_double_buffered(flags) && !camera_state_post_processing(flags) {
        OffscreenWriteTarget::host_render_texture_with_self_sampling(
            rt_id,
            RenderTextureSelfSampling::AllowPreviousContents,
        )
    } else {
        OffscreenWriteTarget::host_render_texture(rt_id)
    }
}

fn secondary_camera_layer_policy(flags: u16) -> ViewLayerPolicy {
    ViewLayerPolicy::camera(camera_state_render_private_ui(flags))
}

/// Returns the render-space scope used by a camera-like view with optional selective roots.
pub(in crate::runtime) fn camera_render_space_scope(
    has_selective_roots: bool,
    render_space_id: RenderSpaceId,
) -> ViewRenderSpaceScope {
    if has_selective_roots {
        ViewRenderSpaceScope::single(render_space_id)
    } else {
        ViewRenderSpaceScope::AllActive
    }
}

fn secondary_camera_shadows_enabled(flags: u16) -> bool {
    camera_state_render_shadows(flags)
}

fn camera_portal_write_target(rt_id: i32) -> OffscreenWriteTarget {
    OffscreenWriteTarget::host_render_texture(rt_id)
}

fn camera_portal_clear(flags: i32, mode: CameraClearMode) -> FrameViewClear {
    if camera_portal_has_camera_clear_mode(flags) {
        FrameViewClear {
            mode,
            color: glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    } else {
        FrameViewClear::skybox()
    }
}

/// Logs a missing secondary render-texture depth attachment once per render texture id.
fn log_secondary_rt_missing_depth(rt_id: i32, sid: RenderSpaceId, cam_idx: usize) {
    if SECONDARY_RT_MISSING_DEPTH_LOG.should_log(rt_id) {
        logger::warn!(
            "secondary camera: render texture {rt_id} missing depth; space={sid:?} camera_index={cam_idx}"
        );
    }
}

/// Logs a missing secondary render-texture depth view once per render texture id.
fn log_secondary_rt_missing_depth_view(rt_id: i32, sid: RenderSpaceId, cam_idx: usize) {
    if SECONDARY_RT_MISSING_DEPTH_VIEW_LOG.should_log(rt_id) {
        logger::warn!(
            "secondary camera: render texture {rt_id} missing depth view; space={sid:?} camera_index={cam_idx}"
        );
    }
}

/// Logs a missing camera-portal render-texture depth attachment once per render texture id.
fn log_camera_portal_rt_missing_depth(rt_id: i32, sid: RenderSpaceId, portal_idx: usize) {
    if CAMERA_PORTAL_RT_MISSING_DEPTH_LOG.should_log(rt_id) {
        logger::warn!(
            "camera portal: render texture {rt_id} missing depth; space={sid:?} portal_index={portal_idx}"
        );
    }
}

/// Logs a missing camera-portal render-texture depth view once per render texture id.
fn log_camera_portal_rt_missing_depth_view(rt_id: i32, sid: RenderSpaceId, portal_idx: usize) {
    if CAMERA_PORTAL_RT_MISSING_DEPTH_VIEW_LOG.should_log(rt_id) {
        logger::warn!(
            "camera portal: render texture {rt_id} missing depth view; space={sid:?} portal_index={portal_idx}"
        );
    }
}

fn log_camera_portal_disable_per_pixel_lights(rt_id: i32) {
    if CAMERA_PORTAL_DISABLE_PIXEL_LIGHTS_LOG.should_log(rt_id) {
        logger::debug!(
            "camera portal: disablePerPixelLights requested for render texture {rt_id}; clustered lights do not expose a per-pixel light-count switch"
        );
    }
}

/// Resident host render texture handles needed to plan one secondary camera view.
#[derive(Clone)]
struct ResidentSecondaryRenderTexture {
    /// Host render texture color storage.
    color_texture: Arc<wgpu::Texture>,
    /// Full host render texture color view.
    color_view: Arc<wgpu::TextureView>,
    /// Full host render texture depth storage.
    depth_texture: Arc<wgpu::Texture>,
    /// Full host render texture depth view.
    depth_view: Arc<wgpu::TextureView>,
    /// Full host render texture extent in pixels.
    extent_px: (u32, u32),
    /// Host render texture color format.
    color_format: wgpu::TextureFormat,
    /// Host render texture depth format.
    depth_format: wgpu::TextureFormat,
}

/// Builds graph target handles for a resolved camera render rect.
fn secondary_rt_handles_for_rect(
    backend: &mut crate::backend::RenderBackend,
    gpu: &GpuContext,
    write_target: OffscreenWriteTarget,
    rt: ResidentSecondaryRenderTexture,
    render_rect: CameraRenderRect,
) -> Option<OffscreenTargetHandles> {
    if render_rect.is_full_target(rt.extent_px) {
        return Some(OffscreenTargetHandles::new(
            write_target,
            rt.color_texture.as_ref().clone(),
            rt.color_view.as_ref().clone(),
            rt.depth_texture.as_ref().clone(),
            rt.depth_view.as_ref().clone(),
            rt.extent_px,
            rt.color_format,
        ));
    }

    let scratch = backend.secondary_render_rect_scratch(
        gpu.device().as_ref(),
        render_rect.extent_px,
        rt.color_format,
        rt.depth_format,
    )?;
    Some(
        OffscreenTargetHandles::new(
            write_target,
            scratch.color_texture.as_ref().clone(),
            scratch.color_view.as_ref().clone(),
            scratch.depth_texture.as_ref().clone(),
            scratch.depth_view.as_ref().clone(),
            render_rect.extent_px,
            rt.color_format,
        )
        .with_copy_to_color(OffscreenColorCopy {
            destination_texture: rt.color_texture.as_ref().clone(),
            destination_origin_px: render_rect.origin_px,
            extent_px: render_rect.extent_px,
        }),
    )
}

impl RendererRuntime {
    /// Snapshots the GPU handles for a resident secondary render texture.
    fn resident_secondary_render_texture(
        &self,
        rt_id: i32,
        sid: RenderSpaceId,
        cam_idx: usize,
    ) -> Option<ResidentSecondaryRenderTexture> {
        let Some(rt) = self.backend.render_texture_pool().get(rt_id) else {
            logger::trace!("secondary camera: render texture asset {rt_id} not resident; skipping");
            return None;
        };
        let Some(depth_texture) = rt.depth_texture.clone() else {
            log_secondary_rt_missing_depth(rt_id, sid, cam_idx);
            return None;
        };
        let Some(depth_view) = rt.depth_view.clone() else {
            log_secondary_rt_missing_depth_view(rt_id, sid, cam_idx);
            return None;
        };
        Some(ResidentSecondaryRenderTexture {
            color_texture: rt.color_texture.clone(),
            color_view: rt.color_view.clone(),
            depth_format: depth_texture.format(),
            depth_texture,
            depth_view,
            extent_px: (rt.width, rt.height),
            color_format: rt.wgpu_color_format,
        })
    }

    /// Snapshots the GPU handles for a resident camera-portal render texture.
    fn resident_camera_portal_render_texture(
        &self,
        rt_id: i32,
        sid: RenderSpaceId,
        portal_idx: usize,
    ) -> Option<ResidentSecondaryRenderTexture> {
        let Some(rt) = self.backend.render_texture_pool().get(rt_id) else {
            logger::trace!("camera portal: render texture asset {rt_id} not resident; skipping");
            return None;
        };
        let Some(depth_texture) = rt.depth_texture.clone() else {
            log_camera_portal_rt_missing_depth(rt_id, sid, portal_idx);
            return None;
        };
        let Some(depth_view) = rt.depth_view.clone() else {
            log_camera_portal_rt_missing_depth_view(rt_id, sid, portal_idx);
            return None;
        };
        Some(ResidentSecondaryRenderTexture {
            color_texture: rt.color_texture.clone(),
            color_view: rt.color_view.clone(),
            depth_format: depth_texture.format(),
            depth_texture,
            depth_view,
            extent_px: (rt.width, rt.height),
            color_format: rt.wgpu_color_format,
        })
    }

    /// Collects every active view for this tick into a single ordered list.
    ///
    /// Ordering -- preserved so the mesh-deform skip flag on
    /// [`crate::backend::FrameResourceManager`] still runs deform exactly once per tick:
    /// 1. HMD stereo multiview (when requested as the primary view).
    /// 2. Secondary render-texture cameras, sorted by camera depth.
    /// 3. Main desktop swapchain (when requested as the primary view).
    pub(in crate::runtime) fn collect_prepared_views<'a>(
        &mut self,
        gpu: &GpuContext,
        primary: PrimaryViewRequest<'a>,
        main_extent_px: (u32, u32),
        main_profile: RenderPathProfile,
        fallback_frame_global_profile: RenderPathProfile,
        main_offscreen_target: Option<OffscreenTargetHandles>,
    ) -> ViewFamilyPlan<'a> {
        let secondary_views = self.collect_offscreen_scene_views(gpu, main_extent_px);
        self.assemble_prepared_views(
            primary,
            main_extent_px,
            main_profile,
            fallback_frame_global_profile,
            main_offscreen_target,
            secondary_views,
        )
    }

    /// Collects active views without GPU-backed secondary render targets for CPU-only tests.
    #[cfg(test)]
    pub(in crate::runtime) fn collect_prepared_views_without_secondaries<'a>(
        &self,
        primary: PrimaryViewRequest<'a>,
        main_extent_px: (u32, u32),
        main_profile: RenderPathProfile,
        fallback_frame_global_profile: RenderPathProfile,
    ) -> ViewFamilyPlan<'a> {
        self.assemble_prepared_views(
            primary,
            main_extent_px,
            main_profile,
            fallback_frame_global_profile,
            None,
            Vec::new(),
        )
    }

    /// Appends HMD, pre-collected offscreen, and main desktop views in submission order.
    fn assemble_prepared_views<'a>(
        &self,
        primary: PrimaryViewRequest<'a>,
        main_extent_px: (u32, u32),
        main_profile: RenderPathProfile,
        fallback_frame_global_profile: RenderPathProfile,
        main_offscreen_target: Option<OffscreenTargetHandles>,
        mut secondary_views: Vec<FrameViewPlan<'a>>,
    ) -> ViewFamilyPlan<'a> {
        let (includes_main, hmd_target) = match primary {
            PrimaryViewRequest::DesktopMain => (true, None),
            PrimaryViewRequest::HmdExternalMultiview(ext) => (false, Some(ext)),
            PrimaryViewRequest::None => (false, None),
        };

        let est_capacity =
            usize::from(hmd_target.is_some()) + secondary_views.len() + usize::from(includes_main);
        let mut views: Vec<FrameViewPlan<'a>> = Vec::with_capacity(est_capacity);
        let main_render_context = self.scene.active_main_render_context();
        let mut frame_global =
            self.frame_global_from_runtime(main_render_context, fallback_frame_global_profile);

        if let Some(ext) = hmd_target {
            let extent_px = ext.extent_px;
            let hmd_view = FrameViewPlan::new(
                &self.host_camera,
                FrameViewPlanParams {
                    render_context: main_render_context,
                    frame_time_seconds: self.tick_state.frame_time_seconds(),
                    view_id: ViewId::Main,
                    viewport_px: extent_px,
                    clear: FrameViewClear::skybox(),
                    profile: RenderPathProfile::xr_hmd(),
                    target: FrameViewPlanTarget::ExternalMultiview(ext),
                },
            );
            frame_global = hmd_view.frame_global_view();
            views.push(hmd_view);
        }

        views.append(&mut secondary_views);

        if includes_main {
            let main_view = self.build_main_view_with_profile(
                main_extent_px,
                main_profile,
                main_offscreen_target,
            );
            frame_global = main_view.frame_global_view();
            views.push(main_view);
        }

        ViewFamilyPlan::new(&frame_global, views)
    }

    /// Builds fallback primary-view metadata for frame-global passes when no HMD or main view is
    /// submitted.
    fn frame_global_from_runtime(
        &self,
        render_context: RenderingContext,
        profile: RenderPathProfile,
    ) -> FrameGlobalView {
        FrameGlobalView::new(
            &self.host_camera,
            render_context,
            self.tick_state.frame_time_seconds(),
            FrameViewClear::skybox(),
            profile.post_processing(),
        )
    }

    /// Builds prepared views for every enabled secondary render-texture camera in the scene,
    /// skipping cameras whose host render texture is not yet resident on the GPU.
    ///
    /// Reuses [`RendererRuntime::secondary_view_tasks_scratch`] for the depth-sort scratch buffer
    /// so a frame with secondary cameras does not allocate a fresh `Vec` for the sort each tick.
    fn collect_secondary_rt_views<'a>(&mut self, gpu: &GpuContext) -> Vec<FrameViewPlan<'a>> {
        let mut tasks = std::mem::take(&mut self.tick_state.secondary_view_tasks_scratch);
        tasks.clear();
        let result = self.collect_secondary_rt_views_using(gpu, &mut tasks);
        self.tick_state.secondary_view_tasks_scratch = tasks;
        result
    }

    /// Collects camera portals followed by secondary cameras into the offscreen view prefix.
    fn collect_offscreen_scene_views<'a>(
        &mut self,
        gpu: &GpuContext,
        main_extent_px: (u32, u32),
    ) -> Vec<FrameViewPlan<'a>> {
        let mut portal_views = self.collect_camera_portal_views(gpu, main_extent_px);
        let mut secondary_views = self.collect_secondary_rt_views(gpu);
        portal_views.append(&mut secondary_views);
        portal_views
    }

    /// Collects active camera portals that render into resident host render textures.
    fn collect_camera_portal_views<'a>(
        &mut self,
        gpu: &GpuContext,
        main_extent_px: (u32, u32),
    ) -> Vec<FrameViewPlan<'a>> {
        let mut tasks = std::mem::take(&mut self.tick_state.camera_portal_view_tasks_scratch);
        tasks.clear();
        let result = self.collect_camera_portal_views_using(gpu, main_extent_px, &mut tasks);
        self.tick_state.camera_portal_view_tasks_scratch = tasks;
        result
    }

    /// Inner helper that consumes the supplied camera-portal scratch buffer.
    fn collect_camera_portal_views_using<'a>(
        &mut self,
        gpu: &GpuContext,
        main_extent_px: (u32, u32),
        tasks: &mut Vec<(RenderSpaceId, usize)>,
    ) -> Vec<FrameViewPlan<'a>> {
        for sid in self.scene.render_space_ids() {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            if !space.is_active() {
                continue;
            }
            for (idx, portal) in space.camera_portals().iter().enumerate() {
                if portal.state.render_texture_id >= 0 {
                    tasks.push((sid, idx));
                }
            }
        }
        sort_camera_portal_view_tasks(tasks);

        let mut views: Vec<FrameViewPlan<'a>> = Vec::with_capacity(tasks.len().saturating_mul(2));
        for (sid, portal_idx) in tasks.drain(..) {
            self.append_camera_portal_views_for_task(
                gpu,
                main_extent_px,
                sid,
                portal_idx,
                &mut views,
            );
        }
        views
    }

    fn append_camera_portal_views_for_task<'a>(
        &mut self,
        gpu: &GpuContext,
        main_extent_px: (u32, u32),
        sid: RenderSpaceId,
        portal_idx: usize,
        views: &mut Vec<FrameViewPlan<'a>>,
    ) {
        let Some(task) = self.camera_portal_view_task(sid, portal_idx) else {
            return;
        };
        let Some(rt) = self.resident_camera_portal_render_texture(
            task.render_texture_id,
            task.render_space_id,
            task.portal_index,
        ) else {
            return;
        };
        let Some(source_plans) = self.camera_portal_source_view_plans(main_extent_px, rt.extent_px)
        else {
            logger::trace!(
                "camera portal: render texture {} has zero extent; space={:?} portal_index={}",
                task.render_texture_id,
                task.render_space_id,
                task.portal_index
            );
            return;
        };
        if camera_portal_disable_per_pixel_lights(task.state.flags) {
            log_camera_portal_disable_per_pixel_lights(task.render_texture_id);
        }
        for source_plan in source_plans.iter() {
            if let Some(plan) =
                self.camera_portal_frame_view_plan(gpu, task, rt.clone(), source_plan)
            {
                views.push(plan);
            }
        }
    }

    fn camera_portal_view_task(
        &self,
        sid: RenderSpaceId,
        portal_idx: usize,
    ) -> Option<CameraPortalViewTask> {
        let space = self.scene.space(sid)?;
        let entry = space.camera_portals().get(portal_idx)?;
        let state = entry.state;
        let mode = if camera_portal_portal_mode(state.flags) {
            CameraPortalMode::Portal
        } else {
            CameraPortalMode::Mirror
        };
        let render_context = camera_portal_render_context(mode);
        let Some(surface_world_matrix) =
            self.camera_portal_surface_world_matrix(sid, state.mesh_renderer_index, render_context)
        else {
            logger::trace!(
                "camera portal: invalid mesh renderer index {}; space={sid:?} portal_index={portal_idx}",
                state.mesh_renderer_index
            );
            return None;
        };
        Some(CameraPortalViewTask {
            render_space_id: sid,
            portal_index: portal_idx,
            state,
            renderable_index: entry.renderable_index,
            render_texture_id: state.render_texture_id,
            mode,
            render_context,
            surface_world_matrix,
        })
    }

    fn camera_portal_frame_view_plan<'a>(
        &mut self,
        gpu: &GpuContext,
        task: CameraPortalViewTask,
        rt: ResidentSecondaryRenderTexture,
        source_plan: CameraPortalSourceViewPlan,
    ) -> Option<FrameViewPlan<'a>> {
        let Some(hc) = host_camera_frame_for_camera_portal(
            &self.host_camera,
            &task.state,
            source_plan.source,
            CameraPortalSurface::new(task.surface_world_matrix),
            task.mode,
            camera_portal_has_far_clip_value(task.state.flags),
        ) else {
            logger::trace!(
                "camera portal: invalid camera matrices; render texture {} space={:?} portal_index={} eye_index={}",
                task.render_texture_id,
                task.render_space_id,
                task.portal_index,
                source_plan.eye_index
            );
            return None;
        };
        let Some(target_handles) = secondary_rt_handles_for_rect(
            &mut self.backend,
            gpu,
            camera_portal_write_target(task.render_texture_id),
            rt,
            source_plan.render_rect,
        ) else {
            logger::trace!(
                "camera portal: render texture {} could not allocate split target; space={:?} portal_index={} eye_index={}",
                task.render_texture_id,
                task.render_space_id,
                task.portal_index,
                source_plan.eye_index
            );
            return None;
        };
        let view_id = if source_plan.eye_index == 0 {
            camera_portal_view_id(
                task.render_space_id,
                task.renderable_index,
                task.portal_index,
            )
        } else {
            camera_portal_eye_view_id(
                task.render_space_id,
                task.renderable_index,
                task.portal_index,
                source_plan.eye_index,
            )
        };
        let mut plan = FrameViewPlan::new(
            &hc,
            FrameViewPlanParams {
                render_context: task.render_context,
                frame_time_seconds: self.tick_state.frame_time_seconds(),
                view_id,
                viewport_px: source_plan.render_rect.extent_px,
                clear: camera_portal_clear(task.state.flags, task.state.override_clear_flag_value),
                profile: RenderPathProfile::secondary_camera(ViewPostProcessing::disabled()),
                target: FrameViewPlanTarget::offscreen(target_handles),
            },
        );
        if task.mode == CameraPortalMode::Mirror {
            plan.view_winding = ViewWinding::mirror_reflection();
        }
        plan.render_space_scope = ViewRenderSpaceScope::AllActive;
        plan.layer_policy = ViewLayerPolicy::camera(false);
        plan.render_shadows = !camera_portal_disable_shadows(task.state.flags);
        Some(plan)
    }

    fn camera_portal_source_view_plans(
        &self,
        main_extent_px: (u32, u32),
        target_extent_px: (u32, u32),
    ) -> Option<CameraPortalSourceViewPlans> {
        if let Some(stereo) = self.host_camera.active_stereo() {
            let left = CameraPortalSourceView::new(
                stereo.left,
                self.host_camera.clip,
                self.host_camera.projection_kind,
            );
            let right = CameraPortalSourceView::new(
                stereo.right,
                self.host_camera.clip,
                self.host_camera.projection_kind,
            );
            if let Some(plans) = CameraPortalSourceViewPlans::stereo(left, right, target_extent_px)
            {
                return Some(plans);
            }
        }
        CameraPortalSourceViewPlans::mono(
            self.camera_portal_mono_source_view(main_extent_px),
            target_extent_px,
        )
    }

    fn camera_portal_mono_source_view(&self, main_extent_px: (u32, u32)) -> CameraPortalSourceView {
        let projections =
            WorldProjectionSet::from_scene_host(&self.scene, main_extent_px, &self.host_camera);
        if let Some(explicit) = self.host_camera.explicit_view {
            return CameraPortalSourceView::new(
                explicit,
                projections.clip,
                self.host_camera.projection_kind,
            );
        }
        let view = self
            .scene
            .active_main_space()
            .map_or(glam::Mat4::IDENTITY, |space| {
                view_matrix_for_world_mesh_render_space(&self.scene, space)
            });
        let world_position = self.host_camera.view_origin_world();
        let eye = EyeView::new(
            view,
            projections.world_proj,
            projections.world_proj * view,
            world_position,
        );
        CameraPortalSourceView::symmetric_perspective(
            eye,
            projections.clip,
            Viewport::from_tuple(main_extent_px),
            self.host_camera.desktop_fov_degrees,
        )
    }

    fn camera_portal_surface_world_matrix(
        &self,
        sid: RenderSpaceId,
        mesh_renderer_index: i32,
        render_context: RenderingContext,
    ) -> Option<glam::Mat4> {
        if mesh_renderer_index < 0 {
            return None;
        }
        let space = self.scene.space(sid)?;
        let renderer = space
            .static_mesh_renderers()
            .get(mesh_renderer_index as usize)?;
        if renderer.node_id < 0 {
            return None;
        }
        self.scene.world_matrix_for_render_context(
            sid,
            renderer.node_id as usize,
            render_context,
            self.host_camera.head_output_transform,
        )
    }

    /// Inner helper that consumes the supplied scratch `tasks` buffer; split out so the outer
    /// caller can keep the scratch field reachable across the immutable borrow taken here.
    fn collect_secondary_rt_views_using<'a>(
        &mut self,
        gpu: &GpuContext,
        tasks: &mut Vec<(RenderSpaceId, f32, usize)>,
    ) -> Vec<FrameViewPlan<'a>> {
        for sid in self.scene.render_space_ids() {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            if !space.is_active() {
                continue;
            }
            for (idx, cam) in space.cameras().iter().enumerate() {
                if !camera_state_enabled(cam.state.flags) {
                    continue;
                }
                if cam.state.render_texture_asset_id < 0 {
                    continue;
                }
                tasks.push((sid, cam.state.depth, idx));
            }
        }
        sort_secondary_view_tasks(tasks);

        let mut views: Vec<FrameViewPlan<'a>> = Vec::with_capacity(tasks.len());
        for (sid, _, cam_idx) in tasks.drain(..) {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            let Some(entry) = space.cameras().get(cam_idx) else {
                continue;
            };
            if !camera_state_enabled(entry.state.flags) {
                continue;
            }
            let rt_id = entry.state.render_texture_asset_id;
            let Some(rt) = self.resident_secondary_render_texture(rt_id, sid, cam_idx) else {
                continue;
            };
            let Some(render_rect) = CameraRenderRect::resolve(entry.state.viewport, rt.extent_px)
            else {
                logger::trace!(
                    "secondary camera: render texture asset {rt_id} viewport {:?} resolved empty; skipping",
                    entry.state.viewport
                );
                continue;
            };
            let viewport_px = render_rect.extent_px;
            let write_target = secondary_camera_write_target(rt_id, entry.state.flags);
            let Some(rt_handles) = secondary_rt_handles_for_rect(
                &mut self.backend,
                gpu,
                write_target,
                rt,
                render_rect,
            ) else {
                logger::trace!(
                    "secondary camera: render texture asset {rt_id} viewport {:?} scratch unavailable; skipping",
                    entry.state.viewport
                );
                continue;
            };
            // Use the render-context world matrix (not the bare hierarchy matrix). For overlay
            // render spaces (userspace world: dash camera, interactive-camera mirrors, avatar
            // previews), this multiplies in `head_output_transform` so the camera follows the
            // user's head -- matching how `world_mesh` draws in the same space are positioned.
            // Without this the secondary camera sits at userspace-local coords while its
            // selective-render meshes track the user, so any movement away from origin shifts
            // the dash UI out of the camera's view (dash content drifts off the rendered RT).
            let Some(world_m) = self.scene.world_matrix_for_render_context(
                sid,
                entry.transform_id as usize,
                secondary_camera_render_context(),
                self.host_camera.head_output_transform,
            ) else {
                continue;
            };
            let mut hc = host_camera_frame_for_render_texture(
                &self.host_camera,
                &entry.state,
                viewport_px,
                world_m,
            );
            let filter = draw_filter_from_camera_entry(entry);
            let has_selective_roots = filter.has_selective_roots();
            // Selective secondary cameras (dashboards, in-world UI panels, mirrors on specific
            // subtrees) render tens of draws, not thousands. Hi-Z snapshots + occlusion temporal
            // cost a per-camera readback path with negligible payoff at that scale -- skip them.
            if has_selective_roots {
                hc.suppress_occlusion_temporal = true;
            }
            let post_processing = if space.is_overlay() && has_selective_roots {
                ViewPostProcessing::disabled()
            } else {
                ViewPostProcessing::from_camera_state(&entry.state)
            };
            let mut plan = FrameViewPlan::new(
                &hc,
                FrameViewPlanParams {
                    render_context: secondary_camera_render_context(),
                    frame_time_seconds: self.tick_state.frame_time_seconds(),
                    view_id: secondary_camera_view_id(sid, entry.renderable_index, cam_idx),
                    viewport_px,
                    clear: FrameViewClear::from_camera_state(&entry.state),
                    profile: RenderPathProfile::secondary_camera(post_processing),
                    target: FrameViewPlanTarget::offscreen(rt_handles),
                },
            );
            plan.draw_filter = Some(filter);
            plan.transform_filter_space = Some(sid);
            plan.render_space_scope = camera_render_space_scope(has_selective_roots, sid);
            plan.layer_policy = secondary_camera_layer_policy(entry.state.flags);
            plan.render_shadows = secondary_camera_shadows_enabled(entry.state.flags);
            views.push(plan);
        }
        views
    }

    /// Builds the main desktop/headless [`FrameViewPlan`] from the cached
    /// [`RendererRuntime::host_camera`].
    ///
    /// `main_extent_px` must match the current main-view target extent: it feeds
    /// [`crate::world_mesh::build_world_mesh_cull_proj_params`] on the pre-dispatch CPU cull
    /// path. A stale or zero extent produces a degenerate frustum and random scene-object
    /// culling.
    #[cfg(test)]
    pub(in crate::runtime) fn build_main_desktop_view<'a>(
        &self,
        main_extent_px: (u32, u32),
    ) -> FrameViewPlan<'a> {
        self.build_main_view_with_profile(main_extent_px, RenderPathProfile::desktop_main(), None)
    }

    fn build_main_view_with_profile<'a>(
        &self,
        main_extent_px: (u32, u32),
        profile: RenderPathProfile,
        offscreen_target: Option<OffscreenTargetHandles>,
    ) -> FrameViewPlan<'a> {
        let target = offscreen_target.map_or(
            FrameViewPlanTarget::Swapchain,
            FrameViewPlanTarget::offscreen,
        );
        FrameViewPlan::new(
            &self.host_camera,
            FrameViewPlanParams {
                render_context: self.scene.active_main_render_context(),
                frame_time_seconds: self.tick_state.frame_time_seconds(),
                view_id: ViewId::Main,
                viewport_px: main_extent_px,
                clear: FrameViewClear::skybox(),
                profile,
                target,
            },
        )
    }
}

#[cfg(test)]
mod tests;
