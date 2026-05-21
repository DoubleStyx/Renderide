//! Per-tick view collection on [`super::RendererRuntime`].
//!
//! Builds the ordered list of [`FrameViewPlan`]s that drive draw collection and graph
//! execution: HMD stereo multiview (when present), then enabled secondary render-texture
//! cameras sorted by camera depth, then the main desktop swapchain (when included). Logic
//! sits between the render entry point in [`super::render`] and the per-view extraction
//! pipeline in [`super::extract`].

use std::sync::{Arc, LazyLock};

use crate::camera::{
    CameraRenderRect, ViewId, camera_state_enabled, host_camera_frame_for_render_texture,
};
use crate::diagnostics::log_once::KeyedLogOnce;
use crate::gpu::GpuContext;
use crate::render_graph::{FrameViewClear, RenderPathProfile, ViewPostProcessing};
use crate::scene::RenderSpaceId;
use crate::shared::RenderingContext;
use crate::world_mesh::draw_filter_from_camera_entry;

use super::super::RendererRuntime;
use super::render::FrameRenderMode;
use super::view_plan::{
    FrameViewPlan, FrameViewPlanTarget, OffscreenRtColorCopy, OffscreenRtHandles, ViewFamilyPlan,
};

/// Once-only diagnostic gate for secondary render textures without a depth texture.
static SECONDARY_RT_MISSING_DEPTH_LOG: LazyLock<KeyedLogOnce<i32>> =
    LazyLock::new(KeyedLogOnce::new);

/// Once-only diagnostic gate for secondary render textures without a depth view.
static SECONDARY_RT_MISSING_DEPTH_VIEW_LOG: LazyLock<KeyedLogOnce<i32>> =
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

fn sort_secondary_view_tasks(tasks: &mut [(RenderSpaceId, f32, usize)]) {
    tasks.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.2.cmp(&b.2))
    });
}

fn secondary_camera_render_context() -> RenderingContext {
    RenderingContext::Camera
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

/// Resident host render texture handles needed to plan one secondary camera view.
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
    rt_id: i32,
    rt: ResidentSecondaryRenderTexture,
    render_rect: CameraRenderRect,
) -> Option<OffscreenRtHandles> {
    if render_rect.is_full_target(rt.extent_px) {
        return Some(OffscreenRtHandles {
            rt_id,
            color_texture: rt.color_texture,
            color_view: rt.color_view,
            depth_texture: rt.depth_texture,
            depth_view: rt.depth_view,
            color_format: rt.color_format,
            copy_to_color: None,
        });
    }

    let scratch = backend.secondary_render_rect_scratch(
        gpu.device().as_ref(),
        render_rect.extent_px,
        rt.color_format,
        rt.depth_format,
    )?;
    Some(OffscreenRtHandles {
        rt_id,
        color_texture: scratch.color_texture,
        color_view: scratch.color_view,
        depth_texture: scratch.depth_texture,
        depth_view: scratch.depth_view,
        color_format: rt.color_format,
        copy_to_color: Some(OffscreenRtColorCopy {
            destination_texture: rt.color_texture,
            destination_origin_px: render_rect.origin_px,
            extent_px: render_rect.extent_px,
        }),
    })
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

    /// Collects every active view for this tick into a single ordered list.
    ///
    /// Ordering -- preserved so the mesh-deform skip flag on
    /// [`crate::backend::FrameResourceManager`] still runs deform exactly once per tick:
    /// 1. HMD stereo multiview (when `mode = VrWithHmd`).
    /// 2. Secondary render-texture cameras, sorted by camera depth.
    /// 3. Main desktop swapchain (when `mode = DesktopPlusSecondaries`).
    pub(in crate::runtime) fn collect_prepared_views<'a>(
        &mut self,
        gpu: &GpuContext,
        mode: FrameRenderMode<'a>,
        swapchain_extent_px: (u32, u32),
        main_profile: RenderPathProfile,
    ) -> ViewFamilyPlan<'a> {
        let secondary_views = self.collect_secondary_rt_views(gpu);
        self.assemble_prepared_views(mode, swapchain_extent_px, main_profile, secondary_views)
    }

    /// Collects active views without GPU-backed secondary render targets for CPU-only tests.
    #[cfg(test)]
    pub(in crate::runtime) fn collect_prepared_views_without_secondaries<'a>(
        &self,
        mode: FrameRenderMode<'a>,
        swapchain_extent_px: (u32, u32),
        main_profile: RenderPathProfile,
    ) -> ViewFamilyPlan<'a> {
        self.assemble_prepared_views(mode, swapchain_extent_px, main_profile, Vec::new())
    }

    /// Appends HMD, pre-collected secondary, and main swapchain views in submission order.
    fn assemble_prepared_views<'a>(
        &self,
        mode: FrameRenderMode<'a>,
        swapchain_extent_px: (u32, u32),
        main_profile: RenderPathProfile,
        mut secondary_views: Vec<FrameViewPlan<'a>>,
    ) -> ViewFamilyPlan<'a> {
        let (includes_main, hmd_target) = match mode {
            FrameRenderMode::DesktopPlusSecondaries => (true, None),
            FrameRenderMode::VrWithHmd(ext) => (false, Some(ext)),
            FrameRenderMode::VrSecondariesOnly => (false, None),
        };

        let est_capacity =
            usize::from(hmd_target.is_some()) + secondary_views.len() + usize::from(includes_main);
        let mut views: Vec<FrameViewPlan<'a>> = Vec::with_capacity(est_capacity);
        let main_render_context = self.scene.active_main_render_context();

        if let Some(ext) = hmd_target {
            let extent_px = ext.extent_px;
            views.push(FrameViewPlan {
                host_camera: self.host_camera,
                render_context: main_render_context,
                frame_time_seconds: self.tick_state.frame_time_seconds(),
                draw_filter: None,
                render_space_filter: None,
                view_id: ViewId::Main,
                viewport_px: extent_px,
                clear: FrameViewClear::skybox(),
                profile: RenderPathProfile::xr_hmd(),
                target: FrameViewPlanTarget::Hmd(ext),
            });
        }

        views.append(&mut secondary_views);

        if includes_main {
            views.push(
                self.build_main_swapchain_view_with_profile(swapchain_extent_px, main_profile),
            );
        }

        ViewFamilyPlan::new(views)
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
            let Some(rt_handles) =
                secondary_rt_handles_for_rect(&mut self.backend, gpu, rt_id, rt, render_rect)
            else {
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
            // Selective secondary cameras (dashboards, in-world UI panels, mirrors on specific
            // subtrees) render tens of draws, not thousands. Hi-Z snapshots + occlusion temporal
            // cost a per-camera readback path with negligible payoff at that scale -- skip them.
            if !entry.selective_transform_ids.is_empty() {
                hc.suppress_occlusion_temporal = true;
            }
            let post_processing = if space.is_overlay() && !entry.selective_transform_ids.is_empty()
            {
                ViewPostProcessing::disabled()
            } else {
                ViewPostProcessing::from_camera_state(&entry.state)
            };
            views.push(FrameViewPlan {
                host_camera: hc,
                render_context: secondary_camera_render_context(),
                frame_time_seconds: self.tick_state.frame_time_seconds(),
                draw_filter: Some(filter),
                render_space_filter: Some(sid),
                view_id: secondary_camera_view_id(sid, entry.renderable_index, cam_idx),
                viewport_px,
                clear: FrameViewClear::from_camera_state(&entry.state),
                profile: RenderPathProfile::secondary_camera(post_processing),
                target: FrameViewPlanTarget::SecondaryRt(rt_handles),
            });
        }
        views
    }

    /// Builds the main desktop swapchain [`FrameViewPlan`] from the cached
    /// [`RendererRuntime::host_camera`].
    ///
    /// `swapchain_extent_px` must be the current GPU surface extent: it feeds
    /// [`crate::world_mesh::build_world_mesh_cull_proj_params`] on the pre-dispatch CPU cull
    /// path. A stale or zero extent produces a degenerate frustum and random scene-object
    /// culling. The render graph resolves its own rendering extent from
    /// [`crate::render_graph::FrameViewTarget::Swapchain::extent_px`] at record time -- that is a
    /// separate concern from cull math, which has already run by then.
    #[cfg(test)]
    pub(in crate::runtime) fn build_main_swapchain_view<'a>(
        &self,
        swapchain_extent_px: (u32, u32),
    ) -> FrameViewPlan<'a> {
        self.build_main_swapchain_view_with_profile(
            swapchain_extent_px,
            RenderPathProfile::desktop_main(),
        )
    }

    fn build_main_swapchain_view_with_profile<'a>(
        &self,
        swapchain_extent_px: (u32, u32),
        profile: RenderPathProfile,
    ) -> FrameViewPlan<'a> {
        FrameViewPlan {
            host_camera: self.host_camera,
            render_context: self.scene.active_main_render_context(),
            frame_time_seconds: self.tick_state.frame_time_seconds(),
            draw_filter: None,
            render_space_filter: None,
            view_id: ViewId::Main,
            viewport_px: swapchain_extent_px,
            clear: FrameViewClear::skybox(),
            profile,
            target: FrameViewPlanTarget::MainSwapchain,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Data-only tests for [`RendererRuntime::collect_prepared_views`]. No GPU is created.

    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::config::{RendererSettings, RendererSettingsHandle};
    use crate::connection::ConnectionParams;
    use crate::gpu::OutputDepthMode;
    use crate::materials::ShaderPermutation;

    fn build_runtime() -> RendererRuntime {
        let settings: RendererSettingsHandle =
            Arc::new(std::sync::RwLock::new(RendererSettings::default()));
        RendererRuntime::new(
            Option::<ConnectionParams>::None,
            settings,
            PathBuf::from("test_config.toml"),
        )
    }

    const TEST_EXTENT: (u32, u32) = (1920, 1080);

    fn collect_default_desktop_views(runtime: &RendererRuntime) -> ViewFamilyPlan<'_> {
        runtime.collect_prepared_views_without_secondaries(
            FrameRenderMode::DesktopPlusSecondaries,
            TEST_EXTENT,
            RenderPathProfile::desktop_main(),
        )
    }

    #[test]
    fn secondary_cameras_use_single_sample_profile_policy() {
        assert_eq!(
            RenderPathProfile::secondary_camera(ViewPostProcessing::primary_view())
                .sample_count_policy(),
            crate::render_graph::compiled::RenderPathSampleCountPolicy::SingleSample
        );
    }

    #[test]
    fn secondary_cameras_use_camera_render_context() {
        assert_eq!(secondary_camera_render_context(), RenderingContext::Camera);
    }

    #[test]
    fn empty_scene_desktop_mode_yields_only_main_view() {
        let runtime = build_runtime();
        let views = collect_default_desktop_views(&runtime);
        let plans = views.plans();
        assert_eq!(plans.len(), 1);
        assert!(matches!(
            plans[0].target,
            FrameViewPlanTarget::MainSwapchain
        ));
        assert_eq!(plans[0].view_id, ViewId::Main);
        assert!(plans[0].draw_filter.is_none());
        assert!(views.requirements().any_post_processing);
    }

    #[test]
    fn empty_scene_vr_secondaries_only_yields_empty_vec() {
        let runtime = build_runtime();
        let views = runtime.collect_prepared_views_without_secondaries(
            FrameRenderMode::VrSecondariesOnly,
            TEST_EXTENT,
            RenderPathProfile::desktop_main(),
        );
        assert!(
            views.is_empty(),
            "no HMD, no secondaries, and main swapchain excluded -- nothing to render"
        );
    }

    #[test]
    fn main_view_carries_runtime_host_camera() {
        let mut runtime = build_runtime();
        runtime.host_camera.frame_index = 42;
        runtime.host_camera.desktop_fov_degrees = 75.0;
        let views = collect_default_desktop_views(&runtime);
        let main = &views.plans()[0];
        assert_eq!(main.host_camera.frame_index, 42);
        assert_eq!(main.host_camera.desktop_fov_degrees, 75.0);
    }

    /// Pins the contract from the April 2026 cull regression: the main desktop `FrameViewPlan`
    /// must carry the swapchain extent supplied to `collect_prepared_views`. A zero or stale
    /// extent produces a degenerate `build_world_mesh_cull_proj_params` frustum and flickering
    /// scene-object culling.
    #[test]
    fn main_view_viewport_matches_supplied_swapchain_extent() {
        let runtime = build_runtime();
        let views = runtime.collect_prepared_views_without_secondaries(
            FrameRenderMode::DesktopPlusSecondaries,
            (1280, 720),
            RenderPathProfile::desktop_main(),
        );
        let main = views
            .plans()
            .iter()
            .find(|v| matches!(v.target, FrameViewPlanTarget::MainSwapchain))
            .expect("DesktopPlusSecondaries yields a MainSwapchain view");
        assert_eq!(main.viewport_px, (1280, 720));
    }

    #[test]
    fn main_view_uses_default_shader_permutation_and_depth_mode() {
        let runtime = build_runtime();
        let view = runtime.build_main_swapchain_view(TEST_EXTENT);
        assert_eq!(view.shader_permutation(), ShaderPermutation(0));
        assert_eq!(view.output_depth_mode(), OutputDepthMode::DesktopSingle);
        assert_eq!(view.clear.mode, crate::shared::CameraClearMode::Skybox);
        assert_eq!(view.post_processing(), ViewPostProcessing::primary_view());
        assert_eq!(
            view.profile.id(),
            crate::render_graph::compiled::RenderPathProfileId::DesktopMain
        );
    }

    #[test]
    fn main_view_uses_headless_profile_for_headless_output() {
        let runtime = build_runtime();
        let view = runtime.build_main_swapchain_view_with_profile(
            TEST_EXTENT,
            RenderPathProfile::headless_main(),
        );

        assert_eq!(view.post_processing(), ViewPostProcessing::disabled());
        assert_eq!(
            view.profile.id(),
            crate::render_graph::compiled::RenderPathProfileId::HeadlessMain
        );
    }

    /// Secondary view identity follows camera identity even when cameras share a render target.
    #[test]
    fn secondary_camera_view_ids_do_not_alias_shared_render_targets() {
        let first = secondary_camera_view_id(RenderSpaceId(9), 12, 0);
        let second = secondary_camera_view_id(RenderSpaceId(9), 13, 1);
        let fallback = secondary_camera_view_id(RenderSpaceId(9), -1, 2);

        assert_ne!(first, second);
        assert_ne!(first, fallback);
        assert_eq!(
            fallback,
            ViewId::SecondaryCamera(crate::camera::SecondaryCameraId::new(RenderSpaceId(9), 2))
        );
    }

    #[test]
    fn secondary_view_tasks_sort_by_depth_then_space_then_camera() {
        let mut tasks = vec![
            (RenderSpaceId(2), 0.0, 4),
            (RenderSpaceId(1), 0.0, 3),
            (RenderSpaceId(1), -5.0, 9),
            (RenderSpaceId(1), 0.0, 1),
        ];

        sort_secondary_view_tasks(&mut tasks);

        assert_eq!(
            tasks,
            vec![
                (RenderSpaceId(1), -5.0, 9),
                (RenderSpaceId(1), 0.0, 1),
                (RenderSpaceId(1), 0.0, 3),
                (RenderSpaceId(2), 0.0, 4),
            ]
        );
    }

    #[test]
    fn prepared_view_helpers_honor_explicit_camera_view_origin() {
        let runtime = build_runtime();
        let mut view = runtime.build_main_swapchain_view(TEST_EXTENT);
        view.host_camera.head_output_transform =
            glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(view.view_origin_world(), glam::Vec3::new(1.0, 2.0, 3.0));
        view.host_camera.explicit_view = Some(crate::camera::EyeView::new(
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Vec3::new(7.0, 8.0, 9.0),
        ));
        assert_eq!(view.view_origin_world(), glam::Vec3::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn prepared_view_helpers_prefer_eye_world_position_over_head_output() {
        let runtime = build_runtime();
        let mut view = runtime.build_main_swapchain_view(TEST_EXTENT);
        view.host_camera.head_output_transform =
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.0));
        view.host_camera.eye_world_position = Some(glam::Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(
            view.view_origin_world(),
            glam::Vec3::new(4.0, 5.0, 6.0),
            "eye_world_position must override the head-output (render-space root) translation \
             so PBS view-direction math sees the eye, not the floor anchor"
        );
        view.host_camera.explicit_view = Some(crate::camera::EyeView::new(
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Vec3::new(7.0, 8.0, 9.0),
        ));
        assert_eq!(
            view.view_origin_world(),
            glam::Vec3::new(7.0, 8.0, 9.0),
            "explicit camera view still wins over eye_world_position"
        );
    }
}
