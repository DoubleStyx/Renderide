//! Per-tick view collection on [`super::RendererRuntime`].
//!
//! Builds the ordered list of [`FrameViewPlan`]s that drive draw collection and graph
//! execution: HMD stereo multiview (when present), then enabled secondary render-texture
//! cameras sorted by camera depth, then the main desktop swapchain (when included). Logic
//! sits between the render entry point in [`super::frame_render`] and the per-view extraction
//! pipeline in [`super::frame_extract`].

use crate::camera::ViewId;
use crate::render_graph::{
    FrameViewClear, camera_state_enabled, host_camera_frame_for_render_texture,
};
use crate::scene::RenderSpaceId;
use crate::world_mesh::draw_filter_from_camera_entry;

use super::RendererRuntime;
use super::frame_render::FrameRenderMode;
use super::frame_view_plan::{FrameViewPlan, FrameViewPlanTarget, OffscreenRtHandles};

/// Returns the stable logical identity for one secondary camera view.
pub(super) fn secondary_camera_view_id(
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

impl RendererRuntime {
    /// Collects every active view for this tick into a single ordered list.
    ///
    /// Ordering -- preserved so the mesh-deform skip flag on
    /// [`crate::backend::FrameResourceManager`] still runs deform exactly once per tick:
    /// 1. HMD stereo multiview (when `mode = VrWithHmd`).
    /// 2. Secondary render-texture cameras, sorted by camera depth.
    /// 3. Main desktop swapchain (when `mode = DesktopPlusSecondaries`).
    pub(super) fn collect_prepared_views<'a>(
        &mut self,
        mode: FrameRenderMode<'a>,
        swapchain_extent_px: (u32, u32),
    ) -> Vec<FrameViewPlan<'a>> {
        let (includes_main, hmd_target) = match mode {
            FrameRenderMode::DesktopPlusSecondaries => (true, None),
            FrameRenderMode::VrWithHmd(ext) => (false, Some(ext)),
            FrameRenderMode::VrSecondariesOnly => (false, None),
        };

        let mut secondary_views = self.collect_secondary_rt_views();
        let est_capacity =
            usize::from(hmd_target.is_some()) + secondary_views.len() + usize::from(includes_main);
        let mut views: Vec<FrameViewPlan<'a>> = Vec::with_capacity(est_capacity);

        if let Some(ext) = hmd_target {
            let extent_px = ext.extent_px;
            views.push(FrameViewPlan {
                host_camera: self.host_camera,
                draw_filter: None,
                view_id: ViewId::Main,
                viewport_px: extent_px,
                clear: FrameViewClear::skybox(),
                target: FrameViewPlanTarget::Hmd(ext),
            });
        }

        views.append(&mut secondary_views);

        if includes_main {
            views.push(self.build_main_swapchain_view(swapchain_extent_px));
        }

        views
    }

    /// Builds prepared views for every enabled secondary render-texture camera in the scene,
    /// skipping cameras whose host render texture is not yet resident on the GPU.
    ///
    /// Reuses [`RendererRuntime::secondary_view_tasks_scratch`] for the depth-sort scratch buffer
    /// so a frame with secondary cameras does not allocate a fresh `Vec` for the sort each tick.
    fn collect_secondary_rt_views<'a>(&mut self) -> Vec<FrameViewPlan<'a>> {
        let mut tasks = std::mem::take(&mut self.secondary_view_tasks_scratch);
        tasks.clear();
        let result = self.collect_secondary_rt_views_using(&mut tasks);
        self.secondary_view_tasks_scratch = tasks;
        result
    }

    /// Inner helper that consumes the supplied scratch `tasks` buffer; split out so the outer
    /// caller can keep the scratch field reachable across the immutable borrow taken here.
    fn collect_secondary_rt_views_using<'a>(
        &self,
        tasks: &mut Vec<(RenderSpaceId, f32, usize)>,
    ) -> Vec<FrameViewPlan<'a>> {
        for sid in self.scene.render_space_ids() {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            if !space.is_active {
                continue;
            }
            for (idx, cam) in space.cameras.iter().enumerate() {
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
            let Some(entry) = space.cameras.get(cam_idx) else {
                continue;
            };
            if !camera_state_enabled(entry.state.flags) {
                continue;
            }
            let rt_id = entry.state.render_texture_asset_id;
            let (color_view, depth_texture, depth_view, viewport, color_format) = {
                let Some(rt) = self.backend.render_texture_pool().get(rt_id) else {
                    logger::trace!(
                        "secondary camera: render texture asset {rt_id} not resident; skipping"
                    );
                    continue;
                };
                let Some(dt) = rt.depth_texture.clone() else {
                    logger::warn!("secondary camera: render texture {rt_id} missing depth");
                    continue;
                };
                let Some(dv) = rt.depth_view.clone() else {
                    logger::warn!("secondary camera: render texture {rt_id} missing depth view");
                    continue;
                };
                (
                    rt.color_view.clone(),
                    dt,
                    dv,
                    (rt.width, rt.height),
                    rt.wgpu_color_format,
                )
            };
            let Some(world_m) = self.scene.world_matrix(sid, entry.transform_id as usize) else {
                continue;
            };
            let mut hc = host_camera_frame_for_render_texture(
                &self.host_camera,
                &entry.state,
                viewport,
                world_m,
                &self.scene,
            );
            let filter = draw_filter_from_camera_entry(entry);
            // Selective secondary cameras (dashboards, in-world UI panels, mirrors on specific
            // subtrees) render tens of draws, not thousands. Hi-Z snapshots + occlusion temporal
            // cost a per-camera readback path with negligible payoff at that scale -- skip them.
            if !entry.selective_transform_ids.is_empty() {
                hc.suppress_occlusion_temporal = true;
            }
            views.push(FrameViewPlan {
                host_camera: hc,
                draw_filter: Some(filter),
                view_id: secondary_camera_view_id(sid, entry.renderable_index, cam_idx),
                viewport_px: viewport,
                clear: FrameViewClear::from_camera_state(&entry.state),
                target: FrameViewPlanTarget::SecondaryRt(OffscreenRtHandles {
                    rt_id,
                    color_view,
                    depth_texture,
                    depth_view,
                    color_format,
                }),
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
    pub(super) fn build_main_swapchain_view<'a>(
        &self,
        swapchain_extent_px: (u32, u32),
    ) -> FrameViewPlan<'a> {
        FrameViewPlan {
            host_camera: self.host_camera,
            draw_filter: None,
            view_id: ViewId::Main,
            viewport_px: swapchain_extent_px,
            clear: FrameViewClear::skybox(),
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

    #[test]
    fn empty_scene_desktop_mode_yields_only_main_view() {
        let mut runtime = build_runtime();
        let views =
            runtime.collect_prepared_views(FrameRenderMode::DesktopPlusSecondaries, TEST_EXTENT);
        assert_eq!(views.len(), 1);
        assert!(matches!(
            views[0].target,
            FrameViewPlanTarget::MainSwapchain
        ));
        assert_eq!(views[0].view_id, ViewId::Main);
        assert!(views[0].draw_filter.is_none());
    }

    #[test]
    fn empty_scene_vr_secondaries_only_yields_empty_vec() {
        let mut runtime = build_runtime();
        let views = runtime.collect_prepared_views(FrameRenderMode::VrSecondariesOnly, TEST_EXTENT);
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
        let views =
            runtime.collect_prepared_views(FrameRenderMode::DesktopPlusSecondaries, TEST_EXTENT);
        let main = &views[0];
        assert_eq!(main.host_camera.frame_index, 42);
        assert_eq!(main.host_camera.desktop_fov_degrees, 75.0);
    }

    /// Pins the contract from the April 2026 cull regression: the main desktop `FrameViewPlan`
    /// must carry the swapchain extent supplied to `collect_prepared_views`. A zero or stale
    /// extent produces a degenerate `build_world_mesh_cull_proj_params` frustum and flickering
    /// scene-object culling.
    #[test]
    fn main_view_viewport_matches_supplied_swapchain_extent() {
        let mut runtime = build_runtime();
        let views =
            runtime.collect_prepared_views(FrameRenderMode::DesktopPlusSecondaries, (1280, 720));
        let main = views
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
