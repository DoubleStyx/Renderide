//! OpenXR helpers used by the winit [`crate::app::RenderideApp`] loop: frame tick state and HMD multiview submission.

use crate::gpu::GpuContext;
use crate::render_graph::{effective_head_output_clip_planes, ExternalFrameTargets};
use crate::runtime::RendererRuntime;
use crate::xr::{
    create_stereo_depth_texture, XrStereoSwapchain, XrWgpuHandles, XR_COLOR_FORMAT, XR_VIEW_COUNT,
};
use glam::Mat4;
use openxr as xr;
use winit::window::Window;

/// Cached OpenXR frame state after a single `wait_frame` (no second wait per tick).
pub struct OpenxrFrameTick {
    /// Predicted display time for this frame (input sampling, `end_frame`).
    pub predicted_display_time: xr::Time,
    /// Whether the runtime expects rendering work this frame.
    pub should_render: bool,
    /// Stereo views from `locate_views` (may be empty when `should_render` is false).
    pub views: Vec<xr::View>,
    /// Single-view matrix for the desktop mirror when VR is active (left-eye center), else `None`.
    pub desktop_mirror_view_proj: Option<Mat4>,
}

/// Single `wait_frame` + `locate_views` for stereo uniforms; used for both mirror and HMD paths.
pub fn openxr_begin_frame_tick(
    handles: &mut XrWgpuHandles,
    runtime: &mut RendererRuntime,
) -> Option<OpenxrFrameTick> {
    let _ = handles.xr_session.poll_events();
    let fs = handles.xr_session.wait_frame().ok()??;
    let views = if fs.should_render {
        handles
            .xr_session
            .locate_views(fs.predicted_display_time)
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    if views.len() >= 2 {
        if runtime.host_camera.vr_active {
            let (near, far) = effective_head_output_clip_planes(
                runtime.host_camera.near_clip,
                runtime.host_camera.far_clip,
                runtime.host_camera.output_device,
                runtime
                    .scene
                    .active_main_space()
                    .map(|space| space.root_transform.scale),
            );
            let center_pose = crate::xr::headset_center_pose_from_stereo_views(&views);
            let world_from_tracking = runtime
                .scene
                .active_main_space()
                .map(|space| {
                    crate::xr::tracking_space_to_world_matrix(
                        &space.root_transform,
                        &space.view_transform,
                        space.override_view_position,
                        center_pose,
                    )
                })
                .unwrap_or(glam::Mat4::IDENTITY);
            runtime.set_head_output_transform(world_from_tracking);
            let l = crate::xr::view_projection_from_xr_view_aligned(
                &views[0],
                near,
                far,
                world_from_tracking,
            );
            let r = crate::xr::view_projection_from_xr_view_aligned(
                &views[1],
                near,
                far,
                world_from_tracking,
            );
            runtime.set_stereo_view_proj(Some((l, r)));
            let desktop_mirror_view_proj =
                crate::xr::center_view_projection_from_stereo_views_aligned(
                    &views,
                    near,
                    far,
                    world_from_tracking,
                );
            return Some(OpenxrFrameTick {
                predicted_display_time: fs.predicted_display_time,
                should_render: fs.should_render,
                views,
                desktop_mirror_view_proj,
            });
        }
        // Desktop (`!vr_active`): keep [`HostCameraFrame::head_output_transform`] from
        // [`RendererRuntime::on_frame_submit`] (host `root_transform`), matching Unity
        // `HeadOutput.UpdatePositioning`. OpenXR still supplies views for IPC pose.
        return Some(OpenxrFrameTick {
            predicted_display_time: fs.predicted_display_time,
            should_render: fs.should_render,
            views,
            desktop_mirror_view_proj: None,
        });
    }
    Some(OpenxrFrameTick {
        predicted_display_time: fs.predicted_display_time,
        should_render: fs.should_render,
        views,
        desktop_mirror_view_proj: None,
    })
}

/// Renders to the OpenXR stereo swapchain and calls [`crate::xr::session::XrSessionState::end_frame_projection`].
///
/// Uses the same [`xr::FrameState`] as [`openxr_begin_frame_tick`] — no second `wait_frame`.
pub fn try_openxr_hmd_multiview_submit(
    gpu: &mut GpuContext,
    handles: &mut XrWgpuHandles,
    runtime: &mut RendererRuntime,
    xr_swapchain: &mut Option<XrStereoSwapchain>,
    xr_stereo_depth: &mut Option<(wgpu::Texture, wgpu::TextureView)>,
    window: &Window,
    tick: &OpenxrFrameTick,
) -> bool {
    if !handles.xr_session.session_running() {
        return false;
    }
    if !runtime.host_camera.vr_active {
        return false;
    }
    if !gpu.device().features().contains(wgpu::Features::MULTIVIEW) {
        return false;
    }
    if !tick.should_render || tick.views.len() < 2 {
        return false;
    }
    if xr_swapchain.is_none() {
        let sys_id = handles.xr_system_id;
        let session = handles.xr_session.xr_vulkan_session();
        let inst = handles.xr_session.xr_instance();
        let dev = handles.device.as_ref();
        let res = unsafe { XrStereoSwapchain::new(session, inst, sys_id, dev) };
        match res {
            Ok(sc) => {
                logger::info!(
                    "OpenXR swapchain {}×{} (stereo array)",
                    sc.resolution.0,
                    sc.resolution.1
                );
                *xr_swapchain = Some(sc);
            }
            Err(e) => {
                logger::debug!("OpenXR swapchain not created: {e}");
                return false;
            }
        }
    }
    let sc = match xr_swapchain.as_mut() {
        Some(s) => s,
        None => return false,
    };
    let image_index = match sc.handle.acquire_image() {
        Ok(i) => i as usize,
        Err(_) => return false,
    };
    if sc.handle.wait_image(xr::Duration::INFINITE).is_err() {
        return false;
    }
    let Some(color_view) = sc.color_view_for_image(image_index) else {
        let _ = sc.handle.release_image();
        return false;
    };
    let extent = sc.resolution;
    let need_new_depth = xr_stereo_depth
        .as_ref()
        .map(|(tex, _)| {
            tex.size().width != extent.0
                || tex.size().height != extent.1
                || tex.size().depth_or_array_layers != XR_VIEW_COUNT
        })
        .unwrap_or(true);
    if need_new_depth {
        let (dt, dv) = create_stereo_depth_texture(gpu.device().as_ref(), extent);
        *xr_stereo_depth = Some((dt, dv));
    }
    let depth_pair = &xr_stereo_depth
        .as_ref()
        .expect("xr_stereo_depth set above when missing")
        .1;
    let ext = ExternalFrameTargets {
        color_view,
        depth_view: depth_pair,
        extent_px: extent,
        surface_format: XR_COLOR_FORMAT,
    };
    let rect = xr::Rect2Di {
        offset: xr::Offset2Di { x: 0, y: 0 },
        extent: xr::Extent2Di {
            width: extent.0 as i32,
            height: extent.1 as i32,
        },
    };
    let views_ref = tick.views.as_slice();
    if runtime
        .execute_frame_graph_external_multiview(gpu, window, ext)
        .is_err()
    {
        let _ = sc.handle.release_image();
        return false;
    }
    if sc.handle.release_image().is_err() {
        return false;
    }
    if handles
        .xr_session
        .end_frame_projection(tick.predicted_display_time, &sc.handle, views_ref, rect)
        .is_err()
    {
        return false;
    }
    true
}
