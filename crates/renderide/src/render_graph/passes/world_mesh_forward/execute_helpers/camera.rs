//! Camera and pipeline-state helpers for world-mesh forward views.

use std::num::NonZeroU32;

use glam::Mat4;

use crate::gpu::GpuLimits;
use crate::materials::MaterialPipelineDesc;
use crate::pipelines::{ShaderPermutation, SHADER_PERM_MULTIVIEW_STEREO};
use crate::render_graph::camera::{
    effective_head_output_clip_planes, reverse_z_orthographic, reverse_z_perspective,
};
use crate::render_graph::clamp_desktop_fov_degrees;
use crate::render_graph::frame_params::{HostCameraFrame, WorldMeshForwardPipelineState};
use crate::render_graph::world_mesh_draw_prep::WorldMeshDrawItem;
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;

/// Selects the camera world-space position fed into `frame.camera_world_pos` for shader view-direction math.
///
/// Preference order:
/// 1. `explicit_camera_world_position` — secondary RT cameras carry their own pose.
/// 2. `eye_world_position` — main-space eye derived from the active render space's `view_transform`.
/// 3. `head_output_transform.col(3)` — last-ditch fallback (the render-space *root*, used by overlay
///    positioning) for any path that has not yet propagated the eye position. Using this as the
///    camera caused PBS specular highlights to converge at the space root (typically "the player's
///    feet") because every fragment's `v = normalize(cam - world_pos)` then pointed at the root.
pub(super) fn resolve_camera_world(hc: &HostCameraFrame) -> glam::Vec3 {
    hc.explicit_camera_world_position
        .or(hc.eye_world_position)
        .unwrap_or_else(|| hc.head_output_transform.col(3).truncate())
}

/// Resolves multiview use, [`MaterialPipelineDesc`], and [`ShaderPermutation`].
pub(super) fn resolve_pass_config(
    hc: HostCameraFrame,
    multiview_stereo: bool,
    scene_color_format: wgpu::TextureFormat,
    depth_stencil_format: wgpu::TextureFormat,
    gpu_limits: &GpuLimits,
    sample_count: u32,
) -> WorldMeshForwardPipelineState {
    let use_multiview =
        multiview_stereo && hc.vr_active && hc.stereo.is_some() && gpu_limits.supports_multiview;

    let sc = sample_count.max(1);

    let pass_desc = MaterialPipelineDesc {
        surface_format: scene_color_format,
        depth_stencil_format: Some(depth_stencil_format),
        sample_count: sc,
        multiview_mask: if use_multiview {
            NonZeroU32::new(3)
        } else {
            None
        },
    };

    let shader_perm = if use_multiview {
        SHADER_PERM_MULTIVIEW_STEREO
    } else {
        ShaderPermutation(0)
    };

    WorldMeshForwardPipelineState {
        use_multiview,
        pass_desc,
        shader_perm,
    }
}

/// Main render-space context, perspective projection for world draws, and optional ortho for overlays.
pub(super) fn compute_view_projections(
    scene: &SceneCoordinator,
    hc: HostCameraFrame,
    viewport_px: (u32, u32),
    draws: &[WorldMeshDrawItem],
) -> (RenderingContext, Mat4, Option<Mat4>) {
    let render_context = scene.active_main_render_context();
    let (vw, vh) = viewport_px;
    let aspect = vw as f32 / vh.max(1) as f32;
    let (near, far) = effective_head_output_clip_planes(
        hc.near_clip,
        hc.far_clip,
        hc.output_device,
        scene
            .active_main_space()
            .map(|space| space.root_transform.scale),
    );
    let fov_rad = clamp_desktop_fov_degrees(hc.desktop_fov_degrees).to_radians();
    let world_proj = reverse_z_perspective(aspect, fov_rad, near, far);

    let has_overlay = !draws.is_empty() && draws.iter().any(|d| d.is_overlay);
    let overlay_proj = if has_overlay {
        Some(if let Some((half_h, on, of)) = hc.primary_ortho_task {
            reverse_z_orthographic(half_h * aspect, half_h, on, of)
        } else {
            reverse_z_orthographic(1.0 * aspect, 1.0, near, far)
        })
    } else {
        None
    };

    (render_context, world_proj, overlay_proj)
}
