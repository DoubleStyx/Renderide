//! OpenXR session and Vulkan device bootstrap (Vulkan + `KHR_vulkan_enable2`).
//!
//! When the runtime exposes [`XR_EXT_debug_utils`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_EXT_debug_utils),
//! [`bootstrap::init_wgpu_openxr`] registers a messenger so those messages go to the Renderide log
//! files. On Unix, [`init_wgpu_openxr`](bootstrap::init_wgpu_openxr) also replaces libc **stderr**
//! with a pipe and forwards lines to the file logger so native `fprintf(stderr, ...)` from the
//! runtime does not reach the terminal.

mod app_integration;
mod bootstrap;
mod debug_utils;
mod input;
mod session;
mod stderr_forward;
mod swapchain;

pub use bootstrap::{init_wgpu_openxr, XrWgpuHandles};
pub use input::OpenxrInput;
pub use session::{
    center_view_projection_from_stereo_views_aligned, headset_center_pose_from_stereo_views,
    headset_pose_from_xr_view, openxr_pose_to_engine, openxr_pose_to_host_tracking,
    tracking_space_to_world_matrix, view_projection_from_xr_view,
    view_projection_from_xr_view_aligned, XrSessionState,
};
pub use swapchain::{
    create_stereo_depth_texture, XrStereoSwapchain, XrSwapchainError, XR_COLOR_FORMAT,
    XR_VIEW_COUNT,
};

pub use app_integration::{
    openxr_begin_frame_tick, try_openxr_hmd_multiview_submit, OpenxrFrameTick,
};
