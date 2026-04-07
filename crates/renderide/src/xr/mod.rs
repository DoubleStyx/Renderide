//! OpenXR session and Vulkan device bootstrap (Vulkan + `KHR_vulkan_enable2`).

mod bootstrap;
mod input;
mod session;
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
