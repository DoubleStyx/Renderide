//! OpenXR session and Vulkan device bootstrap (Vulkan + `KHR_vulkan_enable2`).
//!
//! Vulkan validation layers are requested only when [`crate::config::DebugSettings::gpu_validation_layers`]
//! (and env overrides) say so; see [`bootstrap::init_wgpu_openxr`].
//!
//! When the runtime exposes [`XR_EXT_debug_utils`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_EXT_debug_utils),
//! [`bootstrap::init_wgpu_openxr`] registers a messenger so those messages go to the Renderide log
//! files. Native `printf` / `fprintf(stderr, ...)` (runtime or drivers) is forwarded via
//! [`crate::native_stdio::ensure_stdio_forwarded_to_logger`] ([`bootstrap::init_wgpu_openxr`];
//! [`crate::app::run`] installs it unconditionally after file logging starts).
//!
//! Khronos OpenXR **loader** discovery (runtime `LoadLibrary` / `dlopen`) is implemented in
//! [`openxr_loader_paths`], including [`openxr_loader_paths::RENDERIDE_OPENXR_LOADER`] and
//! [`openxr_loader_paths::openxr_loader_candidate_paths`].
//!
//! ## Layout
//!
//! - **`bootstrap`** -- Vulkan + OpenXR + wgpu init (`init_wgpu_openxr`).
//! - **`session`** -- `view_math` submodule (poses, view-projection, tracking alignment); [`XrSessionState`]
//!   (wait / submit frame loop).
//! - **`input`** -- OpenXR actions, profiles, and [`OpenxrInput`].
//! - **`swapchain`** / **`app_integration`** -- stereo targets and frame-tick glue for the render loop.
//! - **`debug_utils`**, **`openxr_loader_paths`**, **`host_camera_sync`** -- debug messenger, loader paths, IPC-facing traits.

mod app_integration;
mod bootstrap;
mod debug_utils;
mod host_camera_sync;
mod input;
mod openxr_loader_paths;
pub mod output_device;
pub(crate) mod session;
mod swapchain;

pub use bootstrap::{XrWgpuHandles, init_wgpu_openxr};
pub use input::{OpenxrInput, synthesize_hand_states};
pub use session::{
    TrackedSessionState, XrSessionState, center_view_projection_from_stereo_views_aligned,
    eye_view_from_xr_view_aligned, eye_world_position_from_xr_view_aligned,
    headset_center_pose_from_stereo_views, headset_pose_from_xr_view, openxr_pose_to_engine,
    openxr_pose_to_host_tracking, tracking_space_to_world_matrix, view_from_xr_view_aligned,
    view_projection_from_xr_view, view_projection_from_xr_view_aligned,
};
pub use swapchain::{
    XR_COLOR_FORMAT, XR_VIEW_COUNT, XrStereoSwapchain, XrSwapchainError,
    create_stereo_depth_texture,
};

pub use app_integration::{
    OpenxrFrameTick, XrSessionBundle, openxr_begin_frame_tick, try_openxr_hmd_multiview_submit,
};
pub use host_camera_sync::{XrFrameRenderer, XrHostCameraSync};
