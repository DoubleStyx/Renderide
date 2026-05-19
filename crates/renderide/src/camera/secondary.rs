//! Secondary render-texture camera support: state-flag decoders and HostCameraFrame construction.

mod flags;
mod host_frame;

pub use flags::{
    camera_state_enabled, camera_state_motion_blur, camera_state_post_processing,
    camera_state_render_shadows, camera_state_screen_space_reflections,
};
pub use host_frame::host_camera_frame_for_render_texture;
