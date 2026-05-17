//! OpenXR helpers used by the winit app driver: frame tick state and HMD multiview submission.

mod frame_tick;
mod resources;
mod submit;
mod types;

pub use frame_tick::openxr_begin_frame_tick;
pub use submit::{HmdSubmitOutcome, try_openxr_hmd_multiview_submit};
pub use types::{OpenxrFrameTick, XrSessionBundle};
