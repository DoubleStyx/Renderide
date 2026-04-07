//! Presentation split: desktop window vs OpenXR mirror + headset submission.
//!
//! The concrete loop lives in [`crate::app`]. VR intent comes from host
//! [`RendererInitData`](crate::shared::RendererInitData) via
//! [`crate::output_device::head_output_device_wants_openxr`].

pub use crate::output_device::{head_output_device_is_vr, head_output_device_wants_openxr};
