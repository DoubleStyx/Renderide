//! Per-frame pipeline that turns host scene state into a render-graph submit.
//!
//! - [`render`] -- view-family dispatch (desktop, HMD, and secondary-only submissions) and MSAA prep.
//! - [`main_view_360`] -- experimental cubemap-backed 360 projection for the desktop main view.
//! - [`schedule`] -- explicit CPU render schedule phases shared by main and offscreen graph submissions.
//! - [`view_planning`] -- collects the HMD, secondary render-texture, and main swapchain views for one tick.
//! - [`view_plan`] -- per-view CPU intent types (target, clear, viewport, host camera).
//! - [`extract`] -- immutable per-tick view extraction, draw collection, and submit packet construction.
//! - [`submit`] -- runtime-side application of host frame-submit payloads.

pub(in crate::runtime) mod extract;
pub(in crate::runtime) mod main_view_360;
pub(crate) mod render;
pub(in crate::runtime) mod schedule;
pub(in crate::runtime) mod skinned_bounds;
pub(in crate::runtime) mod submit;
pub(in crate::runtime) mod view_plan;
pub(in crate::runtime) mod view_planning;
