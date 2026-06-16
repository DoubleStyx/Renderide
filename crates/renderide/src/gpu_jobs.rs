//! Reusable lifecycle helpers for backend-owned nonblocking GPU jobs.
//!
//! These helpers are for one-off backend compute or copy work that needs completion
//! notification or a CPU readback outside the render graph. Frame-shape render work stays in
//! the graph so transient resources, barriers, and pass ordering remain explicit there.

mod readback;
mod submit;

pub(crate) use readback::{GpuReadbackJobs, GpuReadbackOutcomes, SubmittedReadbackJob};
pub(crate) use submit::{GpuSubmitJobTracker, SubmittedGpuJob};

/// GPU resources retained until an asynchronous backend job is known to be complete.
pub(crate) type GpuJobResources = crate::gpu::GpuRetainedResources;
