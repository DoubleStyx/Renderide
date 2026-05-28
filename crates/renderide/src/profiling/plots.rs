//! Tracy plot helpers for the renderer.
//!
//! Public plot fns are grouped by subsystem in the sibling files and re-exported here so the
//! parent `profiling` module's surface (`crate::profiling::plot_*`, the sample structs) stays
//! flat. Per-fn `cfg(feature = "tracy")` gating is centralized in [`tracy_plot::tracy_plot`].

mod asset_integration;
mod command_encoding;
mod event_loop;
mod frame_upload;
mod ipc;
mod mesh_deform;
mod tracy_plot;

pub use asset_integration::{AssetIntegrationProfileSample, plot_asset_integration};
pub use command_encoding::{CommandEncodingProfileSample, plot_command_encoding};
pub use event_loop::{
    plot_driver_submit_backlog, plot_event_loop_idle_ms, plot_event_loop_wait_ms,
    plot_fps_cap_active, plot_surface_acquire_outcome, plot_window_focused,
};
pub use frame_upload::{plot_frame_upload_batch, plot_world_mesh_subpass};
pub use ipc::{IpcPollProfileSample, plot_ipc_poll};
pub use mesh_deform::{MeshDeformProfileSample, plot_mesh_deform};
