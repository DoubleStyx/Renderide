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
mod lockstep_pipeline;
mod mesh_deform;
mod mesh_upload;
mod rayon_admission;
mod render_world;
mod shadow_atlas;
mod tracy_plot;
mod world_mesh_prepare;

pub use asset_integration::{AssetIntegrationProfileSample, plot_asset_integration};
pub use command_encoding::{CommandEncodingProfileSample, plot_command_encoding};
pub use event_loop::{
    plot_driver_submit_backlog, plot_event_loop_idle_ms, plot_event_loop_wait_ms,
    plot_fps_cap_active, plot_surface_acquire_outcome, plot_surface_get_current_texture_ms,
    plot_surface_in_flight_count, plot_surface_previous_present_wait_ms, plot_window_focused,
};
pub use frame_upload::{
    FrameUploadArenaProfileSample, plot_frame_upload_arena, plot_frame_upload_batch,
    plot_world_mesh_subpass,
};
pub use ipc::{IpcPollProfileSample, plot_ipc_poll};
pub use lockstep_pipeline::{LockstepPipelineProfileSample, plot_lockstep_pipeline};
pub use mesh_deform::{MeshDeformProfileSample, plot_mesh_deform};
pub(crate) use mesh_upload::{
    MeshUploadBatchProfileSample, plot_mesh_derived_stream_masks, plot_mesh_upload_batch,
};
pub use rayon_admission::{RayonAdmissionProfileSample, plot_rayon_admission};
pub use render_world::{RenderWorldMaintenanceProfileSample, plot_render_world_maintenance};
pub use shadow_atlas::{plot_frame_global_split, plot_shadow_atlas};
pub use world_mesh_prepare::plot_world_mesh_prepare;
