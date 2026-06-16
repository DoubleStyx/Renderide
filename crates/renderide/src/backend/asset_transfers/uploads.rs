//! Shared-memory ingestion and queue draining for [`super::AssetTransferQueue`].
//!
//! Split into [`allocations`], [`attach`], [`texture2d`], [`mesh`], and [`render_texture`] for clarity.

mod allocations;
mod attach;
mod auxiliary;
mod cubemap;
mod mesh;
mod render_texture;
mod texture2d;
mod texture3d;
mod texture_common;
mod video_texture;

pub use attach::attach_flush_pending_asset_uploads;
pub use auxiliary::{
    on_desktop_texture_properties_update, on_gaussian_splat_config,
    on_gaussian_splat_upload_encoded, on_gaussian_splat_upload_raw, on_point_render_buffer_unload,
    on_point_render_buffer_upload, on_set_desktop_texture_properties,
    on_trail_render_buffer_unload, on_trail_render_buffer_upload, on_unload_desktop_texture,
    on_unload_gaussian_splat,
};
pub use cubemap::{
    on_set_cubemap_data, on_set_cubemap_format, on_set_cubemap_properties, on_unload_cubemap,
};
pub use mesh::{on_mesh_unload, try_process_mesh_upload};
pub use render_texture::{on_set_render_texture_format, on_unload_render_texture};
pub use texture2d::{
    on_set_texture_2d_data, on_set_texture_2d_format, on_set_texture_2d_properties,
    on_unload_texture_2d,
};
pub use texture3d::{
    on_set_texture_3d_data, on_set_texture_3d_format, on_set_texture_3d_properties,
    on_unload_texture_3d,
};
pub use video_texture::{
    on_unload_video_texture, on_video_texture_load, on_video_texture_properties,
    on_video_texture_start_audio_track, on_video_texture_update,
};

/// Deferred [`MeshUploadData`](crate::shared::MeshUploadData) count that emits queue-pressure diagnostics.
pub const PENDING_MESH_UPLOAD_WARN_THRESHOLD: usize = 256;

/// Deferred texture data command count that emits queue-pressure diagnostics.
pub const PENDING_TEXTURE_UPLOAD_WARN_THRESHOLD: usize = 256;

/// Deferred Texture3D data command count that emits queue-pressure diagnostics.
pub const PENDING_TEXTURE3D_UPLOAD_WARN_THRESHOLD: usize = 256;

/// Deferred cubemap data command count that emits queue-pressure diagnostics.
pub const PENDING_CUBEMAP_UPLOAD_WARN_THRESHOLD: usize = 256;
