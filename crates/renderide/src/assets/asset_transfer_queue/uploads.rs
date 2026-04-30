//! Shared-memory ingestion and queue draining for [`super::AssetTransferQueue`].
//!
//! Split into [`allocations`], [`attach`], [`texture2d`], [`mesh`], and [`render_texture`] for clarity.

mod allocations;
mod attach;
mod cubemap;
mod mesh;
mod render_texture;
mod texture2d;
mod texture3d;
mod texture_common;
mod video_texture;

pub use attach::attach_flush_pending_asset_uploads;
pub use cubemap::{
    on_set_cubemap_data, on_set_cubemap_format, on_set_cubemap_properties, on_unload_cubemap,
    try_cubemap_upload_with_device,
};
pub use mesh::{on_mesh_unload, try_process_mesh_upload};
pub use render_texture::{on_set_render_texture_format, on_unload_render_texture};
pub use texture2d::{
    on_set_texture_2d_data, on_set_texture_2d_format, on_set_texture_2d_properties,
    on_unload_texture_2d, try_texture_upload_with_device,
};
pub use texture3d::{
    on_set_texture_3d_data, on_set_texture_3d_format, on_set_texture_3d_properties,
    on_unload_texture_3d, try_texture3d_upload_with_device,
};
pub use video_texture::{
    on_unload_video_texture, on_video_texture_load, on_video_texture_properties,
    on_video_texture_start_audio_track, on_video_texture_update,
};

/// Max queued [`MeshUploadData`](crate::shared::MeshUploadData) when GPU is not ready yet (host data stays in shared memory).
pub const MAX_PENDING_MESH_UPLOADS: usize = 256;

/// Max queued texture data commands when GPU or format is not ready.
pub const MAX_PENDING_TEXTURE_UPLOADS: usize = 256;

/// Max queued Texture3D data commands when GPU or format is not ready.
pub const MAX_PENDING_TEXTURE3D_UPLOADS: usize = 256;

/// Max queued cubemap data commands when GPU or format is not ready.
pub const MAX_PENDING_CUBEMAP_UPLOADS: usize = 256;
