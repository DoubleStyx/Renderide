//! Background-channel asset upload helpers.
//!
//! Writes mesh, shader, texture, and material-update data on the background queue while pumping
//! lockstep so renderer frame-start messages are still answered during asset acknowledgement waits.

use std::time::Duration;

mod material;
mod mesh;
mod shader;
mod shared;
mod texture;

pub(super) use material::{
    BoundMaterial, MaterialBatchRequest, MaterialUpdateOp, PropertyIdLookup, apply_material_batch,
    pack_texture2d_handle, request_property_ids,
};
pub(super) use mesh::{MeshUploadRequest, UploadedMesh, upload_mesh};
pub(super) use shader::upload_shader;
pub(super) use texture::{Texture2DUploadRequest, upload_texture2d_rgba8};

/// Default deadline for receiving an asset acknowledgement after sending upload data.
pub(super) const DEFAULT_ASSET_UPLOAD_TIMEOUT: Duration = Duration::from_secs(10);
