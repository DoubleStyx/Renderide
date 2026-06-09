//! Upload commands deferred until GPU state, formats, or shared memory are available.

use std::collections::VecDeque;

use hashbrown::HashMap;

use crate::shared::{
    MeshUploadData, SetCubemapData, SetTexture2DData, SetTexture3DData, VideoTextureLoad,
};

/// Deferred texture-family upload command with the format generation it was received under.
pub(crate) struct PendingTextureUpload<T> {
    /// Host upload command.
    pub(crate) data: T,
    /// Renderer-local format generation, or `None` when data arrived before any format.
    pub(crate) generation: Option<u64>,
}

impl<T> PendingTextureUpload<T> {
    /// Wraps `data` with the current format generation at the deferral point.
    pub(crate) fn new(data: T, generation: Option<u64>) -> Self {
        Self { data, generation }
    }
}

/// Pre-GPU or not-yet-resident upload commands awaiting replay.
#[derive(Default)]
pub(crate) struct PendingAssetUploads {
    /// Mesh payloads waiting for GPU or shared memory.
    pub(crate) pending_mesh_uploads: VecDeque<MeshUploadData>,
    /// Texture2D payloads waiting for GPU allocation or shared memory.
    pub(crate) pending_texture_uploads: VecDeque<PendingTextureUpload<SetTexture2DData>>,
    /// Texture3D payloads waiting for GPU allocation or shared memory.
    pub(crate) pending_texture3d_uploads: VecDeque<PendingTextureUpload<SetTexture3DData>>,
    /// Cubemap payloads waiting for GPU allocation or shared memory.
    pub(crate) pending_cubemap_uploads: VecDeque<PendingTextureUpload<SetCubemapData>>,
    /// Latest video load commands received before GPU attach.
    pub(crate) pending_video_texture_loads: HashMap<i32, VideoTextureLoad>,
}
