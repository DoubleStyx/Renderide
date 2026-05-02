//! Host-authored asset descriptors and sampler/property catalogs.

use glam::IVec2;
use hashbrown::HashMap;
use std::sync::Arc;

use crate::shared::{
    SetCubemapFormat, SetCubemapProperties, SetRenderTextureFormat, SetTexture2DFormat,
    SetTexture2DProperties, SetTexture3DFormat, SetTexture3DProperties, VideoTextureProperties,
};

/// CPU copy of one host point-render buffer payload.
#[derive(Clone, Debug, Default)]
#[expect(
    dead_code,
    reason = "Point-render buffer metadata is staged for upcoming parity work and read in follow-up changes"
)]
pub(crate) struct PointRenderBufferPayload {
    /// Host asset id.
    pub asset_id: i32,
    /// Number of points in this payload.
    pub count: i32,
    /// Byte offset of position rows in [`Self::payload`].
    pub positions_offset: i32,
    /// Byte offset of rotation rows in [`Self::payload`].
    pub rotations_offset: i32,
    /// Byte offset of size rows in [`Self::payload`].
    pub sizes_offset: i32,
    /// Byte offset of color rows in [`Self::payload`].
    pub colors_offset: i32,
    /// Byte offset of frame-index rows in [`Self::payload`].
    pub frame_indexes_offset: i32,
    /// Host-provided frame grid size metadata.
    pub frame_grid_size: IVec2,
    /// Raw shared-memory payload copy for this point buffer upload.
    pub payload: Arc<[u8]>,
}

/// Latest host format/property rows keyed by asset id.
#[derive(Default)]
pub(crate) struct AssetCatalogs {
    /// Latest render-texture format rows.
    pub(crate) render_texture_formats: HashMap<i32, SetRenderTextureFormat>,
    /// Latest Texture2D format rows.
    pub(crate) texture_formats: HashMap<i32, SetTexture2DFormat>,
    /// Latest Texture2D sampler/property rows.
    pub(crate) texture_properties: HashMap<i32, SetTexture2DProperties>,
    /// Latest Texture3D format rows.
    pub(crate) texture3d_formats: HashMap<i32, SetTexture3DFormat>,
    /// Latest Texture3D sampler/property rows.
    pub(crate) texture3d_properties: HashMap<i32, SetTexture3DProperties>,
    /// Latest cubemap format rows.
    pub(crate) cubemap_formats: HashMap<i32, SetCubemapFormat>,
    /// Latest cubemap sampler/property rows.
    pub(crate) cubemap_properties: HashMap<i32, SetCubemapProperties>,
    /// Latest video texture sampler/property rows.
    pub(crate) video_texture_properties: HashMap<i32, VideoTextureProperties>,
    /// Latest point-render buffer payload copied from shared memory.
    pub(crate) point_render_buffers: HashMap<i32, PointRenderBufferPayload>,
}

impl AssetCatalogs {
    /// Returns cached video texture properties, or stable defaults tagged with `asset_id`.
    pub(crate) fn video_texture_properties_or_default(
        &self,
        asset_id: i32,
    ) -> VideoTextureProperties {
        self.video_texture_properties
            .get(&asset_id)
            .cloned()
            .unwrap_or(VideoTextureProperties {
                asset_id,
                ..VideoTextureProperties::default()
            })
    }
}
