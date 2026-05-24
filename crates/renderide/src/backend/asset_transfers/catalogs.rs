//! Host-authored asset descriptors and sampler/property catalogs.

use hashbrown::HashMap;

use crate::particles::{PointRenderBufferAsset, TrailRenderBufferAsset};
use crate::shared::{
    DesktopTexturePropertiesUpdate, GaussianSplatConfig, SetCubemapFormat, SetCubemapProperties,
    SetDesktopTextureProperties, SetRenderTextureFormat, SetTexture2DFormat,
    SetTexture2DProperties, SetTexture3DFormat, SetTexture3DProperties, VideoTextureProperties,
};

/// Latest Gaussian splat payload family known for an asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GaussianSplatUploadKind {
    /// Raw Gaussian attribute buffers.
    Raw,
    /// Encoded Gaussian attribute buffers.
    Encoded,
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
    /// Latest desktop texture display properties.
    pub(crate) desktop_texture_properties: HashMap<i32, SetDesktopTextureProperties>,
    /// Latest desktop texture size updates.
    pub(crate) desktop_texture_updates: HashMap<i32, DesktopTexturePropertiesUpdate>,
    /// Resident point render-buffer metadata keyed by source asset id.
    pub(crate) point_render_buffers: HashMap<i32, PointRenderBufferAsset>,
    /// Resident trail render-buffer metadata keyed by source asset id.
    pub(crate) trail_render_buffers: HashMap<i32, TrailRenderBufferAsset>,
    /// Latest Gaussian splat upload family per asset.
    pub(crate) gaussian_splat_uploads: HashMap<i32, GaussianSplatUploadKind>,
    /// Latest Gaussian splat renderer config.
    pub(crate) gaussian_splat_config: GaussianSplatConfig,
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
            .unwrap_or_else(|| VideoTextureProperties {
                asset_id,
                ..VideoTextureProperties::default()
            })
    }
}
