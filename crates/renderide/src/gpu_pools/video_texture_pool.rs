use crate::gpu_pools::resource_pool::{GpuResourcePool, PoolResourceAccess};
use crate::gpu_pools::{GpuResource, Texture2dSamplerState, VramAccounting, VramResourceKind};
use hashbrown::HashMap;
use renderide_shared::VideoTextureProperties;
use std::sync::Arc;

/// Bytes per resident RGBA8 video pixel.
const RGBA8_BYTES_PER_PIXEL: u64 = 4;

/// Host video texture,
/// holds a dummy texture before an external view gets assigned from the video player.
#[derive(Debug)]
pub struct GpuVideoTexture {
    /// Host VideoTexture asset id.
    pub asset_id: i32,
    /// The 1x1 placeholder texture used before the first [`Self::set_view`] call.
    dummy_texture: Option<Arc<wgpu::Texture>>,
    /// Current view, initially from `dummy_texture` and then replaced by [`Self::set_view`].
    pub view: Arc<wgpu::TextureView>,
    /// Pixel width of the current frame.
    pub width: u32,
    /// Pixel height of the current frame.
    pub height: u32,
    /// Estimated VRAM for the current view.
    pub resident_bytes: u64,
    /// Sampler state mirrored from host format for material binds.
    pub sampler: Texture2dSamplerState,
}

impl GpuResource for GpuVideoTexture {
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn asset_id(&self) -> i32 {
        self.asset_id
    }
}

impl GpuVideoTexture {
    /// Creates a 1x1 dummy texture. The real view is installed later via [`Self::set_view`].
    pub fn new(device: &wgpu::Device, asset_id: i32, props: &VideoTextureProperties) -> Self {
        let dummy = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("VideoTexture {asset_id} dummy")),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));

        let view = Arc::new(dummy.create_view(&wgpu::TextureViewDescriptor::default()));

        Self {
            asset_id,
            dummy_texture: Some(dummy),
            view,
            width: 1,
            height: 1,
            resident_bytes: RGBA8_BYTES_PER_PIXEL,
            sampler: sampler_from_props(props),
        }
    }

    /// Replaces the current view with one pointing at an externally-managed texture.
    pub fn set_view(
        &mut self,
        view: Arc<wgpu::TextureView>,
        width: u32,
        height: u32,
        resident_bytes: u64,
    ) {
        self.dummy_texture = None;
        self.view = view;
        self.width = width;
        self.height = height;
        self.resident_bytes = resident_bytes;
    }

    /// Set props from [`VideoTextureProperties`].
    pub fn set_props(&mut self, props: &VideoTextureProperties) {
        self.sampler = sampler_from_props(props);
    }

    /// `true` when the color target exists and can be sampled (always after successful creation).
    #[inline]
    pub fn is_sampleable(&self) -> bool {
        true
    }
}

/// Converts host video texture properties into the shared 2D sampler state.
fn sampler_from_props(props: &VideoTextureProperties) -> Texture2dSamplerState {
    Texture2dSamplerState {
        filter_mode: props.filter_mode,
        aniso_level: props.aniso_level.max(0),
        wrap_u: props.wrap_u,
        wrap_v: props.wrap_v,
        mipmap_bias: 0.0,
    }
}

/// Render-texture access behavior without streaming hooks.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct VideoTexturePoolAccess;

impl PoolResourceAccess for VideoTexturePoolAccess {
    const RESOURCE_KIND: VramResourceKind = VramResourceKind::Texture;

    fn note_access(&mut self, _asset_id: i32) {}
}

/// Pool of [`GpuVideoTexture`] entries keyed by host asset id (per-type id; disambiguate with packed texture type in materials).
#[derive(Debug)]
pub struct VideoTexturePool {
    inner: GpuResourcePool<GpuVideoTexture, VideoTexturePoolAccess>,
}

impl Default for VideoTexturePool {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoTexturePool {
    /// Empty pool.
    pub fn new() -> Self {
        Self {
            inner: GpuResourcePool::new(VideoTexturePoolAccess),
        }
    }

    /// VRAM accounting for resident video textures.
    pub fn accounting(&self) -> &VramAccounting {
        self.inner.accounting()
    }

    /// Inserts or replaces a video texture; returns `true` if replaced.
    pub fn insert_texture(&mut self, tex: GpuVideoTexture) -> bool {
        self.inner.insert(tex)
    }

    /// Removes by asset id; returns `true` if present.
    pub fn remove(&mut self, asset_id: i32) -> bool {
        self.inner.remove(asset_id)
    }

    /// Borrows a resident video texture by asset id.
    #[inline]
    pub fn get(&self, asset_id: i32) -> Option<&GpuVideoTexture> {
        self.inner.get(asset_id)
    }

    /// Mutably borrows a resident video texture.
    #[inline]
    pub fn get_mut(&mut self, asset_id: i32) -> Option<&mut GpuVideoTexture> {
        self.inner.get_mut(asset_id)
    }

    /// Full map for diagnostics and iteration.
    #[inline]
    pub fn textures(&self) -> &HashMap<i32, GpuVideoTexture> {
        self.inner.resources()
    }

    /// Number of video textures currently resident.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the pool is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use renderide_shared::{TextureFilterMode, TextureWrapMode};

    #[test]
    fn sampler_from_props_clamps_negative_anisotropy() {
        let props = VideoTextureProperties {
            filter_mode: TextureFilterMode::Anisotropic,
            aniso_level: -4,
            wrap_u: TextureWrapMode::Mirror,
            wrap_v: TextureWrapMode::Clamp,
            asset_id: 12,
        };

        let sampler = sampler_from_props(&props);
        assert_eq!(sampler.filter_mode, TextureFilterMode::Anisotropic);
        assert_eq!(sampler.aniso_level, 0);
        assert_eq!(sampler.wrap_u, TextureWrapMode::Mirror);
        assert_eq!(sampler.wrap_v, TextureWrapMode::Clamp);
        assert_eq!(sampler.mipmap_bias, 0.0);
    }
}
