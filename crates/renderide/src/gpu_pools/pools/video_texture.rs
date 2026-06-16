//! GPU-resident video texture pool: dummy 1x1 storage replaced by external views from the host.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use renderide_shared::VideoTextureProperties;

use crate::gpu_pools::VramResourceKind;
use crate::gpu_pools::impl_gpu_resource;
use crate::gpu_pools::resource_pool::{
    GpuResourcePool, UntrackedAccess, impl_resident_pool_facade,
};
use crate::gpu_pools::sampler_state::SamplerState;

/// Bytes per resident RGBA8 video pixel.
const RGBA8_BYTES_PER_PIXEL: u64 = 4;

static NEXT_VIDEO_TEXTURE_VIEW_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Host video texture; holds a dummy texture before an external view gets assigned from the
/// video player.
#[derive(Debug)]
pub struct GpuVideoTexture {
    /// Host VideoTexture asset id.
    pub asset_id: i32,
    /// The 1x1 placeholder texture used before the first [`Self::set_view`] call.
    _dummy_texture: Option<Arc<wgpu::Texture>>,
    /// Current view, initially from `dummy_texture` and then replaced by [`Self::set_view`].
    pub view: Arc<wgpu::TextureView>,
    /// Current video width in texels.
    pub width: u32,
    /// Current video height in texels.
    pub height: u32,
    /// Monotonic identifier for the current bindable view.
    pub view_generation: u64,
    /// Estimated VRAM for the current view.
    pub resident_bytes: u64,
    /// Sampler state mirrored from host format for material binds.
    pub sampler: SamplerState,
}

impl_gpu_resource!(GpuVideoTexture);

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
        crate::profiling::note_resource_churn!(TextureView, "gpu_pools::video_dummy_view");

        Self {
            asset_id,
            _dummy_texture: Some(dummy),
            view,
            width: 1,
            height: 1,
            view_generation: NEXT_VIDEO_TEXTURE_VIEW_GENERATION.fetch_add(1, Ordering::Relaxed),
            resident_bytes: RGBA8_BYTES_PER_PIXEL,
            sampler: SamplerState::from_video_props(props),
        }
    }

    /// Replaces the current view with one pointing at an externally-managed texture.
    #[cfg(feature = "video-textures")]
    pub fn set_view(
        &mut self,
        view: Arc<wgpu::TextureView>,
        width: u32,
        height: u32,
        resident_bytes: u64,
    ) {
        if !Arc::ptr_eq(&self.view, &view) {
            self.view_generation =
                NEXT_VIDEO_TEXTURE_VIEW_GENERATION.fetch_add(1, Ordering::Relaxed);
        }
        self._dummy_texture = None;
        self.view = view;
        self.width = width.max(1);
        self.height = height.max(1);
        self.resident_bytes = resident_bytes;
    }

    /// Updates sampler fields from [`VideoTextureProperties`].
    pub fn set_props(&mut self, props: &VideoTextureProperties) {
        self.sampler = SamplerState::from_video_props(props);
    }

    /// `true` when the color target exists and can be sampled (always after successful creation).
    #[inline]
    pub fn is_sampleable(&self) -> bool {
        true
    }
}

/// Pool of [`GpuVideoTexture`] entries keyed by host asset id.
#[derive(Debug)]
pub struct VideoTexturePool {
    /// Shared resident GPU resource table.
    inner: GpuResourcePool<GpuVideoTexture, UntrackedAccess>,
}

impl_resident_pool_facade!(VideoTexturePool, GpuVideoTexture, VramResourceKind::Texture,);

impl VideoTexturePool {
    /// Mutably borrows a resident video texture by host asset id.
    #[inline]
    pub fn get_mut(&mut self, asset_id: i32) -> Option<&mut GpuVideoTexture> {
        self.inner.get_mut(asset_id)
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu_pools::sampler_state::SamplerState;
    use renderide_shared::{TextureFilterMode, TextureWrapMode, VideoTextureProperties};

    #[test]
    fn sampler_from_props_clamps_negative_anisotropy() {
        let props = VideoTextureProperties {
            filter_mode: TextureFilterMode::Anisotropic,
            aniso_level: -4,
            wrap_u: TextureWrapMode::Mirror,
            wrap_v: TextureWrapMode::Clamp,
            asset_id: 12,
        };

        let sampler = SamplerState::from_video_props(&props);
        assert_eq!(sampler.filter_mode, TextureFilterMode::Anisotropic);
        assert_eq!(sampler.aniso_level, 0);
        assert_eq!(sampler.wrap_u, TextureWrapMode::Mirror);
        assert_eq!(sampler.wrap_v, TextureWrapMode::Clamp);
        assert_eq!(sampler.mipmap_bias, 0.0);
    }
}
