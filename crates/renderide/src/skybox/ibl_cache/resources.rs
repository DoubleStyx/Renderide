//! IBL cubemap resources retained across asynchronous GPU submits.

use std::sync::Arc;

use crate::gpu_pools::SamplerState;
use crate::shared::{TextureFilterMode, TextureWrapMode};

/// IBL cubemap format. Matches the analytic skybox bake; supports STORAGE_BINDING.
pub(super) const IBL_CUBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Completed prefiltered cubemap that the frame-global binding owns.
pub(super) struct PrefilteredCube {
    /// Texture backing [`Self::view`]. Held to keep the storage alive while the view is bound.
    pub(super) _texture: Arc<wgpu::Texture>,
    /// Full mip-chain cube view.
    pub(super) view: Arc<wgpu::TextureView>,
    /// Sampler state used when binding for material sampling.
    pub(super) sampler: SamplerState,
    /// Mip count of [`Self::view`].
    pub(super) mip_levels: u32,
}

/// Pending bake retained until the submit callback fires.
pub(super) struct PendingBake {
    /// Completed cube that becomes visible after submit completion.
    pub(super) cube: PrefilteredCube,
    /// Transient resources retained until the queued commands complete.
    pub(super) _resources: PendingBakeResources,
}

/// Transient command resources that must survive until submit completion.
#[derive(Default)]
pub(super) struct PendingBakeResources {
    /// Uniform and transient buffers retained until the queued commands complete.
    pub(super) buffers: Vec<wgpu::Buffer>,
    /// Bind groups retained until the queued commands complete.
    pub(super) bind_groups: Vec<wgpu::BindGroup>,
    /// Per-mip texture views retained until the queued commands complete.
    pub(super) texture_views: Vec<wgpu::TextureView>,
    /// Source asset views/textures retained for the duration of the bake.
    pub(super) source_views: Vec<Arc<wgpu::TextureView>>,
    /// Cube sampling view of the destination retained for the convolve passes.
    pub(super) dst_sample_view: Option<Arc<wgpu::TextureView>>,
}

/// Sampler state used when the prefiltered cube is bound for material sampling.
pub(super) fn prefiltered_sampler_state() -> SamplerState {
    SamplerState {
        filter_mode: TextureFilterMode::Trilinear,
        aniso_level: 1,
        wrap_u: TextureWrapMode::Clamp,
        wrap_v: TextureWrapMode::Clamp,
        wrap_w: TextureWrapMode::default(),
        mipmap_bias: 0.0,
    }
}

/// IBL cube texture handles produced by [`create_ibl_cube`].
pub(super) struct IblCubeTexture {
    /// Texture backing the destination cubemap.
    pub(super) texture: Arc<wgpu::Texture>,
    /// Full mip-chain cube view bound at runtime.
    pub(super) full_view: Arc<wgpu::TextureView>,
}

/// Allocates the destination Rgba16Float cube and its full sampling view.
pub(super) fn create_ibl_cube(
    device: &wgpu::Device,
    face_size: u32,
    mip_levels: u32,
) -> IblCubeTexture {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skybox_ibl_cube"),
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: IBL_CUBE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    let full_view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox_ibl_cube_view"),
        format: Some(IBL_CUBE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(mip_levels),
        base_array_layer: 0,
        array_layer_count: Some(6),
    }));
    IblCubeTexture { texture, full_view }
}

/// Creates a cube-dimension sampling view of mip 0 only, used as the convolve input source.
///
/// The view must not overlap any storage-bound mip in the same compute dispatch -- wgpu treats
/// overlapping subresources as a usage conflict between `RESOURCE` and `STORAGE_WRITE_ONLY`.
/// A mip-0-only view is non-overlapping with every per-mip storage view (mip >= 1).
pub(super) fn create_mip0_cube_sample_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox_ibl_cube_mip0_sample_view"),
        format: Some(IBL_CUBE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
    })
}

/// Creates a per-mip storage view for one face-array of the destination cube.
pub(super) fn create_mip_storage_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox_ibl_mip_storage_view"),
        format: Some(IBL_CUBE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
    })
}
