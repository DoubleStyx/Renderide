//! IBL cubemap resources retained across asynchronous GPU submits.

use std::sync::Arc;

/// Copies cube face mip 0 from `source` into `destination`, including all six layers.
pub(super) fn copy_cube_mip0(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Texture,
    destination: &wgpu::Texture,
    face_size: u32,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) {
    let copy_query = profiler.map(|p| p.begin_query("skybox_ibl::copy_cube_mip0", encoder));
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: destination,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
    );
    if let (Some(profiler), Some(query)) = (profiler, copy_query) {
        profiler.end_query(encoder, query);
    }
}

/// IBL cubemap format. Matches the analytic skybox bake; supports STORAGE_BINDING.
pub(super) const IBL_CUBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Completed prefiltered cubemap that the frame-global binding owns.
pub(crate) struct PrefilteredCube {
    /// Texture backing the completed prefiltered cubemap.
    pub(crate) texture: Arc<wgpu::Texture>,
    /// Mip count resident in [`Self::texture`].
    pub(crate) mip_levels: u32,
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
    /// Transient textures retained until the queued commands complete.
    pub(super) textures: Vec<Arc<wgpu::Texture>>,
    /// Uniform and transient buffers retained until the queued commands complete.
    pub(super) buffers: Vec<wgpu::Buffer>,
    /// Bind groups retained until the queued commands complete.
    pub(super) bind_groups: Vec<wgpu::BindGroup>,
    /// Per-mip texture views retained until the queued commands complete.
    pub(super) texture_views: Vec<wgpu::TextureView>,
    /// Source asset views/textures retained for the duration of the bake.
    pub(super) source_views: Vec<Arc<wgpu::TextureView>>,
    /// 2D-array sampling view of the source pyramid retained for the convolve passes.
    pub(super) source_sample_view: Option<Arc<wgpu::TextureView>>,
}

/// IBL cube texture handles produced by [`create_ibl_cube`].
pub(super) struct IblCubeTexture {
    /// Texture backing the destination cubemap.
    pub(super) texture: Arc<wgpu::Texture>,
}

/// Allocates one Rgba16Float IBL cube.
pub(super) fn create_ibl_cube(
    device: &wgpu::Device,
    label: &'static str,
    face_size: u32,
    mip_levels: u32,
) -> IblCubeTexture {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
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
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    IblCubeTexture { texture }
}

/// Creates a 2D-array sampling view of every mip in the source pyramid.
pub(super) fn create_full_array_sample_view(
    texture: &wgpu::Texture,
    mip_levels: u32,
) -> wgpu::TextureView {
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox_ibl_array_full_sample_view"),
        format: Some(IBL_CUBE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(mip_levels),
        base_array_layer: 0,
        array_layer_count: Some(6),
    });
    crate::profiling::note_resource_churn!(TextureView, "skybox::ibl_full_array_sample_view");
    view
}

/// Creates a 2D-array sampling view of one mip, used by source-pyramid downsampling.
pub(super) fn create_mip_array_sample_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox_ibl_mip_array_sample_view"),
        format: Some(IBL_CUBE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
    });
    crate::profiling::note_resource_churn!(TextureView, "skybox::ibl_mip_array_sample_view");
    view
}

/// Creates a per-mip storage view for one face-array of the destination cube.
pub(super) fn create_mip_storage_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("skybox_ibl_mip_storage_view"),
        format: Some(IBL_CUBE_FORMAT),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
    });
    crate::profiling::note_resource_churn!(TextureView, "skybox::ibl_mip_storage_view");
    view
}
