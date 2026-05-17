//! IBL source identity and per-variant payload definitions.

use std::sync::Arc;

/// Reflection-probe source to be baked into a GGX-prefiltered cubemap.
pub(crate) enum SkyboxIblSource {
    /// Resident host-uploaded cubemap read directly from a baked reflection probe.
    Cubemap(CubemapIblSource),
    /// Constant-color reflection-probe source.
    SolidColor(SolidColorIblSource),
    /// Renderer-captured cubemap source for an OnChanges reflection probe.
    RuntimeCubemap(RuntimeCubemapIblSource),
}

/// Resident cubemap source identity and GPU handle.
pub(crate) struct CubemapIblSource {
    /// Material asset id when this source came from a material, or `-1` for direct probe sources.
    pub material_asset_id: i32,
    /// Material property generation when this source came from a material.
    pub material_generation: u64,
    /// Stable hash of the shader route stem when this source came from a material.
    pub route_hash: u64,
    /// Source cubemap asset id.
    pub asset_id: i32,
    /// Source GPU allocation generation; invalidates when an asset id is reallocated.
    pub allocation_generation: u64,
    /// Resident cubemap face edge in texels (mip 0).
    pub face_size: u32,
    /// Resident mip count of the source cubemap.
    pub mip_levels_resident: u32,
    /// Source cubemap content generation; invalidates bakes when texels are re-uploaded.
    pub content_generation: u64,
    /// Whether sampling needs V-axis storage compensation.
    pub storage_v_inverted: bool,
    /// Cube-dimension texture view used by cube-sampling systems such as SH projection.
    pub view: Arc<wgpu::TextureView>,
    /// 2D-array texture view used by manual seam-aware specular IBL filtering.
    pub array_view: Arc<wgpu::TextureView>,
}

/// Constant-color source identity and color.
pub(crate) struct SolidColorIblSource {
    /// Renderer-side identity for this color source.
    pub identity: u64,
    /// Linear RGB color with alpha padding.
    pub color: [f32; 4],
}

/// Renderer-owned cubemap source identity and GPU handle.
pub(crate) struct RuntimeCubemapIblSource {
    /// Render space that owns the captured probe.
    pub render_space_id: i32,
    /// Dense reflection-probe renderable index.
    pub renderable_index: i32,
    /// Monotonic renderer-side capture generation.
    pub generation: u64,
    /// Source cubemap face edge in texels.
    pub face_size: u32,
    /// Mip count allocated on the captured cubemap.
    pub mip_levels: u32,
    /// Whether sampling needs V-axis storage compensation.
    pub storage_v_inverted: bool,
    /// Captured texture retained with the source view.
    pub texture: Arc<wgpu::Texture>,
    /// Cube-dimension texture view used by cube-sampling systems such as SH projection.
    pub view: Arc<wgpu::TextureView>,
    /// 2D-array texture view used by manual seam-aware specular IBL filtering.
    pub array_view: Arc<wgpu::TextureView>,
}
