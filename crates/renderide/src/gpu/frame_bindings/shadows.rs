//! GPU shadow metadata layouts consumed through frame `@group(0)` bindings.

use bytemuck::{Pod, Zeroable};

/// Maximum shadow-map views rendered for one logical render view.
pub const MAX_SHADOW_VIEWS: usize = 16;

/// Default shadow atlas layer count.
pub const SHADOW_ARRAY_LAYERS: u32 = MAX_SHADOW_VIEWS as u32;

/// Depth format used by realtime raster shadow maps.
pub const SHADOW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Per-light shadow indirection row.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuShadowLight {
    /// First shadow-view row for this light, or `u32::MAX` when unshadowed.
    pub first_view: u32,
    /// Number of shadow-view rows for this light.
    pub view_count: u32,
    /// Reserved flags for future cached/static shadow modes.
    pub flags: u32,
    /// Aligns the row to a 16-byte WGSL storage stride.
    pub _pad0: u32,
}

impl Default for GpuShadowLight {
    fn default() -> Self {
        Self {
            first_view: u32::MAX,
            view_count: 0,
            flags: 0,
            _pad0: 0,
        }
    }
}

/// One shadow-map view matrix and sampling parameters.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuShadowView {
    /// World-to-shadow clip matrix.
    pub view_proj: [[f32; 4]; 4],
    /// `x = depth bias`, `y = normal bias`, `z = texel size`, `w = slope bias`.
    pub params: [f32; 4],
}

impl Default for GpuShadowView {
    fn default() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            params: [0.0; 4],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn shadow_metadata_rows_match_wgsl_storage_stride() {
        assert_eq!(size_of::<GpuShadowLight>(), 16);
        assert_eq!(size_of::<GpuShadowView>(), 80);
    }
}
