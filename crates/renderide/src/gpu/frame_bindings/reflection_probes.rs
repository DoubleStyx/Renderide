//! Reflection-probe metadata row layout, params encoding, and the IBL atlas texture format.

use bytemuck::{Pod, Zeroable};

/// Texture format used by prefiltered reflection-probe IBL cubemaps.
pub const REFLECTION_PROBE_ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Probe metadata flag for box-projected reflection sampling.
pub const REFLECTION_PROBE_METADATA_BOX_PROJECTION: u32 = 1;
/// Probe metadata parameter value for local reflection-probe SH2 coefficients.
pub const REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL: f32 = 1.0;

/// One reflection-probe metadata row consumed by PBS fragment shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuReflectionProbeMetadata {
    /// World-space AABB minimum; `.w` stores the sanitized blend distance.
    pub box_min: [f32; 4],
    /// World-space AABB maximum, padded to a vec4.
    pub box_max: [f32; 4],
    /// World-space probe position; `.w` stores the deduplicated specular-atlas texture slot.
    pub position: [f32; 4],
    /// `.x` intensity, `.y` max LOD, `.z` flags, `.w` SH2 source kind.
    pub params: [f32; 4],
    /// Probe SH2 coefficients in host order, padded to vec4 rows.
    pub sh2: [[f32; 4]; 9],
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn reflection_probe_metadata_row_size_matches_wgsl_storage_stride() {
        assert_eq!(size_of::<GpuReflectionProbeMetadata>(), 208);
    }
}
