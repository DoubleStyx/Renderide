//! Depth clear and compare conventions for the main **reverse-Z** forward pass and matching
//! raster pipelines (near fragments win with greater depth values in NDC).

/// Chooses the main forward depth-stencil attachment format.
///
/// `Depth32FloatStencil8` preserves the old 32-bit depth precision when the optional WebGPU
/// feature was enabled. `Depth24PlusStencil8` is the portable stencil-capable fallback.
pub fn main_forward_depth_stencil_format(features: wgpu::Features) -> wgpu::TextureFormat {
    if features.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8) {
        wgpu::TextureFormat::Depth32FloatStencil8
    } else {
        wgpu::TextureFormat::Depth24PlusStencil8
    }
}

/// Clear value for the main forward depth attachment.
///
/// With reverse-Z and a perspective projection that maps **near** clip to depth **1** and **far**
/// to **0**, clearing to `0` initializes the buffer to the farthest representable depth so the
/// first fragment can always pass a [`MAIN_FORWARD_DEPTH_COMPARE`] test.
pub const MAIN_FORWARD_DEPTH_CLEAR: f32 = 0.0;

/// Depth comparison for reverse-Z: fragments closer to the camera have **greater** interpolated
/// depth, so new geometry replaces the buffer when it is greater than or equal to the stored
/// value.
pub const MAIN_FORWARD_DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::GreaterEqual;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_forward_reverse_z_depth_invariants() {
        assert_eq!(MAIN_FORWARD_DEPTH_CLEAR, 0.0);
        assert_eq!(
            MAIN_FORWARD_DEPTH_COMPARE,
            wgpu::CompareFunction::GreaterEqual
        );
    }

    #[test]
    fn main_forward_depth_stencil_format_prefers_32f_when_enabled() {
        assert_eq!(
            main_forward_depth_stencil_format(wgpu::Features::DEPTH32FLOAT_STENCIL8),
            wgpu::TextureFormat::Depth32FloatStencil8
        );
        assert_eq!(
            main_forward_depth_stencil_format(wgpu::Features::empty()),
            wgpu::TextureFormat::Depth24PlusStencil8
        );
    }
}
