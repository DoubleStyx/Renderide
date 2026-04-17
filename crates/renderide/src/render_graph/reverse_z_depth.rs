//! Depth clear and compare conventions for the main **reverse-Z** forward pass and matching
//! raster pipelines (near fragments win with greater depth values in NDC).

/// Clear value for the main forward depth attachment (`Depth32Float`).
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
}
