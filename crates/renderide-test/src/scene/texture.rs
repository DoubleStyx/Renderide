//! Deterministic procedural texture (checkerboard) used to color the test sphere.
//!
//! Currently retained as a building block for a richer textured-material golden test in the
//! future; the active sphere golden uses the renderer's `DebugWorldNormals` fallback shader and
//! does not bind a texture.

/// A small RGBA8 texture in row-major order (top-down).
#[expect(
    dead_code,
    reason = "test helper type; not all test scenarios upload a texture"
)]
#[derive(Clone, Debug)]
pub(crate) struct CheckerboardTexture {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Tightly packed RGBA8 bytes (`width * height * 4` long).
    pub pixels: Vec<u8>,
}

#[cfg_attr(not(test), expect(dead_code, reason = "only used by unit tests today"))]
impl CheckerboardTexture {
    /// Generates a checkerboard with `tile` pixels per cell and the two given colors.
    ///
    /// Choosing the two colors to be visually distinct (e.g. magenta and dark blue) produces a
    /// stable golden image that highlights UV mapping bugs immediately.
    pub(crate) fn generate(
        width: u32,
        height: u32,
        tile: u32,
        color_a: [u8; 4],
        color_b: [u8; 4],
    ) -> Self {
        let tile = tile.max(1);
        let mut pixels = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for y in 0..height {
            for x in 0..width {
                let cx = x / tile;
                let cy = y / tile;
                let c = if (cx + cy).is_multiple_of(2) {
                    color_a
                } else {
                    color_b
                };
                pixels.extend_from_slice(&c);
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CheckerboardTexture;

    #[test]
    fn checkerboard_is_deterministic_for_same_inputs() {
        let a = CheckerboardTexture::generate(32, 32, 8, [255, 0, 255, 255], [16, 16, 64, 255]);
        let b = CheckerboardTexture::generate(32, 32, 8, [255, 0, 255, 255], [16, 16, 64, 255]);
        assert_eq!(a.pixels, b.pixels);
    }

    #[test]
    fn checkerboard_size_matches() {
        let t = CheckerboardTexture::generate(16, 32, 4, [0; 4], [255; 4]);
        assert_eq!(t.pixels.len(), 16 * 32 * 4);
    }

    #[test]
    fn first_tile_uses_color_a() {
        let a = [200u8, 50, 100, 255];
        let b = [10u8, 220, 30, 255];
        let t = CheckerboardTexture::generate(8, 8, 4, a, b);
        assert_eq!(&t.pixels[0..4], &a);
    }
}
