//! Deterministic CPU texture generators for visual integration cases.

use image::{Rgba, RgbaImage};

/// Builds a high-contrast checker texture with colored axes useful for UV-orientation checks.
pub fn checker_rgba(width: u32, height: u32, cells: u32, a: [u8; 4], b: [u8; 4]) -> RgbaImage {
    let cells = cells.max(1);
    let cell_w = (width / cells).max(1);
    let cell_h = (height / cells).max(1);
    let mut img = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / cell_w) + (y / cell_h)).is_multiple_of(2);
            let mut color = if checker { a } else { b };
            if x < cell_w / 2 {
                color = [230, color[1] / 2, color[2] / 2, color[3]];
            }
            if y < cell_h / 2 {
                color = [color[0] / 2, 230, color[2] / 2, color[3]];
            }
            img.put_pixel(x, y, Rgba(color));
        }
    }
    img
}

/// Builds an RGBA UV ramp: red tracks U and green tracks V.
pub fn uv_ramp_rgba(width: u32, height: u32) -> RgbaImage {
    let mut img = RgbaImage::new(width, height);
    let max_x = width.saturating_sub(1).max(1);
    let max_y = height.saturating_sub(1).max(1);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / max_x) as u8;
            let g = ((y * 255) / max_y) as u8;
            img.put_pixel(x, y, Rgba([r, g, 180, 255]));
        }
    }
    img
}

/// Builds a binary alpha mask with opaque rings and transparent gaps.
pub fn alpha_rings_rgba(width: u32, height: u32) -> RgbaImage {
    let mut img = RgbaImage::new(width, height);
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    let max_radius = cx.min(cy).max(1.0);
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let r = (dx * dx + dy * dy).sqrt() / max_radius;
            let ring = (r * 8.0).floor() as u32;
            let alpha = if r < 0.95 && ring.is_multiple_of(2) {
                255
            } else {
                0
            };
            let u = ((x * 255) / width.saturating_sub(1).max(1)) as u8;
            let v = ((y * 255) / height.saturating_sub(1).max(1)) as u8;
            let band = (ring * 31).min(255) as u8;
            img.put_pixel(x, y, Rgba([255u8.saturating_sub(u / 2), v, band, alpha]));
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_is_not_flat() {
        let img = checker_rgba(32, 32, 4, [255, 255, 255, 255], [0, 0, 0, 255]);
        let first = img.get_pixel(0, 0);
        assert!(img.pixels().any(|p| p != first));
    }

    #[test]
    fn uv_ramp_reaches_corners() {
        let img = uv_ramp_rgba(16, 16);
        assert_eq!(img.get_pixel(0, 0).0[0], 0);
        assert_eq!(img.get_pixel(15, 15).0[0], 255);
        assert_eq!(img.get_pixel(15, 15).0[1], 255);
    }

    #[test]
    fn alpha_rings_have_opaque_and_transparent_pixels() {
        let img = alpha_rings_rgba(64, 64);
        assert!(img.pixels().any(|p| p.0[3] == 0));
        assert!(img.pixels().any(|p| p.0[3] == 255));
    }
}
