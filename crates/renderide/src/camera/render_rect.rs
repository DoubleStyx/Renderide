//! Normalized camera viewport rect resolution for render-texture targets.

use crate::shared::RenderRect;

/// Pixel-space render rectangle resolved from a normalized host camera viewport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CameraRenderRect {
    /// Destination origin in render-texture storage coordinates.
    pub origin_px: (u32, u32),
    /// Pixel extent rendered by the camera.
    pub extent_px: (u32, u32),
}

impl CameraRenderRect {
    /// Resolves a normalized host viewport against a render texture extent.
    pub fn resolve(rect: RenderRect, target_extent_px: (u32, u32)) -> Option<Self> {
        if target_extent_px.0 == 0 || target_extent_px.1 == 0 {
            return None;
        }
        if !render_rect_fields_are_finite(rect) || rect.width <= 0.0 || rect.height <= 0.0 {
            return None;
        }

        let left = rect.x.clamp(0.0, 1.0);
        let bottom = rect.y.clamp(0.0, 1.0);
        let right = (rect.x + rect.width).clamp(0.0, 1.0);
        let top = (rect.y + rect.height).clamp(0.0, 1.0);
        if right <= left || top <= bottom {
            return None;
        }

        let target_width = target_extent_px.0 as f32;
        let target_height = target_extent_px.1 as f32;
        let origin_x = normalized_min_to_pixel(left, target_width);
        let origin_y = normalized_min_to_pixel(bottom, target_height);
        let end_x = normalized_max_to_pixel(right, target_width);
        let end_y = normalized_max_to_pixel(top, target_height);
        let width = end_x.saturating_sub(origin_x);
        let height = end_y.saturating_sub(origin_y);
        if width == 0 || height == 0 {
            return None;
        }

        Some(Self {
            origin_px: (origin_x, origin_y),
            extent_px: (width, height),
        })
    }

    /// Returns `true` when this rect covers the full render texture.
    #[inline]
    pub fn is_full_target(self, target_extent_px: (u32, u32)) -> bool {
        self.origin_px == (0, 0) && self.extent_px == target_extent_px
    }
}

/// Returns `true` when every normalized rect component can participate in pixel math.
fn render_rect_fields_are_finite(rect: RenderRect) -> bool {
    rect.x.is_finite() && rect.y.is_finite() && rect.width.is_finite() && rect.height.is_finite()
}

/// Converts a normalized lower edge to the first covered pixel.
fn normalized_min_to_pixel(value: f32, target_extent: f32) -> u32 {
    (value * target_extent).floor() as u32
}

/// Converts a normalized upper edge to the first pixel after the covered range.
fn normalized_max_to_pixel(value: f32, target_extent: f32) -> u32 {
    (value * target_extent).ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a normalized render rect for tests.
    fn rect(x: f32, y: f32, width: f32, height: f32) -> RenderRect {
        RenderRect {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn full_rect_resolves_to_target_extent() {
        let resolved = CameraRenderRect::resolve(rect(0.0, 0.0, 1.0, 1.0), (1280, 720))
            .expect("full rect should resolve");

        assert_eq!(resolved.origin_px, (0, 0));
        assert_eq!(resolved.extent_px, (1280, 720));
        assert!(resolved.is_full_target((1280, 720)));
    }

    #[test]
    fn centered_rect_resolves_to_pixel_origin_and_extent() {
        let resolved = CameraRenderRect::resolve(rect(0.25, 0.25, 0.5, 0.5), (800, 600))
            .expect("center rect should resolve");

        assert_eq!(resolved.origin_px, (200, 150));
        assert_eq!(resolved.extent_px, (400, 300));
        assert!(!resolved.is_full_target((800, 600)));
    }

    #[test]
    fn bottom_half_keeps_render_texture_storage_y_origin() {
        let resolved = CameraRenderRect::resolve(rect(0.0, 0.0, 1.0, 0.5), (256, 128))
            .expect("bottom half should resolve");

        assert_eq!(resolved.origin_px, (0, 0));
        assert_eq!(resolved.extent_px, (256, 64));
    }

    #[test]
    fn top_half_uses_positive_storage_y_origin() {
        let resolved = CameraRenderRect::resolve(rect(0.0, 0.5, 1.0, 0.5), (256, 128))
            .expect("top half should resolve");

        assert_eq!(resolved.origin_px, (0, 64));
        assert_eq!(resolved.extent_px, (256, 64));
    }

    #[test]
    fn out_of_bounds_rect_clips_to_target() {
        let resolved = CameraRenderRect::resolve(rect(-0.25, 0.5, 0.75, 0.75), (400, 200))
            .expect("clipped rect should resolve");

        assert_eq!(resolved.origin_px, (0, 100));
        assert_eq!(resolved.extent_px, (200, 100));
    }

    #[test]
    fn empty_or_non_finite_rects_do_not_resolve() {
        assert_eq!(
            CameraRenderRect::resolve(rect(0.0, 0.0, 0.0, 1.0), (100, 100)),
            None
        );
        assert_eq!(
            CameraRenderRect::resolve(rect(0.0, 0.0, 1.0, f32::NAN), (100, 100)),
            None
        );
        assert_eq!(
            CameraRenderRect::resolve(rect(0.0, 0.0, 1.0, 1.0), (0, 100)),
            None
        );
    }
}
