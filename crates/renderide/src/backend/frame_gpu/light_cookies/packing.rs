use crate::gpu::GpuLightCookieRect;

use super::atlas::LightCookieAtlasExtent;

/// Source cookie rectangle request before atlas packing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LightCookiePackItem {
    /// Metadata row that will point at the packed rectangle.
    pub(super) rect_index: u32,
    /// Source width in texels.
    pub(super) width: u32,
    /// Source height in texels.
    pub(super) height: u32,
}

/// Integer texel rectangle inside a packed atlas.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct LightCookieAtlasRect {
    /// Left texel.
    pub(super) x: u32,
    /// Top texel.
    pub(super) y: u32,
    /// Width in texels.
    pub(super) width: u32,
    /// Height in texels.
    pub(super) height: u32,
}

impl LightCookieAtlasRect {
    /// Converts this texel rectangle to normalized atlas metadata.
    pub(super) fn metadata(self, extent: LightCookieAtlasExtent) -> GpuLightCookieRect {
        let extent = extent.sanitized();
        GpuLightCookieRect {
            origin_scale: [
                self.x as f32 / extent.width as f32,
                self.y as f32 / extent.height as f32,
                self.width as f32 / extent.width as f32,
                self.height as f32 / extent.height as f32,
            ],
        }
    }
}

/// Packed output for one cookie source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LightCookiePackedRect {
    /// Metadata row that points at `rect`.
    pub(super) rect_index: u32,
    /// Packed atlas rectangle.
    pub(super) rect: LightCookieAtlasRect,
}

/// Packed atlas layout and its used extent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LightCookiePackPlan {
    /// Smallest atlas extent containing all packed rectangles.
    pub(super) extent: LightCookieAtlasExtent,
    /// Successfully packed rectangles.
    pub(super) rects: Vec<LightCookiePackedRect>,
    /// Number of items skipped because they could not fit inside the device atlas limit.
    pub(super) overflow_count: usize,
}

/// Greedily packs source cookies into rows no wider or taller than `max_extent`.
pub(super) fn pack_light_cookie_rects(
    items: &[LightCookiePackItem],
    max_extent: u32,
) -> LightCookiePackPlan {
    let max_extent = max_extent.max(1);
    let mut sorted = items
        .iter()
        .copied()
        .filter(|item| item.width > 0 && item.height > 0)
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| {
        b.height
            .cmp(&a.height)
            .then_with(|| b.width.cmp(&a.width))
            .then_with(|| a.rect_index.cmp(&b.rect_index))
    });

    let mut rects = Vec::with_capacity(sorted.len());
    let mut overflow_count = items.len().saturating_sub(sorted.len());
    let mut x = 0u32;
    let mut y = 0u32;
    let mut row_height = 0u32;
    let mut used_width = 0u32;

    for item in sorted {
        if item.width > max_extent || item.height > max_extent {
            overflow_count += 1;
            continue;
        }
        if x > 0 && x.saturating_add(item.width) > max_extent {
            y = y.saturating_add(row_height);
            x = 0;
            row_height = 0;
        }
        if y.saturating_add(item.height) > max_extent {
            overflow_count += 1;
            continue;
        }
        rects.push(LightCookiePackedRect {
            rect_index: item.rect_index,
            rect: LightCookieAtlasRect {
                x,
                y,
                width: item.width,
                height: item.height,
            },
        });
        x = x.saturating_add(item.width);
        used_width = used_width.max(x);
        row_height = row_height.max(item.height);
    }

    LightCookiePackPlan {
        extent: LightCookieAtlasExtent {
            width: used_width.max(1),
            height: y.saturating_add(row_height).max(1),
        },
        rects,
        overflow_count,
    }
}
