//! Transient texture/buffer extent and mip resolution against the active viewport.

use crate::gpu_resource::{TextureViewCache, TextureViewDescriptorKey};
use crate::render_graph::pool::TextureKey;
use crate::render_graph::resources::{BufferSizePolicy, TransientExtent};

/// Resolves backbuffer-relative descriptors into concrete extents using the current viewport.
pub(in crate::render_graph::compiled) fn resolve_transient_extent(
    extent: TransientExtent,
    viewport_px: (u32, u32),
    array_layers: u32,
) -> TransientExtent {
    match extent {
        TransientExtent::Backbuffer if array_layers > 1 => TransientExtent::MultiLayer {
            width: viewport_px.0.max(1),
            height: viewport_px.1.max(1),
            layers: array_layers,
        },
        TransientExtent::Backbuffer => TransientExtent::Custom {
            width: viewport_px.0.max(1),
            height: viewport_px.1.max(1),
        },
        TransientExtent::BackbufferDivisor { divisor } => {
            resolve_backbuffer_divisor_extent(divisor, 0, viewport_px, array_layers)
        }
        TransientExtent::BackbufferDivisorMip { divisor, mip } => {
            resolve_backbuffer_divisor_extent(divisor, mip, viewport_px, array_layers)
        }
        TransientExtent::BackbufferScaledMip { max_dim, mip } => {
            resolve_backbuffer_scaled_mip_extent(max_dim, mip, viewport_px, array_layers)
        }
        other => other,
    }
}

/// Clamps a requested mip count to the number of mips representable by the resolved extent.
pub(in crate::render_graph::compiled) fn clamp_mip_levels_for_transient_extent(
    requested_mips: u32,
    extent: TransientExtent,
    dimension: wgpu::TextureDimension,
    array_layers: u32,
) -> u32 {
    requested_mips
        .max(1)
        .min(max_mip_levels_for_transient_extent(
            extent,
            dimension,
            array_layers,
        ))
}

fn max_mip_levels_for_transient_extent(
    extent: TransientExtent,
    dimension: wgpu::TextureDimension,
    array_layers: u32,
) -> u32 {
    let (width, height, depth) = match extent {
        TransientExtent::Custom { width, height } => {
            (width.max(1), height.max(1), array_layers.max(1))
        }
        TransientExtent::MultiLayer {
            width,
            height,
            layers,
        } => (width.max(1), height.max(1), layers.max(1)),
        TransientExtent::Backbuffer
        | TransientExtent::BackbufferDivisor { .. }
        | TransientExtent::BackbufferDivisorMip { .. }
        | TransientExtent::BackbufferScaledMip { .. } => (1, 1, 1),
    };
    let max_axis = match dimension {
        wgpu::TextureDimension::D1 => width,
        wgpu::TextureDimension::D2 => width.max(height),
        wgpu::TextureDimension::D3 => width.max(height).max(depth),
    };
    u32::BITS - max_axis.max(1).leading_zeros()
}

/// Resolves a backbuffer extent divided by an integer and optionally shifted by mip level.
fn resolve_backbuffer_divisor_extent(
    divisor: u32,
    mip: u32,
    viewport_px: (u32, u32),
    array_layers: u32,
) -> TransientExtent {
    let divisor = divisor.max(1);
    let base_w = viewport_px.0.max(1).div_ceil(divisor).max(1);
    let base_h = viewport_px.1.max(1).div_ceil(divisor).max(1);
    let w = mip_axis_extent(base_w, mip);
    let h = mip_axis_extent(base_h, mip);
    if array_layers > 1 {
        TransientExtent::MultiLayer {
            width: w,
            height: h,
            layers: array_layers,
        }
    } else {
        TransientExtent::Custom {
            width: w,
            height: h,
        }
    }
}

/// Resolves a bloom-style backbuffer-relative mip extent without exceeding the current viewport.
fn resolve_backbuffer_scaled_mip_extent(
    max_dim: u32,
    mip: u32,
    viewport_px: (u32, u32),
    array_layers: u32,
) -> TransientExtent {
    let (vw, vh) = (viewport_px.0.max(1), viewport_px.1.max(1));
    let requested_h = power_of_two_at_or_below(max_dim.max(1));
    let viewport_h = power_of_two_at_or_below(vh);
    let base_h = requested_h.min(viewport_h).max(1);
    let ratio = f64::from(base_h) / f64::from(vh);
    let base_w = ((f64::from(vw) * ratio).round() as u32).max(1).min(vw);
    let w = mip_axis_extent(base_w, mip);
    let h = mip_axis_extent(base_h, mip);
    if array_layers > 1 {
        TransientExtent::MultiLayer {
            width: w,
            height: h,
            layers: array_layers,
        }
    } else {
        TransientExtent::Custom {
            width: w,
            height: h,
        }
    }
}

/// Returns the largest power of two less than or equal to `value`.
fn power_of_two_at_or_below(value: u32) -> u32 {
    let value = value.max(1);
    1_u32 << (u32::BITS - value.leading_zeros() - 1)
}

/// Halves a base extent by `mip`, clamping degenerate high mips to one pixel.
fn mip_axis_extent(base: u32, mip: u32) -> u32 {
    match base.checked_shr(mip) {
        Some(size) => size.max(1),
        None => 1,
    }
}

/// Clamps viewport dimensions to [`wgpu::Limits::max_texture_dimension_2d`] before transient texture
/// or buffer allocation from viewport-derived sizes.
pub(in crate::render_graph::compiled) fn clamp_viewport_for_transient_alloc(
    viewport_px: (u32, u32),
    max_texture_dimension_2d: u32,
) -> (u32, u32) {
    let ow = viewport_px.0.max(1);
    let oh = viewport_px.1.max(1);
    let w = ow.min(max_texture_dimension_2d);
    let h = oh.min(max_texture_dimension_2d);
    if w != ow || h != oh {
        logger::warn!(
            "transient alloc: viewport {}x{} clamped to {}x{} (max_texture_dimension_2d={max_texture_dimension_2d})",
            ow,
            oh,
            w,
            h,
        );
    }
    (w, h)
}

pub(in crate::render_graph::compiled) fn resolve_buffer_size(
    size_policy: BufferSizePolicy,
    _viewport_px: (u32, u32),
) -> u64 {
    match size_policy {
        BufferSizePolicy::Fixed(size) => size.max(1),
    }
}

pub(in crate::render_graph::compiled) fn create_transient_layer_views(
    texture: &wgpu::Texture,
    view_cache: &TextureViewCache,
    resource_generation: u64,
    key: TextureKey,
) -> Vec<wgpu::TextureView> {
    if key.dimension != wgpu::TextureDimension::D2 || key.array_layers <= 1 {
        return Vec::new();
    }
    (0..key.array_layers)
        .map(|layer| {
            let desc = wgpu::TextureViewDescriptor {
                label: Some("render-graph-transient-layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            };
            let (view, created) = view_cache.get_or_create(
                texture,
                resource_generation,
                TextureViewDescriptorKey::from_descriptor(&desc),
                desc.label,
            );
            if created {
                crate::profiling::note_resource_churn!(
                    TextureView,
                    "render_graph::transient_layer_view"
                );
            }
            view
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbuffer_scaled_mip_clamps_high_max_dimension_to_viewport_pot() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 2048,
                    mip: 0,
                },
                (1920, 1080),
                1,
            ),
            TransientExtent::Custom {
                width: 1820,
                height: 1024,
            }
        );
    }

    #[test]
    fn backbuffer_extent_resolves_to_viewport_or_multilayer_viewport() {
        assert_eq!(
            resolve_transient_extent(TransientExtent::Backbuffer, (0, 0), 1),
            TransientExtent::Custom {
                width: 1,
                height: 1,
            }
        );
        assert_eq!(
            resolve_transient_extent(TransientExtent::Backbuffer, (1280, 720), 2),
            TransientExtent::MultiLayer {
                width: 1280,
                height: 720,
                layers: 2,
            }
        );
    }

    #[test]
    fn backbuffer_divisor_extent_ceil_divides_viewport() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferDivisor { divisor: 2 },
                (1279, 721),
                1,
            ),
            TransientExtent::Custom {
                width: 640,
                height: 361,
            }
        );
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferDivisor { divisor: 4 },
                (1280, 720),
                2,
            ),
            TransientExtent::MultiLayer {
                width: 320,
                height: 180,
                layers: 2,
            }
        );
    }

    #[test]
    fn backbuffer_divisor_mip_halves_from_divided_base() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferDivisorMip { divisor: 2, mip: 2 },
                (1279, 721),
                1,
            ),
            TransientExtent::Custom {
                width: 160,
                height: 90,
            }
        );
    }

    #[test]
    fn backbuffer_scaled_mip_preserves_lower_configured_dimension() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 512,
                    mip: 0,
                },
                (1920, 1080),
                1,
            ),
            TransientExtent::Custom {
                width: 910,
                height: 512,
            }
        );
    }

    #[test]
    fn power_of_two_at_or_below_handles_zero_and_non_powers() {
        assert_eq!(power_of_two_at_or_below(0), 1);
        assert_eq!(power_of_two_at_or_below(1), 1);
        assert_eq!(power_of_two_at_or_below(3), 2);
        assert_eq!(power_of_two_at_or_below(1025), 1024);
    }

    #[test]
    fn mip_axis_extent_clamps_high_mips_to_one() {
        assert_eq!(mip_axis_extent(64, 0), 64);
        assert_eq!(mip_axis_extent(64, 3), 8);
        assert_eq!(mip_axis_extent(64, 32), 1);
    }

    #[test]
    fn backbuffer_scaled_mip_halves_from_clamped_base() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 2048,
                    mip: 1,
                },
                (800, 600),
                1,
            ),
            TransientExtent::Custom {
                width: 341,
                height: 256,
            }
        );
    }

    #[test]
    fn backbuffer_scaled_mip_handles_tiny_viewports_and_high_mips() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 2048,
                    mip: 5,
                },
                (32, 18),
                1,
            ),
            TransientExtent::Custom {
                width: 1,
                height: 1,
            }
        );
    }

    #[test]
    fn backbuffer_scaled_mip_preserves_multiview_layers() {
        assert_eq!(
            resolve_transient_extent(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 2048,
                    mip: 0,
                },
                (1920, 1080),
                2,
            ),
            TransientExtent::MultiLayer {
                width: 1820,
                height: 1024,
                layers: 2,
            }
        );
    }

    #[test]
    fn transient_mip_count_clamps_to_resolved_2d_extent() {
        assert_eq!(
            clamp_mip_levels_for_transient_extent(
                5,
                TransientExtent::Custom {
                    width: 16,
                    height: 9,
                },
                wgpu::TextureDimension::D2,
                1,
            ),
            5
        );
        assert_eq!(
            clamp_mip_levels_for_transient_extent(
                5,
                TransientExtent::Custom {
                    width: 8,
                    height: 8,
                },
                wgpu::TextureDimension::D2,
                1,
            ),
            4
        );
        assert_eq!(
            clamp_mip_levels_for_transient_extent(
                5,
                TransientExtent::Custom {
                    width: 1,
                    height: 1,
                },
                wgpu::TextureDimension::D2,
                1,
            ),
            1
        );
    }

    #[test]
    fn transient_mip_count_never_returns_zero() {
        assert_eq!(
            clamp_mip_levels_for_transient_extent(
                0,
                TransientExtent::Custom {
                    width: 16,
                    height: 16,
                },
                wgpu::TextureDimension::D2,
                1,
            ),
            1
        );
    }

    #[test]
    fn transient_mip_count_uses_array_layers_only_for_3d_textures() {
        let extent = TransientExtent::MultiLayer {
            width: 4,
            height: 4,
            layers: 16,
        };
        assert_eq!(
            clamp_mip_levels_for_transient_extent(8, extent, wgpu::TextureDimension::D2, 16),
            3
        );
        assert_eq!(
            clamp_mip_levels_for_transient_extent(8, extent, wgpu::TextureDimension::D3, 16),
            5
        );
    }

    #[test]
    fn clamp_viewport_for_transient_alloc_applies_minimum_and_device_limit() {
        assert_eq!(clamp_viewport_for_transient_alloc((0, 0), 1024), (1, 1));
        assert_eq!(
            clamp_viewport_for_transient_alloc((2048, 512), 1024),
            (1024, 512)
        );
    }

    #[test]
    fn resolve_buffer_size_keeps_fixed_buffers_nonzero() {
        assert_eq!(
            resolve_buffer_size(BufferSizePolicy::Fixed(0), (1920, 1080)),
            1
        );
        assert_eq!(
            resolve_buffer_size(BufferSizePolicy::Fixed(4096), (1920, 1080)),
            4096
        );
    }
}
