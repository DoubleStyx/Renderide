//! Resource-shape policies used by [`super::Pool`] to specialise the generic LRU + free-list
//! bookkeeping for textures and buffers.
//!
//! A [`PoolKind`] describes the cached value shape (e.g. `(Texture, View)` for textures,
//! `(Buffer, size)` for buffers) and the key that identifies an alias-equivalent slot. The
//! resource-specific bits (limit validation, GPU object construction, lease assembly, "is this
//! cached resource still valid?" checks) live in the texture / buffer specialisations of
//! [`super::TransientPool`] rather than on the trait -- that keeps `Pool<P>` a pure storage primitive
//! and lets the call sites stay readable.

use std::hash::Hash;

use super::super::resources::{BufferSizePolicy, TransientExtent};

/// Bound trait describing the value/key shape stored in a [`super::Pool`].
pub trait PoolKind: 'static {
    /// Alias-equivalence key. Two resources may share a slot only when their keys compare equal.
    type Key: Eq + Hash + Copy + std::fmt::Debug;
    /// Cached value held by each pool entry; typically a tuple of GPU handles.
    type Value: std::fmt::Debug + Default;
}

/// Per-texture-slot value: the GPU texture and its default full-resource view, when attached.
#[derive(Debug, Default)]
pub(super) struct TextureSlotValue {
    /// Cached GPU texture, if attached.
    pub texture: Option<wgpu::Texture>,
    /// Cached default `wgpu::TextureView` over the full resource, if attached.
    pub view: Option<wgpu::TextureView>,
}

impl TextureSlotValue {
    /// Returns whether this slot currently holds GPU resources.
    pub fn is_present(&self) -> bool {
        self.texture.is_some()
    }

    /// Drops cached GPU resources, leaving the slot empty.
    pub fn clear(&mut self) {
        self.texture = None;
        self.view = None;
    }
}

/// Per-buffer-slot value: the GPU buffer and the byte size at which it was allocated.
#[derive(Debug, Default)]
pub(super) struct BufferSlotValue {
    /// Cached GPU buffer, if attached.
    pub buffer: Option<wgpu::Buffer>,
    /// Allocation size in bytes for the cached buffer (0 when unattached).
    pub size: u64,
}

impl BufferSlotValue {
    /// Returns whether this slot currently holds a GPU buffer.
    pub fn is_present(&self) -> bool {
        self.buffer.is_some()
    }

    /// Drops the cached buffer, leaving the slot empty.
    pub fn clear(&mut self) {
        self.buffer = None;
        self.size = 0;
    }
}

/// Concrete texture allocation key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureKey {
    /// Texture format.
    pub format: wgpu::TextureFormat,
    /// Resolved extent policy.
    pub extent: TransientExtent,
    /// Mip count.
    pub mip_levels: u32,
    /// Sample count.
    pub sample_count: u32,
    /// Texture dimension.
    pub dimension: wgpu::TextureDimension,
    /// Array-layer count.
    pub array_layers: u32,
    /// Usage bitset.
    pub usage_bits: u64,
}

/// Concrete buffer allocation key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferKey {
    /// Size policy.
    pub size_policy: BufferSizePolicy,
    /// Usage bitset.
    pub usage_bits: u64,
}

/// Marker type for texture pools.
#[derive(Debug)]
pub(super) struct TextureKind;

impl PoolKind for TextureKind {
    type Key = TextureKey;
    type Value = TextureSlotValue;
}

/// Marker type for buffer pools.
#[derive(Debug)]
pub(super) struct BufferKind;

impl PoolKind for BufferKind {
    type Key = BufferKey;
    type Value = BufferSlotValue;
}

/// Resolves the (width, height, layers) triple for `key.extent`. Backbuffer-relative extents
/// resolve to `(1, 1, array_layers)` because their real size comes from the per-view viewport
/// pre-clamp; this helper is only used for limit validation of the parts of the key that are not
/// viewport-derived.
pub(super) fn texture_key_dims(key: TextureKey) -> (u32, u32, u32) {
    match key.extent {
        TransientExtent::Backbuffer | TransientExtent::BackbufferScaledMip { .. } => {
            (1, 1, key.array_layers)
        }
        TransientExtent::Custom { width, height } => (width, height, key.array_layers),
        TransientExtent::MultiLayer {
            width,
            height,
            layers,
        } => (width, height, layers),
    }
}

/// Allocates a texture matching `key` and a default full-resource view.
pub(super) fn create_texture_and_view(
    device: &wgpu::Device,
    key: TextureKey,
    label: &'static str,
    usage: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let (width, height, layers) = texture_key_dims(key);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: layers.max(1),
        },
        mip_level_count: key.mip_levels.max(1),
        sample_count: key.sample_count.max(1),
        dimension: key.dimension,
        format: key.format,
        usage,
        view_formats: &[],
    });
    let view = if key.dimension == wgpu::TextureDimension::D2 && layers > 1 {
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(layers.max(1)),
            ..Default::default()
        })
    } else {
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    };
    (texture, view)
}

/// Allocates a buffer matching `usage` and `size` bytes.
pub(super) fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    usage: wgpu::BufferUsages,
    size: u64,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(1),
        usage,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture_key(extent: TransientExtent, array_layers: u32) -> TextureKey {
        TextureKey {
            format: wgpu::TextureFormat::Rgba8Unorm,
            extent,
            mip_levels: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            array_layers,
            usage_bits: u64::from(wgpu::TextureUsages::RENDER_ATTACHMENT.bits()),
        }
    }

    #[test]
    fn texture_key_dims_uses_array_layers_for_backbuffer_relative_extents() {
        assert_eq!(
            texture_key_dims(texture_key(TransientExtent::Backbuffer, 2)),
            (1, 1, 2)
        );
        assert_eq!(
            texture_key_dims(texture_key(
                TransientExtent::BackbufferScaledMip {
                    max_dim: 1024,
                    mip: 2,
                },
                4,
            )),
            (1, 1, 4)
        );
    }

    #[test]
    fn texture_key_dims_preserves_custom_dimensions() {
        assert_eq!(
            texture_key_dims(texture_key(
                TransientExtent::Custom {
                    width: 320,
                    height: 180,
                },
                3,
            )),
            (320, 180, 3)
        );
    }

    #[test]
    fn texture_key_dims_prefers_multilayer_extent_layers() {
        assert_eq!(
            texture_key_dims(texture_key(
                TransientExtent::MultiLayer {
                    width: 64,
                    height: 32,
                    layers: 6,
                },
                2,
            )),
            (64, 32, 6)
        );
    }

    #[test]
    fn empty_slot_values_report_absent_and_clear_to_default_state() {
        let mut texture_slot = TextureSlotValue::default();
        texture_slot.clear();
        assert!(!texture_slot.is_present());

        let mut buffer_slot = BufferSlotValue {
            buffer: None,
            size: 4096,
        };
        buffer_slot.clear();
        assert!(!buffer_slot.is_present());
        assert_eq!(buffer_slot.size, 0);
    }
}
