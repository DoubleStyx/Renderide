//! Persistent ping-pong resource registry for render-graph history slots.
//!
//! A **history slot** is a pair of GPU resources (textures or buffers) that the graph's
//! [`crate::render_graph::resources::ImportSource::PingPong`] and
//! [`crate::render_graph::resources::BufferImportSource::PingPong`] reference by
//! [`crate::render_graph::resources::HistorySlotId`]. The previous frame writes slot index
//! `current()`; the next frame swaps and writes the other half. This structure keeps both halves
//! alive across frames so the read of last frame's data is just a lookup, not a re-copy.
//!
//! Hi-Z registers view-scoped texture history here while keeping CPU snapshots and readback policy
//! on [`crate::backend::OcclusionSystem`]. Future TAA, SSR, or cached-shadow systems can declare
//! their persistent resources through the same owner instead of hand-rolling ping-pong pairs.

use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::render_graph::{HistorySlotId, OcclusionViewId};

/// Errors returned by [`HistoryRegistry`] registration APIs.
#[derive(Debug, thiserror::Error)]
pub enum HistoryRegistryError {
    /// A slot with this id was already registered with a different resource kind.
    ///
    /// The registry keeps textures and buffers in separate tables; re-registering the same id
    /// under the opposite kind is a programmer error.
    #[error("history slot {id:?} already registered as {other_kind}")]
    KindMismatch {
        /// Slot id whose registration conflicted.
        id: HistorySlotId,
        /// Kind the slot is already registered as ("texture" or "buffer").
        other_kind: &'static str,
    },
}

/// Scope for a persistent history slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HistoryResourceScope {
    /// One global resource pair shared by all views.
    Global,
    /// One resource pair for one logical occlusion/render view.
    View(OcclusionViewId),
}

/// Concrete key used by the registry's texture and buffer maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct HistoryResourceKey {
    /// Stable subsystem slot id.
    id: HistorySlotId,
    /// Resource ownership scope.
    scope: HistoryResourceScope,
}

impl HistoryResourceKey {
    /// Builds a concrete history key.
    const fn new(id: HistorySlotId, scope: HistoryResourceScope) -> Self {
        Self { id, scope }
    }
}

/// Texture history slot declaration.
#[derive(Clone, Debug)]
pub struct TextureHistorySpec {
    /// Debug label used for the allocated `wgpu::Texture`.
    pub label: &'static str,
    /// Texture format.
    pub format: wgpu::TextureFormat,
    /// Texture extent.
    pub extent: wgpu::Extent3d,
    /// Texture usage flags.
    pub usage: wgpu::TextureUsages,
    /// Mip level count.
    pub mip_level_count: u32,
    /// Sample count.
    pub sample_count: u32,
    /// Texture dimension.
    pub dimension: wgpu::TextureDimension,
}

/// Buffer history slot declaration.
#[derive(Clone, Debug)]
pub struct BufferHistorySpec {
    /// Debug label used for the allocated `wgpu::Buffer`.
    pub label: &'static str,
    /// Byte size.
    pub size: u64,
    /// Buffer usage flags.
    pub usage: wgpu::BufferUsages,
}

/// Per-layer/per-mip views created for one texture-history allocation.
#[derive(Clone, Debug)]
pub struct HistoryTextureMipViews {
    /// Views grouped as `layers[layer][mip]`.
    layers: Arc<[Arc<[wgpu::TextureView]>]>,
    /// Number of mip levels represented by every layer.
    mip_level_count: u32,
}

impl HistoryTextureMipViews {
    /// Returns the number of array layers represented by this view table.
    pub fn layer_count(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Returns the number of mip views represented for every layer.
    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    /// Returns all mip views for one array layer.
    pub fn layer_mip_views(&self, layer: u32) -> Option<&[wgpu::TextureView]> {
        self.layers.get(layer as usize).map(|views| views.as_ref())
    }
}

/// Pure shape data for texture-history view tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryTextureViewShape {
    /// Number of array layers that should receive per-mip D2 views.
    pub layer_count: u32,
    /// Number of mip levels represented per layer.
    pub mip_level_count: u32,
}

/// One half of a ping-pong texture pair: a texture plus reusable views.
#[derive(Clone)]
pub struct HistoryTexture {
    /// Allocated texture.
    pub texture: wgpu::Texture,
    /// Default full-resource view.
    pub view: wgpu::TextureView,
    /// Per-layer/per-mip views for passes that write or sample explicit subresources.
    pub mip_views: HistoryTextureMipViews,
}

/// Ping-pong texture history slot.
pub struct TextureHistorySlot {
    spec: TextureHistorySpec,
    pair: [Option<HistoryTexture>; 2],
}

impl TextureHistorySlot {
    fn ensure(&mut self, device: &wgpu::Device) {
        for slot in &mut self.pair {
            if slot.is_none() {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(self.spec.label),
                    size: self.spec.extent,
                    mip_level_count: self.spec.mip_level_count.max(1),
                    sample_count: self.spec.sample_count.max(1),
                    dimension: self.spec.dimension,
                    format: self.spec.format,
                    usage: self.spec.usage,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mip_views = create_texture_history_mip_views(&texture, &self.spec);
                *slot = Some(HistoryTexture {
                    texture,
                    view,
                    mip_views,
                });
            }
        }
    }

    /// Current spec used for allocation; compared against reallocation requests.
    pub fn spec(&self) -> &TextureHistorySpec {
        &self.spec
    }

    /// Borrows a half of the ping-pong pair; returns [`None`] until the first
    /// [`HistoryRegistry::ensure_resources`] call has allocated it.
    pub fn half(&self, index: usize) -> Option<&HistoryTexture> {
        self.pair.get(index)?.as_ref()
    }
}

/// Ping-pong buffer history slot.
pub struct BufferHistorySlot {
    spec: BufferHistorySpec,
    pair: [Option<wgpu::Buffer>; 2],
}

impl BufferHistorySlot {
    fn ensure(&mut self, device: &wgpu::Device) {
        for slot in &mut self.pair {
            if slot.is_none() {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(self.spec.label),
                    size: self.spec.size.max(1),
                    usage: self.spec.usage,
                    mapped_at_creation: false,
                });
                *slot = Some(buffer);
            }
        }
    }

    /// Current spec used for allocation.
    pub fn spec(&self) -> &BufferHistorySpec {
        &self.spec
    }

    /// Borrows a half of the ping-pong pair; returns [`None`] until the first
    /// [`HistoryRegistry::ensure_resources`] call has allocated it.
    pub fn half(&self, index: usize) -> Option<&wgpu::Buffer> {
        self.pair.get(index)?.as_ref()
    }
}

/// Thread-safe owner of all history slots.
///
/// Subsystems register their slots once at init. The registry is advanced once per frame via
/// [`HistoryRegistry::advance_frame`]; [`HistoryRegistry::current_index`] and
/// [`HistoryRegistry::previous_index`] return the two halves of any slot pair. Reallocation of
/// slots with changed specs is explicit: call [`HistoryRegistry::update_texture_spec`] or
/// [`HistoryRegistry::update_buffer_spec`] with the new shape and drop-then-resize happens on the
/// next [`HistoryRegistry::ensure_resources`].
#[derive(Default)]
pub struct HistoryRegistry {
    textures: HashMap<HistoryResourceKey, Arc<Mutex<TextureHistorySlot>>>,
    buffers: HashMap<HistoryResourceKey, Arc<Mutex<BufferHistorySlot>>>,
    frame_counter: u64,
}

impl HistoryRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a texture slot under `id`. Idempotent when called with the same spec; when
    /// called with a different spec, the stored spec is updated and both halves are dropped so
    /// the next [`HistoryRegistry::ensure_resources`] reallocates.
    pub fn register_texture(
        &mut self,
        id: HistorySlotId,
        spec: TextureHistorySpec,
    ) -> Result<(), HistoryRegistryError> {
        self.register_texture_scoped(id, HistoryResourceScope::Global, spec)
    }

    /// Registers a texture slot under `id` and `scope`. See [`Self::register_texture`].
    pub fn register_texture_scoped(
        &mut self,
        id: HistorySlotId,
        scope: HistoryResourceScope,
        spec: TextureHistorySpec,
    ) -> Result<(), HistoryRegistryError> {
        let key = HistoryResourceKey::new(id, scope);
        if self.buffers.contains_key(&key) {
            return Err(HistoryRegistryError::KindMismatch {
                id,
                other_kind: "buffer",
            });
        }
        match self.textures.get(&key) {
            Some(existing) => {
                let mut guard = existing.lock();
                if !texture_specs_equivalent(&guard.spec, &spec) {
                    guard.spec = spec;
                    guard.pair = [None, None];
                }
            }
            None => {
                self.textures.insert(
                    key,
                    Arc::new(Mutex::new(TextureHistorySlot {
                        spec,
                        pair: [None, None],
                    })),
                );
            }
        }
        Ok(())
    }

    /// Registers a buffer slot under `id`. Idempotent when called with the same spec; when
    /// called with a different spec, the stored spec is updated and both halves are dropped so
    /// the next [`HistoryRegistry::ensure_resources`] reallocates.
    pub fn register_buffer(
        &mut self,
        id: HistorySlotId,
        spec: BufferHistorySpec,
    ) -> Result<(), HistoryRegistryError> {
        self.register_buffer_scoped(id, HistoryResourceScope::Global, spec)
    }

    /// Registers a buffer slot under `id` and `scope`. See [`Self::register_buffer`].
    pub fn register_buffer_scoped(
        &mut self,
        id: HistorySlotId,
        scope: HistoryResourceScope,
        spec: BufferHistorySpec,
    ) -> Result<(), HistoryRegistryError> {
        let key = HistoryResourceKey::new(id, scope);
        if self.textures.contains_key(&key) {
            return Err(HistoryRegistryError::KindMismatch {
                id,
                other_kind: "texture",
            });
        }
        match self.buffers.get(&key) {
            Some(existing) => {
                let mut guard = existing.lock();
                if !buffer_specs_equivalent(&guard.spec, &spec) {
                    guard.spec = spec;
                    guard.pair = [None, None];
                }
            }
            None => {
                self.buffers.insert(
                    key,
                    Arc::new(Mutex::new(BufferHistorySlot {
                        spec,
                        pair: [None, None],
                    })),
                );
            }
        }
        Ok(())
    }

    /// Updates the spec for an already-registered texture slot, dropping its halves so they are
    /// reallocated on the next [`HistoryRegistry::ensure_resources`]. Returns `false` when no
    /// slot was registered under `id`.
    pub fn update_texture_spec(&mut self, id: HistorySlotId, spec: TextureHistorySpec) -> bool {
        self.update_texture_spec_scoped(id, HistoryResourceScope::Global, spec)
    }

    /// Updates the spec for a scoped texture slot. See [`Self::update_texture_spec`].
    pub fn update_texture_spec_scoped(
        &mut self,
        id: HistorySlotId,
        scope: HistoryResourceScope,
        spec: TextureHistorySpec,
    ) -> bool {
        match self.textures.get(&HistoryResourceKey::new(id, scope)) {
            Some(slot) => {
                let mut guard = slot.lock();
                guard.spec = spec;
                guard.pair = [None, None];
                true
            }
            None => false,
        }
    }

    /// Updates the spec for an already-registered buffer slot. See [`Self::update_texture_spec`].
    pub fn update_buffer_spec(&mut self, id: HistorySlotId, spec: BufferHistorySpec) -> bool {
        self.update_buffer_spec_scoped(id, HistoryResourceScope::Global, spec)
    }

    /// Updates the spec for a scoped buffer slot. See [`Self::update_buffer_spec`].
    pub fn update_buffer_spec_scoped(
        &mut self,
        id: HistorySlotId,
        scope: HistoryResourceScope,
        spec: BufferHistorySpec,
    ) -> bool {
        match self.buffers.get(&HistoryResourceKey::new(id, scope)) {
            Some(slot) => {
                let mut guard = slot.lock();
                guard.spec = spec;
                guard.pair = [None, None];
                true
            }
            None => false,
        }
    }

    /// Returns the shared handle for a texture slot, or [`None`] when unregistered.
    pub fn texture_slot(&self, id: HistorySlotId) -> Option<Arc<Mutex<TextureHistorySlot>>> {
        self.texture_slot_scoped(id, HistoryResourceScope::Global)
    }

    /// Returns the shared handle for a scoped texture slot, or [`None`] when unregistered.
    pub fn texture_slot_scoped(
        &self,
        id: HistorySlotId,
        scope: HistoryResourceScope,
    ) -> Option<Arc<Mutex<TextureHistorySlot>>> {
        self.textures
            .get(&HistoryResourceKey::new(id, scope))
            .cloned()
    }

    /// Returns the shared handle for a buffer slot, or [`None`] when unregistered.
    pub fn buffer_slot(&self, id: HistorySlotId) -> Option<Arc<Mutex<BufferHistorySlot>>> {
        self.buffer_slot_scoped(id, HistoryResourceScope::Global)
    }

    /// Returns the shared handle for a scoped buffer slot, or [`None`] when unregistered.
    pub fn buffer_slot_scoped(
        &self,
        id: HistorySlotId,
        scope: HistoryResourceScope,
    ) -> Option<Arc<Mutex<BufferHistorySlot>>> {
        self.buffers
            .get(&HistoryResourceKey::new(id, scope))
            .cloned()
    }

    /// Allocates every slot's resources against `device` if they are not already present.
    /// Idempotent; cheap when all halves are resident.
    pub fn ensure_resources(&self, device: &wgpu::Device) {
        for slot in self.textures.values() {
            slot.lock().ensure(device);
        }
        for slot in self.buffers.values() {
            slot.lock().ensure(device);
        }
    }

    /// Advances the frame counter. Call once per tick, before reading or writing history slots.
    pub fn advance_frame(&mut self) {
        self.frame_counter = self.frame_counter.wrapping_add(1);
    }

    /// Current frame counter value, for callers that want to plot or log registry churn.
    pub fn frame_counter(&self) -> u64 {
        self.frame_counter
    }

    /// Index (0 or 1) of the slot half that is current-frame writable.
    pub fn current_index(&self) -> usize {
        (self.frame_counter & 1) as usize
    }

    /// Index (0 or 1) of the slot half that contains the previous frame's data.
    pub fn previous_index(&self) -> usize {
        ((self.frame_counter.wrapping_add(1)) & 1) as usize
    }

    /// Number of registered texture slots.
    pub fn texture_slot_count(&self) -> usize {
        self.textures.len()
    }

    /// Number of registered buffer slots.
    pub fn buffer_slot_count(&self) -> usize {
        self.buffers.len()
    }
}

fn texture_specs_equivalent(a: &TextureHistorySpec, b: &TextureHistorySpec) -> bool {
    a.label == b.label
        && a.format == b.format
        && a.extent == b.extent
        && a.usage == b.usage
        && a.mip_level_count == b.mip_level_count
        && a.sample_count == b.sample_count
        && a.dimension == b.dimension
}

fn buffer_specs_equivalent(a: &BufferHistorySpec, b: &BufferHistorySpec) -> bool {
    a.label == b.label && a.size == b.size && a.usage == b.usage
}

/// Computes the per-layer/per-mip view table shape for a texture history spec.
fn texture_history_view_shape(spec: &TextureHistorySpec) -> HistoryTextureViewShape {
    let layer_count = match spec.dimension {
        wgpu::TextureDimension::D2 => spec.extent.depth_or_array_layers.max(1),
        wgpu::TextureDimension::D1 | wgpu::TextureDimension::D3 => 0,
    };
    HistoryTextureViewShape {
        layer_count,
        mip_level_count: spec.mip_level_count.max(1),
    }
}

/// Creates D2 per-mip views for every array layer of a texture history allocation.
fn create_texture_history_mip_views(
    texture: &wgpu::Texture,
    spec: &TextureHistorySpec,
) -> HistoryTextureMipViews {
    let shape = texture_history_view_shape(spec);
    let mut layers = Vec::with_capacity(shape.layer_count as usize);
    for layer in 0..shape.layer_count {
        let mut mips = Vec::with_capacity(shape.mip_level_count as usize);
        for mip in 0..shape.mip_level_count {
            mips.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("history_texture_mip_layer_view"),
                format: Some(spec.format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            }));
        }
        layers.push(Arc::<[wgpu::TextureView]>::from(mips));
    }
    HistoryTextureMipViews {
        layers: Arc::<[Arc<[wgpu::TextureView]>]>::from(layers),
        mip_level_count: shape.mip_level_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex_spec() -> TextureHistorySpec {
        TextureHistorySpec {
            label: "test_tex",
            format: wgpu::TextureFormat::Rgba16Float,
            extent: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        }
    }

    fn buf_spec() -> BufferHistorySpec {
        BufferHistorySpec {
            label: "test_buf",
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
        }
    }

    const SLOT_A: HistorySlotId = HistorySlotId::new("test_a");
    const SLOT_B: HistorySlotId = HistorySlotId::new("test_b");

    #[test]
    fn texture_slot_registration_is_idempotent() {
        let mut reg = HistoryRegistry::new();
        reg.register_texture(SLOT_A, tex_spec()).unwrap();
        reg.register_texture(SLOT_A, tex_spec()).unwrap();
        assert_eq!(reg.texture_slot_count(), 1);
    }

    #[test]
    fn buffer_slot_registration_is_idempotent() {
        let mut reg = HistoryRegistry::new();
        reg.register_buffer(SLOT_A, buf_spec()).unwrap();
        reg.register_buffer(SLOT_A, buf_spec()).unwrap();
        assert_eq!(reg.buffer_slot_count(), 1);
    }

    #[test]
    fn registering_same_id_as_other_kind_is_rejected() {
        let mut reg = HistoryRegistry::new();
        reg.register_texture(SLOT_A, tex_spec()).unwrap();
        let err = reg.register_buffer(SLOT_A, buf_spec()).unwrap_err();
        assert!(matches!(
            err,
            HistoryRegistryError::KindMismatch {
                other_kind: "texture",
                ..
            }
        ));
    }

    #[test]
    fn different_ids_live_side_by_side() {
        let mut reg = HistoryRegistry::new();
        reg.register_texture(SLOT_A, tex_spec()).unwrap();
        reg.register_buffer(SLOT_B, buf_spec()).unwrap();
        assert_eq!(reg.texture_slot_count(), 1);
        assert_eq!(reg.buffer_slot_count(), 1);
        assert!(reg.texture_slot(SLOT_A).is_some());
        assert!(reg.buffer_slot(SLOT_B).is_some());
        assert!(reg.texture_slot(SLOT_B).is_none());
    }

    #[test]
    fn scoped_texture_slots_do_not_alias_global_or_other_views() {
        let mut reg = HistoryRegistry::new();
        reg.register_texture(SLOT_A, tex_spec()).unwrap();
        reg.register_texture_scoped(
            SLOT_A,
            HistoryResourceScope::View(OcclusionViewId::Main),
            tex_spec(),
        )
        .unwrap();
        reg.register_texture_scoped(
            SLOT_A,
            HistoryResourceScope::View(OcclusionViewId::OffscreenRenderTexture(7)),
            tex_spec(),
        )
        .unwrap();

        assert_eq!(reg.texture_slot_count(), 3);
        assert!(reg.texture_slot(SLOT_A).is_some());
        assert!(reg
            .texture_slot_scoped(SLOT_A, HistoryResourceScope::View(OcclusionViewId::Main))
            .is_some());
        assert!(reg
            .texture_slot_scoped(
                SLOT_A,
                HistoryResourceScope::View(OcclusionViewId::OffscreenRenderTexture(7)),
            )
            .is_some());
    }

    #[test]
    fn ping_pong_indices_alternate_per_frame() {
        let mut reg = HistoryRegistry::new();
        let first_cur = reg.current_index();
        let first_prev = reg.previous_index();
        assert_ne!(first_cur, first_prev);
        reg.advance_frame();
        assert_eq!(reg.current_index(), first_prev);
        assert_eq!(reg.previous_index(), first_cur);
        reg.advance_frame();
        assert_eq!(reg.current_index(), first_cur);
    }

    #[test]
    fn update_spec_drops_existing_halves() {
        // We can't exercise GPU allocation in unit tests, but we can verify that update_spec
        // returns true for a registered slot and that the new spec is stored.
        let mut reg = HistoryRegistry::new();
        reg.register_texture(SLOT_A, tex_spec()).unwrap();
        let mut new_spec = tex_spec();
        new_spec.extent.width = 128;
        assert!(reg.update_texture_spec(SLOT_A, new_spec));
        let slot = reg.texture_slot(SLOT_A).unwrap();
        assert_eq!(slot.lock().spec().extent.width, 128);
        // Unregistered id returns false.
        assert!(!reg.update_texture_spec(SLOT_B, tex_spec()));
    }

    #[test]
    fn registering_texture_with_different_spec_resets_pair() {
        let mut reg = HistoryRegistry::new();
        reg.register_texture(SLOT_A, tex_spec()).unwrap();
        let mut changed = tex_spec();
        changed.extent.width = 200;
        reg.register_texture(SLOT_A, changed).unwrap();
        let slot = reg.texture_slot(SLOT_A).unwrap();
        {
            let guard = slot.lock();
            assert_eq!(guard.spec().extent.width, 200);
            assert!(guard.half(0).is_none());
            assert!(guard.half(1).is_none());
            drop(guard);
        }
    }

    #[test]
    fn texture_history_view_shape_tracks_d2_layers_and_mips() {
        let mut spec = tex_spec();
        spec.extent.depth_or_array_layers = 2;
        spec.mip_level_count = 6;

        assert_eq!(
            texture_history_view_shape(&spec),
            HistoryTextureViewShape {
                layer_count: 2,
                mip_level_count: 6,
            }
        );
    }

    #[test]
    fn texture_history_view_shape_skips_non_d2_subresource_tables() {
        let mut spec = tex_spec();
        spec.dimension = wgpu::TextureDimension::D3;
        spec.extent.depth_or_array_layers = 4;
        spec.mip_level_count = 3;

        assert_eq!(
            texture_history_view_shape(&spec),
            HistoryTextureViewShape {
                layer_count: 0,
                mip_level_count: 3,
            }
        );
    }
}
