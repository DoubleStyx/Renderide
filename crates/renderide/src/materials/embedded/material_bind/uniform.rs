//! Embedded `@group(1)` material uniform arena and upload.

use std::num::NonZeroU64;
use std::sync::Arc;

use hashbrown::{HashMap, HashSet};

use super::super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::super::layout::StemMaterialLayout;
use super::super::texture_pools::EmbeddedTexturePools;
use super::super::uniform_pack::{
    MaterialUniformPackMetadata, UniformPackTextureContext,
    build_embedded_uniform_bytes_with_material_defaults,
};
use crate::frame_upload_batch::GraphUploadSink;
use crate::materials::host_data::{MaterialPropertyLookupIds, MaterialPropertyStore};

const INITIAL_MATERIAL_UNIFORM_ARENA_BYTES: u64 = 64 * 1024;

/// Exact per-source property generations used by persistent material caches.
///
/// Keeping the three counters separate avoids the collision ambiguity of a lossy combined
/// fingerprint.
pub(super) type MaterialPropertyGenerations = [u64; 3];

pub(super) fn material_property_generations(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> MaterialPropertyGenerations {
    [
        store.material_generation(lookup.material_asset_id),
        lookup
            .mesh_property_block_slot0
            .map_or(0, |id| store.property_block_generation(id)),
        lookup
            .mesh_renderer_property_block_id
            .map_or(0, |id| store.property_block_generation(id)),
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) struct MaterialUniformCacheKey {
    pub(super) stem_hash: u64,
    pub(super) material_asset_id: i32,
    pub(super) property_block_slot0: Option<i32>,
    /// Renderer-level `MaterialPropertyBlock` id (applies to every material on the renderer).
    pub(super) renderer_property_block_id: Option<i32>,
    pub(super) texture_2d_asset_id: i32,
    pub(super) shader_variant_bits: Option<u32>,
}

/// GPU arena slot selected for a material uniform block.
#[derive(Clone)]
pub(super) struct MaterialUniformArenaSlotBinding {
    pub(super) buffer: Arc<wgpu::Buffer>,
    pub(super) dynamic_offset: u32,
    pub(super) size: NonZeroU64,
    pub(super) arena_shard: u8,
    pub(super) buffer_generation: u64,
}

#[derive(Clone, Copy, Debug)]
struct MaterialUniformSlot {
    offset: u64,
    size: u64,
    last_written_generations: MaterialPropertyGenerations,
    last_written_texture_state_sig: u64,
    buffer_generation: u64,
}

#[derive(Clone, Copy, Debug)]
struct MaterialUniformSlotResolution {
    offset: u64,
    size: NonZeroU64,
    buffer_generation: u64,
    needs_write: bool,
}

/// Pure allocator state for material uniform arena slots.
#[derive(Debug)]
struct MaterialUniformArenaAllocator {
    slots: HashMap<MaterialUniformCacheKey, MaterialUniformSlot>,
    cursor: u64,
    capacity: u64,
    max_bytes: u64,
    alignment: u64,
    generation: u64,
}

impl MaterialUniformArenaAllocator {
    fn new(capacity: u64, max_bytes: u64, alignment: u64) -> Self {
        Self {
            slots: HashMap::new(),
            cursor: 0,
            capacity,
            max_bytes,
            alignment: alignment.max(1),
            generation: 0,
        }
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Returns an existing slot only when it needs no allocation or refresh.
    fn stable_slot(
        &self,
        key: &MaterialUniformCacheKey,
        size: NonZeroU64,
        property_generations: MaterialPropertyGenerations,
        texture_state_sig: u64,
    ) -> Option<MaterialUniformSlotResolution> {
        let slot = self.slots.get(key)?;
        if slot.size != size.get()
            || slot.last_written_generations != property_generations
            || slot.last_written_texture_state_sig != texture_state_sig
            || slot.buffer_generation != self.generation
        {
            return None;
        }
        Some(MaterialUniformSlotResolution {
            offset: slot.offset,
            size,
            buffer_generation: self.generation,
            needs_write: false,
        })
    }

    fn resolve_slot(
        &mut self,
        key: MaterialUniformCacheKey,
        size: NonZeroU64,
        property_generations: MaterialPropertyGenerations,
        texture_state_sig: u64,
    ) -> Result<MaterialUniformSlotResolution, EmbeddedMaterialBindError> {
        let size = size.get();
        if let Some(slot) = self.slots.get(&key).copied() {
            if slot.size != size {
                return Err(EmbeddedMaterialBindError::from(format!(
                    "material uniform arena slot size changed for key {:?}: old={} new={}",
                    key, slot.size, size
                )));
            }
            let needs_write = slot.last_written_generations != property_generations
                || slot.last_written_texture_state_sig != texture_state_sig
                || slot.buffer_generation != self.generation;
            return Ok(MaterialUniformSlotResolution {
                offset: slot.offset,
                size: non_zero_u64(size)?,
                buffer_generation: self.generation,
                needs_write,
            });
        }

        let offset = align_up(self.cursor, self.alignment);
        let end = offset.checked_add(size).ok_or_else(|| {
            EmbeddedMaterialBindError::from("material uniform arena offset overflow".to_string())
        })?;
        self.ensure_capacity(end)?;
        self.cursor = end;
        self.slots.insert(
            key,
            MaterialUniformSlot {
                offset,
                size,
                last_written_generations: [u64::MAX; 3],
                last_written_texture_state_sig: u64::MAX,
                buffer_generation: self.generation,
            },
        );
        Ok(MaterialUniformSlotResolution {
            offset,
            size: non_zero_u64(size)?,
            buffer_generation: self.generation,
            needs_write: true,
        })
    }

    fn mark_written(
        &mut self,
        key: &MaterialUniformCacheKey,
        buffer_generation: u64,
        property_generations: MaterialPropertyGenerations,
        texture_state_sig: u64,
    ) {
        let Some(slot) = self.slots.get_mut(key) else {
            return;
        };
        if slot.buffer_generation > buffer_generation {
            return;
        }
        slot.buffer_generation = buffer_generation;
        slot.last_written_generations = property_generations;
        slot.last_written_texture_state_sig = texture_state_sig;
    }

    fn purge_material_assets(
        &mut self,
        material_ids: &HashSet<i32>,
        property_block_ids: &HashSet<i32>,
    ) -> usize {
        let before = self.slots.len();
        self.slots.retain(|key, _slot| {
            !material_ids.contains(&key.material_asset_id)
                && key
                    .property_block_slot0
                    .is_none_or(|id| !property_block_ids.contains(&id))
                && key
                    .renderer_property_block_id
                    .is_none_or(|id| !property_block_ids.contains(&id))
        });
        let removed = before.saturating_sub(self.slots.len());
        self.reset_empty_arena_after_purge(removed);
        removed
    }

    /// Clears every slot after a global material-layout cache invalidation.
    ///
    /// The allocator is bump-only, so removing one stem while retaining other slots would leave
    /// holes that repeated shader reloads could never reclaim.
    fn clear_slots(&mut self) -> usize {
        let removed = self.slots.len();
        self.slots.clear();
        self.reset_empty_arena_after_purge(removed);
        removed
    }

    fn reset_empty_arena_after_purge(&mut self, removed: usize) {
        if self.slots.is_empty() && removed > 0 {
            self.cursor = 0;
            self.generation = self.generation.wrapping_add(1);
        }
    }

    fn ensure_capacity(&mut self, needed: u64) -> Result<(), EmbeddedMaterialBindError> {
        if needed <= self.capacity {
            return Ok(());
        }
        if needed > self.max_bytes {
            return Err(EmbeddedMaterialBindError::from(format!(
                "material uniform arena needs {needed} bytes, exceeding cap {}",
                self.max_bytes
            )));
        }
        let grown = align_up(self.capacity.saturating_mul(2).max(needed), self.alignment);
        self.capacity = grown.min(self.max_bytes).max(needed);
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }
}

/// Shared GPU buffer plus slot allocator for embedded material uniform constants.
pub(super) struct MaterialUniformArena {
    device: Arc<wgpu::Device>,
    allocator: MaterialUniformArenaAllocator,
    buffer: Arc<wgpu::Buffer>,
}

impl MaterialUniformArena {
    pub(super) fn new(device: Arc<wgpu::Device>, limits: Arc<crate::gpu::GpuLimits>) -> Self {
        let alignment = u64::from(limits.min_uniform_buffer_offset_alignment()).max(1);
        let max_bytes = limits.max_buffer_size().min(u64::from(u32::MAX));
        let initial = align_up(
            INITIAL_MATERIAL_UNIFORM_ARENA_BYTES
                .min(max_bytes)
                .max(alignment),
            alignment,
        );
        let buffer = Arc::new(create_material_uniform_arena_buffer(
            device.as_ref(),
            initial,
        ));
        Self {
            device,
            allocator: MaterialUniformArenaAllocator::new(initial, max_bytes, alignment),
            buffer,
        }
    }

    /// Returns an exact existing binding without mutating allocator or buffer state.
    fn stable_binding(
        &self,
        key: &MaterialUniformCacheKey,
        size: NonZeroU64,
        property_generations: MaterialPropertyGenerations,
        texture_state_sig: u64,
        arena_shard: u8,
    ) -> Result<Option<MaterialUniformArenaSlotBinding>, EmbeddedMaterialBindError> {
        let Some(resolved) =
            self.allocator
                .stable_slot(key, size, property_generations, texture_state_sig)
        else {
            return Ok(None);
        };
        self.binding_for_resolution(resolved, arena_shard).map(Some)
    }

    fn resolve_binding(
        &mut self,
        key: MaterialUniformCacheKey,
        size: NonZeroU64,
        property_generations: MaterialPropertyGenerations,
        texture_state_sig: u64,
        arena_shard: u8,
    ) -> Result<(MaterialUniformArenaSlotBinding, bool), EmbeddedMaterialBindError> {
        let previous_generation = self.allocator.generation();
        let resolved =
            self.allocator
                .resolve_slot(key, size, property_generations, texture_state_sig)?;
        if self.allocator.generation() != previous_generation {
            self.buffer = Arc::new(create_material_uniform_arena_buffer(
                self.device.as_ref(),
                self.allocator.capacity(),
            ));
            logger::debug!(
                "material uniform arena: grew to {} bytes (generation {})",
                self.allocator.capacity(),
                self.allocator.generation()
            );
        }
        self.binding_for_resolution(resolved, arena_shard)
            .map(|binding| (binding, resolved.needs_write))
    }

    fn binding_for_resolution(
        &self,
        resolved: MaterialUniformSlotResolution,
        arena_shard: u8,
    ) -> Result<MaterialUniformArenaSlotBinding, EmbeddedMaterialBindError> {
        let dynamic_offset = u32::try_from(resolved.offset).map_err(|_err| {
            EmbeddedMaterialBindError::from(format!(
                "material uniform arena offset {} exceeds dynamic offset range",
                resolved.offset
            ))
        })?;
        Ok(MaterialUniformArenaSlotBinding {
            buffer: self.buffer.clone(),
            dynamic_offset,
            size: resolved.size,
            arena_shard,
            buffer_generation: resolved.buffer_generation,
        })
    }

    fn mark_written(
        &mut self,
        key: &MaterialUniformCacheKey,
        binding: &MaterialUniformArenaSlotBinding,
        property_generations: MaterialPropertyGenerations,
        texture_state_sig: u64,
    ) {
        self.allocator.mark_written(
            key,
            binding.buffer_generation,
            property_generations,
            texture_state_sig,
        );
    }

    pub(super) fn purge_material_assets(
        &mut self,
        material_ids: &HashSet<i32>,
        property_block_ids: &HashSet<i32>,
    ) -> usize {
        self.allocator
            .purge_material_assets(material_ids, property_block_ids)
    }

    pub(super) fn clear_slots(&mut self) -> usize {
        self.allocator.clear_slots()
    }
}

fn create_material_uniform_arena_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("embedded_material_uniform_arena"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    crate::profiling::note_resource_churn!(Buffer, "materials::embedded_material_uniform_arena");
    buffer
}

fn align_up(value: u64, alignment: u64) -> u64 {
    let alignment = alignment.max(1);
    value.div_ceil(alignment).saturating_mul(alignment)
}

fn non_zero_u64(value: u64) -> Result<NonZeroU64, EmbeddedMaterialBindError> {
    NonZeroU64::new(value).ok_or_else(|| {
        EmbeddedMaterialBindError::from("material uniform block has zero size".to_string())
    })
}

/// Uniform arena lookup and upload request for [`super::EmbeddedMaterialBindResources`].
pub(super) struct EmbeddedUniformArenaRequest<'a> {
    pub(super) uploads: GraphUploadSink<'a>,
    pub(super) stem: &'a str,
    pub(super) shader_variant_bits: Option<u32>,
    pub(super) layout: &'a Arc<StemMaterialLayout>,
    pub(super) uniform_key: &'a MaterialUniformCacheKey,
    pub(super) property_generations: MaterialPropertyGenerations,
    pub(super) store: &'a MaterialPropertyStore,
    pub(super) lookup: MaterialPropertyLookupIds,
    pub(super) pools: &'a EmbeddedTexturePools<'a>,
    pub(super) primary_texture_2d: i32,
    pub(super) texture_bind_signature: u64,
    pub(super) texture_state_sig: u64,
}

use super::EmbeddedMaterialBindResources;

impl EmbeddedMaterialBindResources {
    /// Shared dynamic uniform arena slot for embedded `@group(1)` material constants.
    ///
    /// Refreshes bytes when [`MaterialPropertyStore`] mutates, when texture-derived uniform state
    /// changes, or when the arena grows to a new backing buffer generation.
    ///
    /// Exact no-write hits only clone an existing binding under a shared shard lock. Every other
    /// path reacquires the shard exclusively and revalidates before allocation or refresh.
    /// `resolve_binding`, the uniform byte build, `GraphUploadSink::write_buffer`, and
    /// `mark_written` stay in that one exclusive section so recorded generations cannot disagree
    /// with uploaded bytes.
    ///
    /// The arena is sharded by compatible `(stem, shader variant, texture signature)` tuples so
    /// concurrent rayon recording workers usually land on distinct shards. A given tuple always
    /// routes to the same shard, so the per-shard slot table and `buffer_generation` counter
    /// remain self-consistent.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the arena shard write lock is intentionally held across resolve_binding, uniform packing, upload, and mark_written"
    )]
    pub(super) fn get_or_update_embedded_uniform_arena_slot(
        &self,
        req: EmbeddedUniformArenaRequest<'_>,
    ) -> Result<MaterialUniformArenaSlotBinding, EmbeddedMaterialBindError> {
        profiling::scope!("materials::embedded_uniform_arena");
        let EmbeddedUniformArenaRequest {
            uploads,
            stem,
            shader_variant_bits,
            layout,
            uniform_key,
            property_generations,
            store,
            lookup,
            pools,
            primary_texture_2d,
            texture_bind_signature,
            texture_state_sig,
        } = req;
        let uniform_size = non_zero_u64(
            layout
                .reflected
                .material_uniform
                .as_ref()
                .ok_or_else(|| {
                    EmbeddedMaterialBindError::from(format!(
                        "stem {stem}: uniform block missing (shader has no material uniform)"
                    ))
                })?
                .total_size
                .into(),
        )?;
        let tex_ctx = UniformPackTextureContext {
            pools,
            primary_texture_2d,
        };

        let (arena_shard, shard) = self.uniform_arena_shard(uniform_key, texture_bind_signature);
        {
            profiling::scope!("materials::embedded_uniform_arena_read");
            let arena = shard.read();
            if let Some(binding) = arena.stable_binding(
                uniform_key,
                uniform_size,
                property_generations,
                texture_state_sig,
                arena_shard,
            )? {
                return Ok(binding);
            }
        }

        profiling::scope!("materials::embedded_uniform_arena_critical_section");
        let mut arena = shard.write();
        let (binding, needs_write) = arena.resolve_binding(
            *uniform_key,
            uniform_size,
            property_generations,
            texture_state_sig,
            arena_shard,
        )?;
        if needs_write {
            profiling::scope!("materials::embedded_uniform_arena_write");
            let uniform_bytes = build_embedded_uniform_bytes_with_material_defaults(
                &layout.reflected,
                layout.ids.as_ref(),
                &MaterialUniformPackMetadata {
                    value_spaces: &layout.uniform_value_spaces,
                    material_defaults: &layout.uniform_default_by_field,
                },
                store,
                lookup,
                &tex_ctx,
                shader_variant_bits,
            )
            .ok_or_else(|| {
                format!("stem {stem}: uniform block missing (shader has no material uniform)")
            })?;
            uploads.write_buffer(
                binding.buffer.as_ref(),
                u64::from(binding.dynamic_offset),
                &uniform_bytes,
            );
            arena.mark_written(
                uniform_key,
                &binding,
                property_generations,
                texture_state_sig,
            );
        } else {
            profiling::scope!("materials::embedded_uniform_arena_hit");
        }
        Ok(binding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(material_asset_id: i32) -> MaterialUniformCacheKey {
        MaterialUniformCacheKey {
            stem_hash: 7,
            material_asset_id,
            property_block_slot0: None,
            renderer_property_block_id: None,
            texture_2d_asset_id: -1,
            shader_variant_bits: None,
        }
    }

    fn key_with_block(material_asset_id: i32, property_block_id: i32) -> MaterialUniformCacheKey {
        MaterialUniformCacheKey {
            stem_hash: 7,
            material_asset_id,
            property_block_slot0: Some(property_block_id),
            renderer_property_block_id: None,
            texture_2d_asset_id: -1,
            shader_variant_bits: None,
        }
    }

    fn key_with_renderer_block(
        material_asset_id: i32,
        property_block_id: i32,
    ) -> MaterialUniformCacheKey {
        MaterialUniformCacheKey {
            stem_hash: 7,
            material_asset_id,
            property_block_slot0: None,
            renderer_property_block_id: Some(property_block_id),
            texture_2d_asset_id: -1,
            shader_variant_bits: None,
        }
    }

    fn generations(value: u64) -> MaterialPropertyGenerations {
        [value; 3]
    }

    #[test]
    fn arena_allocator_aligns_offsets() {
        let mut allocator = MaterialUniformArenaAllocator::new(1024, 4096, 256);
        let size = NonZeroU64::new(80).unwrap();

        let a = allocator
            .resolve_slot(key(1), size, generations(0), 0)
            .unwrap();
        let b = allocator
            .resolve_slot(key(2), size, generations(0), 0)
            .unwrap();

        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 256);
    }

    #[test]
    fn arena_allocator_reuses_stable_slot() {
        let mut allocator = MaterialUniformArenaAllocator::new(1024, 4096, 256);
        let size = NonZeroU64::new(80).unwrap();

        let first = allocator
            .resolve_slot(key(1), size, generations(3), 5)
            .unwrap();
        allocator.mark_written(&key(1), first.buffer_generation, generations(3), 5);
        let second = allocator
            .resolve_slot(key(1), size, generations(3), 5)
            .unwrap();

        assert_eq!(first.offset, second.offset);
        assert!(!second.needs_write);
    }

    #[test]
    fn arena_allocator_stable_slot_requires_exact_state() {
        let mut allocator = MaterialUniformArenaAllocator::new(1024, 4096, 256);
        let size = NonZeroU64::new(80).unwrap();
        let cache_key = key(1);
        let first = allocator
            .resolve_slot(cache_key, size, generations(3), 5)
            .unwrap();
        allocator.mark_written(&cache_key, first.buffer_generation, generations(3), 5);

        assert!(
            allocator
                .stable_slot(&cache_key, size, generations(3), 5)
                .is_some()
        );
        assert!(
            allocator
                .stable_slot(&cache_key, size, [3, 4, 3], 5)
                .is_none()
        );
        assert!(
            allocator
                .stable_slot(&cache_key, size, generations(3), 6)
                .is_none()
        );

        let changed_size = NonZeroU64::new(96).unwrap();
        assert!(
            allocator
                .stable_slot(&cache_key, changed_size, generations(3), 5)
                .is_none()
        );
        let error = allocator
            .resolve_slot(cache_key, changed_size, generations(3), 5)
            .expect_err("exclusive resolution must retain size mismatch validation");
        assert!(error.to_string().contains("slot size changed"));
    }

    #[test]
    fn arena_allocator_growth_invalidates_existing_slots_for_new_generation() {
        let mut allocator = MaterialUniformArenaAllocator::new(256, 2048, 256);
        let size = NonZeroU64::new(128).unwrap();

        let first = allocator
            .resolve_slot(key(1), size, generations(1), 1)
            .unwrap();
        allocator.mark_written(&key(1), first.buffer_generation, generations(1), 1);
        assert_eq!(allocator.generation(), 0);
        assert!(
            allocator
                .stable_slot(&key(1), size, generations(1), 1)
                .is_some()
        );

        let _second = allocator
            .resolve_slot(key(2), size, generations(1), 1)
            .unwrap();
        assert_eq!(allocator.generation(), 1);
        assert!(
            allocator
                .stable_slot(&key(1), size, generations(1), 1)
                .is_none()
        );
        let first_after_growth = allocator
            .resolve_slot(key(1), size, generations(1), 1)
            .unwrap();

        assert_eq!(first_after_growth.offset, first.offset);
        assert_eq!(first_after_growth.buffer_generation, 1);
        assert!(first_after_growth.needs_write);
    }

    #[test]
    fn concurrent_stale_readers_revalidate_to_one_refresh() {
        const WORKERS: usize = 8;

        let size = NonZeroU64::new(80).unwrap();
        let cache_key = key(1);
        let mut allocator = MaterialUniformArenaAllocator::new(1024, 4096, 256);
        let first = allocator
            .resolve_slot(cache_key, size, generations(1), 2)
            .unwrap();
        allocator.mark_written(&cache_key, first.buffer_generation, generations(1), 2);

        let shared = Arc::new(parking_lot::RwLock::new(allocator));
        let ready = Arc::new(std::sync::Barrier::new(WORKERS));
        let refreshes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut threads = Vec::with_capacity(WORKERS);

        for _ in 0..WORKERS {
            let shared = shared.clone();
            let ready = ready.clone();
            let refreshes = refreshes.clone();
            threads.push(std::thread::spawn(move || {
                {
                    let allocator = shared.read();
                    assert!(
                        allocator
                            .stable_slot(&cache_key, size, generations(3), 4)
                            .is_none()
                    );
                }
                ready.wait();

                let mut allocator = shared.write();
                let resolved = allocator
                    .resolve_slot(cache_key, size, generations(3), 4)
                    .unwrap();
                if resolved.needs_write {
                    refreshes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    allocator.mark_written(
                        &cache_key,
                        resolved.buffer_generation,
                        generations(3),
                        4,
                    );
                }
            }));
        }

        for thread in threads {
            thread.join().expect("uniform arena worker");
        }

        assert_eq!(refreshes.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(
            shared
                .read()
                .stable_slot(&cache_key, size, generations(3), 4)
                .is_some()
        );
    }

    #[test]
    fn arena_allocator_purges_material_and_property_block_slots() {
        let mut allocator = MaterialUniformArenaAllocator::new(1024, 4096, 256);
        let size = NonZeroU64::new(80).unwrap();
        allocator
            .resolve_slot(key(1), size, generations(0), 0)
            .unwrap();
        allocator
            .resolve_slot(key_with_block(2, 20), size, generations(0), 0)
            .unwrap();
        allocator
            .resolve_slot(key_with_renderer_block(3, 30), size, generations(0), 0)
            .unwrap();
        allocator
            .resolve_slot(key(4), size, generations(0), 0)
            .unwrap();

        let mut materials = HashSet::new();
        materials.insert(1);
        let mut blocks = HashSet::new();
        blocks.insert(20);
        blocks.insert(30);
        assert_eq!(allocator.purge_material_assets(&materials, &blocks), 3);

        assert!(!allocator.slots.contains_key(&key(1)));
        assert!(!allocator.slots.contains_key(&key_with_block(2, 20)));
        assert!(
            !allocator
                .slots
                .contains_key(&key_with_renderer_block(3, 30))
        );
        assert!(allocator.slots.contains_key(&key(4)));
    }

    #[test]
    fn arena_allocator_clear_forces_equal_size_rewrite() {
        let mut allocator = MaterialUniformArenaAllocator::new(1024, 4096, 256);
        let size = NonZeroU64::new(80).unwrap();
        let cache_key = key(1);
        let first = allocator
            .resolve_slot(cache_key, size, generations(3), 5)
            .unwrap();
        allocator.mark_written(&cache_key, first.buffer_generation, generations(3), 5);
        assert!(
            allocator
                .stable_slot(&cache_key, size, generations(3), 5)
                .is_some()
        );

        assert_eq!(allocator.clear_slots(), 1);
        let refreshed = allocator
            .resolve_slot(cache_key, size, generations(3), 5)
            .unwrap();

        assert!(refreshed.needs_write);
    }

    #[test]
    fn arena_allocator_clear_accepts_changed_uniform_size() {
        let mut allocator = MaterialUniformArenaAllocator::new(1024, 4096, 256);
        let old_size = NonZeroU64::new(80).unwrap();
        let new_size = NonZeroU64::new(96).unwrap();
        let cache_key = key(1);
        let first = allocator
            .resolve_slot(cache_key, old_size, generations(3), 5)
            .unwrap();
        allocator.mark_written(&cache_key, first.buffer_generation, generations(3), 5);

        assert_eq!(allocator.clear_slots(), 1);
        let refreshed = allocator
            .resolve_slot(cache_key, new_size, generations(3), 5)
            .expect("invalidating the stem must discard the old slot size");

        assert_eq!(refreshed.size, new_size);
        assert!(refreshed.needs_write);
    }
}
