//! `@group(1)` bind groups for embedded raster materials (WGSL targets shipped with the renderer).
//!
//! Layouts and uniform packing come from [`crate::materials::reflect_raster_material_wgsl`] (naga).
//! WGSL identifiers in `@group(1)` match Unity [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)
//! names; [`crate::materials::host_data::PropertyIdRegistry`] resolves them to batch property ids.
//! Multi-compile keyword state is shipped as a single `_RenderideVariantBits: u32` uniform field
//! and decoded by WGSL via `renderide::material::variant_bits::enabled`.

mod assemble;
mod cache;
mod resolve;
mod texture_signature;
mod uniform;
mod white_texture;

pub(crate) use cache::MaterialBindCacheKey;

use ahash::RandomState;
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

use super::bind_kind::TextureBindKind;
use super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::layout::StemMaterialLayout;
use super::texture_pools::EmbeddedTexturePools;
use super::texture_resolve::default_embedded_sampler;
use crate::gpu_resource::ShardedLru;
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};
use crate::render_graph::frame_upload_batch::GraphUploadSink;

use assemble::build_embedded_bind_group_entries;
use cache::{
    EMBEDDED_CACHE_SHARDS, EmbeddedSamplerCacheKey, TextureDebugCacheKey,
    max_cached_embedded_bind_groups, max_cached_embedded_samplers, max_cached_texture_debug_ids,
};
use texture_signature::compute_uniform_texture_state_signature;
use uniform::{
    EmbeddedUniformArenaRequest, MaterialUniformArena, MaterialUniformArenaSlotBinding,
    MaterialUniformCacheKey,
};
use white_texture::{
    PlaceholderTexture, create_black, create_flat_normal, create_gray, create_red, create_white,
    upload_black, upload_flat_normal, upload_gray, upload_red, upload_white,
};

use resolve::EmbeddedBindInputResolution;

/// Resolved embedded material `@group(1)` bind group plus its optional dynamic uniform offset.
#[derive(Clone)]
pub(crate) struct EmbeddedMaterialBindGroup {
    /// Bind group containing the material uniform arena buffer and resolved texture/sampler bindings.
    pub(crate) bind_group: Arc<wgpu::BindGroup>,
    /// Dynamic offset for `@group(1) @binding(0)` when the shader has a material uniform block.
    pub(crate) uniform_dynamic_offset: Option<u32>,
}

/// Embedded shader identity needed when resolving a material bind group.
#[derive(Clone, Copy)]
pub(crate) struct EmbeddedMaterialBindShader<'a> {
    /// Embedded WGSL stem selected for the shader asset.
    pub(crate) stem: &'a str,
    /// Froox shader-specific variant bitmask decoded from the uploaded Unity shader asset.
    pub(crate) shader_variant_bits: Option<u32>,
}

fn material_bind_group_result(
    bind_key: MaterialBindCacheKey,
    bind_group: Arc<wgpu::BindGroup>,
    uniform_binding: Option<&MaterialUniformArenaSlotBinding>,
) -> (MaterialBindCacheKey, EmbeddedMaterialBindGroup) {
    (
        bind_key,
        EmbeddedMaterialBindGroup {
            bind_group,
            uniform_dynamic_offset: uniform_binding.map(|binding| binding.dynamic_offset),
        },
    )
}

struct EmbeddedBindCacheMissInputs<'a> {
    layout: &'a Arc<StemMaterialLayout>,
    stem_hash: u64,
    texture_2d_asset_id: i32,
    pools: &'a EmbeddedTexturePools<'a>,
    store: &'a MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    offscreen_write_render_texture_asset_id: Option<i32>,
    lookup_bind_key: MaterialBindCacheKey,
    lookup_texture_bind_signature: u64,
    uniform_binding: Option<&'a MaterialUniformArenaSlotBinding>,
}

fn build_material_bind_cache_key(
    stem_hash: u64,
    lookup: MaterialPropertyLookupIds,
    texture_bind_signature: u64,
    offscreen_write_render_texture_asset_id: Option<i32>,
    uniform_binding: Option<&MaterialUniformArenaSlotBinding>,
) -> MaterialBindCacheKey {
    MaterialBindCacheKey {
        stem_hash,
        material_asset_id: lookup.material_asset_id,
        property_block_slot0: lookup.mesh_property_block_slot0,
        renderer_property_block_id: lookup.mesh_renderer_property_block_id,
        texture_bind_signature,
        offscreen_write_render_texture_asset_id,
        uniform_arena_generation: uniform_binding
            .as_ref()
            .map_or(0, |binding| binding.buffer_generation),
    }
}

/// GPU resources shared by embedded material bind groups (layouts, default textures, sampler).
pub struct EmbeddedMaterialBindResources {
    device: Arc<wgpu::Device>,
    white_2d: PlaceholderTexture,
    black_2d: PlaceholderTexture,
    gray_2d: PlaceholderTexture,
    red_2d: PlaceholderTexture,
    flat_normal_2d: PlaceholderTexture,
    white_3d: PlaceholderTexture,
    black_3d: PlaceholderTexture,
    gray_3d: PlaceholderTexture,
    red_3d: PlaceholderTexture,
    white_cube: PlaceholderTexture,
    black_cube: PlaceholderTexture,
    gray_cube: PlaceholderTexture,
    red_cube: PlaceholderTexture,
    default_sampler: Arc<wgpu::Sampler>,
    property_registry: Arc<PropertyIdRegistry>,
    stem_cache: Mutex<HashMap<String, Arc<StemMaterialLayout>>>,
    /// Sharded dynamic uniform arenas for `@group(1) @binding(0)` material constants.
    ///
    /// Each shard owns an independent growable GPU buffer and slot allocator; a uniform cache
    /// key always routes to the same shard, so concurrent rayon recording workers hitting
    /// distinct keys never block each other. Replaces the previous single
    /// `Mutex<MaterialUniformArena>` whose lock was the dominant contention point during
    /// `graph::per_view_fan_out` (3 ms `embedded_uniform_arena_critical_section` zone).
    ///
    /// Per-shard growth bumps that shard's `buffer_generation`; the bind group cache key already
    /// includes the slot's `buffer_generation`, so per-shard generations are self-consistent.
    uniform_arena_shards: Box<[Mutex<MaterialUniformArena>]>,
    /// Deterministic per-process hasher routing a [`MaterialUniformCacheKey`] to its shard.
    uniform_arena_hasher: RandomState,
    /// Sharded LRU caches for `@group(1)` bind groups and samplers.
    /// Each shard is a `parking_lot::Mutex<LruCache<...>>` so per-view rayon workers contend
    /// only with workers whose cache key hashes into the same shard. Replaces the previous
    /// single-`Mutex<LruCache<...>>` whose lock was the dominant contention point during
    /// `graph::per_view_fan_out`.
    bind_cache: ShardedLru<MaterialBindCacheKey, Arc<wgpu::BindGroup>>,
    sampler_cache: ShardedLru<EmbeddedSamplerCacheKey, Arc<wgpu::Sampler>>,
    /// Texture-debug HUD cache stays a single mutex: per-stem call frequency is low and the
    /// HUD does not run inside the per-view rayon fan-out, so sharding has no benefit here.
    texture_debug_cache: Mutex<LruCache<TextureDebugCacheKey, Arc<[i32]>>>,
}

impl EmbeddedMaterialBindResources {
    /// Builds layouts and placeholder textures.
    pub fn new(
        device: Arc<wgpu::Device>,
        property_registry: Arc<PropertyIdRegistry>,
        limits: Arc<crate::gpu::GpuLimits>,
    ) -> Result<Self, EmbeddedMaterialBindError> {
        let white_2d = create_white(device.as_ref(), TextureBindKind::Tex2D);
        let black_2d = create_black(device.as_ref(), TextureBindKind::Tex2D);
        let gray_2d = create_gray(device.as_ref(), TextureBindKind::Tex2D);
        let red_2d = create_red(device.as_ref(), TextureBindKind::Tex2D);
        let flat_normal_2d = create_flat_normal(device.as_ref(), TextureBindKind::Tex2D);
        let white_3d = create_white(device.as_ref(), TextureBindKind::Tex3D);
        let black_3d = create_black(device.as_ref(), TextureBindKind::Tex3D);
        let gray_3d = create_gray(device.as_ref(), TextureBindKind::Tex3D);
        let red_3d = create_red(device.as_ref(), TextureBindKind::Tex3D);
        let white_cube = create_white(device.as_ref(), TextureBindKind::Cube);
        let black_cube = create_black(device.as_ref(), TextureBindKind::Cube);
        let gray_cube = create_gray(device.as_ref(), TextureBindKind::Cube);
        let red_cube = create_red(device.as_ref(), TextureBindKind::Cube);

        let default_sampler = Arc::new(default_embedded_sampler(device.as_ref()));

        Ok(Self {
            device: device.clone(),
            white_2d,
            black_2d,
            gray_2d,
            red_2d,
            flat_normal_2d,
            white_3d,
            black_3d,
            gray_3d,
            red_3d,
            white_cube,
            black_cube,
            gray_cube,
            red_cube,
            default_sampler,
            property_registry,
            stem_cache: Mutex::new(HashMap::new()),
            uniform_arena_shards: (0..EMBEDDED_CACHE_SHARDS)
                .map(|_| Mutex::new(MaterialUniformArena::new(device.clone(), limits.clone())))
                .collect(),
            uniform_arena_hasher: RandomState::new(),
            bind_cache: ShardedLru::new(max_cached_embedded_bind_groups(), EMBEDDED_CACHE_SHARDS),
            sampler_cache: ShardedLru::new(max_cached_embedded_samplers(), EMBEDDED_CACHE_SHARDS),
            texture_debug_cache: Mutex::new(LruCache::new(max_cached_texture_debug_ids())),
        })
    }

    /// Uploads texels into every placeholder texture (call once after creation with queue).
    pub fn write_default_textures(&self, queue: &wgpu::Queue) {
        upload_white(queue, &self.white_2d, TextureBindKind::Tex2D);
        upload_black(queue, &self.black_2d, TextureBindKind::Tex2D);
        upload_gray(queue, &self.gray_2d, TextureBindKind::Tex2D);
        upload_red(queue, &self.red_2d, TextureBindKind::Tex2D);
        upload_flat_normal(queue, &self.flat_normal_2d, TextureBindKind::Tex2D);
        upload_white(queue, &self.white_3d, TextureBindKind::Tex3D);
        upload_black(queue, &self.black_3d, TextureBindKind::Tex3D);
        upload_gray(queue, &self.gray_3d, TextureBindKind::Tex3D);
        upload_red(queue, &self.red_3d, TextureBindKind::Tex3D);
        upload_white(queue, &self.white_cube, TextureBindKind::Cube);
        upload_black(queue, &self.black_cube, TextureBindKind::Cube);
        upload_gray(queue, &self.gray_cube, TextureBindKind::Cube);
        upload_red(queue, &self.red_cube, TextureBindKind::Cube);
    }

    /// Purges embedded GPU cache entries tied to one unloaded material.
    pub(crate) fn purge_material_asset(&self, material_id: i32) {
        let mut materials = HashSet::new();
        materials.insert(material_id);
        self.purge_material_and_property_block_assets(&materials, &HashSet::new());
    }

    /// Purges embedded GPU cache entries tied to one unloaded property block.
    pub(crate) fn purge_property_block_asset(&self, property_block_id: i32) {
        let mut property_blocks = HashSet::new();
        property_blocks.insert(property_block_id);
        self.purge_material_and_property_block_assets(&HashSet::new(), &property_blocks);
    }

    /// Purges embedded GPU cache entries tied to unloaded material or property-block ids.
    pub(crate) fn purge_material_and_property_block_assets(
        &self,
        material_ids: &HashSet<i32>,
        property_block_ids: &HashSet<i32>,
    ) {
        if material_ids.is_empty() && property_block_ids.is_empty() {
            return;
        }
        profiling::scope!("materials::embedded_purge_material_assets");
        self.bind_cache.clear();
        self.texture_debug_cache.lock().clear();
        for shard in &self.uniform_arena_shards {
            shard
                .lock()
                .purge_material_assets(material_ids, property_block_ids);
        }
    }

    /// Purges bind groups that may retain texture views after texture assets unload.
    pub(crate) fn purge_texture_reference_caches(&self) {
        profiling::scope!("materials::embedded_purge_texture_reference_caches");
        self.bind_cache.clear();
        self.texture_debug_cache.lock().clear();
    }

    /// Returns or builds a `@group(1)` bind group for the composed embedded `stem`. Callers
    /// must thread the shader-specific Froox variant bitmask through
    /// [`EmbeddedMaterialBindShader::shader_variant_bits`]: hard-coding `None` zeroes
    /// `_RenderideVariantBits` in the packed uniform and breaks every keyword-driven
    /// branch in the shader (this is how the Projection360 skybox-pass black-render
    /// regression reached production).
    pub(crate) fn embedded_material_bind_group_with_cache_key(
        &self,
        shader: EmbeddedMaterialBindShader<'_>,
        uploads: GraphUploadSink<'_>,
        store: &MaterialPropertyStore,
        pools: &EmbeddedTexturePools<'_>,
        lookup: MaterialPropertyLookupIds,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Result<(MaterialBindCacheKey, EmbeddedMaterialBindGroup), EmbeddedMaterialBindError> {
        profiling::scope!("materials::embedded_bind_group");
        let EmbeddedBindInputResolution {
            layout,
            uniform_key,
            stem_hash,
            texture_bind_signature,
            texture_2d_asset_id,
        } = self.resolve_embedded_bind_inputs(
            shader.stem,
            shader.shader_variant_bits,
            store,
            pools,
            lookup,
            offscreen_write_render_texture_asset_id,
        )?;

        let mutation_gen = store.mutation_generation(lookup);
        let texture_state_sig = {
            profiling::scope!("materials::embedded_uniform_texture_signature");
            compute_uniform_texture_state_signature(
                &layout,
                pools,
                store,
                lookup,
                texture_2d_asset_id,
            )
        };
        let uniform_binding = if layout.reflected.material_uniform.is_some() {
            Some(
                self.get_or_update_embedded_uniform_arena_slot(EmbeddedUniformArenaRequest {
                    uploads,
                    stem: shader.stem,
                    shader_variant_bits: shader.shader_variant_bits,
                    layout: &layout,
                    uniform_key: &uniform_key,
                    mutation_gen,
                    store,
                    lookup,
                    pools,
                    primary_texture_2d: texture_2d_asset_id,
                    texture_state_sig,
                })?,
            )
        } else {
            None
        };
        let bind_key = build_material_bind_cache_key(
            stem_hash,
            lookup,
            texture_bind_signature,
            offscreen_write_render_texture_asset_id,
            uniform_binding.as_ref(),
        );

        let hit_bg = {
            profiling::scope!("materials::embedded_bind_cache_lookup");
            self.bind_cache.get_cloned(&bind_key)
        };
        if let Some(bg) = hit_bg {
            profiling::scope!("materials::embedded_bind_cache_hit");
            return Ok(material_bind_group_result(
                bind_key,
                bg,
                uniform_binding.as_ref(),
            ));
        }

        profiling::scope!("materials::embedded_bind_cache_miss");
        self.build_and_cache_embedded_bind_group(EmbeddedBindCacheMissInputs {
            layout: &layout,
            stem_hash,
            texture_2d_asset_id,
            pools,
            store,
            lookup,
            offscreen_write_render_texture_asset_id,
            lookup_bind_key: bind_key,
            lookup_texture_bind_signature: texture_bind_signature,
            uniform_binding: uniform_binding.as_ref(),
        })
    }

    fn build_and_cache_embedded_bind_group(
        &self,
        inputs: EmbeddedBindCacheMissInputs<'_>,
    ) -> Result<(MaterialBindCacheKey, EmbeddedMaterialBindGroup), EmbeddedMaterialBindError> {
        let EmbeddedBindCacheMissInputs {
            layout,
            stem_hash,
            texture_2d_asset_id,
            pools,
            store,
            lookup,
            offscreen_write_render_texture_asset_id,
            lookup_bind_key,
            lookup_texture_bind_signature,
            uniform_binding,
        } = inputs;

        let snapshot = self.snapshot_group1_textures_samplers(
            layout,
            texture_2d_asset_id,
            pools,
            store,
            lookup,
            offscreen_write_render_texture_asset_id,
        )?;

        // If pool state shifted between the lookup-side signature compute and the snapshot,
        // the bind group we are about to build matches the snapshot's signature, not the
        // lookup key. Re-key under the snapshot's signature and re-check the cache so we
        // never file a bind group whose key does not describe its contents.
        let final_bind_key = if snapshot.texture_bind_signature == lookup_texture_bind_signature {
            lookup_bind_key
        } else {
            let updated = build_material_bind_cache_key(
                stem_hash,
                lookup,
                snapshot.texture_bind_signature,
                offscreen_write_render_texture_asset_id,
                uniform_binding,
            );
            if let Some(bg) = self.bind_cache.get_cloned(&updated) {
                return Ok(material_bind_group_result(updated, bg, uniform_binding));
            }
            updated
        };

        let entries = build_embedded_bind_group_entries(
            layout,
            uniform_binding,
            &snapshot.views,
            &snapshot.samplers,
        )?;
        let bind_group = {
            profiling::scope!("materials::embedded_create_bind_group");
            let bind_group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("embedded_material_bind"),
                layout: &layout.bind_group_layout,
                entries: &entries,
            }));
            crate::profiling::note_resource_churn!(BindGroup, "materials::embedded_material_bind");
            bind_group
        };
        let evicted = self.bind_cache.put(final_bind_key, bind_group.clone());
        if let Some(evicted) = evicted {
            drop(evicted);
            logger::trace!("EmbeddedMaterialBindResources: evicted LRU bind group cache entry");
        }
        Ok(material_bind_group_result(
            final_bind_key,
            bind_group,
            uniform_binding,
        ))
    }

    /// Returns the reflected `@group(1)` bind-group layout for an embedded material stem.
    pub(crate) fn embedded_material_bind_group_layout(
        &self,
        stem: &str,
    ) -> Result<wgpu::BindGroupLayout, EmbeddedMaterialBindError> {
        self.stem_layout(stem)
            .map(|layout| layout.bind_group_layout.clone())
    }

    /// Routes a uniform cache key to its arena shard. A given key always maps to the same shard,
    /// so the per-shard generation tracked by [`MaterialUniformArena`] is self-consistent.
    fn uniform_arena_shard(&self, key: &MaterialUniformCacheKey) -> &Mutex<MaterialUniformArena> {
        let idx = (self.uniform_arena_hasher.hash_one(key) as usize) & (EMBEDDED_CACHE_SHARDS - 1);
        &self.uniform_arena_shards[idx]
    }
}
