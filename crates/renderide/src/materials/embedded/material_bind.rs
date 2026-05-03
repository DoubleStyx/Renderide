//! `@group(1)` bind groups for embedded raster materials (WGSL targets shipped with the renderer).
//!
//! Layouts and uniform packing come from [`crate::materials::reflect_raster_material_wgsl`] (naga).
//! WGSL identifiers in `@group(1)` match Unity [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)
//! names; [`crate::materials::host_data::PropertyIdRegistry`] resolves them to batch property ids.
//!
//! **Text/UI controls (`_TextMode`, `_RectClip`, `_OVERLAY`):** packing uses canonical
//! `set_float` values when present. Missing `_TextMode` is inferred from `_FontAtlas` texture
//! profile, while UI-only controls default to zero instead of keyword-style alias inference.

mod assemble;
mod cache;
mod resolve;
mod texture_signature;
mod uniform;
mod white_texture;

pub(crate) use cache::MaterialBindCacheKey;

use hashbrown::HashMap;
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

use super::bind_kind::TextureBindKind;
use super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::layout::{EmbeddedSharedKeywordIds, StemMaterialLayout};
use super::texture_pools::EmbeddedTexturePools;
use super::texture_resolve::default_embedded_sampler;
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};

use assemble::build_embedded_bind_group_entries;
use cache::{
    EMBEDDED_CACHE_SHARDS, EmbeddedSamplerCacheKey, ShardedLru, TextureDebugCacheKey,
    max_cached_embedded_bind_groups, max_cached_embedded_samplers, max_cached_embedded_uniforms,
    max_cached_texture_debug_ids,
};
use texture_signature::compute_uniform_texture_state_signature;
use uniform::{CachedUniformEntry, EmbeddedUniformBufferRequest, MaterialUniformCacheKey};
use white_texture::{WhiteTexture, create_white, upload_white};

use resolve::EmbeddedBindInputResolution;

/// GPU resources shared by embedded material bind groups (layouts, default texture, sampler).
pub struct EmbeddedMaterialBindResources {
    device: Arc<wgpu::Device>,
    white_2d: WhiteTexture,
    white_3d: WhiteTexture,
    white_cube: WhiteTexture,
    default_sampler: Arc<wgpu::Sampler>,
    property_registry: Arc<PropertyIdRegistry>,
    shared_keyword_ids: Arc<EmbeddedSharedKeywordIds>,
    stem_cache: Mutex<HashMap<String, Arc<StemMaterialLayout>>>,
    /// Sharded LRU caches for `@group(1)` bind groups, packed uniform buffers, and samplers.
    /// Each shard is a `parking_lot::Mutex<LruCache<...>>` so per-view rayon workers contend
    /// only with workers whose cache key hashes into the same shard. Replaces the previous
    /// single-`Mutex<LruCache<...>>` whose lock was the dominant contention point during
    /// `graph::per_view_fan_out`.
    bind_cache: ShardedLru<MaterialBindCacheKey, Arc<wgpu::BindGroup>>,
    uniform_cache: ShardedLru<MaterialUniformCacheKey, CachedUniformEntry>,
    sampler_cache: ShardedLru<EmbeddedSamplerCacheKey, Arc<wgpu::Sampler>>,
    /// Texture-debug HUD cache stays a single mutex: per-stem call frequency is low and the
    /// HUD does not run inside the per-view rayon fan-out, so sharding has no benefit here.
    texture_debug_cache: Mutex<LruCache<TextureDebugCacheKey, Arc<[i32]>>>,
}

impl EmbeddedMaterialBindResources {
    /// Builds layouts and placeholder texture.
    pub fn new(
        device: Arc<wgpu::Device>,
        property_registry: Arc<PropertyIdRegistry>,
    ) -> Result<Self, EmbeddedMaterialBindError> {
        let white_2d = create_white(device.as_ref(), TextureBindKind::Tex2D);
        let white_3d = create_white(device.as_ref(), TextureBindKind::Tex3D);
        let white_cube = create_white(device.as_ref(), TextureBindKind::Cube);

        let default_sampler = Arc::new(default_embedded_sampler(device.as_ref()));

        let shared_keyword_ids =
            Arc::new(EmbeddedSharedKeywordIds::new(property_registry.as_ref()));

        Ok(Self {
            device,
            white_2d,
            white_3d,
            white_cube,
            default_sampler,
            property_registry,
            shared_keyword_ids,
            stem_cache: Mutex::new(HashMap::new()),
            bind_cache: ShardedLru::new(max_cached_embedded_bind_groups(), EMBEDDED_CACHE_SHARDS),
            uniform_cache: ShardedLru::new(max_cached_embedded_uniforms(), EMBEDDED_CACHE_SHARDS),
            sampler_cache: ShardedLru::new(max_cached_embedded_samplers(), EMBEDDED_CACHE_SHARDS),
            texture_debug_cache: Mutex::new(LruCache::new(max_cached_texture_debug_ids())),
        })
    }

    /// Uploads white texel into every placeholder texture (call once after creation with queue).
    pub fn write_default_white(&self, queue: &wgpu::Queue) {
        upload_white(queue, &self.white_2d, TextureBindKind::Tex2D);
        upload_white(queue, &self.white_3d, TextureBindKind::Tex3D);
        upload_white(queue, &self.white_cube, TextureBindKind::Cube);
    }

    /// Returns or builds a `@group(1)` bind group for the composed embedded `stem` (e.g. `unlit_default`).
    #[inline]
    pub fn embedded_material_bind_group(
        &self,
        stem: &str,
        queue: &wgpu::Queue,
        store: &MaterialPropertyStore,
        pools: &EmbeddedTexturePools<'_>,
        lookup: MaterialPropertyLookupIds,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Result<Arc<wgpu::BindGroup>, EmbeddedMaterialBindError> {
        self.embedded_material_bind_group_with_cache_key(
            stem,
            queue,
            store,
            pools,
            lookup,
            offscreen_write_render_texture_asset_id,
        )
        .map(|(_, g)| g)
    }

    /// Same as [`Self::embedded_material_bind_group`], plus the cache key so callers can skip redundant
    /// [`wgpu::RenderPass::set_bind_group`] calls when the key matches the previous draw.
    pub(crate) fn embedded_material_bind_group_with_cache_key(
        &self,
        stem: &str,
        queue: &wgpu::Queue,
        store: &MaterialPropertyStore,
        pools: &EmbeddedTexturePools<'_>,
        lookup: MaterialPropertyLookupIds,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Result<(MaterialBindCacheKey, Arc<wgpu::BindGroup>), EmbeddedMaterialBindError> {
        profiling::scope!("materials::embedded_bind_group");
        let EmbeddedBindInputResolution {
            layout,
            uniform_key,
            bind_key,
            texture_2d_asset_id,
        } = self.resolve_embedded_bind_inputs(
            stem,
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

        let hit_bg = {
            profiling::scope!("materials::embedded_bind_cache_lookup");
            self.bind_cache.get_cloned(&bind_key)
        };
        if let Some(bg) = hit_bg {
            profiling::scope!("materials::embedded_bind_cache_hit");
            // Bind group is unchanged; still refresh the uniform slab if the material store mutated.
            let _uniform_buf =
                self.get_or_create_embedded_uniform_buffer(EmbeddedUniformBufferRequest {
                    queue,
                    stem,
                    layout: &layout,
                    uniform_key: &uniform_key,
                    mutation_gen,
                    store,
                    lookup,
                    pools,
                    primary_texture_2d: texture_2d_asset_id,
                    texture_state_sig,
                })?;
            return Ok((bind_key, bg));
        }

        profiling::scope!("materials::embedded_bind_cache_miss");
        let uniform_buf =
            self.get_or_create_embedded_uniform_buffer(EmbeddedUniformBufferRequest {
                queue,
                stem,
                layout: &layout,
                uniform_key: &uniform_key,
                mutation_gen,
                store,
                lookup,
                pools,
                primary_texture_2d: texture_2d_asset_id,
                texture_state_sig,
            })?;

        let (keepalive_views, keepalive_samplers) = self.resolve_group1_textures_and_samplers(
            &layout,
            texture_2d_asset_id,
            pools,
            store,
            lookup,
            offscreen_write_render_texture_asset_id,
        )?;

        let entries = build_embedded_bind_group_entries(
            &layout,
            &uniform_buf,
            &keepalive_views,
            &keepalive_samplers,
        )?;

        let bind_group = {
            profiling::scope!("materials::embedded_create_bind_group");
            Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("embedded_material_bind"),
                layout: &layout.bind_group_layout,
                entries: &entries,
            }))
        };
        let evicted = self.bind_cache.put(bind_key, bind_group.clone());
        if let Some(evicted) = evicted {
            drop(evicted);
            logger::trace!("EmbeddedMaterialBindResources: evicted LRU bind group cache entry");
        }
        Ok((bind_key, bind_group))
    }

    /// Returns the reflected `@group(1)` bind-group layout for an embedded material stem.
    pub(crate) fn embedded_material_bind_group_layout(
        &self,
        stem: &str,
    ) -> Result<wgpu::BindGroupLayout, EmbeddedMaterialBindError> {
        self.stem_layout(stem)
            .map(|layout| layout.bind_group_layout.clone())
    }
}
