//! Embedded `@group(1)` uniform buffer LRU and upload.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use super::super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::super::layout::StemMaterialLayout;
use super::super::texture_pools::EmbeddedTexturePools;
use super::super::uniform_pack::{build_embedded_uniform_bytes, UniformPackTextureContext};
use crate::assets::material::{MaterialPropertyLookupIds, MaterialPropertyStore};

/// Cached GPU uniform buffer, last store-mutation generation, and last bound-texture bias signature.
///
/// Bias signature tracks host `mipmap_bias` for the currently-bound textures; the store's
/// mutation generation does not bump on texture-property updates, so buffered LOD-bias fields
/// would otherwise become stale after a host bias write. Both must match to skip reupload.
pub(super) struct CachedUniformEntry {
    pub(super) buffer: Arc<wgpu::Buffer>,
    pub(super) last_written_generation: u64,
    pub(super) last_written_bias_sig: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) struct MaterialUniformCacheKey {
    pub(super) stem_hash: u64,
    pub(super) material_asset_id: i32,
    pub(super) property_block_slot0: Option<i32>,
    pub(super) texture_2d_asset_id: i32,
}

/// LRU uniform buffer create/refresh for [`super::EmbeddedMaterialBindResources::get_or_create_embedded_uniform_buffer`].
pub(super) struct EmbeddedUniformBufferRequest<'a> {
    pub(super) queue: &'a wgpu::Queue,
    pub(super) stem: &'a str,
    pub(super) layout: &'a Arc<StemMaterialLayout>,
    pub(super) uniform_key: &'a MaterialUniformCacheKey,
    pub(super) mutation_gen: u64,
    pub(super) store: &'a MaterialPropertyStore,
    pub(super) lookup: MaterialPropertyLookupIds,
    pub(super) pools: &'a EmbeddedTexturePools<'a>,
    pub(super) primary_texture_2d: i32,
    pub(super) bias_sig: u64,
}

use super::EmbeddedMaterialBindResources;

impl EmbeddedMaterialBindResources {
    /// LRU uniform buffer for embedded `@group(1)`; refreshes bytes when [`MaterialPropertyStore`] mutates
    /// or when the bound-texture `mipmap_bias` signature changes (relevant for `_<Tex>_LodBias` fields).
    pub(super) fn get_or_create_embedded_uniform_buffer(
        &self,
        req: EmbeddedUniformBufferRequest<'_>,
    ) -> Result<Arc<wgpu::Buffer>, EmbeddedMaterialBindError> {
        let EmbeddedUniformBufferRequest {
            queue,
            stem,
            layout,
            uniform_key,
            mutation_gen,
            store,
            lookup,
            pools,
            primary_texture_2d,
            bias_sig,
        } = req;
        let tex_ctx = UniformPackTextureContext {
            pools,
            primary_texture_2d,
        };
        let mut uniform_cache = self.uniform_cache.lock();
        if let Some(entry) = uniform_cache.get_mut(uniform_key) {
            if entry.last_written_generation == mutation_gen
                && entry.last_written_bias_sig == bias_sig
            {
                return Ok(entry.buffer.clone());
            }
            let uniform_bytes = build_embedded_uniform_bytes(
                &layout.reflected,
                layout.ids.as_ref(),
                store,
                lookup,
                &tex_ctx,
            )
            .ok_or_else(|| {
                format!("stem {stem}: uniform block missing (shader has no material uniform)")
            })?;
            queue.write_buffer(entry.buffer.as_ref(), 0, &uniform_bytes);
            entry.last_written_generation = mutation_gen;
            entry.last_written_bias_sig = bias_sig;
            return Ok(entry.buffer.clone());
        }
        let uniform_bytes = build_embedded_uniform_bytes(
            &layout.reflected,
            layout.ids.as_ref(),
            store,
            lookup,
            &tex_ctx,
        )
        .ok_or_else(|| {
            format!("stem {stem}: uniform block missing (shader has no material uniform)")
        })?;
        let buf = Arc::new(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("embedded_material_uniform"),
                    contents: &uniform_bytes,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }),
        );
        let entry = CachedUniformEntry {
            buffer: buf.clone(),
            last_written_generation: mutation_gen,
            last_written_bias_sig: bias_sig,
        };
        if let Some(evicted) = uniform_cache.put(*uniform_key, entry) {
            drop(evicted);
            logger::trace!("EmbeddedMaterialBindResources: evicted LRU uniform cache entry");
        }
        drop(uniform_cache);
        Ok(buf)
    }
}
