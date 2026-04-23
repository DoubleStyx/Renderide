//! LRU caches and stem layout memoization for embedded `@group(1)` bind groups.

use std::num::NonZeroUsize;
use std::sync::Arc;

use super::super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::super::layout::{build_stem_material_layout, stem_hash, StemMaterialLayout};
use super::super::texture_resolve::{
    primary_texture_2d_asset_id, resolved_texture_binding_for_host,
    texture_property_ids_for_binding, ResolvedTextureBinding,
};
use crate::assets::material::{MaterialPropertyLookupIds, MaterialPropertyStore};
use crate::resources::{CubemapSamplerState, Texture2dSamplerState, Texture3dSamplerState};

/// LRU cap for `@group(1)` bind groups (per unique material/texture signature).
pub(super) const MAX_CACHED_EMBEDDED_BIND_GROUPS: usize = 512;
/// LRU cap for embedded material uniform buffers.
pub(super) const MAX_CACHED_EMBEDDED_UNIFORMS: usize = 512;
/// LRU cap for embedded samplers.
pub(super) const MAX_CACHED_EMBEDDED_SAMPLERS: usize = 512;
/// LRU cap for texture HUD asset-id scans.
pub(super) const MAX_CACHED_TEXTURE_DEBUG_IDS: usize = 512;

pub(super) const MAX_CACHED_EMBEDDED_BIND_GROUPS_NZ: NonZeroUsize = {
    match NonZeroUsize::new(MAX_CACHED_EMBEDDED_BIND_GROUPS) {
        Some(n) => n,
        None => panic!("MAX_CACHED_EMBEDDED_BIND_GROUPS must be non-zero"),
    }
};
pub(super) const MAX_CACHED_EMBEDDED_UNIFORMS_NZ: NonZeroUsize = {
    match NonZeroUsize::new(MAX_CACHED_EMBEDDED_UNIFORMS) {
        Some(n) => n,
        None => panic!("MAX_CACHED_EMBEDDED_UNIFORMS must be non-zero"),
    }
};
pub(super) const MAX_CACHED_EMBEDDED_SAMPLERS_NZ: NonZeroUsize = {
    match NonZeroUsize::new(MAX_CACHED_EMBEDDED_SAMPLERS) {
        Some(n) => n,
        None => panic!("MAX_CACHED_EMBEDDED_SAMPLERS must be non-zero"),
    }
};
pub(super) const MAX_CACHED_TEXTURE_DEBUG_IDS_NZ: NonZeroUsize = {
    match NonZeroUsize::new(MAX_CACHED_TEXTURE_DEBUG_IDS) {
        Some(n) => n,
        None => panic!("MAX_CACHED_TEXTURE_DEBUG_IDS must be non-zero"),
    }
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) struct EmbeddedSamplerCacheKey {
    pub(super) dimension: u8,
    pub(super) filter_mode: i32,
    pub(super) aniso_level: i32,
    pub(super) wrap_u: i32,
    pub(super) wrap_v: i32,
    pub(super) wrap_w: i32,
    pub(super) mipmap_bias_bits: u32,
    pub(super) mip_levels_resident: u32,
}

impl EmbeddedSamplerCacheKey {
    pub(super) fn texture2d(state: &Texture2dSamplerState, mip_levels_resident: u32) -> Self {
        Self {
            dimension: 2,
            filter_mode: state.filter_mode as i32,
            aniso_level: state.aniso_level,
            wrap_u: state.wrap_u as i32,
            wrap_v: state.wrap_v as i32,
            wrap_w: state.wrap_u as i32,
            mipmap_bias_bits: state.mipmap_bias.to_bits(),
            mip_levels_resident,
        }
    }

    pub(super) fn texture3d(state: &Texture3dSamplerState) -> Self {
        Self {
            dimension: 3,
            filter_mode: state.filter_mode as i32,
            aniso_level: state.aniso_level,
            wrap_u: state.wrap_u as i32,
            wrap_v: state.wrap_v as i32,
            wrap_w: state.wrap_w as i32,
            mipmap_bias_bits: state.mipmap_bias.to_bits(),
            mip_levels_resident: 0,
        }
    }

    pub(super) fn cubemap(state: &CubemapSamplerState) -> Self {
        Self {
            dimension: 4,
            filter_mode: state.filter_mode as i32,
            aniso_level: state.aniso_level,
            wrap_u: state.wrap_u as i32,
            wrap_v: state.wrap_v as i32,
            wrap_w: state.wrap_u as i32,
            mipmap_bias_bits: state.mipmap_bias.to_bits(),
            mip_levels_resident: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) struct TextureDebugCacheKey {
    pub(super) stem_hash: u64,
    pub(super) material_asset_id: i32,
    pub(super) property_block_slot0: Option<i32>,
    pub(super) mutation_generation: u64,
}

/// Key for [`EmbeddedMaterialBindResources`](super::EmbeddedMaterialBindResources) `@group(1)` bind-group cache (matches internal hashing).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct MaterialBindCacheKey {
    pub(super) stem_hash: u64,
    pub(super) material_asset_id: i32,
    pub(super) property_block_slot0: Option<i32>,
    pub(super) texture_bind_signature: u64,
    /// Distinguishes main vs secondary-RT passes when self-sampling is masked.
    pub(super) offscreen_write_render_texture_asset_id: Option<i32>,
}

use super::EmbeddedMaterialBindResources;

impl EmbeddedMaterialBindResources {
    pub(super) fn stem_layout(
        &self,
        stem: &str,
    ) -> Result<Arc<StemMaterialLayout>, EmbeddedMaterialBindError> {
        let mut cache = self.stem_cache.lock();
        if let Some(s) = cache.get(stem) {
            return Ok(s.clone());
        }

        let layout = build_stem_material_layout(
            self.device.as_ref(),
            stem,
            &self.shared_keyword_ids,
            self.property_registry.as_ref(),
        )?;
        cache.insert(stem.to_string(), layout.clone());
        drop(cache);
        Ok(layout)
    }

    /// Returns Texture2D asset ids referenced by a material draw for the texture debug HUD.
    pub(crate) fn texture2d_asset_ids_for_stem(
        &self,
        stem: &str,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
    ) -> Vec<i32> {
        let Ok(layout) = self.stem_layout(stem) else {
            return Vec::new();
        };
        let cache_key = TextureDebugCacheKey {
            stem_hash: stem_hash(stem),
            material_asset_id: lookup.material_asset_id,
            property_block_slot0: lookup.mesh_property_block_slot0,
            mutation_generation: store.mutation_generation(lookup),
        };
        {
            let mut cache = self.texture_debug_cache.lock();
            if let Some(hit) = cache.get(&cache_key) {
                return hit.to_vec();
            }
        }
        let primary_texture_2d =
            primary_texture_2d_asset_id(&layout.reflected, layout.ids.as_ref(), store, lookup);
        let mut out = Vec::new();
        for entry in &layout.reflected.material_entries {
            if !matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
                continue;
            }
            let Some(host_name) = layout.reflected.material_group1_names.get(&entry.binding) else {
                continue;
            };
            let texture_pids = texture_property_ids_for_binding(layout.ids.as_ref(), entry.binding);
            if texture_pids.is_empty() {
                continue;
            }
            let ResolvedTextureBinding::Texture2D { asset_id } = resolved_texture_binding_for_host(
                host_name.as_str(),
                texture_pids,
                primary_texture_2d,
                store,
                lookup,
            ) else {
                continue;
            };
            if asset_id >= 0 && !out.contains(&asset_id) {
                out.push(asset_id);
            }
        }
        //perf xlinka: texture HUD can scan thousands of draws; cache by material mutation.
        self.texture_debug_cache
            .lock()
            .put(cache_key, Arc::from(out.clone()));
        out
    }

    pub(super) fn cached_sampler(
        &self,
        key: EmbeddedSamplerCacheKey,
        create: impl FnOnce() -> wgpu::Sampler,
    ) -> Arc<wgpu::Sampler> {
        {
            let mut cache = self.sampler_cache.lock();
            if let Some(hit) = cache.get(&key) {
                return hit.clone();
            }
        }
        //perf xlinka: sampler objects are cheap-ish, but bind misses can make lots of them.
        let sampler = Arc::new(create());
        if let Some(evicted) = self.sampler_cache.lock().put(key, sampler.clone()) {
            drop(evicted);
        }
        sampler
    }
}
