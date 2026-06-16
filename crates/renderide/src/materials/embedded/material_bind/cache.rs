//! LRU caches and stem layout memoization for embedded `@group(1)` bind groups.

use std::num::NonZeroUsize;
use std::sync::Arc;

use super::super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::super::layout::{StemMaterialLayout, build_stem_material_layout, stem_hash};
use super::super::texture_resolve::{
    ResolvedTextureBinding, primary_texture_2d_asset_id, resolved_texture_binding_for_host,
    texture_property_ids_for_binding,
};
use crate::frame_contract::OffscreenWriteTarget;
use crate::gpu_pools::SamplerState;
use crate::materials::host_data::{MaterialPropertyLookupIds, MaterialPropertyStore};

/// Number of shards across which the embedded `@group(1)` bind, uniform, and sampler caches are
/// split. Each shard owns its own [`parking_lot::Mutex`] over [`lru::LruCache`]; per-view rayon workers
/// hash their cache key into a shard index and only contend with workers whose keys hash into the
/// same shard. 16 is enough to keep contention sub-linear up through ~16-core rayon pools while
/// keeping the per-shard LRU large enough to track the working set.
pub(super) const EMBEDDED_CACHE_SHARDS: usize = 16;

/// LRU cap for `@group(1)` bind groups (per stem/texture signature/arena generation).
///
/// Inspector-heavy worlds can expose thousands of distinct material/texture signatures in a
/// single frame. A small cache churns and recreates bind groups during draw preparation.
pub(super) const MAX_CACHED_EMBEDDED_BIND_GROUPS: usize = 16_384;
/// LRU cap for embedded samplers.
pub(super) const MAX_CACHED_EMBEDDED_SAMPLERS: usize = 512;
/// LRU cap for texture HUD asset-id scans.
pub(super) const MAX_CACHED_TEXTURE_DEBUG_IDS: usize = 512;

/// Non-zero bind-group cache capacity.
pub(super) fn max_cached_embedded_bind_groups() -> NonZeroUsize {
    NonZeroUsize::new(MAX_CACHED_EMBEDDED_BIND_GROUPS).unwrap_or(NonZeroUsize::MIN)
}

/// Non-zero sampler cache capacity.
pub(super) fn max_cached_embedded_samplers() -> NonZeroUsize {
    NonZeroUsize::new(MAX_CACHED_EMBEDDED_SAMPLERS).unwrap_or(NonZeroUsize::MIN)
}

/// Non-zero texture debug-id cache capacity.
pub(super) fn max_cached_texture_debug_ids() -> NonZeroUsize {
    NonZeroUsize::new(MAX_CACHED_TEXTURE_DEBUG_IDS).unwrap_or(NonZeroUsize::MIN)
}

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
    /// Builds a Texture2D sampler cache key. `wrap_w` is intentionally set to `wrap_u` to
    /// preserve the prior cache distribution; 2D bind paths never sample on the W axis.
    pub(super) fn texture2d(state: &SamplerState, mip_levels_resident: u32) -> Self {
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

    /// Builds a Texture3D sampler cache key, including the W wrap mode.
    pub(super) fn texture3d(state: &SamplerState, mip_levels_resident: u32) -> Self {
        Self {
            dimension: 3,
            filter_mode: state.filter_mode as i32,
            aniso_level: state.aniso_level,
            wrap_u: state.wrap_u as i32,
            wrap_v: state.wrap_v as i32,
            wrap_w: state.wrap_w as i32,
            mipmap_bias_bits: state.mipmap_bias.to_bits(),
            mip_levels_resident,
        }
    }

    /// Builds a cubemap sampler cache key. `wrap_w` follows `wrap_u` because the host cubemap
    /// properties carry no third axis.
    pub(super) fn cubemap(state: &SamplerState, mip_levels_resident: u32) -> Self {
        Self {
            dimension: 4,
            filter_mode: state.filter_mode as i32,
            aniso_level: state.aniso_level,
            wrap_u: state.wrap_u as i32,
            wrap_v: state.wrap_v as i32,
            wrap_w: state.wrap_u as i32,
            mipmap_bias_bits: state.mipmap_bias.to_bits(),
            mip_levels_resident,
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
    /// Host material asset id when the reflected layout has a material uniform block.
    /// No-uniform layouts canonicalize this field because the bind group only depends on
    /// texture/sampler state.
    pub(super) material_asset_id: i32,
    /// Optional per-slot `MaterialPropertyBlock` id for uniform-backed layouts.
    pub(super) property_block_slot0: Option<i32>,
    /// Optional renderer-level `MaterialPropertyBlock` id for uniform-backed layouts.
    pub(super) renderer_property_block_id: Option<i32>,
    pub(super) texture_bind_signature: u64,
    /// Distinguishes direct, untracked offscreen, and host render-texture write targets.
    pub(super) offscreen_write_target: OffscreenWriteTarget,
    /// Bumps whenever the shared material uniform arena reallocates to a new GPU buffer.
    pub(super) uniform_arena_generation: u64,
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
        // texture HUD can scan thousands of draws; cache by material mutation.
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
        if let Some(hit) = self.sampler_cache.get_cloned(&key) {
            return hit;
        }
        // sampler objects are cheap-ish, but bind misses can make lots of them.
        let sampler = Arc::new(create());
        let evicted = self.sampler_cache.put(key, sampler.clone());
        drop(evicted);
        sampler
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{TextureFilterMode, TextureWrapMode};

    fn texture2d_state() -> SamplerState {
        SamplerState {
            filter_mode: TextureFilterMode::Bilinear,
            aniso_level: 4,
            wrap_u: TextureWrapMode::Repeat,
            wrap_v: TextureWrapMode::Clamp,
            wrap_w: TextureWrapMode::default(),
            mipmap_bias: 0.25,
        }
    }

    fn texture3d_state() -> SamplerState {
        SamplerState {
            filter_mode: TextureFilterMode::Trilinear,
            aniso_level: 8,
            wrap_u: TextureWrapMode::Repeat,
            wrap_v: TextureWrapMode::Mirror,
            wrap_w: TextureWrapMode::Clamp,
            mipmap_bias: 0.0,
        }
    }

    fn cubemap_state() -> SamplerState {
        SamplerState {
            filter_mode: TextureFilterMode::Anisotropic,
            aniso_level: 12,
            wrap_u: TextureWrapMode::Repeat,
            wrap_v: TextureWrapMode::Repeat,
            wrap_w: TextureWrapMode::default(),
            mipmap_bias: -0.5,
        }
    }

    #[test]
    fn texture2d_sampler_cache_key_tracks_mode_affecting_fields() {
        let base = texture2d_state();
        let base_key = EmbeddedSamplerCacheKey::texture2d(&base, 4);

        let mut changed = base.clone();
        changed.filter_mode = TextureFilterMode::Trilinear;
        assert_ne!(base_key, EmbeddedSamplerCacheKey::texture2d(&changed, 4));

        let mut changed = base.clone();
        changed.aniso_level = 16;
        assert_ne!(base_key, EmbeddedSamplerCacheKey::texture2d(&changed, 4));

        let mut changed = base.clone();
        changed.wrap_v = TextureWrapMode::Mirror;
        assert_ne!(base_key, EmbeddedSamplerCacheKey::texture2d(&changed, 4));

        let mut changed = base.clone();
        changed.mipmap_bias = -1.0;
        assert_ne!(base_key, EmbeddedSamplerCacheKey::texture2d(&changed, 4));

        assert_ne!(base_key, EmbeddedSamplerCacheKey::texture2d(&base, 3));
    }

    #[test]
    fn texture3d_and_cubemap_sampler_cache_keys_track_residency() {
        let texture3d = texture3d_state();
        assert_ne!(
            EmbeddedSamplerCacheKey::texture3d(&texture3d, 2),
            EmbeddedSamplerCacheKey::texture3d(&texture3d, 3)
        );

        let cubemap = cubemap_state();
        assert_ne!(
            EmbeddedSamplerCacheKey::cubemap(&cubemap, 5),
            EmbeddedSamplerCacheKey::cubemap(&cubemap, 6)
        );
    }
}
