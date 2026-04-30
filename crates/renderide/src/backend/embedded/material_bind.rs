//! `@group(1)` bind groups for embedded raster materials (WGSL targets shipped with the renderer).
//!
//! Layouts and uniform packing come from [`crate::materials::reflect_raster_material_wgsl`] (naga).
//! WGSL identifiers in `@group(1)` match Unity [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)
//! names; [`crate::materials::host_data::PropertyIdRegistry`] resolves them to batch property ids.
//!
//! **UI text (`_TextMode`, `_RectClip`):** When a reflected uniform field is named `_TextMode` or `_RectClip`,
//! packing uses explicit `set_float` when present; otherwise keyword-style floats (`MSDF`, `RASTER`, `SDF`,
//! `RECTCLIP`, case variants) are interpreted the same way FrooxEngine/Unity keyword bindings are—without
//! hard-coding a particular shader stem in the draw pass.

mod cache;
mod resolve;
mod uniform;

pub(crate) use cache::MaterialBindCacheKey;

use hashbrown::HashMap;
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

use super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::layout::{EmbeddedSharedKeywordIds, StemMaterialLayout};
use super::texture_pools::EmbeddedTexturePools;
use super::texture_resolve::default_embedded_sampler;
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};

use cache::{
    EmbeddedSamplerCacheKey, TextureDebugCacheKey, max_cached_embedded_bind_groups,
    max_cached_embedded_samplers, max_cached_embedded_uniforms, max_cached_texture_debug_ids,
};
use uniform::{CachedUniformEntry, EmbeddedUniformBufferRequest, MaterialUniformCacheKey};

use resolve::EmbeddedBindInputResolution;

/// GPU resources shared by embedded material bind groups (layouts, default texture, sampler).
pub struct EmbeddedMaterialBindResources {
    device: Arc<wgpu::Device>,
    white_texture: Arc<wgpu::Texture>,
    white_texture_view: Arc<wgpu::TextureView>,
    white_texture3d: Arc<wgpu::Texture>,
    white_texture3d_view: Arc<wgpu::TextureView>,
    white_cubemap_texture: Arc<wgpu::Texture>,
    white_cubemap_view: Arc<wgpu::TextureView>,
    default_sampler: Arc<wgpu::Sampler>,
    property_registry: Arc<PropertyIdRegistry>,
    shared_keyword_ids: Arc<EmbeddedSharedKeywordIds>,
    stem_cache: Mutex<HashMap<String, Arc<StemMaterialLayout>>>,
    bind_cache: Mutex<LruCache<MaterialBindCacheKey, Arc<wgpu::BindGroup>>>,
    uniform_cache: Mutex<LruCache<MaterialUniformCacheKey, CachedUniformEntry>>,
    sampler_cache: Mutex<LruCache<EmbeddedSamplerCacheKey, Arc<wgpu::Sampler>>>,
    texture_debug_cache: Mutex<LruCache<TextureDebugCacheKey, Arc<[i32]>>>,
}

impl EmbeddedMaterialBindResources {
    /// Builds layouts and placeholder texture.
    pub fn new(
        device: Arc<wgpu::Device>,
        property_registry: Arc<PropertyIdRegistry>,
    ) -> Result<Self, EmbeddedMaterialBindError> {
        let white_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("embedded_default_white"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let white_texture_view =
            Arc::new(white_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let white_texture3d = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("embedded_default_white_3d"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let white_texture3d_view =
            Arc::new(white_texture3d.create_view(&wgpu::TextureViewDescriptor {
                label: Some("embedded_default_white_3d_view"),
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            }));
        let white_cubemap_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("embedded_default_white_cube"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let white_cubemap_view = Arc::new(white_cubemap_texture.create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("embedded_default_white_cube_view"),
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            },
        ));

        let default_sampler = Arc::new(default_embedded_sampler(device.as_ref()));

        let shared_keyword_ids =
            Arc::new(EmbeddedSharedKeywordIds::new(property_registry.as_ref()));

        Ok(Self {
            device,
            white_texture,
            white_texture_view,
            white_texture3d,
            white_texture3d_view,
            white_cubemap_texture,
            white_cubemap_view,
            default_sampler,
            property_registry,
            shared_keyword_ids,
            stem_cache: Mutex::new(HashMap::new()),
            bind_cache: Mutex::new(LruCache::new(max_cached_embedded_bind_groups())),
            uniform_cache: Mutex::new(LruCache::new(max_cached_embedded_uniforms())),
            sampler_cache: Mutex::new(LruCache::new(max_cached_embedded_samplers())),
            texture_debug_cache: Mutex::new(LruCache::new(max_cached_texture_debug_ids())),
        })
    }

    /// Uploads white texel into the placeholder texture (call once after creation with queue).
    pub fn write_default_white(&self, queue: &wgpu::Queue) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.white_texture.as_ref(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.white_texture3d.as_ref(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.white_cubemap_texture.as_ref(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[
                255u8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255,
            ],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
        );
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
            let mut cache = self.bind_cache.lock();
            cache.get(&bind_key).cloned()
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
        let evicted = self.bind_cache.lock().put(bind_key, bind_group.clone());
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

/// Second pass: assemble [`wgpu::BindGroupEntry`] list matching reflected material entry order.
fn build_embedded_bind_group_entries<'a>(
    layout: &'a Arc<StemMaterialLayout>,
    uniform_buf: &'a Arc<wgpu::Buffer>,
    keepalive_views: &'a [Arc<wgpu::TextureView>],
    keepalive_samplers: &'a [Arc<wgpu::Sampler>],
) -> Result<Vec<wgpu::BindGroupEntry<'a>>, EmbeddedMaterialBindError> {
    profiling::scope!("materials::embedded_build_bind_entries");
    let mut view_i = 0usize;
    let mut samp_i = 0usize;
    let mut entries: Vec<wgpu::BindGroupEntry<'a>> =
        Vec::with_capacity(layout.reflected.material_entries.len());
    for entry in &layout.reflected.material_entries {
        let b = entry.binding;
        match entry.ty {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                ..
            } => {
                entries.push(wgpu::BindGroupEntry {
                    binding: b,
                    resource: uniform_buf.as_entire_binding(),
                });
            }
            wgpu::BindingType::Texture { .. } => {
                let tv = keepalive_views
                    .get(view_i)
                    .ok_or_else(|| format!("internal: texture view index {view_i}"))?;
                view_i += 1;
                entries.push(wgpu::BindGroupEntry {
                    binding: b,
                    resource: wgpu::BindingResource::TextureView(tv.as_ref()),
                });
            }
            wgpu::BindingType::Sampler(_) => {
                let s = keepalive_samplers
                    .get(samp_i)
                    .ok_or_else(|| format!("internal: sampler index {samp_i}"))?;
                samp_i += 1;
                entries.push(wgpu::BindGroupEntry {
                    binding: b,
                    resource: wgpu::BindingResource::Sampler(s.as_ref()),
                });
            }
            _ => {
                return Err(EmbeddedMaterialBindError::from(format!(
                    "unsupported binding type for @binding({b})"
                )));
            }
        }
    }
    Ok(entries)
}

#[inline]
pub(super) fn sampler_pairs_texture_binding(sampler_binding: u32) -> u32 {
    sampler_binding.saturating_sub(1)
}

/// Hashes current texture metadata used by reflected material-uniform packing.
///
/// Uniform packing sources `_<Tex>_LodBias` and `_<Tex>_StorageVInverted` fields from texture
/// state, but [`MaterialPropertyStore::mutation_generation`] does not bump when textures update,
/// so this signature drives uniform-buffer refresh on those metadata changes.
fn compute_uniform_texture_state_signature(
    layout: &Arc<StemMaterialLayout>,
    pools: &EmbeddedTexturePools<'_>,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    primary_texture_2d: i32,
) -> u64 {
    use ahash::AHasher;
    use std::hash::{Hash, Hasher};

    use super::texture_resolve::{
        ResolvedTextureBinding, resolved_texture_binding_for_host, texture_property_ids_for_binding,
    };

    let mut h = AHasher::default();
    for entry in &layout.reflected.material_entries {
        if !matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            continue;
        }
        let Some(name) = layout.reflected.material_group1_names.get(&entry.binding) else {
            continue;
        };
        let pids = texture_property_ids_for_binding(layout.ids.as_ref(), entry.binding);
        if pids.is_empty() {
            continue;
        }
        let binding = resolved_texture_binding_for_host(
            name.as_str(),
            pids,
            primary_texture_2d,
            store,
            lookup,
        );
        entry.binding.hash(&mut h);
        let (bias, storage_v_inverted): (f32, bool) = match binding {
            ResolvedTextureBinding::Texture2D { asset_id } => pools
                .texture
                .get_texture(asset_id)
                .map_or((0.0, false), |t| {
                    (t.sampler.mipmap_bias, t.storage_v_inverted)
                }),
            ResolvedTextureBinding::Texture3D { asset_id } => pools
                .texture3d
                .get_texture(asset_id)
                .map_or((0.0, false), |t| (t.sampler.mipmap_bias, false)),
            ResolvedTextureBinding::Cubemap { asset_id } => pools
                .cubemap
                .get_texture(asset_id)
                .map_or((0.0, false), |t| {
                    (t.sampler.mipmap_bias, t.storage_v_inverted)
                }),
            ResolvedTextureBinding::RenderTexture { .. } => (0.0, true),
            ResolvedTextureBinding::VideoTexture { .. } => (0.0, false),
            ResolvedTextureBinding::None => (0.0, false),
        };
        bias.to_bits().hash(&mut h);
        storage_v_inverted.hash(&mut h);
    }
    h.finish()
}
