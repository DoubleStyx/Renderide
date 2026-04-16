//! `@group(1)` bind groups for embedded raster materials (WGSL targets shipped with the renderer).
//!
//! Layouts and uniform packing come from [`crate::materials::reflect_raster_material_wgsl`] (naga).
//! WGSL identifiers in `@group(1)` match Unity [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)
//! names; [`crate::assets::material::PropertyIdRegistry`] resolves them to batch property ids.
//!
//! **UI text (`_TextMode`, `_RectClip`):** When a reflected uniform field is named `_TextMode` or `_RectClip`,
//! packing uses explicit `set_float` when present; otherwise keyword-style floats (`MSDF`, `RASTER`, `SDF`,
//! `RECTCLIP`, case variants) are interpreted the same way as legacy FrooxEngine/Unity keyword bindings—without
//! hard-coding a particular shader stem in the draw pass.

use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use wgpu::util::DeviceExt;

use super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::layout::{
    build_stem_material_layout, stem_hash, EmbeddedSharedKeywordIds, StemMaterialLayout,
};
use super::texture_pools::EmbeddedTexturePools;
use super::texture_resolve::{
    primary_texture_2d_asset_id, primary_texture_any_kind_present,
    resolved_texture_binding_for_host, sampler_from_cubemap_state, sampler_from_state,
    sampler_from_texture3d_state, texture_bind_signature, ResolvedTextureBinding,
};
use super::uniform_pack::build_embedded_uniform_bytes;
use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};

/// LRU cap for `@group(1)` bind groups (per unique material/texture signature).
const MAX_CACHED_EMBEDDED_BIND_GROUPS: usize = 512;
/// LRU cap for embedded material uniform buffers.
const MAX_CACHED_EMBEDDED_UNIFORMS: usize = 512;

type EmbeddedGroup1TexturesAndSamplers = (Vec<Arc<wgpu::TextureView>>, Vec<Arc<wgpu::Sampler>>);

/// GPU resources shared by embedded material bind groups (layouts, default texture, sampler).
pub struct EmbeddedMaterialBindResources {
    device: Arc<wgpu::Device>,
    white_texture: Arc<wgpu::Texture>,
    white_texture_view: Arc<wgpu::TextureView>,
    default_sampler: Arc<wgpu::Sampler>,
    property_registry: Arc<PropertyIdRegistry>,
    shared_keyword_ids: Arc<EmbeddedSharedKeywordIds>,
    stem_cache: RefCell<HashMap<String, Arc<StemMaterialLayout>>>,
    bind_cache: RefCell<LruCache<MaterialBindCacheKey, Arc<wgpu::BindGroup>>>,
    uniform_cache: RefCell<LruCache<MaterialUniformCacheKey, CachedUniformEntry>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct MaterialUniformCacheKey {
    stem_hash: u64,
    material_asset_id: i32,
    property_block_slot0: Option<i32>,
    texture_2d_asset_id: i32,
    /// Distinguishes RT-only primary slot from empty (`flags` bit 0).
    primary_texture_any_kind_present: bool,
}

/// Key for [`EmbeddedMaterialBindResources`] `@group(1)` bind-group cache (matches internal hashing).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct MaterialBindCacheKey {
    stem_hash: u64,
    material_asset_id: i32,
    property_block_slot0: Option<i32>,
    texture_bind_signature: u64,
    /// Distinguishes main vs secondary-RT passes when self-sampling is masked.
    offscreen_write_render_texture_asset_id: Option<i32>,
}

/// Cached GPU uniform buffer and last [`crate::assets::material::MaterialPropertyStore::mutation_generation`] uploaded to it.
struct CachedUniformEntry {
    buffer: Arc<wgpu::Buffer>,
    last_written_generation: u64,
}

/// Stem layout, uniform/bind cache keys, and resolved primary texture ids for embedded `@group(1)` wiring.
struct EmbeddedBindInputResolution {
    layout: Arc<StemMaterialLayout>,
    uniform_key: MaterialUniformCacheKey,
    bind_key: MaterialBindCacheKey,
    texture_2d_asset_id: i32,
    primary_texture_any_kind_present: bool,
}

/// Host texture binding lookup for [`EmbeddedMaterialBindResources::resolve_texture_view_for_host`] and
/// [`EmbeddedMaterialBindResources::resolve_sampler_for_host`].
struct HostTexturePropertyQuery<'a> {
    host_name: &'a str,
    texture_property_id: i32,
    primary_texture_2d: i32,
    pools: &'a EmbeddedTexturePools<'a>,
    store: &'a MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    offscreen_write_render_texture_asset_id: Option<i32>,
}

/// LRU uniform buffer create/refresh for [`EmbeddedMaterialBindResources::get_or_create_embedded_uniform_buffer`].
struct EmbeddedUniformBufferRequest<'a> {
    queue: &'a wgpu::Queue,
    stem: &'a str,
    layout: &'a Arc<StemMaterialLayout>,
    uniform_key: &'a MaterialUniformCacheKey,
    mutation_gen: u64,
    store: &'a MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    primary_texture_any_kind_present: bool,
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

        let default_sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("embedded_default_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }));

        let shared_keyword_ids =
            Arc::new(EmbeddedSharedKeywordIds::new(property_registry.as_ref()));

        Ok(Self {
            device,
            white_texture,
            white_texture_view,
            default_sampler,
            property_registry,
            shared_keyword_ids,
            stem_cache: RefCell::new(HashMap::new()),
            bind_cache: RefCell::new(LruCache::new(
                NonZeroUsize::new(MAX_CACHED_EMBEDDED_BIND_GROUPS)
                    .expect("MAX_CACHED_EMBEDDED_BIND_GROUPS > 0"),
            )),
            uniform_cache: RefCell::new(LruCache::new(
                NonZeroUsize::new(MAX_CACHED_EMBEDDED_UNIFORMS)
                    .expect("MAX_CACHED_EMBEDDED_UNIFORMS > 0"),
            )),
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
        let EmbeddedBindInputResolution {
            layout,
            uniform_key,
            bind_key,
            texture_2d_asset_id,
            primary_texture_any_kind_present,
        } = self.resolve_embedded_bind_inputs(
            stem,
            store,
            pools,
            lookup,
            offscreen_write_render_texture_asset_id,
        )?;

        let mutation_gen = store.mutation_generation(lookup);

        let uniform_buf =
            self.get_or_create_embedded_uniform_buffer(EmbeddedUniformBufferRequest {
                queue,
                stem,
                layout: &layout,
                uniform_key: &uniform_key,
                mutation_gen,
                store,
                lookup,
                primary_texture_any_kind_present,
            })?;

        let mut cache = self.bind_cache.borrow_mut();
        if let Some(bg) = cache.get(&bind_key) {
            return Ok((bind_key, bg.clone()));
        }

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

        let bind_group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedded_material_bind"),
            layout: &layout.bind_group_layout,
            entries: &entries,
        }));
        if let Some(evicted) = cache.put(bind_key, bind_group.clone()) {
            drop(evicted);
            logger::trace!("EmbeddedMaterialBindResources: evicted LRU bind group cache entry");
        }
        Ok((bind_key, bind_group))
    }

    /// Resolves stem layout, primary texture ids, texture signature, and LRU cache keys for embedded binds.
    fn resolve_embedded_bind_inputs(
        &self,
        stem: &str,
        store: &MaterialPropertyStore,
        pools: &EmbeddedTexturePools<'_>,
        lookup: MaterialPropertyLookupIds,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Result<EmbeddedBindInputResolution, EmbeddedMaterialBindError> {
        let layout = self.stem_layout(stem)?;
        let sh = stem_hash(stem);

        let texture_2d_asset_id =
            primary_texture_2d_asset_id(&layout.reflected, layout.ids.as_ref(), store, lookup);
        let primary_texture_any_kind_present =
            primary_texture_any_kind_present(&layout.reflected, layout.ids.as_ref(), store, lookup);
        let texture_bind_signature = texture_bind_signature(
            &layout.reflected,
            layout.ids.as_ref(),
            store,
            lookup,
            pools,
            texture_2d_asset_id,
            offscreen_write_render_texture_asset_id,
        );

        let uniform_key = MaterialUniformCacheKey {
            stem_hash: sh,
            material_asset_id: lookup.material_asset_id,
            property_block_slot0: lookup.mesh_property_block_slot0,
            texture_2d_asset_id,
            primary_texture_any_kind_present,
        };
        let bind_key = MaterialBindCacheKey {
            stem_hash: sh,
            material_asset_id: lookup.material_asset_id,
            property_block_slot0: lookup.mesh_property_block_slot0,
            texture_bind_signature,
            offscreen_write_render_texture_asset_id,
        };

        Ok(EmbeddedBindInputResolution {
            layout,
            uniform_key,
            bind_key,
            texture_2d_asset_id,
            primary_texture_any_kind_present,
        })
    }

    /// Resolves every non-uniform `@group(1)` texture view and sampler in reflection order.
    fn resolve_group1_textures_and_samplers(
        &self,
        layout: &Arc<StemMaterialLayout>,
        texture_2d_asset_id: i32,
        pools: &EmbeddedTexturePools<'_>,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Result<EmbeddedGroup1TexturesAndSamplers, EmbeddedMaterialBindError> {
        let mut keepalive_views: Vec<Arc<wgpu::TextureView>> = Vec::new();
        let mut keepalive_samplers: Vec<Arc<wgpu::Sampler>> = Vec::new();
        for entry in &layout.reflected.material_entries {
            let b = entry.binding;
            match entry.ty {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                } => {}
                wgpu::BindingType::Texture { .. } => {
                    let host_name = layout
                        .reflected
                        .material_group1_names
                        .get(&b)
                        .map(String::as_str)
                        .ok_or_else(|| {
                            format!("reflection: no WGSL name for texture @binding({b})")
                        })?;
                    let tex_pid = layout
                        .ids
                        .texture_binding_to_property_id
                        .get(&b)
                        .copied()
                        .ok_or_else(|| {
                            format!("reflection: missing property id for texture @binding({b})")
                        })?;
                    let tex_view = self
                        .resolve_texture_view_for_host(HostTexturePropertyQuery {
                            host_name,
                            texture_property_id: tex_pid,
                            primary_texture_2d: texture_2d_asset_id,
                            pools,
                            store,
                            lookup,
                            offscreen_write_render_texture_asset_id,
                        })
                        .unwrap_or_else(|| self.white_texture_view.clone());
                    keepalive_views.push(tex_view);
                }
                wgpu::BindingType::Sampler(_) => {
                    let tex_binding = sampler_pairs_texture_binding(b);
                    let host_name = layout
                        .reflected
                        .material_group1_names
                        .get(&tex_binding)
                        .map(String::as_str)
                        .ok_or_else(|| {
                            format!("reflection: no texture global for sampler @binding({b})")
                        })?;
                    let tex_pid = layout
                        .ids
                        .texture_binding_to_property_id
                        .get(&tex_binding)
                        .copied()
                        .ok_or_else(|| {
                            format!(
                                "reflection: missing property id for texture @binding({tex_binding})"
                            )
                        })?;
                    let sampler = self.resolve_sampler_for_host(HostTexturePropertyQuery {
                        host_name,
                        texture_property_id: tex_pid,
                        primary_texture_2d: texture_2d_asset_id,
                        pools,
                        store,
                        lookup,
                        offscreen_write_render_texture_asset_id,
                    });
                    keepalive_samplers.push(sampler);
                }
                _ => {
                    return Err(EmbeddedMaterialBindError::from(format!(
                        "unsupported binding type for @binding({b})"
                    )));
                }
            }
        }
        Ok((keepalive_views, keepalive_samplers))
    }

    /// LRU uniform buffer for embedded `@group(1)`; refreshes bytes when [`MaterialPropertyStore`] mutates.
    fn get_or_create_embedded_uniform_buffer(
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
            primary_texture_any_kind_present,
        } = req;
        let mut uniform_cache = self.uniform_cache.borrow_mut();
        if let Some(entry) = uniform_cache.get_mut(uniform_key) {
            if entry.last_written_generation == mutation_gen {
                return Ok(entry.buffer.clone());
            }
            let uniform_bytes = build_embedded_uniform_bytes(
                &layout.reflected,
                layout.ids.as_ref(),
                store,
                lookup,
                primary_texture_any_kind_present,
            )
            .ok_or_else(|| {
                format!("stem {stem}: uniform block missing (shader has no material uniform)")
            })?;
            queue.write_buffer(entry.buffer.as_ref(), 0, &uniform_bytes);
            entry.last_written_generation = mutation_gen;
            return Ok(entry.buffer.clone());
        }
        let uniform_bytes = build_embedded_uniform_bytes(
            &layout.reflected,
            layout.ids.as_ref(),
            store,
            lookup,
            primary_texture_any_kind_present,
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
        };
        if let Some(evicted) = uniform_cache.put(*uniform_key, entry) {
            drop(evicted);
            logger::trace!("EmbeddedMaterialBindResources: evicted LRU uniform cache entry");
        }
        Ok(buf)
    }

    fn stem_layout(
        &self,
        stem: &str,
    ) -> Result<Arc<StemMaterialLayout>, EmbeddedMaterialBindError> {
        let mut cache = self.stem_cache.borrow_mut();
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
        Ok(layout)
    }

    fn resolve_texture_view_for_host(
        &self,
        q: HostTexturePropertyQuery<'_>,
    ) -> Option<Arc<wgpu::TextureView>> {
        let binding = resolved_texture_binding_for_host(
            q.host_name,
            q.texture_property_id,
            q.primary_texture_2d,
            q.store,
            q.lookup,
        );
        self.resolve_texture_view(q.pools, binding, q.offscreen_write_render_texture_asset_id)
    }

    fn resolve_sampler_for_host(&self, q: HostTexturePropertyQuery<'_>) -> Arc<wgpu::Sampler> {
        let binding = resolved_texture_binding_for_host(
            q.host_name,
            q.texture_property_id,
            q.primary_texture_2d,
            q.store,
            q.lookup,
        );
        self.resolve_sampler(q.pools, binding, q.offscreen_write_render_texture_asset_id)
    }

    fn resolve_texture_view(
        &self,
        pools: &EmbeddedTexturePools<'_>,
        binding: ResolvedTextureBinding,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Option<Arc<wgpu::TextureView>> {
        match binding {
            ResolvedTextureBinding::None => None,
            ResolvedTextureBinding::Texture2D { asset_id } => {
                if asset_id < 0 {
                    return None;
                }
                pools
                    .texture
                    .get_texture(asset_id)
                    .filter(|t| t.mip_levels_resident > 0)
                    .map(|t| t.view.clone())
            }
            ResolvedTextureBinding::Texture3D { .. } | ResolvedTextureBinding::Cubemap { .. } => {
                // Embedded stems use `texture_2d` bindings; 3D/cube assets need shader/layout variants.
                None
            }
            ResolvedTextureBinding::RenderTexture { asset_id } => {
                if asset_id < 0 {
                    return None;
                }
                if offscreen_write_render_texture_asset_id == Some(asset_id) {
                    return None;
                }
                pools
                    .render_texture
                    .get(asset_id)
                    .filter(|t| t.is_sampleable())
                    .map(|t| t.color_view.clone())
            }
        }
    }

    fn resolve_sampler(
        &self,
        pools: &EmbeddedTexturePools<'_>,
        binding: ResolvedTextureBinding,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Arc<wgpu::Sampler> {
        match binding {
            ResolvedTextureBinding::None => self.default_sampler.clone(),
            ResolvedTextureBinding::Texture2D { asset_id } => {
                if asset_id < 0 {
                    return self.default_sampler.clone();
                }
                let Some(tex) = pools.texture.get_texture(asset_id) else {
                    return self.default_sampler.clone();
                };
                Arc::new(sampler_from_state(&self.device, &tex.sampler))
            }
            ResolvedTextureBinding::Texture3D { asset_id } => {
                if asset_id < 0 {
                    return self.default_sampler.clone();
                }
                let Some(tex) = pools.texture3d.get_texture(asset_id) else {
                    return self.default_sampler.clone();
                };
                Arc::new(sampler_from_texture3d_state(&self.device, &tex.sampler))
            }
            ResolvedTextureBinding::Cubemap { asset_id } => {
                if asset_id < 0 {
                    return self.default_sampler.clone();
                }
                let Some(tex) = pools.cubemap.get_texture(asset_id) else {
                    return self.default_sampler.clone();
                };
                Arc::new(sampler_from_cubemap_state(&self.device, &tex.sampler))
            }
            ResolvedTextureBinding::RenderTexture { asset_id } => {
                if asset_id < 0 {
                    return self.default_sampler.clone();
                }
                if offscreen_write_render_texture_asset_id == Some(asset_id) {
                    return self.default_sampler.clone();
                }
                let Some(tex) = pools.render_texture.get(asset_id) else {
                    return self.default_sampler.clone();
                };
                Arc::new(sampler_from_state(&self.device, &tex.sampler))
            }
        }
    }
}

/// Second pass: assemble [`wgpu::BindGroupEntry`] list matching reflected material entry order.
fn build_embedded_bind_group_entries<'a>(
    layout: &'a Arc<StemMaterialLayout>,
    uniform_buf: &'a Arc<wgpu::Buffer>,
    keepalive_views: &'a [Arc<wgpu::TextureView>],
    keepalive_samplers: &'a [Arc<wgpu::Sampler>],
) -> Result<Vec<wgpu::BindGroupEntry<'a>>, EmbeddedMaterialBindError> {
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
fn sampler_pairs_texture_binding(sampler_binding: u32) -> u32 {
    sampler_binding.saturating_sub(1)
}
