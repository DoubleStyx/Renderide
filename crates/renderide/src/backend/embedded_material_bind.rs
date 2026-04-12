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
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};
use crate::resources::TexturePool;

use super::embedded_material_layout::{
    build_stem_material_layout, stem_hash, EmbeddedSharedKeywordIds, StemMaterialLayout,
};
use super::embedded_texture_resolve::{
    primary_texture_2d_asset_id, resolved_texture_asset_id_for_host, sampler_from_state,
    texture_bind_signature,
};
use super::embedded_uniform_pack::build_embedded_uniform_bytes;

/// GPU resources shared by embedded material bind groups (layouts, default texture, sampler).
pub struct EmbeddedMaterialBindResources {
    device: Arc<wgpu::Device>,
    white_texture: Arc<wgpu::Texture>,
    white_texture_view: Arc<wgpu::TextureView>,
    default_sampler: Arc<wgpu::Sampler>,
    property_registry: Arc<PropertyIdRegistry>,
    shared_keyword_ids: Arc<EmbeddedSharedKeywordIds>,
    stem_cache: RefCell<HashMap<String, Arc<StemMaterialLayout>>>,
    bind_cache: RefCell<HashMap<MaterialBindCacheKey, Arc<wgpu::BindGroup>>>,
    uniform_cache: RefCell<HashMap<MaterialUniformCacheKey, CachedUniformEntry>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct MaterialUniformCacheKey {
    stem_hash: u64,
    material_asset_id: i32,
    property_block_slot0: Option<i32>,
    texture_2d_asset_id: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct MaterialBindCacheKey {
    stem_hash: u64,
    material_asset_id: i32,
    property_block_slot0: Option<i32>,
    texture_bind_signature: u64,
}

/// Cached GPU uniform buffer and last [`crate::assets::material::MaterialPropertyStore::mutation_generation`] uploaded to it.
struct CachedUniformEntry {
    buffer: Arc<wgpu::Buffer>,
    last_written_generation: u64,
}

impl EmbeddedMaterialBindResources {
    /// Builds layouts and placeholder texture.
    pub fn new(
        device: Arc<wgpu::Device>,
        property_registry: Arc<PropertyIdRegistry>,
    ) -> Result<Self, String> {
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
            bind_cache: RefCell::new(HashMap::new()),
            uniform_cache: RefCell::new(HashMap::new()),
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
    pub fn embedded_material_bind_group(
        &self,
        stem: &str,
        queue: &wgpu::Queue,
        store: &MaterialPropertyStore,
        texture_pool: &TexturePool,
        lookup: MaterialPropertyLookupIds,
    ) -> Result<Arc<wgpu::BindGroup>, String> {
        let layout = self.stem_layout(stem)?;
        let sh = stem_hash(stem);

        let texture_2d_asset_id =
            primary_texture_2d_asset_id(&layout.reflected, layout.ids.as_ref(), store, lookup);
        let texture_bind_signature = texture_bind_signature(
            &layout.reflected,
            layout.ids.as_ref(),
            store,
            lookup,
            texture_pool,
            texture_2d_asset_id,
        );

        let uniform_key = MaterialUniformCacheKey {
            stem_hash: sh,
            material_asset_id: lookup.material_asset_id,
            property_block_slot0: lookup.mesh_property_block_slot0,
            texture_2d_asset_id,
        };
        let bind_key = MaterialBindCacheKey {
            stem_hash: sh,
            material_asset_id: lookup.material_asset_id,
            property_block_slot0: lookup.mesh_property_block_slot0,
            texture_bind_signature,
        };

        let mutation_gen = store.mutation_generation(lookup);

        let uniform_buf = {
            let mut uniform_cache = self.uniform_cache.borrow_mut();
            match uniform_cache.entry(uniform_key) {
                Entry::Occupied(mut o) => {
                    if o.get().last_written_generation == mutation_gen {
                        o.get().buffer.clone()
                    } else {
                        let uniform_bytes = build_embedded_uniform_bytes(
                            &layout.reflected,
                            layout.ids.as_ref(),
                            store,
                            lookup,
                            texture_2d_asset_id,
                        )
                        .ok_or_else(|| {
                            format!(
                                "stem {stem}: uniform block missing (shader has no material uniform)"
                            )
                        })?;
                        let e = o.get_mut();
                        queue.write_buffer(e.buffer.as_ref(), 0, &uniform_bytes);
                        e.last_written_generation = mutation_gen;
                        e.buffer.clone()
                    }
                }
                Entry::Vacant(v) => {
                    let uniform_bytes = build_embedded_uniform_bytes(
                        &layout.reflected,
                        layout.ids.as_ref(),
                        store,
                        lookup,
                        texture_2d_asset_id,
                    )
                    .ok_or_else(|| {
                        format!(
                            "stem {stem}: uniform block missing (shader has no material uniform)"
                        )
                    })?;
                    let buf = Arc::new(self.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("embedded_material_uniform"),
                            contents: &uniform_bytes,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        },
                    ));
                    v.insert(CachedUniformEntry {
                        buffer: buf.clone(),
                        last_written_generation: mutation_gen,
                    });
                    buf
                }
            }
        };

        let mut cache = self.bind_cache.borrow_mut();
        if let Some(bg) = cache.get(&bind_key) {
            return Ok(bg.clone());
        }

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
                        .resolve_texture_view_for_host(
                            host_name,
                            tex_pid,
                            texture_2d_asset_id,
                            texture_pool,
                            store,
                            lookup,
                        )
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
                    let sampler = self.resolve_sampler_for_host(
                        host_name,
                        tex_pid,
                        texture_2d_asset_id,
                        texture_pool,
                        store,
                        lookup,
                    );
                    keepalive_samplers.push(sampler);
                }
                _ => {
                    return Err(format!("unsupported binding type for @binding({b})"));
                }
            }
        }

        let mut view_i = 0usize;
        let mut samp_i = 0usize;
        let mut entries: Vec<wgpu::BindGroupEntry<'_>> =
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
                    return Err(format!("unsupported binding type for @binding({b})"));
                }
            }
        }

        let bind_group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedded_material_bind"),
            layout: &layout.bind_group_layout,
            entries: &entries,
        }));
        cache.insert(bind_key, bind_group.clone());
        Ok(bind_group)
    }

    fn stem_layout(&self, stem: &str) -> Result<Arc<StemMaterialLayout>, String> {
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
        host_name: &str,
        texture_property_id: i32,
        primary_texture_2d: i32,
        texture_pool: &TexturePool,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
    ) -> Option<Arc<wgpu::TextureView>> {
        let id = resolved_texture_asset_id_for_host(
            host_name,
            texture_property_id,
            primary_texture_2d,
            store,
            lookup,
        );
        self.resolve_texture_view(texture_pool, id)
    }

    fn resolve_sampler_for_host(
        &self,
        host_name: &str,
        texture_property_id: i32,
        primary_texture_2d: i32,
        texture_pool: &TexturePool,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
    ) -> Arc<wgpu::Sampler> {
        let tid = resolved_texture_asset_id_for_host(
            host_name,
            texture_property_id,
            primary_texture_2d,
            store,
            lookup,
        );
        self.resolve_sampler(texture_pool, tid)
    }

    fn resolve_texture_view(
        &self,
        texture_pool: &TexturePool,
        texture_asset_id: i32,
    ) -> Option<Arc<wgpu::TextureView>> {
        if texture_asset_id < 0 {
            return None;
        }
        texture_pool
            .get_texture(texture_asset_id)
            .filter(|t| t.mip_levels_resident > 0)
            .map(|t| t.view.clone())
    }

    fn resolve_sampler(
        &self,
        texture_pool: &TexturePool,
        texture_asset_id: i32,
    ) -> Arc<wgpu::Sampler> {
        if texture_asset_id < 0 {
            return self.default_sampler.clone();
        }
        let Some(tex) = texture_pool.get_texture(texture_asset_id) else {
            return self.default_sampler.clone();
        };
        Arc::new(sampler_from_state(&self.device, &tex.sampler))
    }
}

fn sampler_pairs_texture_binding(sampler_binding: u32) -> u32 {
    sampler_binding.saturating_sub(1)
}
