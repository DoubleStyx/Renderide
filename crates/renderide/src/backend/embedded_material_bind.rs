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
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::assets::texture::texture2d_asset_id_from_packed;
use crate::embedded_shaders;
use crate::materials::{
    reflect_raster_material_wgsl, ReflectedRasterLayout, ReflectedUniformField,
    ReflectedUniformScalarKind,
};
use crate::resources::{Texture2dSamplerState, TexturePool};

/// GPU resources shared by embedded material bind groups (layouts, default texture, sampler).
pub struct EmbeddedMaterialBindResources {
    device: Arc<wgpu::Device>,
    white_texture: Arc<wgpu::Texture>,
    white_texture_view: Arc<wgpu::TextureView>,
    default_sampler: Arc<wgpu::Sampler>,
    property_registry: Arc<PropertyIdRegistry>,
    stem_cache: RefCell<HashMap<String, Arc<StemMaterialLayout>>>,
    bind_cache: RefCell<HashMap<MaterialBindCacheKey, Arc<wgpu::BindGroup>>>,
    uniform_cache: RefCell<HashMap<MaterialUniformCacheKey, Arc<wgpu::Buffer>>>,
}

/// Cached reflection for one composed stem.
struct StemMaterialLayout {
    bind_group_layout: wgpu::BindGroupLayout,
    reflected: ReflectedRasterLayout,
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
    texture_2d_asset_id: i32,
    texture_gpu_resident: bool,
}

fn stem_hash(stem: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    stem.hash(&mut h);
    h.finish()
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

        Ok(Self {
            device,
            white_texture,
            white_texture_view,
            default_sampler,
            property_registry,
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

        let texture_2d_asset_id = primary_texture_2d_asset_id(
            &layout.reflected,
            store,
            lookup,
            self.property_registry.as_ref(),
        );
        let texture_gpu_resident =
            texture_2d_asset_id >= 0 && texture_pool.get_texture(texture_2d_asset_id).is_some();

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
            texture_2d_asset_id,
            texture_gpu_resident,
        };

        let uniform_bytes = self
            .build_uniform_bytes(&layout, store, lookup, texture_2d_asset_id)
            .ok_or_else(|| {
                format!("stem {stem}: uniform block missing (shader has no material uniform)")
            })?;

        let uniform_buf = {
            let mut uniform_cache = self.uniform_cache.borrow_mut();
            if let Some(buf) = uniform_cache.get(&uniform_key) {
                queue.write_buffer(buf.as_ref(), 0, &uniform_bytes);
                buf.clone()
            } else {
                let buf = Arc::new(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("embedded_material_uniform"),
                        contents: &uniform_bytes,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                uniform_cache.insert(uniform_key, buf.clone());
                buf
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
                    let tex_view = self
                        .resolve_texture_view_for_host(
                            host_name,
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
                    let sampler = self.resolve_sampler_for_host(
                        host_name,
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

        let wgsl = embedded_shaders::embedded_target_wgsl(stem)
            .ok_or_else(|| format!("embedded WGSL missing for stem {stem}"))?;
        let reflected =
            reflect_raster_material_wgsl(wgsl).map_err(|e| format!("reflect {stem}: {e}"))?;

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("embedded_raster_material"),
                    entries: &reflected.material_entries,
                });

        let layout = Arc::new(StemMaterialLayout {
            bind_group_layout,
            reflected,
        });
        cache.insert(stem.to_string(), layout.clone());
        Ok(layout)
    }

    fn build_uniform_bytes(
        &self,
        layout: &StemMaterialLayout,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        texture_2d_for_key: i32,
    ) -> Option<Vec<u8>> {
        let u = layout.reflected.material_uniform.as_ref()?;
        let mut buf = vec![0u8; u.total_size as usize];

        // `_Cutoff` is tracked when present so `flags` packing can mirror Unity alpha-test heuristics for any
        // material whose WGSL declares these identifiers (reflection-driven, not per-shader names in Rust).
        let mut cutoff = 0.5f32;
        for (field_name, field) in &u.fields {
            let pid = self.property_registry.intern(field_name);
            match field.kind {
                ReflectedUniformScalarKind::Vec4 => {
                    let mut v = [1.0f32, 1.0, 1.0, 1.0];
                    if let Some(MaterialPropertyValue::Float4(c)) = store.get_merged(lookup, pid) {
                        v = *c;
                    }
                    write_f32x4_at(&mut buf, field, &v);
                }
                ReflectedUniformScalarKind::F32 => {
                    let v = if field_name == "_TextMode" {
                        packed_text_mode_f32(store, lookup, self.property_registry.as_ref())
                    } else if field_name == "_RectClip" {
                        packed_rect_clip_f32(store, lookup, self.property_registry.as_ref())
                    } else if let Some(MaterialPropertyValue::Float(f)) =
                        store.get_merged(lookup, pid)
                    {
                        *f
                    } else {
                        0.5f32
                    };
                    if field_name == "_Cutoff" {
                        cutoff = v;
                    }
                    write_f32_at(&mut buf, field, v);
                }
                ReflectedUniformScalarKind::U32 => {
                    if field_name == "flags" {
                        let flags = pack_flags_u32(
                            field_name,
                            store,
                            lookup,
                            self.property_registry.as_ref(),
                            texture_2d_for_key,
                            cutoff,
                        );
                        write_u32_at(&mut buf, field, flags);
                    }
                }
                ReflectedUniformScalarKind::Unsupported => {}
            }
        }

        Some(buf)
    }

    fn resolve_texture_view_for_host(
        &self,
        host_name: &str,
        primary_texture_2d: i32,
        texture_pool: &TexturePool,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
    ) -> Option<Arc<wgpu::TextureView>> {
        let pid = self.property_registry.intern(host_name);
        let tid = match store.get_merged(lookup, pid) {
            Some(MaterialPropertyValue::Texture(packed)) => {
                texture2d_asset_id_from_packed(*packed).unwrap_or(-1)
            }
            _ => -1,
        };
        let id = if tid >= 0 { tid } else { primary_texture_2d };
        self.resolve_texture_view(texture_pool, id)
    }

    fn resolve_sampler_for_host(
        &self,
        host_name: &str,
        primary_texture_2d: i32,
        texture_pool: &TexturePool,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
    ) -> Arc<wgpu::Sampler> {
        let pid = self.property_registry.intern(host_name);
        let tid = store
            .get_merged(lookup, pid)
            .and_then(|v| match v {
                MaterialPropertyValue::Texture(p) => texture2d_asset_id_from_packed(*p),
                _ => None,
            })
            .unwrap_or(primary_texture_2d);
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

/// True when the host material has a `set_float` for `name` with value ≥ 0.5 (Unity shader keyword pattern).
fn keyword_float_enabled(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    registry: &PropertyIdRegistry,
    name: &str,
) -> bool {
    let pid = registry.intern(name);
    matches!(
        store.get_merged(lookup, pid),
        Some(MaterialPropertyValue::Float(f)) if *f >= 0.5
    )
}

/// Packs `UI_TextUnlit`-style `_TextMode`: explicit `0`/`1`/`2`, else keyword floats `MSDF` / `SDF` / `RASTER`, else `0` (MSDF default).
fn packed_text_mode_f32(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    registry: &PropertyIdRegistry,
) -> f32 {
    let pid = registry.intern("_TextMode");
    if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, pid) {
        return *f;
    }
    if keyword_float_enabled(store, lookup, registry, "MSDF") {
        return 0.0;
    }
    if keyword_float_enabled(store, lookup, registry, "SDF") {
        return 2.0;
    }
    if keyword_float_enabled(store, lookup, registry, "RASTER") {
        return 1.0;
    }
    if keyword_float_enabled(store, lookup, registry, "msdf") {
        return 0.0;
    }
    if keyword_float_enabled(store, lookup, registry, "sdf") {
        return 2.0;
    }
    if keyword_float_enabled(store, lookup, registry, "raster") {
        return 1.0;
    }
    0.0
}

/// Packs `_RectClip`: explicit value, else `RECTCLIP` / `rectclip` keyword floats, else `0`.
fn packed_rect_clip_f32(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    registry: &PropertyIdRegistry,
) -> f32 {
    let pid = registry.intern("_RectClip");
    if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, pid) {
        return *f;
    }
    if keyword_float_enabled(store, lookup, registry, "RECTCLIP") {
        return 1.0;
    }
    if keyword_float_enabled(store, lookup, registry, "rectclip") {
        return 1.0;
    }
    0.0
}

fn primary_texture_2d_asset_id(
    reflected: &ReflectedRasterLayout,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_registry: &PropertyIdRegistry,
) -> i32 {
    for entry in &reflected.material_entries {
        if matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            let Some(name) = reflected.material_group1_names.get(&entry.binding) else {
                continue;
            };
            let pid = property_registry.intern(name);
            if let Some(MaterialPropertyValue::Texture(packed)) = store.get_merged(lookup, pid) {
                return texture2d_asset_id_from_packed(*packed).unwrap_or(-1);
            }
        }
    }
    -1
}

/// Packs `flags` from `_Flags` when present; otherwise derives texture + alpha-test bits by convention.
fn pack_flags_u32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_registry: &PropertyIdRegistry,
    texture_2d_for_key: i32,
    cutoff: f32,
) -> u32 {
    if field_name != "flags" {
        return 0;
    }
    let flags_pid = property_registry.intern("_Flags");
    if let Some(v) = store.get_merged(lookup, flags_pid) {
        match v {
            MaterialPropertyValue::Float(f) => return (*f).max(0.0) as u32,
            MaterialPropertyValue::Float4(a) => return a[0].max(0.0) as u32,
            _ => {}
        }
    }
    let mut flags = 0u32;
    if texture_2d_for_key >= 0 {
        flags |= 1u32;
    }
    if cutoff > 0.0 && cutoff < 1.0 {
        flags |= 2u32;
    }
    flags
}

fn write_f32_at(buf: &mut [u8], field: &ReflectedUniformField, v: f32) {
    let off = field.offset as usize;
    if off + 4 <= buf.len() && field.size >= 4 {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn write_f32x4_at(buf: &mut [u8], field: &ReflectedUniformField, v: &[f32; 4]) {
    let off = field.offset as usize;
    if off + 16 <= buf.len() && field.size >= 16 {
        for (i, c) in v.iter().enumerate() {
            let o = off + i * 4;
            buf[o..o + 4].copy_from_slice(&c.to_le_bytes());
        }
    }
}

fn write_u32_at(buf: &mut [u8], field: &ReflectedUniformField, v: u32) {
    let off = field.offset as usize;
    if off + 4 <= buf.len() && field.size >= 4 {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn sampler_from_state(device: &wgpu::Device, state: &Texture2dSamplerState) -> wgpu::Sampler {
    let address_mode_u = match state.wrap_u {
        crate::shared::TextureWrapMode::repeat => wgpu::AddressMode::Repeat,
        crate::shared::TextureWrapMode::clamp => wgpu::AddressMode::ClampToEdge,
        crate::shared::TextureWrapMode::mirror => wgpu::AddressMode::MirrorRepeat,
        crate::shared::TextureWrapMode::mirror_once => wgpu::AddressMode::ClampToEdge,
    };
    let address_mode_v = match state.wrap_v {
        crate::shared::TextureWrapMode::repeat => wgpu::AddressMode::Repeat,
        crate::shared::TextureWrapMode::clamp => wgpu::AddressMode::ClampToEdge,
        crate::shared::TextureWrapMode::mirror => wgpu::AddressMode::MirrorRepeat,
        crate::shared::TextureWrapMode::mirror_once => wgpu::AddressMode::ClampToEdge,
    };
    let (mag, min, mipmap) = match state.filter_mode {
        crate::shared::TextureFilterMode::point => (
            wgpu::FilterMode::Nearest,
            wgpu::FilterMode::Nearest,
            wgpu::MipmapFilterMode::Nearest,
        ),
        crate::shared::TextureFilterMode::bilinear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
        crate::shared::TextureFilterMode::trilinear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
        crate::shared::TextureFilterMode::anisotropic => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("embedded_texture_sampler"),
        address_mode_u,
        address_mode_v,
        address_mode_w: address_mode_u,
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: mipmap,
        ..Default::default()
    })
}

#[cfg(test)]
mod text_uniform_packing_tests {
    use super::*;
    use crate::assets::material::{MaterialPropertyLookupIds, MaterialPropertyStore};

    fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id: material_id,
            mesh_property_block_slot0: None,
        }
    }

    #[test]
    fn packed_text_mode_defaults_to_msdf_when_empty() {
        let store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        assert_eq!(
            packed_text_mode_f32(&store, lookup(1), &reg),
            0.0,
            "FrooxEngine default GlyphRenderMethod is MSDF"
        );
    }

    #[test]
    fn packed_text_mode_explicit_overrides_keywords() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let pid_tm = reg.intern("_TextMode");
        let pid_msdf = reg.intern("MSDF");
        store.set_material(1, pid_msdf, MaterialPropertyValue::Float(1.0));
        store.set_material(1, pid_tm, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_text_mode_f32(&store, lookup(1), &reg), 1.0);
    }

    #[test]
    fn packed_text_mode_raster_from_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let pid = reg.intern("RASTER");
        store.set_material(2, pid, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_text_mode_f32(&store, lookup(2), &reg), 1.0);
    }

    #[test]
    fn packed_rect_clip_from_rectclip_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let pid = reg.intern("RECTCLIP");
        store.set_material(3, pid, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_rect_clip_f32(&store, lookup(3), &reg), 1.0);
    }
}
