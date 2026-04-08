//! `@group(1)` bind groups for manifest raster materials (e.g. [`crate::materials::MANIFEST_RASTER_FAMILY_ID`] world Unlit).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::embedded_shaders;
use crate::materials::{reflect_raster_material_wgsl, WorldUnlitPropertyIds};
use crate::resources::{Texture2dSamplerState, TexturePool};

/// GPU resources shared by manifest material bind groups (layouts, default texture, sampler).
pub struct ManifestMaterialBindResources {
    device: Arc<wgpu::Device>,
    world_unlit_bind_group_layout: wgpu::BindGroupLayout,
    white_texture: Arc<wgpu::Texture>,
    white_texture_view: Arc<wgpu::TextureView>,
    default_sampler: Arc<wgpu::Sampler>,
    world_unlit_ids: WorldUnlitPropertyIds,
    cache: RefCell<HashMap<WorldUnlitBindKey, Arc<wgpu::BindGroup>>>,
    uniform_cache: RefCell<HashMap<WorldUnlitBindKey, Arc<wgpu::Buffer>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct WorldUnlitBindKey {
    material_asset_id: i32,
    property_block_slot0: Option<i32>,
    /// Host texture asset id for `_Tex`, or `-1` when using the default white texture.
    texture_asset_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UnlitMaterialGpu {
    color: [f32; 4],
    tex_st: [f32; 4],
    cutoff: f32,
    flags: u32,
    _pad: [f32; 2],
}

impl ManifestMaterialBindResources {
    /// Builds layouts and placeholder texture; interns Unlit property names on `property_registry`.
    pub fn new(
        device: Arc<wgpu::Device>,
        property_registry: &PropertyIdRegistry,
    ) -> Result<Self, String> {
        let wgsl = embedded_shaders::embedded_target_wgsl("world_unlit_default")
            .ok_or_else(|| "embedded world_unlit_default missing".to_string())?;
        let reflected =
            reflect_raster_material_wgsl(wgsl).map_err(|e| format!("reflect world_unlit: {e}"))?;
        let world_unlit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("world_unlit_material"),
                entries: &reflected.material_entries,
            });

        let white_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("manifest_default_white"),
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
            label: Some("manifest_default_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }));

        Ok(Self {
            device,
            world_unlit_bind_group_layout,
            white_texture,
            white_texture_view,
            default_sampler,
            world_unlit_ids: WorldUnlitPropertyIds::new(property_registry),
            cache: RefCell::new(HashMap::new()),
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

    /// Returns or builds a `@group(1)` bind group for world Unlit using the current material store.
    pub fn world_unlit_bind_group(
        &self,
        queue: &wgpu::Queue,
        store: &MaterialPropertyStore,
        texture_pool: &TexturePool,
        lookup: MaterialPropertyLookupIds,
    ) -> Arc<wgpu::BindGroup> {
        let ids = &self.world_unlit_ids;
        let texture_asset_id = match store.get_merged(lookup, ids.tex) {
            Some(MaterialPropertyValue::Texture(t)) if *t >= 0 => *t,
            _ => -1,
        };
        let key = WorldUnlitBindKey {
            material_asset_id: lookup.material_asset_id,
            property_block_slot0: lookup.mesh_property_block_slot0,
            texture_asset_id,
        };

        let u = self.build_unlit_uniform(store, lookup);
        let uniform_buf = {
            let mut uniform_cache = self.uniform_cache.borrow_mut();
            if let Some(buf) = uniform_cache.get(&key) {
                queue.write_buffer(buf.as_ref(), 0, bytemuck::bytes_of(&u));
                buf.clone()
            } else {
                let buf = Arc::new(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("world_unlit_material_uniform"),
                        contents: bytemuck::bytes_of(&u),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                uniform_cache.insert(key, buf.clone());
                buf
            }
        };

        let mut cache = self.cache.borrow_mut();
        if let Some(bg) = cache.get(&key) {
            return bg.clone();
        }

        let tex_view = self
            .resolve_texture_view(texture_pool, texture_asset_id)
            .unwrap_or_else(|| self.white_texture_view.clone());

        let sampler = self.resolve_sampler(texture_pool, texture_asset_id);

        let bind_group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world_unlit_material_bind"),
            layout: &self.world_unlit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(tex_view.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler.as_ref()),
                },
            ],
        }));
        cache.insert(key, bind_group.clone());
        bind_group
    }

    fn build_unlit_uniform(
        &self,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
    ) -> UnlitMaterialGpu {
        let ids = &self.world_unlit_ids;
        let mut color = [1.0f32, 1.0, 1.0, 1.0];
        if let Some(MaterialPropertyValue::Float4(c)) = store.get_merged(lookup, ids.color) {
            color = *c;
        }
        let mut tex_st = [1.0f32, 1.0, 0.0, 0.0];
        if let Some(MaterialPropertyValue::Float4(t)) = store.get_merged(lookup, ids.tex_st) {
            tex_st = *t;
        }
        let mut cutoff = 0.5f32;
        if let Some(MaterialPropertyValue::Float(c)) = store.get_merged(lookup, ids.cutoff) {
            cutoff = *c;
        }
        let mut flags = 0u32;
        if let Some(MaterialPropertyValue::Texture(tid)) = store.get_merged(lookup, ids.tex) {
            if *tid >= 0 {
                flags |= 1u32;
            }
        }
        if cutoff > 0.0 && cutoff < 1.0 {
            flags |= 2u32;
        }
        UnlitMaterialGpu {
            color,
            tex_st,
            cutoff,
            flags,
            _pad: [0.0, 0.0],
        }
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
        label: Some("manifest_texture_sampler"),
        address_mode_u,
        address_mode_v,
        address_mode_w: address_mode_u,
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: mipmap,
        ..Default::default()
    })
}
