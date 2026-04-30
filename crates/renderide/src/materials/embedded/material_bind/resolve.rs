//! Texture view and sampler resolution for embedded `@group(1)` bindings.

use std::sync::Arc;

use super::super::bind_kind::TextureBindKind;
use super::super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::super::layout::{StemMaterialLayout, stem_hash};
use super::super::texture_pools::EmbeddedTexturePools;
use super::super::texture_resolve::{
    ResolvedTextureBinding, create_sampler, primary_texture_2d_asset_id,
    resolved_texture_binding_for_host, texture_bind_signature, texture_property_ids_for_binding,
};
use super::cache::{EmbeddedSamplerCacheKey, MaterialBindCacheKey};
use super::uniform::MaterialUniformCacheKey;
use crate::materials::host_data::{MaterialPropertyLookupIds, MaterialPropertyStore};

pub(super) type EmbeddedGroup1TexturesAndSamplers =
    (Vec<Arc<wgpu::TextureView>>, Vec<Arc<wgpu::Sampler>>);

/// Stem layout, uniform/bind cache keys, and resolved primary texture ids for embedded `@group(1)` wiring.
pub(super) struct EmbeddedBindInputResolution {
    pub(super) layout: Arc<StemMaterialLayout>,
    pub(super) uniform_key: MaterialUniformCacheKey,
    pub(super) bind_key: MaterialBindCacheKey,
    pub(super) texture_2d_asset_id: i32,
}

/// Host texture binding lookup for [`super::EmbeddedMaterialBindResources::resolve_texture_view_for_host`] and
/// [`super::EmbeddedMaterialBindResources::resolve_sampler_for_host`].
pub(super) struct HostTexturePropertyQuery<'a> {
    pub(super) host_name: &'a str,
    pub(super) texture_property_ids: &'a [i32],
    pub(super) primary_texture_2d: i32,
    pub(super) pools: &'a EmbeddedTexturePools<'a>,
    pub(super) store: &'a MaterialPropertyStore,
    pub(super) lookup: MaterialPropertyLookupIds,
    pub(super) offscreen_write_render_texture_asset_id: Option<i32>,
}

use super::EmbeddedMaterialBindResources;

impl EmbeddedMaterialBindResources {
    /// Resolves stem layout, primary texture ids, texture signature, and LRU cache keys for embedded binds.
    ///
    /// The texture bind signature in [`MaterialBindCacheKey`] must reflect pool residency and sampler state.
    /// A cheaper fingerprint that omits it (e.g. keyed only by [`MaterialPropertyStore::mutation_generation`])
    /// would be **unsound**: material mutations do not bump generation when textures stream mips or pools
    /// change without a store write. Any future L1 fast path must include this signature or a dedicated
    /// texture-binding epoch bumped on those events.
    pub(super) fn resolve_embedded_bind_inputs(
        &self,
        stem: &str,
        store: &MaterialPropertyStore,
        pools: &EmbeddedTexturePools<'_>,
        lookup: MaterialPropertyLookupIds,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Result<EmbeddedBindInputResolution, EmbeddedMaterialBindError> {
        profiling::scope!("materials::embedded_resolve_bind_inputs");
        let layout = self.stem_layout(stem)?;
        let sh = stem_hash(stem);

        let texture_2d_asset_id =
            primary_texture_2d_asset_id(&layout.reflected, layout.ids.as_ref(), store, lookup);
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
        })
    }

    /// Resolves every non-uniform `@group(1)` texture view and sampler in reflection order.
    pub(super) fn resolve_group1_textures_and_samplers(
        &self,
        layout: &Arc<StemMaterialLayout>,
        texture_2d_asset_id: i32,
        pools: &EmbeddedTexturePools<'_>,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Result<EmbeddedGroup1TexturesAndSamplers, EmbeddedMaterialBindError> {
        profiling::scope!("materials::embedded_resolve_textures_samplers");
        let mut keepalive_views: Vec<Arc<wgpu::TextureView>> = Vec::new();
        let mut keepalive_samplers: Vec<Arc<wgpu::Sampler>> = Vec::new();
        for entry in &layout.reflected.material_entries {
            let b = entry.binding;
            match entry.ty {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                } => {}
                wgpu::BindingType::Texture { view_dimension, .. } => {
                    let host_name = layout
                        .reflected
                        .material_group1_names
                        .get(&b)
                        .map(String::as_str)
                        .ok_or_else(|| {
                            format!("reflection: no WGSL name for texture @binding({b})")
                        })?;
                    let tex_pids = texture_property_ids_for_binding(layout.ids.as_ref(), b);
                    if tex_pids.is_empty() {
                        return Err(EmbeddedMaterialBindError::from(format!(
                            "reflection: missing property id for texture @binding({b})"
                        )));
                    }
                    let tex_view = Self::resolve_texture_view_for_host(
                        HostTexturePropertyQuery {
                            host_name,
                            texture_property_ids: tex_pids,
                            primary_texture_2d: texture_2d_asset_id,
                            pools,
                            store,
                            lookup,
                            offscreen_write_render_texture_asset_id,
                        },
                        view_dimension,
                    )
                    .unwrap_or_else(|| self.default_texture_view_for_dimension(view_dimension));
                    keepalive_views.push(tex_view);
                }
                wgpu::BindingType::Sampler(_) => {
                    let tex_binding = super::assemble::sampler_pairs_texture_binding(b);
                    let host_name = layout
                        .reflected
                        .material_group1_names
                        .get(&tex_binding)
                        .map(String::as_str)
                        .ok_or_else(|| {
                            format!("reflection: no texture global for sampler @binding({b})")
                        })?;
                    let tex_pids =
                        texture_property_ids_for_binding(layout.ids.as_ref(), tex_binding);
                    if tex_pids.is_empty() {
                        return Err(EmbeddedMaterialBindError::from(format!(
                            "reflection: missing property id for texture @binding({tex_binding})"
                        )));
                    }
                    let sampler = self.resolve_sampler_for_host(HostTexturePropertyQuery {
                        host_name,
                        texture_property_ids: tex_pids,
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

    fn default_texture_view_for_dimension(
        &self,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Arc<wgpu::TextureView> {
        match view_dimension {
            wgpu::TextureViewDimension::D3 => self.white_3d.view.clone(),
            //review: keep a cube fallback here; wgpu rejects a 2D white texture for texture_cube bindings.
            wgpu::TextureViewDimension::Cube => self.white_cube.view.clone(),
            _ => self.white_2d.view.clone(),
        }
    }

    fn resolve_texture_view_for_host(
        q: HostTexturePropertyQuery<'_>,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Option<Arc<wgpu::TextureView>> {
        let binding = resolved_texture_binding_for_host(
            q.host_name,
            q.texture_property_ids,
            q.primary_texture_2d,
            q.store,
            q.lookup,
        );
        Self::resolve_texture_view(
            q.pools,
            view_dimension,
            binding,
            q.offscreen_write_render_texture_asset_id,
        )
    }

    fn resolve_sampler_for_host(&self, q: HostTexturePropertyQuery<'_>) -> Arc<wgpu::Sampler> {
        let binding = resolved_texture_binding_for_host(
            q.host_name,
            q.texture_property_ids,
            q.primary_texture_2d,
            q.store,
            q.lookup,
        );
        self.resolve_sampler(q.pools, binding, q.offscreen_write_render_texture_asset_id)
    }

    fn resolve_texture_view(
        pools: &EmbeddedTexturePools<'_>,
        view_dimension: wgpu::TextureViewDimension,
        binding: ResolvedTextureBinding,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Option<Arc<wgpu::TextureView>> {
        match (view_dimension, binding) {
            (_, ResolvedTextureBinding::None) => None,
            (wgpu::TextureViewDimension::D2, ResolvedTextureBinding::Texture2D { asset_id }) => {
                if asset_id < 0 {
                    return None;
                }
                pools
                    .texture
                    .get(asset_id)
                    .filter(|t| t.mip_levels_resident > 0)
                    .map(|t| t.view.clone())
            }
            (wgpu::TextureViewDimension::D3, ResolvedTextureBinding::Texture3D { asset_id }) => {
                if asset_id < 0 {
                    return None;
                }
                pools
                    .texture3d
                    .get(asset_id)
                    .filter(|t| t.mip_levels_resident > 0)
                    .map(|t| t.view.clone())
            }
            (wgpu::TextureViewDimension::Cube, ResolvedTextureBinding::Cubemap { asset_id }) => {
                if asset_id < 0 {
                    return None;
                }
                pools
                    .cubemap
                    .get(asset_id)
                    .filter(|t| t.mip_levels_resident > 0)
                    .map(|t| t.view.clone())
            }
            (
                wgpu::TextureViewDimension::D2,
                ResolvedTextureBinding::RenderTexture { asset_id },
            ) => {
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
            (wgpu::TextureViewDimension::D2, ResolvedTextureBinding::VideoTexture { asset_id }) => {
                if asset_id < 0 {
                    return None;
                }
                pools.video_texture.get(asset_id).map(|t| t.view.clone())
            }
            _ => None,
        }
    }

    fn resolve_sampler(
        &self,
        pools: &EmbeddedTexturePools<'_>,
        binding: ResolvedTextureBinding,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Arc<wgpu::Sampler> {
        let sampled: Option<Arc<wgpu::Sampler>> = match binding {
            ResolvedTextureBinding::None => None,
            ResolvedTextureBinding::Texture2D { asset_id } => {
                if asset_id < 0 {
                    None
                } else {
                    pools.texture.get(asset_id).map(|tex| {
                        let key = EmbeddedSamplerCacheKey::texture2d(
                            &tex.sampler,
                            tex.mip_levels_resident,
                        );
                        self.cached_sampler(key, || {
                            create_sampler(
                                &self.device,
                                &tex.sampler,
                                TextureBindKind::Tex2D,
                                tex.mip_levels_resident,
                            )
                        })
                    })
                }
            }
            ResolvedTextureBinding::Texture3D { asset_id } => {
                if asset_id < 0 {
                    None
                } else {
                    pools.texture3d.get(asset_id).map(|tex| {
                        let key = EmbeddedSamplerCacheKey::texture3d(
                            &tex.sampler,
                            tex.mip_levels_resident,
                        );
                        self.cached_sampler(key, || {
                            create_sampler(
                                &self.device,
                                &tex.sampler,
                                TextureBindKind::Tex3D,
                                tex.mip_levels_resident,
                            )
                        })
                    })
                }
            }
            ResolvedTextureBinding::Cubemap { asset_id } => {
                if asset_id < 0 {
                    None
                } else {
                    pools.cubemap.get(asset_id).map(|tex| {
                        let key =
                            EmbeddedSamplerCacheKey::cubemap(&tex.sampler, tex.mip_levels_resident);
                        self.cached_sampler(key, || {
                            create_sampler(
                                &self.device,
                                &tex.sampler,
                                TextureBindKind::Cube,
                                tex.mip_levels_resident,
                            )
                        })
                    })
                }
            }
            ResolvedTextureBinding::RenderTexture { asset_id } => {
                if asset_id < 0 || offscreen_write_render_texture_asset_id == Some(asset_id) {
                    None
                } else {
                    pools.render_texture.get(asset_id).map(|tex| {
                        let key = EmbeddedSamplerCacheKey::texture2d(&tex.sampler, 1);
                        self.cached_sampler(key, || {
                            create_sampler(&self.device, &tex.sampler, TextureBindKind::Tex2D, 1)
                        })
                    })
                }
            }
            ResolvedTextureBinding::VideoTexture { asset_id } => {
                pools.video_texture.get(asset_id).map(|tex| {
                    let key = EmbeddedSamplerCacheKey::texture2d(&tex.sampler, 1);
                    self.cached_sampler(key, || {
                        create_sampler(&self.device, &tex.sampler, TextureBindKind::Tex2D, 1)
                    })
                })
            }
        };
        sampled.unwrap_or_else(|| self.default_sampler.clone())
    }
}
