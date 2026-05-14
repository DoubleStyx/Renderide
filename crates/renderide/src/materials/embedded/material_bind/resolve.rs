//! Texture view and sampler resolution for embedded `@group(1)` bindings.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ahash::AHasher;

use super::super::bind_kind::TextureBindKind;
use super::super::embedded_material_bind_error::EmbeddedMaterialBindError;
use super::super::layout::{StemMaterialLayout, stem_hash};
use super::super::texture_pools::EmbeddedTexturePools;
use super::super::texture_resolve::{
    DefaultTextureColor, ResolvedTextureBinding, create_sampler, default_2d_texture_color_for_host,
    hash_texture_entry_signature_contribution, primary_texture_2d_asset_id,
    resolved_texture_binding_for_host, texture_bind_signature, texture_property_ids_for_binding,
};
use super::cache::EmbeddedSamplerCacheKey;
use super::uniform::MaterialUniformCacheKey;
use crate::embedded_shaders::EmbeddedTextureDefaultKind;
use crate::materials::host_data::{MaterialPropertyLookupIds, MaterialPropertyStore};

/// Texture views, samplers, and the matching bind signature captured from one pool read.
pub(super) struct EmbeddedGroup1Snapshot {
    pub(super) views: Vec<Arc<wgpu::TextureView>>,
    pub(super) samplers: Vec<Arc<wgpu::Sampler>>,
    /// Signature hashed from the same pool state used to capture `views` and `samplers`.
    pub(super) texture_bind_signature: u64,
}

/// Stem layout, uniform/bind cache keys, and resolved primary texture ids for embedded `@group(1)` wiring.
pub(super) struct EmbeddedBindInputResolution {
    pub(super) layout: Arc<StemMaterialLayout>,
    pub(super) uniform_key: MaterialUniformCacheKey,
    pub(super) stem_hash: u64,
    pub(super) texture_bind_signature: u64,
    pub(super) texture_2d_asset_id: i32,
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
        shader_variant_bits: Option<u32>,
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
            renderer_property_block_id: lookup.mesh_renderer_property_block_id,
            texture_2d_asset_id,
            shader_variant_bits,
        };
        Ok(EmbeddedBindInputResolution {
            layout,
            uniform_key,
            stem_hash: sh,
            texture_bind_signature,
            texture_2d_asset_id,
        })
    }

    /// Walks `@group(1)` material entries once, capturing texture views, samplers, and the
    /// matching `texture_bind_signature` from a single read of the property store and pools.
    ///
    /// The returned signature is hashed from the same pool entries that produced the captured
    /// views and samplers, so a [`MaterialBindCacheKey`](super::cache::MaterialBindCacheKey)
    /// built from it always describes the assembled bind group's actual contents - even if the
    /// pool state shifted between an earlier lookup-side signature computation and this snapshot.
    pub(super) fn snapshot_group1_textures_samplers(
        &self,
        layout: &Arc<StemMaterialLayout>,
        texture_2d_asset_id: i32,
        pools: &EmbeddedTexturePools<'_>,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        offscreen_write_render_texture_asset_id: Option<i32>,
    ) -> Result<EmbeddedGroup1Snapshot, EmbeddedMaterialBindError> {
        profiling::scope!("materials::embedded_snapshot_textures_samplers");
        let mut views: Vec<Arc<wgpu::TextureView>> = Vec::new();
        let mut samplers: Vec<Arc<wgpu::Sampler>> = Vec::new();
        let mut hasher = AHasher::default();
        offscreen_write_render_texture_asset_id.hash(&mut hasher);
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
                    let resolved = resolved_texture_binding_for_host(
                        host_name,
                        tex_pids,
                        texture_2d_asset_id,
                        store,
                        lookup,
                    );
                    hash_texture_entry_signature_contribution(
                        &mut hasher,
                        b,
                        host_name,
                        resolved,
                        pools,
                        offscreen_write_render_texture_asset_id,
                    );
                    let tex_view = Self::resolve_texture_view(
                        pools,
                        view_dimension,
                        resolved,
                        offscreen_write_render_texture_asset_id,
                    )
                    .unwrap_or_else(|| {
                        self.default_texture_view(layout, b, host_name, view_dimension)
                    });
                    views.push(tex_view);
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
                    let resolved = resolved_texture_binding_for_host(
                        host_name,
                        tex_pids,
                        texture_2d_asset_id,
                        store,
                        lookup,
                    );
                    let sampler = self.resolve_sampler(
                        pools,
                        resolved,
                        offscreen_write_render_texture_asset_id,
                    );
                    samplers.push(sampler);
                }
                _ => {
                    return Err(EmbeddedMaterialBindError::from(format!(
                        "unsupported binding type for @binding({b})"
                    )));
                }
            }
        }
        Ok(EmbeddedGroup1Snapshot {
            views,
            samplers,
            texture_bind_signature: hasher.finish(),
        })
    }

    fn default_texture_view(
        &self,
        layout: &StemMaterialLayout,
        binding: u32,
        host_name: &str,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Arc<wgpu::TextureView> {
        layout
            .texture_default_by_binding
            .get(&binding)
            .copied()
            .map_or_else(
                || self.compatibility_default_texture_view_for_host(host_name, view_dimension),
                |kind| self.default_texture_view_for_kind(kind, view_dimension),
            )
    }

    fn default_texture_view_for_kind(
        &self,
        kind: EmbeddedTextureDefaultKind,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Arc<wgpu::TextureView> {
        match texture_default_placeholder(kind, view_dimension) {
            TextureDefaultPlaceholder::White => match view_dimension {
                wgpu::TextureViewDimension::D3 => self.white_3d.view.clone(),
                wgpu::TextureViewDimension::Cube => self.white_cube.view.clone(),
                _ => self.white_2d.view.clone(),
            },
            TextureDefaultPlaceholder::Black => match view_dimension {
                wgpu::TextureViewDimension::D3 => self.black_3d.view.clone(),
                wgpu::TextureViewDimension::Cube => self.black_cube.view.clone(),
                _ => self.black_2d.view.clone(),
            },
            TextureDefaultPlaceholder::Gray => match view_dimension {
                wgpu::TextureViewDimension::D3 => self.gray_3d.view.clone(),
                wgpu::TextureViewDimension::Cube => self.gray_cube.view.clone(),
                _ => self.gray_2d.view.clone(),
            },
            TextureDefaultPlaceholder::Red => match view_dimension {
                wgpu::TextureViewDimension::D3 => self.red_3d.view.clone(),
                wgpu::TextureViewDimension::Cube => self.red_cube.view.clone(),
                _ => self.red_2d.view.clone(),
            },
            TextureDefaultPlaceholder::FlatNormal => self.flat_normal_2d.view.clone(),
        }
    }

    fn compatibility_default_texture_view_for_host(
        &self,
        host_name: &str,
        view_dimension: wgpu::TextureViewDimension,
    ) -> Arc<wgpu::TextureView> {
        match view_dimension {
            wgpu::TextureViewDimension::D3 => self.white_3d.view.clone(),
            // wgpu rejects a 2D placeholder texture for texture_cube bindings.
            wgpu::TextureViewDimension::Cube => self.white_cube.view.clone(),
            _ => match default_2d_texture_color_for_host(host_name) {
                DefaultTextureColor::White => self.white_2d.view.clone(),
                DefaultTextureColor::Black => self.black_2d.view.clone(),
                DefaultTextureColor::FlatNormal => self.flat_normal_2d.view.clone(),
            },
        }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextureDefaultPlaceholder {
    White,
    Black,
    Gray,
    Red,
    FlatNormal,
}

fn texture_default_placeholder(
    kind: EmbeddedTextureDefaultKind,
    view_dimension: wgpu::TextureViewDimension,
) -> TextureDefaultPlaceholder {
    match kind {
        EmbeddedTextureDefaultKind::White => TextureDefaultPlaceholder::White,
        EmbeddedTextureDefaultKind::Black => TextureDefaultPlaceholder::Black,
        EmbeddedTextureDefaultKind::Gray | EmbeddedTextureDefaultKind::Empty => {
            TextureDefaultPlaceholder::Gray
        }
        EmbeddedTextureDefaultKind::Red => TextureDefaultPlaceholder::Red,
        EmbeddedTextureDefaultKind::Bump => match view_dimension {
            wgpu::TextureViewDimension::D2 => TextureDefaultPlaceholder::FlatNormal,
            _ => TextureDefaultPlaceholder::Gray,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{TextureDefaultPlaceholder, texture_default_placeholder};
    use crate::embedded_shaders::EmbeddedTextureDefaultKind;

    #[test]
    fn texture_default_tokens_map_to_unity_placeholder_colors() {
        assert_eq!(
            texture_default_placeholder(
                EmbeddedTextureDefaultKind::White,
                wgpu::TextureViewDimension::D2
            ),
            TextureDefaultPlaceholder::White
        );
        assert_eq!(
            texture_default_placeholder(
                EmbeddedTextureDefaultKind::Black,
                wgpu::TextureViewDimension::D2
            ),
            TextureDefaultPlaceholder::Black
        );
        assert_eq!(
            texture_default_placeholder(
                EmbeddedTextureDefaultKind::Gray,
                wgpu::TextureViewDimension::D2
            ),
            TextureDefaultPlaceholder::Gray
        );
        assert_eq!(
            texture_default_placeholder(
                EmbeddedTextureDefaultKind::Empty,
                wgpu::TextureViewDimension::D2
            ),
            TextureDefaultPlaceholder::Gray
        );
        assert_eq!(
            texture_default_placeholder(
                EmbeddedTextureDefaultKind::Red,
                wgpu::TextureViewDimension::D2
            ),
            TextureDefaultPlaceholder::Red
        );
        assert_eq!(
            texture_default_placeholder(
                EmbeddedTextureDefaultKind::Bump,
                wgpu::TextureViewDimension::D2
            ),
            TextureDefaultPlaceholder::FlatNormal
        );
        assert_eq!(
            texture_default_placeholder(
                EmbeddedTextureDefaultKind::Bump,
                wgpu::TextureViewDimension::Cube
            ),
            TextureDefaultPlaceholder::Gray
        );
    }
}
