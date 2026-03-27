//! Cached bind groups for native UI material textures (group 2) and world `Shader "Unlit"` (group 1).
//!
//! Bind groups embed the resolved [`wgpu::TextureView`] at creation time. Cache keys therefore
//! include whether each slot uses a **real** GPU view or the **fallback** 1×1 white texture,
//! so an entry created while a host texture was not yet [`crate::assets::TextureAsset::ready_for_gpu`]
//! does not block later draws after upload.
//!
//! [`GpuState::drop_texture2d`](crate::gpu::state::GpuState::drop_texture2d) calls [`evict_texture`] when
//! the host unloads a texture so stale bind groups are not retained.

use std::collections::HashMap;

use crate::assets::{
    MaterialPropertyLookupIds, MaterialPropertyStore, UiTextUnlitPropertyIds, UiUnlitPropertyIds,
    WorldUnlitPropertyIds, ui_text_unlit_material_uniform, ui_unlit_material_uniform,
    world_unlit_material_uniform,
};

use super::pipeline::fallback_white;

/// Maximum entries per map before a full clear (simple safety valve against unbounded growth).
const CACHE_CAP: usize = 512;

/// Key for [`NativeUiMaterialBindCache::ui_unlit`]: texture asset ids plus whether each slot
/// resolved to a GPU view (vs fallback white) when the bind group was created.
type UiUnlitCacheKey = (i32, i32, bool, bool);

/// Key for [`NativeUiMaterialBindCache::ui_text`]: font texture id and whether a GPU view existed.
type UiTextCacheKey = (i32, bool);

/// Key for [`NativeUiMaterialBindCache::world_unlit`]: texture asset ids plus view resolution flags.
type WorldUnlitCacheKey = (i32, i32, bool, bool);

/// Reuses native UI material bind groups keyed by resolved 2D texture asset ids and view state.
pub struct NativeUiMaterialBindCache {
    ui_unlit: HashMap<UiUnlitCacheKey, wgpu::BindGroup>,
    ui_text: HashMap<UiTextCacheKey, wgpu::BindGroup>,
    world_unlit: HashMap<WorldUnlitCacheKey, wgpu::BindGroup>,
}

impl NativeUiMaterialBindCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self {
            ui_unlit: HashMap::new(),
            ui_text: HashMap::new(),
            world_unlit: HashMap::new(),
        }
    }

    fn trim_unlit(map: &mut HashMap<UiUnlitCacheKey, wgpu::BindGroup>) {
        if map.len() > CACHE_CAP {
            map.clear();
        }
    }

    fn trim_text(map: &mut HashMap<UiTextCacheKey, wgpu::BindGroup>) {
        if map.len() > CACHE_CAP {
            map.clear();
        }
    }

    fn trim_world_unlit(map: &mut HashMap<WorldUnlitCacheKey, wgpu::BindGroup>) {
        if map.len() > CACHE_CAP {
            map.clear();
        }
    }

    /// Writes uniform data and binds group 1 for world [`crate::assets::CANONICAL_UNITY_WORLD_UNLIT`].
    #[allow(clippy::too_many_arguments)]
    pub fn write_world_unlit_material_bind(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        material_bgl: &wgpu::BindGroupLayout,
        material_uniform: &wgpu::Buffer,
        linear_sampler: &wgpu::Sampler,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        ids: &WorldUnlitPropertyIds,
        main_view: Option<&wgpu::TextureView>,
        mask_view: Option<&wgpu::TextureView>,
        main_key: i32,
        mask_key: i32,
    ) {
        let (u, _, _) = world_unlit_material_uniform(store, lookup, ids);
        queue.write_buffer(material_uniform, 0, bytemuck::bytes_of(&u));
        let white = fallback_white(device);
        let mv = main_view.unwrap_or(white);
        let xv = mask_view.unwrap_or(white);
        let main_has_view = main_view.is_some();
        let mask_has_view = mask_view.is_some();
        Self::trim_world_unlit(&mut self.world_unlit);
        let key = (main_key, mask_key, main_has_view, mask_has_view);
        self.world_unlit.entry(key).or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("world unlit material BG cached"),
                layout: material_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: material_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(mv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(xv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                ],
            })
        });
        let bg = self.world_unlit.get(&key).expect("just inserted");
        pass.set_bind_group(1, bg, &[]);
    }

    /// Writes uniform data, binds group 2 for `UI_Unlit` using real textures when views exist.
    #[allow(clippy::too_many_arguments)]
    pub fn write_ui_unlit_material_bind(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        material_bgl: &wgpu::BindGroupLayout,
        material_uniform: &wgpu::Buffer,
        linear_sampler: &wgpu::Sampler,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        ids: &UiUnlitPropertyIds,
        main_view: Option<&wgpu::TextureView>,
        mask_view: Option<&wgpu::TextureView>,
        main_key: i32,
        mask_key: i32,
    ) {
        let (u, _, _) = ui_unlit_material_uniform(store, lookup, ids);
        queue.write_buffer(material_uniform, 0, bytemuck::bytes_of(&u));
        let white = fallback_white(device);
        let mv = main_view.unwrap_or(white);
        let xv = mask_view.unwrap_or(white);
        let main_has_view = main_view.is_some();
        let mask_has_view = mask_view.is_some();
        Self::trim_unlit(&mut self.ui_unlit);
        let key = (main_key, mask_key, main_has_view, mask_has_view);
        self.ui_unlit.entry(key).or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ui unlit material BG cached"),
                layout: material_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: material_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(mv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(xv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                ],
            })
        });
        let bg = self.ui_unlit.get(&key).expect("just inserted");
        pass.set_bind_group(2, bg, &[]);
    }

    /// Writes uniform data and binds group 2 for `UI_TextUnlit`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_ui_text_unlit_material_bind(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        material_uniform: &wgpu::Buffer,
        linear_sampler: &wgpu::Sampler,
        material_bgl: &wgpu::BindGroupLayout,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        ids: &UiTextUnlitPropertyIds,
        font_view: Option<&wgpu::TextureView>,
        font_key: i32,
    ) {
        let (u, _) = ui_text_unlit_material_uniform(store, lookup, ids);
        queue.write_buffer(material_uniform, 0, bytemuck::bytes_of(&u));
        let white = fallback_white(device);
        let fv = font_view.unwrap_or(white);
        let font_has_view = font_view.is_some();
        Self::trim_text(&mut self.ui_text);
        let key = (font_key, font_has_view);
        self.ui_text.entry(key).or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ui text unlit material BG cached"),
                layout: material_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: material_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(fv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                ],
            })
        });
        let bg = self.ui_text.get(&key).expect("just inserted");
        pass.set_bind_group(2, bg, &[]);
    }

    /// Drops GPU bind groups for a texture asset (e.g. after unload).
    pub fn evict_texture(&mut self, texture_asset_id: i32) {
        self.ui_unlit
            .retain(|(a, b, _, _), _| *a != texture_asset_id && *b != texture_asset_id);
        self.ui_text.retain(|(k, _), _| *k != texture_asset_id);
        self.world_unlit
            .retain(|(a, b, _, _), _| *a != texture_asset_id && *b != texture_asset_id);
    }
}

impl Default for NativeUiMaterialBindCache {
    fn default() -> Self {
        Self::new()
    }
}
