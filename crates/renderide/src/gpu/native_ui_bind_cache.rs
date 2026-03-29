//! Cached bind groups for native UI material textures (group 2) and world `Shader "Unlit"` (group 1).
//!
//! The public type alias for this table in the render path is [`super::MaterialGpuResources`].
//!
//! Bind groups embed the resolved [`wgpu::TextureView`] at creation time. Cache keys therefore
//! include whether each slot uses a **real** GPU view or the **fallback** 1×1 white texture,
//! so an entry created while a host texture was not yet [`crate::assets::TextureAsset::ready_for_gpu`]
//! does not block later draws after upload.
//!
//! [`GpuState::drop_texture2d`](crate::gpu::state::GpuState::drop_texture2d) calls [`evict_texture`] when
//! the host unloads a texture so stale bind groups are not retained.
//!
//! ## Per-material buffer isolation (world unlit)
//!
//! Each unique `(material_id, main_tex, mask_tex, …)` cache entry owns its **own**
//! [`wgpu::Buffer`] for `WorldUnlitMaterialUniform`. This is critical: wgpu's `write_buffer`
//! calls are all committed before the render pass executes, so if all world-unlit draws shared a
//! single uniform buffer only the *last* material's data would be visible to every draw.
//! Giving each entry its own buffer ensures that each material's uniform write is isolated.

use std::collections::HashMap;

use crate::assets::{
    MaterialPropertyLookupIds, MaterialPropertyStore, UiTextUnlitPropertyIds, UiUnlitPropertyIds,
    WorldUnlitMaterialUniform, WorldUnlitPropertyIds, ui_text_unlit_material_uniform,
    ui_unlit_material_uniform, world_unlit_material_uniform,
};

use super::pipeline::fallback_white;

/// Maximum entries per map before a full clear (simple safety valve against unbounded growth).
const CACHE_CAP: usize = 512;

/// Key for [`NativeUiMaterialBindCache::ui_unlit`]: texture asset ids plus whether each slot
/// resolved to a GPU view (vs fallback white) when the bind group was created.
type UiUnlitCacheKey = (i32, i32, bool, bool);

/// Key for [`NativeUiMaterialBindCache::ui_text`]: font texture id and whether a GPU view existed.
type UiTextCacheKey = (i32, bool);

/// Key for [`NativeUiMaterialBindCache::world_unlit`]: material block id + texture asset ids +
/// view resolution flags. The material block id ensures each distinct material uses its own
/// `WorldUnlitMaterialUniform` buffer so per-draw `write_buffer` calls do not clobber each other.
type WorldUnlitCacheKey = (i32, i32, i32, bool, bool);

/// Per-entry state for cached world-unlit material bind groups.
///
/// Each entry owns its own [`wgpu::Buffer`] so that `write_buffer` calls for different
/// materials in the same frame are isolated from each other.
struct WorldUnlitEntry {
    bind_group: wgpu::BindGroup,
    /// Dedicated per-material uniform buffer (not shared across entries).
    uniform_buffer: wgpu::Buffer,
}

/// Reuses native UI material bind groups keyed by resolved 2D texture asset ids and view state.
pub struct NativeUiMaterialBindCache {
    ui_unlit: HashMap<UiUnlitCacheKey, wgpu::BindGroup>,
    ui_text: HashMap<UiTextCacheKey, wgpu::BindGroup>,
    world_unlit: HashMap<WorldUnlitCacheKey, WorldUnlitEntry>,
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

    fn trim_world_unlit(map: &mut HashMap<WorldUnlitCacheKey, WorldUnlitEntry>) {
        if map.len() > CACHE_CAP {
            map.clear();
        }
    }

    /// Writes uniform data and binds group 1 for world [`crate::assets::CANONICAL_UNITY_WORLD_UNLIT`].
    ///
    /// Each `material_id` gets its **own** [`wgpu::Buffer`] so that multiple world-unlit draws in
    /// the same frame do not overwrite each other's material data. The buffer for `material_id` is
    /// written fresh every frame; draws with the same `material_id` naturally share the same
    /// uniform data (they use the same material).
    #[allow(clippy::too_many_arguments)]
    pub fn write_world_unlit_material_bind(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        material_bgl: &wgpu::BindGroupLayout,
        linear_sampler: &wgpu::Sampler,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        ids: &WorldUnlitPropertyIds,
        main_view: Option<&wgpu::TextureView>,
        mask_view: Option<&wgpu::TextureView>,
        main_key: i32,
        mask_key: i32,
        material_id: i32,
    ) {
        let (u, _, _) = world_unlit_material_uniform(store, lookup, ids);
        let white = fallback_white(device);
        let mv = main_view.unwrap_or(white);
        let xv = mask_view.unwrap_or(white);
        let main_has_view = main_view.is_some();
        let mask_has_view = mask_view.is_some();
        Self::trim_world_unlit(&mut self.world_unlit);
        let key = (
            material_id,
            main_key,
            mask_key,
            main_has_view,
            mask_has_view,
        );
        let entry = self.world_unlit.entry(key).or_insert_with(|| {
            // Each cache entry gets its own uniform buffer so that write_buffer calls for
            // different materials in the same frame don't overwrite each other.
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("world unlit per-material uniform"),
                size: std::mem::size_of::<WorldUnlitMaterialUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("world unlit per-material BG"),
                layout: material_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    // binding 1: _Tex
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(mv),
                    },
                    // binding 2: _Tex_sampler
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    // binding 3: _MaskTex
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(xv),
                    },
                    // binding 4: _MaskTex_sampler
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    // binding 5: _OffsetTex (placeholder white)
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(white),
                    },
                    // binding 6: _OffsetTex_sampler
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                ],
            });
            WorldUnlitEntry {
                bind_group,
                uniform_buffer,
            }
        });
        // Write this material's uniform data into its dedicated buffer.
        queue.write_buffer(&entry.uniform_buffer, 0, bytemuck::bytes_of(&u));
        pass.set_bind_group(1, &entry.bind_group, &[]);
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
            .retain(|(_, a, b, _, _), _| *a != texture_asset_id && *b != texture_asset_id);
    }
}

impl Default for NativeUiMaterialBindCache {
    fn default() -> Self {
        Self::new()
    }
}
