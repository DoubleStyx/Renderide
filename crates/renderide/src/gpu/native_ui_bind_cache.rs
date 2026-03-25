//! Cached bind groups for native UI material textures (group 2).

use std::collections::HashMap;

use crate::assets::{
    MaterialPropertyStore, UiTextUnlitPropertyIds, UiUnlitPropertyIds,
    ui_text_unlit_material_uniform, ui_unlit_material_uniform,
};

use super::pipeline::fallback_white;

/// Maximum entries per map before a full clear (simple safety valve against unbounded growth).
const CACHE_CAP: usize = 512;

/// Reuses native UI material bind groups keyed by resolved 2D texture asset ids.
pub struct NativeUiMaterialBindCache {
    ui_unlit: HashMap<(i32, i32), wgpu::BindGroup>,
    ui_text: HashMap<i32, wgpu::BindGroup>,
}

impl NativeUiMaterialBindCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self {
            ui_unlit: HashMap::new(),
            ui_text: HashMap::new(),
        }
    }

    fn trim_unlit(map: &mut HashMap<(i32, i32), wgpu::BindGroup>) {
        if map.len() > CACHE_CAP {
            map.clear();
        }
    }

    fn trim_text(map: &mut HashMap<i32, wgpu::BindGroup>) {
        if map.len() > CACHE_CAP {
            map.clear();
        }
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
        block_id: i32,
        ids: &UiUnlitPropertyIds,
        main_view: Option<&wgpu::TextureView>,
        mask_view: Option<&wgpu::TextureView>,
        main_key: i32,
        mask_key: i32,
    ) {
        let (u, _, _) = ui_unlit_material_uniform(store, block_id, ids);
        queue.write_buffer(material_uniform, 0, bytemuck::bytes_of(&u));
        let white = fallback_white(device);
        let mv = main_view.unwrap_or(white);
        let xv = mask_view.unwrap_or(white);
        Self::trim_unlit(&mut self.ui_unlit);
        let key = (main_key, mask_key);
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
        block_id: i32,
        ids: &UiTextUnlitPropertyIds,
        font_view: Option<&wgpu::TextureView>,
        font_key: i32,
    ) {
        let (u, _) = ui_text_unlit_material_uniform(store, block_id, ids);
        queue.write_buffer(material_uniform, 0, bytemuck::bytes_of(&u));
        let white = fallback_white(device);
        let fv = font_view.unwrap_or(white);
        Self::trim_text(&mut self.ui_text);
        self.ui_text.entry(font_key).or_insert_with(|| {
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
        let bg = self.ui_text.get(&font_key).expect("just inserted");
        pass.set_bind_group(2, bg, &[]);
    }

    /// Drops GPU bind groups for a texture asset (e.g. after unload).
    pub fn evict_texture(&mut self, texture_asset_id: i32) {
        self.ui_unlit
            .retain(|(a, b), _| *a != texture_asset_id && *b != texture_asset_id);
        self.ui_text.retain(|k, _| *k != texture_asset_id);
    }
}

impl Default for NativeUiMaterialBindCache {
    fn default() -> Self {
        Self::new()
    }
}
