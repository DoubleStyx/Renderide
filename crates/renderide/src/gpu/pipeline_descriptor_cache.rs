//! Secondary index for [`wgpu::RenderPipeline`] instances by a stable numeric descriptor key.
//!
//! Built-in variants are registered with [`PipelineDescriptorCache::builtin_key`]; host-unlit
//! programs use [`PipelineDescriptorCache::host_unlit_key`].

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::assets::NativeUiSurfaceBlend;

use super::PipelineVariant;
use super::pipeline::RenderPipeline;

const HOST_UNLIT_CACHE_TAG: u64 = 0x48_30_53_54_55_4e_4c_54; // "H0STUNLT"
const NATIVE_UI_UNLIT_TAG: u64 = 0x4e_55_49_55_4e_4c_54_31; // "NUIUNLT1"
const NATIVE_UI_TEXT_TAG: u64 = 0x4e_55_49_54_45_58_54_31; // "NUITEXT1"
const NATIVE_UI_UNLIT_STENCIL_TAG: u64 = 0x4e_55_49_55_53_54_45_4e; // "NUIUSTEN"
const NATIVE_UI_TEXT_STENCIL_TAG: u64 = 0x4e_55_49_54_53_54_45_4e; // "NUITSTEN"
const WORLD_UNLIT_TAG: u64 = 0x57_52_4c_44_55_4e_4c_54; // "WRLDUNLT"

/// Maps descriptor hashes to shared pipeline [`Arc`]s.
#[derive(Default)]
pub(crate) struct PipelineDescriptorCache {
    entries: HashMap<u64, Arc<dyn RenderPipeline>>,
}

impl PipelineDescriptorCache {
    /// Hash for a built-in [`PipelineVariant`] at a given swapchain format.
    pub(crate) fn builtin_key(variant: PipelineVariant, format: wgpu::TextureFormat) -> u64 {
        let mut h = DefaultHasher::new();
        0xB0_u8.hash(&mut h);
        variant.hash(&mut h);
        format.hash(&mut h);
        h.finish()
    }

    /// Hash for a host-unlit pipeline sharing WGSL but keyed by shader asset id.
    pub(crate) fn host_unlit_key(shader_asset_id: i32, format: wgpu::TextureFormat) -> u64 {
        let mut h = DefaultHasher::new();
        HOST_UNLIT_CACHE_TAG.hash(&mut h);
        shader_asset_id.hash(&mut h);
        format.hash(&mut h);
        h.finish()
    }

    /// Hash for [`crate::gpu::pipeline::WorldUnlitPipeline`] keyed by shader asset id.
    pub(crate) fn world_unlit_key(shader_asset_id: i32, format: wgpu::TextureFormat) -> u64 {
        let mut h = DefaultHasher::new();
        WORLD_UNLIT_TAG.hash(&mut h);
        shader_asset_id.hash(&mut h);
        format.hash(&mut h);
        h.finish()
    }

    /// Hash for native [`crate::gpu::pipeline::UiUnlitNativePipeline`] keyed by shader asset id.
    pub(crate) fn native_ui_unlit_key(
        shader_asset_id: i32,
        format: wgpu::TextureFormat,
        surface_blend: NativeUiSurfaceBlend,
    ) -> u64 {
        let mut h = DefaultHasher::new();
        NATIVE_UI_UNLIT_TAG.hash(&mut h);
        shader_asset_id.hash(&mut h);
        format.hash(&mut h);
        surface_blend.hash(&mut h);
        h.finish()
    }

    /// Hash for native [`crate::gpu::pipeline::UiTextUnlitNativePipeline`].
    pub(crate) fn native_ui_text_key(
        shader_asset_id: i32,
        format: wgpu::TextureFormat,
        surface_blend: NativeUiSurfaceBlend,
    ) -> u64 {
        let mut h = DefaultHasher::new();
        NATIVE_UI_TEXT_TAG.hash(&mut h);
        shader_asset_id.hash(&mut h);
        format.hash(&mut h);
        surface_blend.hash(&mut h);
        h.finish()
    }

    /// Native `UI_Unlit` with overlay stencil test.
    pub(crate) fn native_ui_unlit_stencil_key(
        shader_asset_id: i32,
        format: wgpu::TextureFormat,
        surface_blend: NativeUiSurfaceBlend,
    ) -> u64 {
        let mut h = DefaultHasher::new();
        NATIVE_UI_UNLIT_STENCIL_TAG.hash(&mut h);
        shader_asset_id.hash(&mut h);
        format.hash(&mut h);
        surface_blend.hash(&mut h);
        h.finish()
    }

    /// Native `UI_TextUnlit` with overlay stencil test.
    pub(crate) fn native_ui_text_stencil_key(
        shader_asset_id: i32,
        format: wgpu::TextureFormat,
        surface_blend: NativeUiSurfaceBlend,
    ) -> u64 {
        let mut h = DefaultHasher::new();
        NATIVE_UI_TEXT_STENCIL_TAG.hash(&mut h);
        shader_asset_id.hash(&mut h);
        format.hash(&mut h);
        surface_blend.hash(&mut h);
        h.finish()
    }

    pub(crate) fn get(&self, key: u64) -> Option<Arc<dyn RenderPipeline>> {
        self.entries.get(&key).cloned()
    }

    pub(crate) fn insert(&mut self, key: u64, pipeline: Arc<dyn RenderPipeline>) {
        self.entries.insert(key, pipeline);
    }

    /// Drops a host-unlit entry when the host unloads that shader asset.
    pub(crate) fn remove_host_unlit(&mut self, shader_asset_id: i32, format: wgpu::TextureFormat) {
        self.entries
            .remove(&Self::host_unlit_key(shader_asset_id, format));
    }

    /// Drops cached world-unlit pipelines for `shader_asset_id` (e.g. shader unload).
    pub(crate) fn remove_world_unlit(&mut self, shader_asset_id: i32, format: wgpu::TextureFormat) {
        self.entries
            .remove(&Self::world_unlit_key(shader_asset_id, format));
    }

    /// Drops cached native UI pipelines for `shader_asset_id` (e.g. shader unload).
    pub(crate) fn remove_native_ui(&mut self, shader_asset_id: i32, format: wgpu::TextureFormat) {
        for b in NativeUiSurfaceBlend::ALL {
            self.entries
                .remove(&Self::native_ui_unlit_key(shader_asset_id, format, b));
            self.entries
                .remove(&Self::native_ui_text_key(shader_asset_id, format, b));
            self.entries.remove(&Self::native_ui_unlit_stencil_key(
                shader_asset_id,
                format,
                b,
            ));
            self.entries.remove(&Self::native_ui_text_stencil_key(
                shader_asset_id,
                format,
                b,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PipelineDescriptorCache;
    use crate::assets::NativeUiSurfaceBlend;
    use wgpu::TextureFormat;

    #[test]
    fn native_ui_cache_keys_differ_from_host_unlit_and_each_other() {
        let fmt = TextureFormat::Bgra8UnormSrgb;
        let sid = 100_i32;
        let b = NativeUiSurfaceBlend::Alpha;
        let u = PipelineDescriptorCache::native_ui_unlit_key(sid, fmt, b);
        let t = PipelineDescriptorCache::native_ui_text_key(sid, fmt, b);
        let h = PipelineDescriptorCache::host_unlit_key(sid, fmt);
        let w = PipelineDescriptorCache::world_unlit_key(sid, fmt);
        assert_ne!(u, t);
        assert_ne!(u, h);
        assert_ne!(t, h);
        assert_ne!(w, h);
        assert_ne!(w, u);
    }

    #[test]
    fn native_ui_blend_changes_cache_key() {
        let fmt = TextureFormat::Bgra8UnormSrgb;
        let a = PipelineDescriptorCache::native_ui_unlit_key(1, fmt, NativeUiSurfaceBlend::Alpha);
        let p = PipelineDescriptorCache::native_ui_unlit_key(
            1,
            fmt,
            NativeUiSurfaceBlend::Premultiplied,
        );
        let m =
            PipelineDescriptorCache::native_ui_unlit_key(1, fmt, NativeUiSurfaceBlend::Additive);
        assert_ne!(a, m);
        assert_ne!(a, p);
        assert_ne!(p, m);
    }
}
