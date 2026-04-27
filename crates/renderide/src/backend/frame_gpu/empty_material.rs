//! Empty `@group(1)` material bind resources.

use std::sync::Arc;

/// Empty `@group(1)` layout for materials that declare no per-material bindings.
pub fn empty_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty_material_slot"),
        entries: &[],
    })
}

/// Single reusable empty bind group for [`empty_material_bind_group_layout`].
fn empty_material_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("empty_material_bind_group"),
        layout,
        entries: &[],
    })
}

/// Cached empty material bind group layout plus instance for one device attach.
pub struct EmptyMaterialBindGroup {
    /// Shared layout for the empty `@group(1)` placeholder.
    pub layout: wgpu::BindGroupLayout,
    /// Bind group with no entries.
    pub bind_group: Arc<wgpu::BindGroup>,
}

impl EmptyMaterialBindGroup {
    /// Builds layout and bind group for the empty `@group(1)` placeholder.
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = empty_material_bind_group_layout(device);
        let bind_group = Arc::new(empty_material_bind_group(device, &layout));
        Self { layout, bind_group }
    }
}
