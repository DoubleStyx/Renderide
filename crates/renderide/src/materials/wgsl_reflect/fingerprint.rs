//! Stable [`ReflectedRasterLayout::layout_fingerprint`](super::types::ReflectedRasterLayout::layout_fingerprint) hashing.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub(super) fn fingerprint_layout(
    material: &[wgpu::BindGroupLayoutEntry],
    per_draw: &[wgpu::BindGroupLayoutEntry],
    vs_max_vertex_location: Option<u32>,
    group1_names: &HashMap<u32, String>,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    1u8.hash(&mut h);
    hash_entries(material, &mut h);
    2u8.hash(&mut h);
    hash_entries(per_draw, &mut h);
    3u8.hash(&mut h);
    vs_max_vertex_location.hash(&mut h);
    let mut keys: Vec<u32> = group1_names.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        k.hash(&mut h);
        group1_names[&k].hash(&mut h);
    }
    h.finish()
}

fn hash_entries(entries: &[wgpu::BindGroupLayoutEntry], h: &mut impl Hasher) {
    entries.len().hash(h);
    for e in entries {
        e.binding.hash(h);
        hash_binding_type(&e.ty, h);
    }
}

fn hash_binding_type(ty: &wgpu::BindingType, h: &mut impl Hasher) {
    match ty {
        wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size,
        } => {
            0u8.hash(h);
            std::mem::discriminant(ty).hash(h);
            has_dynamic_offset.hash(h);
            min_binding_size.map(|n| n.get()).hash(h);
        }
        wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        } => {
            1u8.hash(h);
            std::mem::discriminant(sample_type).hash(h);
            std::mem::discriminant(view_dimension).hash(h);
            multisampled.hash(h);
        }
        wgpu::BindingType::Sampler(ty) => {
            2u8.hash(h);
            std::mem::discriminant(ty).hash(h);
        }
        wgpu::BindingType::StorageTexture { .. } => {
            3u8.hash(h);
        }
        wgpu::BindingType::AccelerationStructure { .. } => {
            6u8.hash(h);
        }
        wgpu::BindingType::ExternalTexture => {
            7u8.hash(h);
        }
    }
}
