//! Transient resource lifetimes and physical alias slot assignment.

use hashbrown::HashMap;

use super::super::compiled::{CompiledBufferResource, CompiledTextureResource, ResourceLifetime};
use super::super::resources::{
    ResourceHandle, TransientBufferDesc, TransientSubresourceDesc, TransientTextureDesc,
};
use super::decl::{BufferAliasKey, SetupEntry, TextureAliasKey};

pub(super) fn compile_textures(
    descs: &[TransientTextureDesc],
    subresources: &[TransientSubresourceDesc],
    setups: &[SetupEntry],
    retained_ord: &HashMap<usize, usize>,
) -> (Vec<CompiledTextureResource>, usize) {
    let mut resources: Vec<CompiledTextureResource> = descs
        .iter()
        .cloned()
        .map(|desc| CompiledTextureResource {
            usage: desc.base_usage,
            desc,
            lifetime: None,
            physical_slot: usize::MAX,
        })
        .collect();

    for (pass_idx, entry) in setups.iter().enumerate() {
        let Some(&ordinal) = retained_ord.get(&pass_idx) else {
            continue;
        };
        for access in &entry.setup.accesses {
            let Some(handle) = transient_texture_for_access(access.resource, subresources) else {
                continue;
            };
            let resource = &mut resources[handle.index()];
            if let Some(usage) = access.texture_usage() {
                resource.usage |= usage;
            }
            resource.lifetime = merge_lifetime(resource.lifetime, ordinal);
        }
    }

    let slot_count = assign_texture_slots(&mut resources);
    (resources, slot_count)
}

/// Resolves a texture access to the parent transient texture that owns lifetime and usage.
fn transient_texture_for_access(
    resource: ResourceHandle,
    subresources: &[TransientSubresourceDesc],
) -> Option<super::super::resources::TextureHandle> {
    match resource {
        ResourceHandle::Texture(_) => resource.transient_texture(),
        ResourceHandle::TextureSubresource(handle) => {
            subresources.get(handle.index()).map(|desc| desc.parent)
        }
        ResourceHandle::Buffer(_) => None,
    }
}

pub(super) fn compile_buffers(
    descs: &[TransientBufferDesc],
    setups: &[SetupEntry],
    retained_ord: &HashMap<usize, usize>,
) -> (Vec<CompiledBufferResource>, usize) {
    let mut resources: Vec<CompiledBufferResource> = descs
        .iter()
        .cloned()
        .map(|desc| CompiledBufferResource {
            usage: desc.base_usage,
            desc,
            lifetime: None,
            physical_slot: usize::MAX,
        })
        .collect();

    for (pass_idx, entry) in setups.iter().enumerate() {
        let Some(&ordinal) = retained_ord.get(&pass_idx) else {
            continue;
        };
        for access in &entry.setup.accesses {
            let Some(handle) = access.resource.transient_buffer() else {
                continue;
            };
            let resource = &mut resources[handle.index()];
            if let Some(usage) = access.buffer_usage() {
                resource.usage |= usage;
            }
            resource.lifetime = merge_lifetime(resource.lifetime, ordinal);
        }
    }

    let slot_count = assign_buffer_slots(&mut resources);
    (resources, slot_count)
}

fn merge_lifetime(existing: Option<ResourceLifetime>, ordinal: usize) -> Option<ResourceLifetime> {
    Some(match existing {
        Some(lifetime) => ResourceLifetime {
            first_pass: lifetime.first_pass.min(ordinal),
            last_pass: lifetime.last_pass.max(ordinal),
        },
        None => ResourceLifetime {
            first_pass: ordinal,
            last_pass: ordinal,
        },
    })
}

fn assign_texture_slots(resources: &mut [CompiledTextureResource]) -> usize {
    let mut slots: Vec<(TextureAliasKey, Vec<ResourceLifetime>)> = Vec::new();
    for resource in resources {
        let Some(lifetime) = resource.lifetime else {
            continue;
        };
        let key = TextureAliasKey {
            format: resource.desc.format,
            extent: resource.desc.extent,
            mip_levels: resource.desc.mip_levels,
            sample_count: resource.desc.sample_count,
            dimension: resource.desc.dimension,
            array_layers: resource.desc.array_layers,
            usage_bits: resource.usage.bits() as u64,
        };
        let existing_slot = resource
            .desc
            .alias
            .then(|| {
                slots.iter().position(|(slot_key, lifetimes)| {
                    *slot_key == key && lifetimes.iter().all(|other| other.disjoint(lifetime))
                })
            })
            .flatten();
        match existing_slot {
            Some(slot) => {
                resource.physical_slot = slot;
                slots[slot].1.push(lifetime);
            }
            None => {
                resource.physical_slot = slots.len();
                slots.push((key, vec![lifetime]));
            }
        }
    }
    slots.len()
}

fn assign_buffer_slots(resources: &mut [CompiledBufferResource]) -> usize {
    let mut slots: Vec<(BufferAliasKey, Vec<ResourceLifetime>)> = Vec::new();
    for resource in resources {
        let Some(lifetime) = resource.lifetime else {
            continue;
        };
        let key = BufferAliasKey {
            size_policy: resource.desc.size_policy,
            usage_bits: resource.usage.bits() as u64,
        };
        let existing_slot = resource
            .desc
            .alias
            .then(|| {
                slots.iter().position(|(slot_key, lifetimes)| {
                    *slot_key == key && lifetimes.iter().all(|other| other.disjoint(lifetime))
                })
            })
            .flatten();
        match existing_slot {
            Some(slot) => {
                resource.physical_slot = slot;
                slots[slot].1.push(lifetime);
            }
            None => {
                resource.physical_slot = slots.len();
                slots.push((key, vec![lifetime]));
            }
        }
    }
    slots.len()
}
