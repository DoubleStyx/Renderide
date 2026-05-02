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

    let slot_count = assign_aliased_slots(&mut resources);
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

    let slot_count = assign_aliased_slots(&mut resources);
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

/// Resource shape required for physical-slot aliasing.
///
/// Texture and buffer compilation share the exact same disjoint-lifetime alias-slot algorithm --
/// only the alias key shape differs. Implementing this trait lets [`assign_aliased_slots`] run
/// once over either kind of resource list.
trait AliasResource {
    /// Alias-equivalence key. Two resources may share a slot only when their keys compare equal.
    type Key: PartialEq;

    /// Lifetime span (first/last retained pass ordinal) when this resource is reachable.
    fn lifetime(&self) -> Option<ResourceLifetime>;

    /// Whether this resource is allowed to share a slot with disjoint equal-key resources.
    fn alias(&self) -> bool;

    /// Builds the alias key from the resource's descriptor and accumulated usage.
    fn alias_key(&self) -> Self::Key;

    /// Records the chosen physical slot index back on the resource.
    fn set_physical_slot(&mut self, slot: usize);
}

impl AliasResource for CompiledTextureResource {
    type Key = TextureAliasKey;

    fn lifetime(&self) -> Option<ResourceLifetime> {
        self.lifetime
    }

    fn alias(&self) -> bool {
        self.desc.alias
    }

    fn alias_key(&self) -> Self::Key {
        TextureAliasKey {
            format: self.desc.format,
            extent: self.desc.extent,
            mip_levels: self.desc.mip_levels,
            sample_count: self.desc.sample_count,
            dimension: self.desc.dimension,
            array_layers: self.desc.array_layers,
            usage_bits: u64::from(self.usage.bits()),
        }
    }

    fn set_physical_slot(&mut self, slot: usize) {
        self.physical_slot = slot;
    }
}

impl AliasResource for CompiledBufferResource {
    type Key = BufferAliasKey;

    fn lifetime(&self) -> Option<ResourceLifetime> {
        self.lifetime
    }

    fn alias(&self) -> bool {
        self.desc.alias
    }

    fn alias_key(&self) -> Self::Key {
        BufferAliasKey {
            size_policy: self.desc.size_policy,
            usage_bits: u64::from(self.usage.bits()),
        }
    }

    fn set_physical_slot(&mut self, slot: usize) {
        self.physical_slot = slot;
    }
}

/// Walks `resources` in order, assigning each a physical slot. Resources whose `alias()` is `true`
/// reuse a prior slot when the keys match and lifetimes are disjoint; otherwise they take a fresh
/// slot. Returns the total number of distinct physical slots.
fn assign_aliased_slots<R: AliasResource>(resources: &mut [R]) -> usize {
    let mut slots: Vec<(R::Key, Vec<ResourceLifetime>)> = Vec::new();
    for resource in resources {
        let Some(lifetime) = resource.lifetime() else {
            continue;
        };
        let key = resource.alias_key();
        let existing_slot = resource
            .alias()
            .then(|| {
                slots.iter().position(|(slot_key, lifetimes)| {
                    *slot_key == key && lifetimes.iter().all(|other| other.disjoint(lifetime))
                })
            })
            .flatten();
        if let Some(slot) = existing_slot {
            resource.set_physical_slot(slot);
            slots[slot].1.push(lifetime);
        } else {
            resource.set_physical_slot(slots.len());
            slots.push((key, vec![lifetime]));
        }
    }
    slots.len()
}
