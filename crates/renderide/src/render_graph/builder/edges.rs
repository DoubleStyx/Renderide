//! Explicit group edges and resource read/write dependency edges.

use hashbrown::HashMap;
use std::collections::HashSet;

use super::super::error::GraphBuildError;
use super::super::ids::{GroupId, PassId};
use super::super::pass::GroupScope;
use super::super::resources::{
    BufferResourceHandle, ResourceHandle, TextureResourceHandle, TextureSubresourceRange,
};
use super::decl::SetupEntry;
use super::GraphBuilder;

pub(super) fn explicit_edges(
    builder: &GraphBuilder,
    n: usize,
) -> Result<HashSet<(usize, usize)>, GraphBuildError> {
    let mut edges = HashSet::new();
    for &(from, to) in &builder.edges {
        if from >= n || to >= n {
            return Err(GraphBuildError::InvalidEdge {
                from: PassId(from),
                to: PassId(to),
            });
        }
        if from != to {
            edges.insert((from, to));
        }
    }
    Ok(edges)
}

/// Adds linear-size relay edges so every pass in set `a` precedes every pass in set `b`.
fn relay_all_before(a: &[usize], b: &[usize], edges: &mut HashSet<(usize, usize)>) {
    if a.is_empty() || b.is_empty() {
        return;
    }
    let mut b_sorted = b.to_vec();
    b_sorted.sort_unstable();
    let rep_b = b_sorted[0];
    for &ai in a {
        if ai != rep_b {
            edges.insert((ai, rep_b));
        }
    }
    for &bi in b_sorted.iter().skip(1) {
        edges.insert((rep_b, bi));
    }
}

pub(super) fn add_group_edges(
    builder: &GraphBuilder,
    setups: &[SetupEntry],
    edges: &mut HashSet<(usize, usize)>,
) -> Result<(), GraphBuildError> {
    for entry in &builder.groups {
        for &dep in &entry.after {
            if dep.0 >= builder.groups.len() {
                return Err(GraphBuildError::CycleDetected);
            }
        }
    }

    let mut frame_global = Vec::new();
    let mut per_view = Vec::new();
    for (idx, setup) in setups.iter().enumerate() {
        match builder.groups[setup.group.0].scope {
            GroupScope::FrameGlobal => frame_global.push(idx),
            GroupScope::PerView => per_view.push(idx),
        }
    }
    frame_global.sort_unstable();
    per_view.sort_unstable();
    relay_all_before(&frame_global, &per_view, edges);

    for (gb_idx, gb) in builder.groups.iter().enumerate() {
        let gb_id = GroupId(gb_idx);
        let mut passes_b: Vec<usize> = setups
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (s.group == gb_id).then_some(i))
            .collect();
        passes_b.sort_unstable();
        for &ga_id in &gb.after {
            let mut passes_a: Vec<usize> = setups
                .iter()
                .enumerate()
                .filter_map(|(i, s)| (s.group == ga_id).then_some(i))
                .collect();
            passes_a.sort_unstable();
            relay_all_before(&passes_a, &passes_b, edges);
        }
    }
    Ok(())
}

pub(super) fn add_resource_edges(
    builder: &GraphBuilder,
    setups: &[SetupEntry],
    edges: &mut HashSet<(usize, usize)>,
) -> Result<(), GraphBuildError> {
    let mut by_domain: HashMap<ResourceDomain, Vec<ResourceAccessEvent>> = HashMap::new();
    for (pass_idx, setup) in setups.iter().enumerate() {
        for access in &setup.setup.accesses {
            let event = ResourceAccessEvent::new(
                builder,
                pass_idx,
                access.resource,
                access.reads(),
                access.writes(),
            );
            by_domain.entry(event.domain).or_default().push(event);
        }
    }

    for (domain, mut accesses) in by_domain {
        accesses.sort_by_key(|access| access.pass_idx);
        accesses.dedup();
        let mut writers: Vec<ResourceAccessEvent> = Vec::new();
        let mut readers: Vec<ResourceAccessEvent> = Vec::new();
        for access in accesses {
            let pass_idx = access.pass_idx;
            if access.reads {
                let mut found_writer = false;
                for writer in writers
                    .iter()
                    .filter(|writer| writer.footprint.overlaps(access.footprint))
                {
                    found_writer = true;
                    if writer.pass_idx != pass_idx {
                        edges.insert((writer.pass_idx, pass_idx));
                    }
                }
                if !found_writer && !domain.is_imported() {
                    return Err(GraphBuildError::MissingDependency {
                        pass: PassId(pass_idx),
                        resource: domain_label(builder, domain),
                    });
                }
                readers.push(access);
            }
            if access.writes {
                for writer in writers
                    .iter()
                    .filter(|writer| writer.footprint.overlaps(access.footprint))
                {
                    if writer.pass_idx != pass_idx {
                        edges.insert((writer.pass_idx, pass_idx));
                    }
                }
                for reader in readers
                    .iter()
                    .filter(|reader| reader.footprint.overlaps(access.footprint))
                {
                    if reader.pass_idx != pass_idx {
                        edges.insert((reader.pass_idx, pass_idx));
                    }
                }
                readers.retain(|reader| !access.footprint.overlaps(reader.footprint));
                writers.retain(|writer| !access.footprint.covers(writer.footprint));
                writers.push(access);
            }
        }
    }
    Ok(())
}

/// Parent resource domain used for overlap-aware dependency synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ResourceDomain {
    /// Transient texture domain.
    TransientTexture(super::super::resources::TextureHandle),
    /// Imported texture domain.
    ImportedTexture(super::super::resources::ImportedTextureHandle),
    /// Buffer domain.
    Buffer(BufferResourceHandle),
}

impl ResourceDomain {
    /// Returns whether this domain is externally owned.
    fn is_imported(self) -> bool {
        matches!(
            self,
            Self::ImportedTexture(_) | Self::Buffer(BufferResourceHandle::Imported(_))
        )
    }
}

/// Access footprint within a resource domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ResourceFootprint {
    /// Entire domain.
    Full,
    /// Texture mip/layer subrange.
    TextureSubresource(TextureSubresourceRange),
}

impl ResourceFootprint {
    /// Returns whether this footprint overlaps `other`.
    fn overlaps(self, other: Self) -> bool {
        match (self, other) {
            (Self::Full, _) | (_, Self::Full) => true,
            (Self::TextureSubresource(a), Self::TextureSubresource(b)) => a.overlaps(b),
        }
    }

    /// Returns whether this footprint fully covers `other`.
    fn covers(self, other: Self) -> bool {
        match (self, other) {
            (Self::Full, _) => true,
            (_, Self::Full) => false,
            (Self::TextureSubresource(a), Self::TextureSubresource(b)) => a.covers(b),
        }
    }
}

/// One pass access projected into a parent domain plus an overlap footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ResourceAccessEvent {
    /// Pass declaration index.
    pass_idx: usize,
    /// Parent domain used for dependency checks.
    domain: ResourceDomain,
    /// Full-resource or texture-subresource span touched by the pass.
    footprint: ResourceFootprint,
    /// Whether the access reads prior contents.
    reads: bool,
    /// Whether the access writes contents.
    writes: bool,
}

impl ResourceAccessEvent {
    /// Builds an access event from a setup declaration.
    fn new(
        builder: &GraphBuilder,
        pass_idx: usize,
        resource: ResourceHandle,
        reads: bool,
        writes: bool,
    ) -> Self {
        match resource {
            ResourceHandle::Texture(TextureResourceHandle::Transient(handle)) => {
                let desc = &builder.textures[handle.index()];
                let layers = match desc.array_layers {
                    super::super::resources::TransientArrayLayers::Fixed(layers) => layers.max(1),
                    super::super::resources::TransientArrayLayers::Frame => 2,
                };
                Self {
                    pass_idx,
                    domain: ResourceDomain::TransientTexture(handle),
                    footprint: ResourceFootprint::TextureSubresource(
                        TextureSubresourceRange::full(desc.mip_levels.max(1), layers),
                    ),
                    reads,
                    writes,
                }
            }
            ResourceHandle::Texture(TextureResourceHandle::Imported(handle)) => Self {
                pass_idx,
                domain: ResourceDomain::ImportedTexture(handle),
                footprint: ResourceFootprint::Full,
                reads,
                writes,
            },
            ResourceHandle::TextureSubresource(handle) => {
                let desc = builder.subresources[handle.index()];
                Self {
                    pass_idx,
                    domain: ResourceDomain::TransientTexture(desc.parent),
                    footprint: ResourceFootprint::TextureSubresource(desc.range()),
                    reads,
                    writes,
                }
            }
            ResourceHandle::Buffer(handle) => Self {
                pass_idx,
                domain: ResourceDomain::Buffer(handle),
                footprint: ResourceFootprint::Full,
                reads,
                writes,
            },
        }
    }
}

fn resource_label(builder: &GraphBuilder, resource: ResourceHandle) -> String {
    match resource {
        ResourceHandle::Texture(TextureResourceHandle::Transient(h)) => builder
            .textures
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("texture#{}", h.index())),
        ResourceHandle::Texture(TextureResourceHandle::Imported(h)) => builder
            .imports_tex
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("imported_texture#{}", h.index())),
        ResourceHandle::Buffer(BufferResourceHandle::Transient(h)) => builder
            .buffers
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("buffer#{}", h.index())),
        ResourceHandle::Buffer(BufferResourceHandle::Imported(h)) => builder
            .imports_buf
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("imported_buffer#{}", h.index())),
        ResourceHandle::TextureSubresource(h) => builder
            .subresources
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("subresource#{}", h.index())),
    }
}

fn domain_label(builder: &GraphBuilder, domain: ResourceDomain) -> String {
    match domain {
        ResourceDomain::TransientTexture(h) => resource_label(
            builder,
            ResourceHandle::Texture(TextureResourceHandle::Transient(h)),
        ),
        ResourceDomain::ImportedTexture(h) => resource_label(
            builder,
            ResourceHandle::Texture(TextureResourceHandle::Imported(h)),
        ),
        ResourceDomain::Buffer(h) => resource_label(builder, ResourceHandle::Buffer(h)),
    }
}
