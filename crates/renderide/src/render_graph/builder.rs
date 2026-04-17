//! Render graph builder: setup collection, dependency synthesis, culling, and alias planning.

use std::collections::{HashMap, HashSet};

use super::compiled::{
    ColorAttachmentTemplate, CompileStats, CompiledBufferResource, CompiledGroup, CompiledPassInfo,
    CompiledRenderGraph, CompiledTextureResource, DepthAttachmentTemplate, RenderPassTemplate,
    ResourceLifetime,
};
use super::error::{GraphBuildError, SetupError};
use super::ids::{GroupId, PassId};
use super::pass::{GroupScope, PassBuilder, PassPhase, PassSetup, RenderPass};
use super::resources::{
    AccessKind, BufferHandle, BufferResourceHandle, ImportSource, ImportedBufferDecl,
    ImportedBufferHandle, ImportedTextureDecl, ImportedTextureHandle, ResourceHandle,
    TextureAccess, TextureHandle, TextureResourceHandle, TransientArrayLayers, TransientBufferDesc,
    TransientExtent, TransientSampleCount, TransientTextureDesc, TransientTextureFormat,
};

struct PassEntry {
    group: GroupId,
    pass: Box<dyn RenderPass>,
}

#[derive(Clone, Debug)]
struct GroupEntry {
    name: &'static str,
    scope: GroupScope,
    after: Vec<GroupId>,
}

struct SetupEntry {
    id: PassId,
    group: GroupId,
    name: String,
    setup: PassSetup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TextureAliasKey {
    format: TransientTextureFormat,
    extent: TransientExtent,
    mip_levels: u32,
    sample_count: TransientSampleCount,
    dimension: wgpu::TextureDimension,
    array_layers: TransientArrayLayers,
    usage_bits: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BufferAliasKey {
    size_policy: super::resources::BufferSizePolicy,
    usage_bits: u64,
}

/// Builder for a typed render graph.
pub struct GraphBuilder {
    textures: Vec<TransientTextureDesc>,
    buffers: Vec<TransientBufferDesc>,
    imports_tex: Vec<ImportedTextureDecl>,
    imports_buf: Vec<ImportedBufferDecl>,
    passes: Vec<PassEntry>,
    edges: Vec<(usize, usize)>,
    groups: Vec<GroupEntry>,
    default_frame_group: GroupId,
    default_per_view_group: GroupId,
}

impl GraphBuilder {
    /// Empty builder with default frame-global and per-view groups.
    pub fn new() -> Self {
        let default_frame_group = GroupId(0);
        let default_per_view_group = GroupId(1);
        Self {
            textures: Vec::new(),
            buffers: Vec::new(),
            imports_tex: Vec::new(),
            imports_buf: Vec::new(),
            passes: Vec::new(),
            edges: Vec::new(),
            groups: vec![
                GroupEntry {
                    name: "frame-global",
                    scope: GroupScope::FrameGlobal,
                    after: Vec::new(),
                },
                GroupEntry {
                    name: "per-view",
                    scope: GroupScope::PerView,
                    after: vec![default_frame_group],
                },
            ],
            default_frame_group,
            default_per_view_group,
        }
    }

    /// Declares a graph-owned transient texture.
    pub fn create_texture(&mut self, desc: TransientTextureDesc) -> TextureHandle {
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(desc);
        handle
    }

    /// Declares a graph-owned transient buffer.
    pub fn create_buffer(&mut self, desc: TransientBufferDesc) -> BufferHandle {
        let handle = BufferHandle(self.buffers.len() as u32);
        self.buffers.push(desc);
        handle
    }

    /// Declares an imported texture.
    pub fn import_texture(&mut self, decl: ImportedTextureDecl) -> ImportedTextureHandle {
        let handle = ImportedTextureHandle(self.imports_tex.len() as u32);
        self.imports_tex.push(decl);
        handle
    }

    /// Declares an imported buffer.
    pub fn import_buffer(&mut self, decl: ImportedBufferDecl) -> ImportedBufferHandle {
        let handle = ImportedBufferHandle(self.imports_buf.len() as u32);
        self.imports_buf.push(decl);
        handle
    }

    /// Creates an explicit scheduling group.
    pub fn group(&mut self, name: &'static str, scope: GroupScope) -> GroupId {
        let id = GroupId(self.groups.len());
        self.groups.push(GroupEntry {
            name,
            scope,
            after: Vec::new(),
        });
        id
    }

    /// Orders `group` after `dependency`.
    pub fn group_after(&mut self, group: GroupId, dependency: GroupId) {
        if let Some(entry) = self.groups.get_mut(group.0) {
            entry.after.push(dependency);
        }
    }

    /// Appends a pass to the default group matching its [`PassPhase`].
    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) -> PassId {
        let group = match pass.phase() {
            PassPhase::FrameGlobal => self.default_frame_group,
            PassPhase::PerView => self.default_per_view_group,
        };
        self.add_pass_to_group(group, pass)
    }

    /// Appends a pass to a specific group.
    pub fn add_pass_to_group(&mut self, group: GroupId, pass: Box<dyn RenderPass>) -> PassId {
        let id = PassId(self.passes.len());
        self.passes.push(PassEntry { group, pass });
        id
    }

    /// Appends a pass only when `condition` is true.
    pub fn add_pass_if(&mut self, condition: bool, pass: Box<dyn RenderPass>) -> Option<PassId> {
        if condition {
            Some(self.add_pass(pass))
        } else {
            None
        }
    }

    /// Ensures `from` is scheduled before `to`.
    pub fn add_edge(&mut self, from: PassId, to: PassId) {
        self.edges.push((from.0, to.0));
    }

    /// Compiles setup declarations into an immutable graph.
    pub fn build(mut self) -> Result<CompiledRenderGraph, GraphBuildError> {
        let n = self.passes.len();
        if n == 0 {
            return Ok(self.empty_graph());
        }

        let setups = self.collect_setup()?;
        let mut edges = self.explicit_edges(n)?;
        self.add_group_edges(&setups, &mut edges)?;
        self.add_resource_edges(&setups, &mut edges)?;

        let (sorted, topo_levels) = topo_sort(n, &edges)?;
        let keep = retained_passes(n, &edges, &setups);
        let culled_count = n.saturating_sub(keep.len());
        let ordered: Vec<usize> = sorted
            .into_iter()
            .filter(|idx| keep.contains(idx))
            .collect();

        let retained_ord = retained_ordinals(&ordered);
        let (compiled_textures, texture_slots) =
            compile_textures(&self.textures, &setups, &retained_ord);
        let (compiled_buffers, buffer_slots) =
            compile_buffers(&self.buffers, &setups, &retained_ord);
        let pass_info = compile_pass_info(&setups, &ordered);
        let groups = compile_groups(&self.groups, &pass_info);
        let needs_surface_acquire = needs_surface_acquire(&pass_info, &self.imports_tex);

        let mut pass_take: Vec<Option<Box<dyn RenderPass>>> = self
            .passes
            .into_iter()
            .map(|entry| Some(entry.pass))
            .collect();
        let mut ordered_passes = Vec::with_capacity(ordered.len());
        for idx in ordered {
            ordered_passes.push(
                pass_take[idx]
                    .take()
                    .expect("pass index taken once during build"),
            );
        }

        Ok(CompiledRenderGraph {
            passes: ordered_passes,
            needs_surface_acquire,
            compile_stats: CompileStats {
                pass_count: pass_info.len(),
                topo_levels,
                culled_count,
                transient_texture_count: self.textures.len(),
                transient_texture_slots: texture_slots,
                transient_buffer_count: self.buffers.len(),
                transient_buffer_slots: buffer_slots,
                imported_texture_count: self.imports_tex.len(),
                imported_buffer_count: self.imports_buf.len(),
            },
            groups,
            pass_info,
            transient_textures: compiled_textures,
            transient_buffers: compiled_buffers,
            imported_textures: self.imports_tex,
            imported_buffers: self.imports_buf,
        })
    }

    fn empty_graph(self) -> CompiledRenderGraph {
        CompiledRenderGraph {
            passes: Vec::new(),
            needs_surface_acquire: false,
            compile_stats: CompileStats {
                transient_texture_count: self.textures.len(),
                transient_buffer_count: self.buffers.len(),
                imported_texture_count: self.imports_tex.len(),
                imported_buffer_count: self.imports_buf.len(),
                ..CompileStats::default()
            },
            groups: Vec::new(),
            pass_info: Vec::new(),
            transient_textures: self
                .textures
                .into_iter()
                .map(|desc| CompiledTextureResource {
                    usage: desc.base_usage,
                    desc,
                    lifetime: None,
                    physical_slot: usize::MAX,
                })
                .collect(),
            transient_buffers: self
                .buffers
                .into_iter()
                .map(|desc| CompiledBufferResource {
                    usage: desc.base_usage,
                    desc,
                    lifetime: None,
                    physical_slot: usize::MAX,
                })
                .collect(),
            imported_textures: self.imports_tex,
            imported_buffers: self.imports_buf,
        }
    }

    fn collect_setup(&mut self) -> Result<Vec<SetupEntry>, GraphBuildError> {
        let texture_count = self.textures.len();
        let buffer_count = self.buffers.len();
        let imported_texture_count = self.imports_tex.len();
        let imported_buffer_count = self.imports_buf.len();

        let mut setups = Vec::with_capacity(self.passes.len());
        for (idx, entry) in self.passes.iter_mut().enumerate() {
            let id = PassId(idx);
            let name = entry.pass.name().to_string();
            let mut builder = PassBuilder::new(&name);
            entry
                .pass
                .setup(&mut builder)
                .map_err(|source| GraphBuildError::Setup {
                    pass: id,
                    name: name.clone(),
                    source,
                })?;
            let setup = builder.finish().map_err(|source| GraphBuildError::Setup {
                pass: id,
                name: name.clone(),
                source,
            })?;
            validate_handles(
                &setup,
                texture_count,
                buffer_count,
                imported_texture_count,
                imported_buffer_count,
            )
            .map_err(|source| GraphBuildError::Setup {
                pass: id,
                name: name.clone(),
                source,
            })?;
            setups.push(SetupEntry {
                id,
                group: entry.group,
                name,
                setup,
            });
        }
        Ok(setups)
    }

    fn explicit_edges(&self, n: usize) -> Result<HashSet<(usize, usize)>, GraphBuildError> {
        let mut edges = HashSet::new();
        for &(from, to) in &self.edges {
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

    fn add_group_edges(
        &self,
        setups: &[SetupEntry],
        edges: &mut HashSet<(usize, usize)>,
    ) -> Result<(), GraphBuildError> {
        for entry in &self.groups {
            for &dep in &entry.after {
                if dep.0 >= self.groups.len() {
                    return Err(GraphBuildError::CycleDetected);
                }
            }
        }

        for (a_idx, a) in setups.iter().enumerate() {
            for (b_idx, b) in setups.iter().enumerate() {
                if a_idx == b_idx {
                    continue;
                }
                let a_group = &self.groups[a.group.0];
                let b_group = &self.groups[b.group.0];
                if a_group.scope == GroupScope::FrameGlobal && b_group.scope == GroupScope::PerView
                {
                    edges.insert((a_idx, b_idx));
                }
                if b_group.after.contains(&a.group) {
                    edges.insert((a_idx, b_idx));
                }
            }
        }
        Ok(())
    }

    fn add_resource_edges(
        &self,
        setups: &[SetupEntry],
        edges: &mut HashSet<(usize, usize)>,
    ) -> Result<(), GraphBuildError> {
        let mut by_resource: HashMap<ResourceHandle, Vec<(usize, bool, bool)>> = HashMap::new();
        for (pass_idx, setup) in setups.iter().enumerate() {
            for access in &setup.setup.accesses {
                by_resource.entry(access.resource).or_default().push((
                    pass_idx,
                    access.reads(),
                    access.writes(),
                ));
            }
        }

        for (resource, mut accesses) in by_resource {
            accesses.sort_by_key(|(pass_idx, _, _)| *pass_idx);
            accesses.dedup();
            let mut last_writer: Option<usize> = None;
            let mut readers: HashSet<usize> = HashSet::new();
            for (pass_idx, reads, writes) in accesses {
                if reads {
                    if let Some(writer) = last_writer {
                        if writer != pass_idx {
                            edges.insert((writer, pass_idx));
                        }
                    } else if !resource.is_imported() {
                        return Err(GraphBuildError::MissingDependency {
                            pass: PassId(pass_idx),
                            resource: self.resource_label(resource),
                        });
                    }
                    readers.insert(pass_idx);
                }
                if writes {
                    if let Some(writer) = last_writer {
                        if writer != pass_idx {
                            edges.insert((writer, pass_idx));
                        }
                    }
                    for reader in readers.drain() {
                        if reader != pass_idx {
                            edges.insert((reader, pass_idx));
                        }
                    }
                    last_writer = Some(pass_idx);
                }
            }
        }
        Ok(())
    }

    fn resource_label(&self, resource: ResourceHandle) -> String {
        match resource {
            ResourceHandle::Texture(TextureResourceHandle::Transient(h)) => self
                .textures
                .get(h.index())
                .map(|d| d.label.to_string())
                .unwrap_or_else(|| format!("texture#{}", h.index())),
            ResourceHandle::Texture(TextureResourceHandle::Imported(h)) => self
                .imports_tex
                .get(h.index())
                .map(|d| d.label.to_string())
                .unwrap_or_else(|| format!("imported_texture#{}", h.index())),
            ResourceHandle::Buffer(BufferResourceHandle::Transient(h)) => self
                .buffers
                .get(h.index())
                .map(|d| d.label.to_string())
                .unwrap_or_else(|| format!("buffer#{}", h.index())),
            ResourceHandle::Buffer(BufferResourceHandle::Imported(h)) => self
                .imports_buf
                .get(h.index())
                .map(|d| d.label.to_string())
                .unwrap_or_else(|| format!("imported_buffer#{}", h.index())),
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_handles(
    setup: &PassSetup,
    texture_count: usize,
    buffer_count: usize,
    imported_texture_count: usize,
    imported_buffer_count: usize,
) -> Result<(), SetupError> {
    for access in &setup.accesses {
        validate_resource_handle(
            access.resource,
            texture_count,
            buffer_count,
            imported_texture_count,
            imported_buffer_count,
        )?;
        if let AccessKind::Texture(TextureAccess::ColorAttachment {
            resolve_to: Some(resolve_to),
            ..
        }) = &access.access
        {
            validate_resource_handle(
                ResourceHandle::Texture(*resolve_to),
                texture_count,
                buffer_count,
                imported_texture_count,
                imported_buffer_count,
            )?;
        }
    }
    Ok(())
}

fn validate_resource_handle(
    resource: ResourceHandle,
    texture_count: usize,
    buffer_count: usize,
    imported_texture_count: usize,
    imported_buffer_count: usize,
) -> Result<(), SetupError> {
    match resource {
        ResourceHandle::Texture(TextureResourceHandle::Transient(h))
            if h.index() >= texture_count =>
        {
            Err(SetupError::UnknownTexture(h))
        }
        ResourceHandle::Buffer(BufferResourceHandle::Transient(h)) if h.index() >= buffer_count => {
            Err(SetupError::UnknownBuffer(h))
        }
        ResourceHandle::Texture(TextureResourceHandle::Imported(h))
            if h.index() >= imported_texture_count =>
        {
            Err(SetupError::UnknownImportedTexture(h))
        }
        ResourceHandle::Buffer(BufferResourceHandle::Imported(h))
            if h.index() >= imported_buffer_count =>
        {
            Err(SetupError::UnknownImportedBuffer(h))
        }
        _ => Ok(()),
    }
}

fn topo_sort(
    n: usize,
    edges: &HashSet<(usize, usize)>,
) -> Result<(Vec<usize>, usize), GraphBuildError> {
    let mut in_degree = vec![0usize; n];
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(from, to) in edges {
        if from != to {
            neighbors[from].push(to);
            in_degree[to] += 1;
        }
    }
    for row in &mut neighbors {
        row.sort_unstable();
        row.dedup();
    }

    let mut current: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut sorted = Vec::with_capacity(n);
    let mut levels = 0usize;
    while !current.is_empty() {
        current.sort_unstable();
        levels += 1;
        let mut next = Vec::new();
        for node in current {
            sorted.push(node);
            for &neighbor in &neighbors[node] {
                in_degree[neighbor] = in_degree[neighbor].saturating_sub(1);
                if in_degree[neighbor] == 0 {
                    next.push(neighbor);
                }
            }
        }
        current = next;
    }
    if sorted.len() != n {
        return Err(GraphBuildError::CycleDetected);
    }
    Ok((sorted, levels))
}

fn retained_passes(
    n: usize,
    edges: &HashSet<(usize, usize)>,
    setups: &[SetupEntry],
) -> HashSet<usize> {
    let mut reverse: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(from, to) in edges {
        reverse[to].push(from);
    }

    let mut keep = HashSet::new();
    let mut stack = Vec::new();
    for (idx, entry) in setups.iter().enumerate() {
        let writes_import = entry
            .setup
            .accesses
            .iter()
            .any(|access| access.resource.is_imported() && access.writes());
        if writes_import || entry.setup.cull_exempt {
            keep.insert(idx);
            stack.push(idx);
        }
    }

    while let Some(idx) = stack.pop() {
        for &pred in &reverse[idx] {
            if keep.insert(pred) {
                stack.push(pred);
            }
        }
    }
    keep
}

fn retained_ordinals(ordered: &[usize]) -> HashMap<usize, usize> {
    ordered
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, original)| (original, ordinal))
        .collect()
}

fn compile_pass_info(setups: &[SetupEntry], ordered: &[usize]) -> Vec<CompiledPassInfo> {
    ordered
        .iter()
        .copied()
        .map(|idx| {
            let setup = &setups[idx];
            let raster_template = compile_raster_template(&setup.setup);
            CompiledPassInfo {
                id: setup.id,
                name: setup.name.clone(),
                group: setup.group,
                kind: setup.setup.kind,
                accesses: setup.setup.accesses.clone(),
                multiview_mask: setup.setup.multiview_mask,
                raster_template,
            }
        })
        .collect()
}

fn compile_raster_template(setup: &PassSetup) -> Option<RenderPassTemplate> {
    let color_attachments: Vec<ColorAttachmentTemplate> = setup
        .color_attachments
        .iter()
        .map(|color| ColorAttachmentTemplate {
            target: color.target,
            load: color.load,
            store: color.store,
            resolve_to: color.resolve_to,
        })
        .collect();
    let depth_stencil_attachment =
        setup
            .depth_stencil_attachment
            .as_ref()
            .map(|depth| DepthAttachmentTemplate {
                target: depth.target,
                depth: depth.depth,
                stencil: depth.stencil,
            });
    (!color_attachments.is_empty() || depth_stencil_attachment.is_some()).then_some(
        RenderPassTemplate {
            color_attachments,
            depth_stencil_attachment,
            multiview_mask: setup.multiview_mask,
        },
    )
}

fn compile_groups(groups: &[GroupEntry], pass_info: &[CompiledPassInfo]) -> Vec<CompiledGroup> {
    groups
        .iter()
        .enumerate()
        .filter_map(|(idx, group)| {
            let id = GroupId(idx);
            let pass_indices: Vec<usize> = pass_info
                .iter()
                .enumerate()
                .filter_map(|(pass_idx, info)| (info.group == id).then_some(pass_idx))
                .collect();
            (!pass_indices.is_empty()).then_some(CompiledGroup {
                id,
                name: group.name,
                scope: group.scope,
                pass_indices,
            })
        })
        .collect()
}

fn needs_surface_acquire(pass_info: &[CompiledPassInfo], imports: &[ImportedTextureDecl]) -> bool {
    pass_info.iter().any(|pass| {
        pass.accesses.iter().any(|access| {
            if !access.writes() {
                return false;
            }
            let ResourceHandle::Texture(TextureResourceHandle::Imported(handle)) = access.resource
            else {
                return false;
            };
            imports.get(handle.index()).is_some_and(|decl| {
                matches!(
                    decl.source,
                    ImportSource::FrameTarget(super::resources::FrameTargetRole::ColorAttachment)
                )
            })
        })
    })
}

fn compile_textures(
    descs: &[TransientTextureDesc],
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
            let Some(handle) = access.resource.transient_texture() else {
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

fn compile_buffers(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::error::RenderPassError;
    use crate::render_graph::resources::{
        BufferAccess, BufferImportSource, BufferSizePolicy, FrameTargetRole, HistorySlotId,
        StorageAccess, TextureAttachmentResolve, TextureAttachmentTarget, TransientTextureFormat,
    };
    use crate::render_graph::{PassKind, RenderPassContext};

    struct TestPass {
        name: &'static str,
        phase: PassPhase,
        kind: PassKind,
        texture_reads: Vec<TextureHandle>,
        texture_writes: Vec<TextureHandle>,
        buffer_reads: Vec<BufferHandle>,
        buffer_writes: Vec<BufferHandle>,
        imported_texture_writes: Vec<ImportedTextureHandle>,
        imported_buffer_writes: Vec<ImportedBufferHandle>,
        raster_color: Option<TextureResourceHandle>,
        cull_exempt: bool,
    }

    impl TestPass {
        fn compute(name: &'static str) -> Self {
            Self {
                name,
                phase: PassPhase::PerView,
                kind: PassKind::Compute,
                texture_reads: Vec::new(),
                texture_writes: Vec::new(),
                buffer_reads: Vec::new(),
                buffer_writes: Vec::new(),
                imported_texture_writes: Vec::new(),
                imported_buffer_writes: Vec::new(),
                raster_color: None,
                cull_exempt: false,
            }
        }

        fn raster(name: &'static str, color: impl Into<TextureResourceHandle>) -> Self {
            let mut pass = Self::compute(name);
            pass.kind = PassKind::Raster;
            pass.raster_color = Some(color.into());
            pass
        }

        fn frame_global(mut self) -> Self {
            self.phase = PassPhase::FrameGlobal;
            self
        }

        fn cull_exempt(mut self) -> Self {
            self.cull_exempt = true;
            self
        }
    }

    impl RenderPass for TestPass {
        fn name(&self) -> &str {
            self.name
        }

        fn phase(&self) -> PassPhase {
            self.phase
        }

        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            match self.kind {
                PassKind::Raster => {
                    let mut r = b.raster();
                    if let Some(color) = self.raster_color {
                        r.color(
                            color,
                            wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            Option::<TextureResourceHandle>::None,
                        );
                    }
                }
                PassKind::Compute => b.compute(),
                PassKind::Copy => b.copy(),
                PassKind::Callback => {}
            }
            if self.cull_exempt {
                b.cull_exempt();
            }
            for &h in &self.texture_reads {
                b.read_texture(
                    h,
                    TextureAccess::Sampled {
                        stages: wgpu::ShaderStages::COMPUTE,
                    },
                );
            }
            for &h in &self.texture_writes {
                b.write_texture(h, TextureAccess::CopyDst);
            }
            for &h in &self.buffer_reads {
                b.read_buffer(
                    h,
                    BufferAccess::Storage {
                        stages: wgpu::ShaderStages::COMPUTE,
                        access: StorageAccess::ReadOnly,
                    },
                );
            }
            for &h in &self.buffer_writes {
                b.write_buffer(h, BufferAccess::CopyDst);
            }
            for &h in &self.imported_texture_writes {
                b.import_texture(h, TextureAccess::Present);
            }
            for &h in &self.imported_buffer_writes {
                b.import_buffer(h, BufferAccess::CopyDst);
            }
            Ok(())
        }

        fn execute(
            &mut self,
            _ctx: &mut RenderPassContext<'_, '_, '_>,
        ) -> Result<(), RenderPassError> {
            Ok(())
        }
    }

    fn tex_desc(label: &'static str) -> TransientTextureDesc {
        TransientTextureDesc::texture_2d(
            label,
            wgpu::TextureFormat::Rgba8Unorm,
            TransientExtent::Custom {
                width: 64,
                height: 64,
            },
            1,
            wgpu::TextureUsages::empty(),
        )
    }

    fn frame_sampled_tex_desc(label: &'static str) -> TransientTextureDesc {
        TransientTextureDesc::frame_sampled_texture_2d(
            label,
            wgpu::TextureFormat::Rgba8Unorm,
            TransientExtent::Custom {
                width: 64,
                height: 64,
            },
            wgpu::TextureUsages::empty(),
        )
    }

    fn backbuffer_import() -> ImportedTextureDecl {
        ImportedTextureDecl {
            label: "backbuffer",
            source: ImportSource::FrameTarget(FrameTargetRole::ColorAttachment),
            initial_access: TextureAccess::ColorAttachment {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
                resolve_to: None,
            },
            final_access: TextureAccess::Present,
        }
    }

    fn depth_import() -> ImportedTextureDecl {
        ImportedTextureDecl {
            label: "depth",
            source: ImportSource::FrameTarget(FrameTargetRole::DepthAttachment),
            initial_access: TextureAccess::DepthAttachment {
                depth: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                stencil: None,
            },
            final_access: TextureAccess::Sampled {
                stages: wgpu::ShaderStages::COMPUTE,
            },
        }
    }

    fn buffer_import(label: &'static str) -> ImportedBufferDecl {
        ImportedBufferDecl {
            label,
            source: BufferImportSource::BackendFrameResource(label),
            initial_access: BufferAccess::CopyDst,
            final_access: BufferAccess::CopyDst,
        }
    }

    #[test]
    fn linear_chain_schedules_in_order() {
        let mut b = GraphBuilder::new();
        let tex = b.create_texture(tex_desc("color"));
        let bb = b.import_texture(backbuffer_import());
        let mut a = TestPass::compute("a");
        a.texture_writes.push(tex);
        let mut c = TestPass::raster("c", bb);
        c.texture_reads.push(tex);
        b.add_pass(Box::new(a));
        b.add_pass(Box::new(c));
        let g = b.build().expect("build");
        assert_eq!(g.pass_count(), 2);
        assert_eq!(g.pass_info[0].name, "a");
        assert_eq!(g.pass_info[1].name, "c");
    }

    #[test]
    fn parallel_passes_single_level() {
        let mut b = GraphBuilder::new();
        let out_a = b.import_texture(backbuffer_import());
        let out_b = b.import_buffer(buffer_import("readback"));
        b.add_pass(Box::new(TestPass::raster("a", out_a)));
        let mut b_pass = TestPass::compute("b");
        b_pass.imported_buffer_writes.push(out_b);
        b.add_pass(Box::new(b_pass));
        let g = b.build().expect("build");
        assert_eq!(g.compile_stats.topo_levels, 1);
        assert_eq!(g.pass_count(), 2);
    }

    #[test]
    fn cycle_detected_through_handle_rw_conflict() {
        let mut b = GraphBuilder::new();
        let tex = b.create_texture(tex_desc("color"));
        let bb = b.import_texture(backbuffer_import());
        let mut a = TestPass::raster("a", bb);
        a.texture_reads.push(tex);
        let mut c = TestPass::compute("c");
        c.texture_writes.push(tex);
        let a_id = b.add_pass(Box::new(a));
        let c_id = b.add_pass(Box::new(c));
        b.add_edge(a_id, c_id);
        assert!(matches!(
            b.build(),
            Err(GraphBuildError::MissingDependency { .. })
        ));
    }

    #[test]
    fn read_without_writer_errors_with_handle_and_access() {
        let mut b = GraphBuilder::new();
        let tex = b.create_texture(tex_desc("orphan"));
        let mut p = TestPass::compute("reader");
        p.texture_reads.push(tex);
        b.add_pass(Box::new(p));
        assert!(matches!(
            b.build(),
            Err(GraphBuildError::MissingDependency { .. })
        ));
    }

    #[test]
    fn aliased_handles_share_slot_when_lifetimes_disjoint() {
        let mut b = GraphBuilder::new();
        let a = b.create_texture(tex_desc("a"));
        let c = b.create_texture(tex_desc("c"));
        let bb = b.import_texture(backbuffer_import());
        let mut p0 = TestPass::compute("write-a");
        p0.texture_writes.push(a);
        let mut p1 = TestPass::raster("export-a", bb);
        p1.texture_reads.push(a);
        let mut p2 = TestPass::compute("write-c");
        p2.texture_writes.push(c);
        let mut p3 = TestPass::raster("export-c", bb);
        p3.texture_reads.push(c);
        b.add_pass(Box::new(p0));
        let p1_id = b.add_pass(Box::new(p1));
        let p2_id = b.add_pass(Box::new(p2));
        b.add_pass(Box::new(p3));
        b.add_edge(p1_id, p2_id);
        let g = b.build().expect("build");
        assert_eq!(
            g.transient_textures[a.index()].physical_slot,
            g.transient_textures[c.index()].physical_slot
        );
        assert_eq!(g.compile_stats.transient_texture_slots, 1);
    }

    #[test]
    fn aliased_handles_do_not_share_when_desc_alias_false() {
        let mut b = GraphBuilder::new();
        let mut d0 = tex_desc("a");
        let mut d1 = tex_desc("c");
        d0.alias = false;
        d1.alias = false;
        let a = b.create_texture(d0);
        let c = b.create_texture(d1);
        let bb = b.import_texture(backbuffer_import());
        let mut p0 = TestPass::compute("write-a");
        p0.texture_writes.push(a);
        let mut p1 = TestPass::raster("export-a", bb);
        p1.texture_reads.push(a);
        let mut p2 = TestPass::compute("write-c");
        p2.texture_writes.push(c);
        let mut p3 = TestPass::raster("export-c", bb);
        p3.texture_reads.push(c);
        b.add_pass(Box::new(p0));
        let p1_id = b.add_pass(Box::new(p1));
        let p2_id = b.add_pass(Box::new(p2));
        b.add_pass(Box::new(p3));
        b.add_edge(p1_id, p2_id);
        let g = b.build().expect("build");
        assert_ne!(
            g.transient_textures[a.index()].physical_slot,
            g.transient_textures[c.index()].physical_slot
        );
    }

    #[test]
    fn usage_union_promotes_transient_to_storage_when_sampled_and_stored() {
        let mut b = GraphBuilder::new();
        let tex = b.create_texture(tex_desc("scratch"));
        let bb = b.import_texture(backbuffer_import());
        let mut p0 = TestPass::compute("write");
        p0.texture_writes.push(tex);
        let mut p1 = TestPass::raster("export", bb);
        p1.texture_reads.push(tex);
        b.add_pass(Box::new(p0));
        b.add_pass(Box::new(p1));
        let g = b.build().expect("build");
        let usage = g.transient_textures[tex.index()].usage;
        assert!(usage.contains(wgpu::TextureUsages::COPY_DST));
        assert!(usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn dead_pass_culled_when_output_unused() {
        let mut b = GraphBuilder::new();
        let tex = b.create_texture(tex_desc("dead"));
        let mut p = TestPass::compute("dead");
        p.texture_writes.push(tex);
        b.add_pass(Box::new(p));
        let g = b.build().expect("build");
        assert_eq!(g.pass_count(), 0);
        assert_eq!(g.compile_stats.culled_count, 1);
    }

    #[test]
    fn dead_pass_retained_when_marked_exempt() {
        let mut b = GraphBuilder::new();
        b.add_pass(Box::new(TestPass::compute("side-effect").cull_exempt()));
        let g = b.build().expect("build");
        assert_eq!(g.pass_count(), 1);
    }

    #[test]
    fn raster_pass_without_attachments_rejected() {
        let mut b = GraphBuilder::new();
        let mut p = TestPass::compute("bad");
        p.kind = PassKind::Raster;
        b.add_pass(Box::new(p));
        assert!(matches!(
            b.build(),
            Err(GraphBuildError::Setup {
                source: SetupError::RasterWithoutAttachments,
                ..
            })
        ));
    }

    #[test]
    fn compute_pass_with_attachment_rejected() {
        struct BadPass(ImportedTextureHandle);
        impl RenderPass for BadPass {
            fn name(&self) -> &str {
                "bad"
            }
            fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
                b.compute();
                b.import_texture(
                    self.0,
                    TextureAccess::ColorAttachment {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                        resolve_to: None,
                    },
                );
                Ok(())
            }
            fn execute(
                &mut self,
                _ctx: &mut RenderPassContext<'_, '_, '_>,
            ) -> Result<(), RenderPassError> {
                Ok(())
            }
        }
        let mut b = GraphBuilder::new();
        let bb = b.import_texture(backbuffer_import());
        b.add_pass(Box::new(BadPass(bb)));
        assert!(matches!(
            b.build(),
            Err(GraphBuildError::Setup {
                source: SetupError::NonRasterPassHasAttachment,
                ..
            })
        ));
    }

    #[test]
    fn frameglobal_runs_before_perview_by_default() {
        let mut b = GraphBuilder::new();
        let bb = b.import_texture(backbuffer_import());
        b.add_pass(Box::new(TestPass::raster("per-view", bb)));
        b.add_pass(Box::new(
            TestPass::compute("frame").frame_global().cull_exempt(),
        ));
        let g = b.build().expect("build");
        assert_eq!(g.pass_info[0].name, "frame");
        assert_eq!(g.pass_info[1].name, "per-view");
    }

    #[test]
    fn group_order_respects_group_after_declarations() {
        let mut b = GraphBuilder::new();
        let bb = b.import_texture(backbuffer_import());
        let a_group = b.group("a", GroupScope::PerView);
        let z_group = b.group("z", GroupScope::PerView);
        b.group_after(z_group, a_group);
        b.add_pass_to_group(z_group, Box::new(TestPass::raster("z", bb)));
        b.add_pass_to_group(a_group, Box::new(TestPass::compute("a").cull_exempt()));
        let g = b.build().expect("build");
        assert_eq!(g.pass_info[0].name, "a");
        assert_eq!(g.pass_info[1].name, "z");
    }

    #[test]
    fn multiview_mask_propagates_into_template() {
        struct MvPass(ImportedTextureHandle);
        impl RenderPass for MvPass {
            fn name(&self) -> &str {
                "mv"
            }
            fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
                let mut r = b.raster();
                r.color(
                    self.0,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    Option::<TextureResourceHandle>::None,
                );
                r.multiview(std::num::NonZeroU32::new(3).unwrap());
                Ok(())
            }
            fn execute(
                &mut self,
                _ctx: &mut RenderPassContext<'_, '_, '_>,
            ) -> Result<(), RenderPassError> {
                Ok(())
            }
        }
        let mut b = GraphBuilder::new();
        let bb = b.import_texture(backbuffer_import());
        b.add_pass(Box::new(MvPass(bb)));
        let g = b.build().expect("build");
        assert_eq!(g.pass_info[0].multiview_mask.unwrap().get(), 3);
        assert_eq!(
            g.pass_info[0]
                .raster_template
                .as_ref()
                .and_then(|template| template.multiview_mask)
                .unwrap()
                .get(),
            3
        );
    }

    #[test]
    fn raster_template_records_color_depth_and_resolve_targets() {
        struct RasterTemplatePass {
            color: ImportedTextureHandle,
            resolve: ImportedTextureHandle,
            depth: ImportedTextureHandle,
        }
        impl RenderPass for RasterTemplatePass {
            fn name(&self) -> &str {
                "templated"
            }
            fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
                let mut r = b.raster();
                r.color(
                    self.color,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Discard,
                    },
                    Some(self.resolve),
                );
                r.depth(
                    self.depth,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.5),
                        store: wgpu::StoreOp::Store,
                    },
                    None,
                );
                Ok(())
            }
            fn execute(
                &mut self,
                _ctx: &mut RenderPassContext<'_, '_, '_>,
            ) -> Result<(), RenderPassError> {
                Ok(())
            }
        }

        let mut b = GraphBuilder::new();
        let color = b.import_texture(backbuffer_import());
        let resolve = b.import_texture(backbuffer_import());
        let depth = b.import_texture(depth_import());
        b.add_pass(Box::new(RasterTemplatePass {
            color,
            resolve,
            depth,
        }));

        let g = b.build().expect("build");
        let template = g.pass_info[0]
            .raster_template
            .as_ref()
            .expect("raster template");
        assert_eq!(template.color_attachments.len(), 1);
        assert_eq!(
            template.color_attachments[0].target,
            TextureAttachmentTarget::Resource(TextureResourceHandle::Imported(color))
        );
        assert_eq!(
            template.color_attachments[0].resolve_to,
            Some(TextureAttachmentResolve::Always(
                TextureResourceHandle::Imported(resolve)
            ))
        );
        assert_eq!(template.color_attachments[0].store, wgpu::StoreOp::Discard);
        assert_eq!(
            template.depth_stencil_attachment.as_ref().map(|d| d.target),
            Some(TextureAttachmentTarget::Resource(
                TextureResourceHandle::Imported(depth)
            ))
        );
    }

    #[test]
    fn raster_template_records_frame_sampled_targets() {
        struct FrameSampledRasterPass {
            color: ImportedTextureHandle,
            resolve: ImportedTextureHandle,
            depth: ImportedTextureHandle,
            msaa_color: TextureHandle,
            msaa_depth: TextureHandle,
        }
        impl RenderPass for FrameSampledRasterPass {
            fn name(&self) -> &str {
                "frame-sampled"
            }
            fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
                let mut r = b.raster();
                r.frame_sampled_color(
                    self.color,
                    self.msaa_color,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                    Some(self.resolve),
                );
                r.frame_sampled_depth(
                    self.depth,
                    self.msaa_depth,
                    wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.5),
                        store: wgpu::StoreOp::Store,
                    },
                    None,
                );
                Ok(())
            }
            fn execute(
                &mut self,
                _ctx: &mut RenderPassContext<'_, '_, '_>,
            ) -> Result<(), RenderPassError> {
                Ok(())
            }
        }

        let mut b = GraphBuilder::new();
        let color = b.import_texture(backbuffer_import());
        let resolve = b.import_texture(backbuffer_import());
        let depth = b.import_texture(depth_import());
        let msaa_color = b.create_texture(frame_sampled_tex_desc("msaa-color"));
        let msaa_depth = b.create_texture(TransientTextureDesc::frame_sampled_texture_2d(
            "msaa-depth",
            wgpu::TextureFormat::Depth32Float,
            TransientExtent::Custom {
                width: 64,
                height: 64,
            },
            wgpu::TextureUsages::empty(),
        ));
        b.add_pass(Box::new(FrameSampledRasterPass {
            color,
            resolve,
            depth,
            msaa_color,
            msaa_depth,
        }));

        let g = b.build().expect("build");
        let template = g.pass_info[0]
            .raster_template
            .as_ref()
            .expect("raster template");
        assert_eq!(
            template.color_attachments[0].target,
            TextureAttachmentTarget::FrameSampled {
                single_sample: TextureResourceHandle::Imported(color),
                multisampled: TextureResourceHandle::Transient(msaa_color),
            }
        );
        assert_eq!(
            template.color_attachments[0].resolve_to,
            Some(TextureAttachmentResolve::FrameMultisampled(
                TextureResourceHandle::Imported(resolve)
            ))
        );
        assert_eq!(
            template.depth_stencil_attachment.as_ref().map(|d| d.target),
            Some(TextureAttachmentTarget::FrameSampled {
                single_sample: TextureResourceHandle::Imported(depth),
                multisampled: TextureResourceHandle::Transient(msaa_depth),
            })
        );
    }

    #[test]
    fn buffer_aliasing_uses_size_and_usage_key() {
        let mut b = GraphBuilder::new();
        let a = b.create_buffer(TransientBufferDesc {
            label: "a",
            size_policy: BufferSizePolicy::Fixed(64),
            base_usage: wgpu::BufferUsages::empty(),
            alias: true,
        });
        let c = b.create_buffer(TransientBufferDesc {
            label: "c",
            size_policy: BufferSizePolicy::Fixed(64),
            base_usage: wgpu::BufferUsages::empty(),
            alias: true,
        });
        let out = b.import_buffer(ImportedBufferDecl {
            label: "history",
            source: BufferImportSource::PingPong(HistorySlotId::HiZ),
            initial_access: BufferAccess::CopyDst,
            final_access: BufferAccess::CopyDst,
        });
        let mut p0 = TestPass::compute("write-a");
        p0.buffer_writes.push(a);
        let mut p1 = TestPass::compute("export-a");
        p1.buffer_reads.push(a);
        p1.imported_buffer_writes.push(out);
        let mut p2 = TestPass::compute("write-c");
        p2.buffer_writes.push(c);
        let mut p3 = TestPass::compute("export-c");
        p3.buffer_reads.push(c);
        p3.imported_buffer_writes.push(out);
        b.add_pass(Box::new(p0));
        let p1_id = b.add_pass(Box::new(p1));
        let p2_id = b.add_pass(Box::new(p2));
        b.add_pass(Box::new(p3));
        b.add_edge(p1_id, p2_id);
        let g = b.build().expect("build");
        assert_eq!(
            g.transient_buffers[a.index()].physical_slot,
            g.transient_buffers[c.index()].physical_slot
        );
    }

    #[test]
    fn texture_aliasing_keys_on_sample_count_policy() {
        let mut b = GraphBuilder::new();
        let frame_sampled = b.create_texture(frame_sampled_tex_desc("frame-sampled"));
        let fixed = b.create_texture(tex_desc("fixed"));
        let bb = b.import_texture(backbuffer_import());
        let mut p0 = TestPass::compute("write-frame-sampled");
        p0.texture_writes.push(frame_sampled);
        let mut p1 = TestPass::raster("export-frame-sampled", bb);
        p1.texture_reads.push(frame_sampled);
        let mut p2 = TestPass::compute("write-fixed");
        p2.texture_writes.push(fixed);
        let mut p3 = TestPass::raster("export-fixed", bb);
        p3.texture_reads.push(fixed);
        b.add_pass(Box::new(p0));
        let p1_id = b.add_pass(Box::new(p1));
        b.add_pass(Box::new(p2));
        b.add_pass(Box::new(p3));
        b.add_edge(p1_id, PassId(2));

        let g = b.build().expect("build");
        assert_ne!(
            g.transient_textures[frame_sampled.index()].physical_slot,
            g.transient_textures[fixed.index()].physical_slot
        );
    }

    #[test]
    fn frame_sample_count_policy_resolves_current_frame_value() {
        assert_eq!(TransientSampleCount::Fixed(0).resolve(4), 1);
        assert_eq!(TransientSampleCount::Fixed(2).resolve(4), 2);
        assert_eq!(TransientSampleCount::Frame.resolve(0), 1);
        assert_eq!(TransientSampleCount::Frame.resolve(4), 4);
    }

    #[test]
    fn frame_texture_format_and_layer_policies_resolve_current_frame_values() {
        assert_eq!(
            TransientTextureFormat::Fixed(wgpu::TextureFormat::Rgba8Unorm)
                .resolve(wgpu::TextureFormat::Bgra8UnormSrgb),
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            TransientTextureFormat::FrameColor.resolve(wgpu::TextureFormat::Bgra8UnormSrgb),
            wgpu::TextureFormat::Bgra8UnormSrgb
        );
        assert_eq!(TransientArrayLayers::Fixed(0).resolve(true), 1);
        assert_eq!(TransientArrayLayers::Fixed(3).resolve(false), 3);
        assert_eq!(TransientArrayLayers::Frame.resolve(false), 1);
        assert_eq!(TransientArrayLayers::Frame.resolve(true), 2);
    }
}
