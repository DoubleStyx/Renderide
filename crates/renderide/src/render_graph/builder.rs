//! Render graph builder: setup collection, dependency synthesis, culling, and alias planning.

mod decl;
mod edges;
mod lifetime;
mod topo;
mod validate;

#[cfg(test)]
mod tests;

use decl::{GroupEntry, PassEntry, SetupEntry};
use edges::{add_blackboard_edges, add_group_edges, add_resource_edges, explicit_edges};
use lifetime::{compile_buffers, compile_textures};
use topo::{retained_ordinals, retained_passes, topo_sort};
use validate::validate_handles;

use std::collections::BTreeSet;

use super::compiled::{
    CompileStats, CompiledBufferResource, CompiledPassInfo, CompiledRenderGraph,
    CompiledTextureResource,
};
use super::error::GraphBuildError;
use super::ids::{GroupId, PassId};
use super::pass::{
    BlackboardSeedDecl, ColorAttachmentTemplate, ComputePass, DepthAttachmentTemplate, EncoderPass,
    GroupScope, PassBuilder, PassMergeHint, PassNode, PassPhase, PassWorkloadFlags, RasterPass,
    RenderPassTemplate,
};
use super::resources::{
    BufferHandle, BufferResourceHandle, FrameTargetRole, ImportSource, ImportedBufferDecl,
    ImportedBufferHandle, ImportedTextureDecl, ImportedTextureHandle, ResourceHandle,
    SubresourceHandle, TextureAccess, TextureHandle, TextureResourceHandle, TransientArrayLayers,
    TransientBufferDesc, TransientSubresourceDesc, TransientTextureDesc,
};
use super::schedule::{
    FrameSchedule, ImportedFinalAccess, ImportedResourceFinalAccess, ImportedScheduleResource,
    RenderPassMergeGroup, ResourceScheduleEvent, ResourceScheduleEventKind, ScheduleHudSnapshot,
    ScheduleStep, ScheduleUploadPhase, ScheduledResource,
};
use super::validation::{GraphValidationReport, RenderGraphValidationMode};

/// Builder for a typed render graph.
pub struct GraphBuilder {
    pub(crate) textures: Vec<TransientTextureDesc>,
    pub(crate) buffers: Vec<TransientBufferDesc>,
    pub(crate) subresources: Vec<TransientSubresourceDesc>,
    pub(crate) imports_tex: Vec<ImportedTextureDecl>,
    pub(crate) imports_buf: Vec<ImportedBufferDecl>,
    pub(crate) passes: Vec<PassEntry>,
    pub(crate) edges: Vec<(usize, usize)>,
    pub(crate) groups: Vec<GroupEntry>,
    pub(crate) blackboard_seeds: Vec<BlackboardSeedDecl>,
    validation_mode: RenderGraphValidationMode,
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
            subresources: Vec::new(),
            imports_tex: Vec::new(),
            imports_buf: Vec::new(),
            passes: Vec::new(),
            edges: Vec::new(),
            groups: vec![
                GroupEntry {
                    scope: GroupScope::FrameGlobal,
                    after: Vec::new(),
                },
                GroupEntry {
                    scope: GroupScope::PerView,
                    after: vec![default_frame_group],
                },
            ],
            blackboard_seeds: Vec::new(),
            validation_mode: RenderGraphValidationMode::default(),
            default_frame_group,
            default_per_view_group,
        }
    }

    /// Empty builder with an explicit validation mode.
    pub fn with_validation_mode(validation_mode: RenderGraphValidationMode) -> Self {
        Self {
            validation_mode,
            ..Self::new()
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

    /// Declares a subresource view of a transient texture.
    ///
    /// The concrete [`wgpu::TextureView`] is created lazily at execute time and cached per-range
    /// on the graph-resources context. Resolve one at encode time via
    /// [`crate::render_graph::GraphResolvedResources::subresource_view`].
    ///
    /// Passes can declare reads and writes against the returned handle with
    /// [`PassBuilder::read_texture_subresource`] and
    /// [`PassBuilder::write_texture_subresource`]. The graph then orders only overlapping
    /// mip/layer ranges and keeps the parent texture alive for lifetime and alias planning.
    pub fn create_subresource(&mut self, desc: TransientSubresourceDesc) -> SubresourceHandle {
        let handle = SubresourceHandle(self.subresources.len() as u32);
        self.subresources.push(desc);
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

    /// Declares a blackboard slot seeded before graph pass recording.
    pub fn seed_blackboard<S: super::blackboard::BlackboardSlot>(
        &mut self,
        producer: &'static str,
    ) {
        self.blackboard_seeds
            .push(BlackboardSeedDecl::new::<S>(producer));
    }

    /// Creates an explicit scheduling group.
    #[cfg(test)]
    pub fn group(&mut self, _name: &'static str, scope: GroupScope) -> GroupId {
        let id = GroupId(self.groups.len());
        self.groups.push(GroupEntry {
            scope,
            after: Vec::new(),
        });
        id
    }

    /// Orders `group` after `dependency`.
    #[cfg(test)]
    pub fn group_after(&mut self, group: GroupId, dependency: GroupId) {
        if let Some(entry) = self.groups.get_mut(group.0) {
            entry.after.push(dependency);
        }
    }

    /// Appends a [`PassNode`] to the default group matching its [`PassPhase`].
    pub fn add_pass(&mut self, pass: PassNode) -> PassId {
        let group = match pass.phase() {
            PassPhase::FrameGlobal => self.default_frame_group,
            PassPhase::PerView => self.default_per_view_group,
        };
        self.add_pass_to_group(group, pass)
    }

    /// Appends a [`PassNode`] to a specific group.
    pub fn add_pass_to_group(&mut self, group: GroupId, pass: PassNode) -> PassId {
        let id = PassId(self.passes.len());
        self.passes.push(PassEntry { group, pass });
        id
    }

    /// Appends a raster pass to the default per-view group.
    pub fn add_raster_pass(&mut self, pass: Box<dyn RasterPass>) -> PassId {
        self.add_pass(PassNode::Raster(pass))
    }

    /// Appends a compute pass to the default group for its phase.
    pub fn add_compute_pass(&mut self, pass: Box<dyn ComputePass>) -> PassId {
        self.add_pass(PassNode::Compute(pass))
    }

    /// Appends an encoder pass to the default group for its phase.
    pub fn add_encoder_pass(&mut self, pass: Box<dyn EncoderPass>) -> PassId {
        self.add_pass(PassNode::Encoder(pass))
    }

    /// Appends a raster pass to a specific group.
    #[cfg(test)]
    pub fn add_raster_pass_to_group(
        &mut self,
        group: GroupId,
        pass: Box<dyn RasterPass>,
    ) -> PassId {
        self.add_pass_to_group(group, PassNode::Raster(pass))
    }

    /// Appends a compute pass to a specific group.
    #[cfg(test)]
    pub fn add_compute_pass_to_group(
        &mut self,
        group: GroupId,
        pass: Box<dyn ComputePass>,
    ) -> PassId {
        self.add_pass_to_group(group, PassNode::Compute(pass))
    }

    /// Ensures `from` is scheduled before `to`.
    pub fn add_edge(&mut self, from: PassId, to: PassId) {
        self.edges.push((from.0, to.0));
    }

    /// Compiles setup declarations into an immutable graph.
    pub fn build(mut self) -> Result<CompiledRenderGraph, GraphBuildError> {
        self.validate_subresource_decls()?;
        let n = self.passes.len();
        if n == 0 {
            return Ok(self.empty_graph());
        }

        let setups = self.collect_setup()?;
        let (edges, validation_report) = self.synthesize_edges_and_validation(&setups, n)?;
        let (sorted, wave_by_node) = topo_sort(n, &edges)?;
        let topo_levels = wave_by_node.iter().copied().max().map_or(0, |max| max + 1);
        #[cfg(debug_assertions)]
        {
            let mut pos = vec![0usize; n];
            for (ord, &node) in sorted.iter().enumerate() {
                pos[node] = ord;
            }
            for &(u, v) in &edges {
                debug_assert!(
                    pos[u] < pos[v],
                    "topological order violates edge ({u} -> {v})"
                );
            }
        }
        let keep = retained_passes(n, &edges, &setups);
        let culled_count = n.saturating_sub(keep.len());
        let ordered: Vec<usize> = sorted
            .into_iter()
            .filter(|idx| keep.contains(idx))
            .collect();

        let retained_ord = retained_ordinals(&ordered);
        let (compiled_textures, texture_slots, texture_lifetime_lanes) =
            compile_textures(&self.textures, &self.subresources, &setups, &retained_ord);
        let (compiled_buffers, buffer_slots, buffer_lifetime_lanes) =
            compile_buffers(&self.buffers, &setups, &retained_ord);
        let pass_info = compile_pass_info(&setups, &ordered);
        let imported_final_accesses =
            compile_imported_final_accesses(&self.imports_tex, &self.imports_buf, &pass_info)?;
        let needs_surface_acquire = needs_surface_acquire(&pass_info, &self.imports_tex);

        // Build passes in retained order, taking ownership from the declaration list.
        let mut pass_take: Vec<Option<PassNode>> = self
            .passes
            .into_iter()
            .map(|entry| Some(entry.pass))
            .collect();
        let mut ordered_passes: Vec<PassNode> = Vec::with_capacity(ordered.len());
        for idx in &ordered {
            let Some(pass) = pass_take[*idx].take() else {
                return Err(GraphBuildError::PassOwnershipInvariant {
                    message: "pass index taken more than once during build",
                });
            };
            ordered_passes.push(pass);
        }

        // Build FrameSchedule: single source of truth for pass ordering and scheduler policy.
        let schedule = build_frame_schedule(
            &ordered_passes,
            &ordered,
            &wave_by_node,
            &compiled_textures,
            &compiled_buffers,
            imported_final_accesses,
            &pass_info,
        )?;
        let schedule_hud = ScheduleHudSnapshot::from_schedule(&schedule);
        let validation_diagnostics = validation_report.len();
        let render_pass_merge_groups = schedule.render_pass_merge_groups.len();
        let render_pass_materialization_groups =
            schedule.render_pass_materialization_plan.groups.len();

        Ok(CompiledRenderGraph {
            passes: ordered_passes,
            needs_surface_acquire,
            compile_stats: CompileStats {
                pass_count: pass_info.len(),
                topo_levels,
                culled_count,
                transient_texture_count: self.textures.len(),
                transient_texture_slots: texture_slots,
                transient_texture_lanes: texture_lifetime_lanes.len(),
                transient_buffer_count: self.buffers.len(),
                transient_buffer_slots: buffer_slots,
                transient_buffer_lanes: buffer_lifetime_lanes.len(),
                imported_texture_count: self.imports_tex.len(),
                imported_buffer_count: self.imports_buf.len(),
                validation_diagnostics,
                render_pass_merge_groups,
                render_pass_materialization_groups,
            },
            pass_info,
            transient_textures: compiled_textures,
            transient_buffers: compiled_buffers,
            texture_lifetime_lanes,
            buffer_lifetime_lanes,
            subresources: self.subresources,
            imported_textures: self.imports_tex,
            imported_buffers: self.imports_buf,
            schedule,
            schedule_hud,
            validation_report,
            validation_mode: self.validation_mode,
            main_graph_msaa_transient_handles: None,
        })
    }

    fn empty_graph(self) -> CompiledRenderGraph {
        let schedule = FrameSchedule::empty();
        let schedule_hud = ScheduleHudSnapshot::from_schedule(&schedule);
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
            texture_lifetime_lanes: Vec::new(),
            buffer_lifetime_lanes: Vec::new(),
            subresources: self.subresources,
            imported_textures: self.imports_tex,
            imported_buffers: self.imports_buf,
            schedule,
            schedule_hud,
            validation_report: GraphValidationReport::new(self.validation_mode),
            validation_mode: self.validation_mode,
            main_graph_msaa_transient_handles: None,
        }
    }

    fn collect_setup(&mut self) -> Result<Vec<SetupEntry>, GraphBuildError> {
        let texture_count = self.textures.len();
        let buffer_count = self.buffers.len();
        let subresource_count = self.subresources.len();
        let imported_texture_count = self.imports_tex.len();
        let imported_buffer_count = self.imports_buf.len();

        let mut setups = Vec::with_capacity(self.passes.len());
        for (idx, entry) in self.passes.iter_mut().enumerate() {
            let id = PassId(idx);
            let name = entry.pass.name().to_string();
            let mut builder = PassBuilder::new(&name);
            entry
                .pass
                .call_setup(&mut builder)
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
                subresource_count,
                imported_texture_count,
                imported_buffer_count,
            )
            .map_err(|source| GraphBuildError::Setup {
                pass: id,
                name: name.clone(),
                source,
            })?;
            setups.push(SetupEntry {
                group: entry.group,
                name,
                setup,
            });
        }
        Ok(setups)
    }

    /// Synthesizes scheduling edges and validates declared blackboard dependencies.
    fn synthesize_edges_and_validation(
        &self,
        setups: &[SetupEntry],
        n: usize,
    ) -> Result<(BTreeSet<(usize, usize)>, GraphValidationReport), GraphBuildError> {
        let mut edges = explicit_edges(self, n)?;
        add_group_edges(self, setups, &mut edges)?;
        add_resource_edges(self, setups, &mut edges)?;
        let mut validation_report = GraphValidationReport::new(self.validation_mode);
        add_blackboard_edges(self, setups, &mut edges, &mut validation_report);
        validation_report.log();
        if self.validation_mode.is_strict() && !validation_report.is_empty() {
            return Err(GraphBuildError::Validation {
                report: validation_report,
            });
        }
        Ok((edges, validation_report))
    }

    /// Validates subresource declarations before pass setup starts using their handles.
    fn validate_subresource_decls(&self) -> Result<(), GraphBuildError> {
        for (idx, subresource) in self.subresources.iter().enumerate() {
            let handle = SubresourceHandle(idx as u32);
            let Some(parent) = self.textures.get(subresource.parent.index()) else {
                return Err(GraphBuildError::InvalidSubresource {
                    handle,
                    reason: "parent texture handle is unknown",
                });
            };
            if subresource.mip_level_count == 0 {
                return Err(GraphBuildError::InvalidSubresource {
                    handle,
                    reason: "mip level count must be at least one",
                });
            }
            if subresource.array_layer_count == 0 {
                return Err(GraphBuildError::InvalidSubresource {
                    handle,
                    reason: "array layer count must be at least one",
                });
            }
            let Some(mip_end) = subresource
                .base_mip_level
                .checked_add(subresource.mip_level_count)
            else {
                return Err(GraphBuildError::InvalidSubresource {
                    handle,
                    reason: "mip range overflows u32",
                });
            };
            if mip_end > parent.mip_levels.max(1) {
                return Err(GraphBuildError::InvalidSubresource {
                    handle,
                    reason: "mip range exceeds parent texture mip count",
                });
            }
            let max_layers = match parent.array_layers {
                TransientArrayLayers::Fixed(layers) => layers.max(1),
                TransientArrayLayers::Frame => 2,
            };
            let Some(layer_end) = subresource
                .base_array_layer
                .checked_add(subresource.array_layer_count)
            else {
                return Err(GraphBuildError::InvalidSubresource {
                    handle,
                    reason: "array layer range overflows u32",
                });
            };
            if layer_end > max_layers {
                return Err(GraphBuildError::InvalidSubresource {
                    handle,
                    reason: "array layer range exceeds parent texture layer count",
                });
            }
        }
        Ok(())
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds the [`FrameSchedule`] from the ordered, retained pass list.
fn build_frame_schedule(
    ordered_passes: &[PassNode],
    ordered: &[usize],
    wave_by_node: &[usize],
    compiled_textures: &[CompiledTextureResource],
    compiled_buffers: &[CompiledBufferResource],
    imported_final_accesses: Vec<ImportedResourceFinalAccess>,
    pass_info: &[CompiledPassInfo],
) -> Result<FrameSchedule, GraphBuildError> {
    let mut steps = Vec::with_capacity(ordered_passes.len());
    for (schedule_idx, pass) in ordered_passes.iter().enumerate() {
        let orig_idx = ordered[schedule_idx];
        let wave_idx = wave_by_node.get(orig_idx).copied().unwrap_or(0);
        let phase = pass.phase();
        let upload_phase = match phase {
            PassPhase::FrameGlobal => ScheduleUploadPhase::FrameGlobal,
            PassPhase::PerView => ScheduleUploadPhase::PerView,
        };
        steps.push(ScheduleStep {
            phase,
            pass_idx: schedule_idx,
            wave_idx,
            upload_phase,
        });
    }
    let waves = build_wave_ranges(&steps);
    let resource_events = compile_resource_schedule_events(compiled_textures, compiled_buffers);
    let render_pass_merge_groups = plan_render_pass_merge_groups(&steps, pass_info);
    let schedule = FrameSchedule::new(
        steps,
        waves,
        resource_events,
        imported_final_accesses,
        render_pass_merge_groups,
    );
    schedule
        .validate()
        .map_err(|source| GraphBuildError::InvalidSchedule { source })?;
    Ok(schedule)
}

/// Compacts step-local wave indices into contiguous step ranges.
fn build_wave_ranges(steps: &[ScheduleStep]) -> Vec<std::ops::Range<usize>> {
    if steps.is_empty() {
        return Vec::new();
    }
    let mut waves = Vec::new();
    let mut start = 0usize;
    let mut current_wave = steps[0].wave_idx;
    for (idx, step) in steps.iter().enumerate().skip(1) {
        if step.wave_idx != current_wave {
            waves.push(start..idx);
            start = idx;
            current_wave = step.wave_idx;
        }
    }
    waves.push(start..steps.len());
    waves
}

/// Emits scheduler-visible first-use and last-use events for transient resources.
fn compile_resource_schedule_events(
    compiled_textures: &[CompiledTextureResource],
    compiled_buffers: &[CompiledBufferResource],
) -> Vec<ResourceScheduleEvent> {
    let mut events = Vec::new();
    for (idx, texture) in compiled_textures.iter().enumerate() {
        if let Some(lifetime) = texture.lifetime {
            let resource = ScheduledResource::Texture(TextureHandle(idx as u32));
            events.push(ResourceScheduleEvent {
                resource,
                pass_idx: lifetime.first_pass,
                kind: ResourceScheduleEventKind::Allocate,
            });
            events.push(ResourceScheduleEvent {
                resource,
                pass_idx: lifetime.last_pass,
                kind: ResourceScheduleEventKind::Release,
            });
        }
    }
    for (idx, buffer) in compiled_buffers.iter().enumerate() {
        if let Some(lifetime) = buffer.lifetime {
            let resource = ScheduledResource::Buffer(BufferHandle(idx as u32));
            events.push(ResourceScheduleEvent {
                resource,
                pass_idx: lifetime.first_pass,
                kind: ResourceScheduleEventKind::Allocate,
            });
            events.push(ResourceScheduleEvent {
                resource,
                pass_idx: lifetime.last_pass,
                kind: ResourceScheduleEventKind::Release,
            });
        }
    }
    events.sort_by_key(|event| {
        let kind_order = match event.kind {
            ResourceScheduleEventKind::Allocate => 0usize,
            ResourceScheduleEventKind::Release => 1usize,
        };
        (event.pass_idx, kind_order)
    });
    events
}

fn compile_pass_info(setups: &[SetupEntry], ordered: &[usize]) -> Vec<CompiledPassInfo> {
    ordered
        .iter()
        .copied()
        .map(|idx| {
            let setup = &setups[idx];
            let raster_template = compile_raster_template(&setup.setup);
            CompiledPassInfo {
                name: setup.name.clone(),
                #[cfg(test)]
                kind: setup.setup.kind,
                workload_flags: setup.setup.workload_flags,
                accesses: setup.setup.accesses.clone(),
                blackboard_accesses: setup.setup.blackboard_accesses.clone(),
                parameter_schema: setup.setup.parameter_schema.clone(),
                #[cfg(test)]
                multiview_mask: setup.setup.multiview_mask,
                raster_template,
                merge_hint: setup.setup.merge_hint,
            }
        })
        .collect()
}

fn compile_raster_template(setup: &super::pass::PassSetup) -> Option<RenderPassTemplate> {
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

/// Builds the imported-resource final access plan and validates presentable frame targets.
fn compile_imported_final_accesses(
    texture_imports: &[ImportedTextureDecl],
    buffer_imports: &[ImportedBufferDecl],
    pass_info: &[CompiledPassInfo],
) -> Result<Vec<ImportedResourceFinalAccess>, GraphBuildError> {
    let mut final_accesses = Vec::with_capacity(texture_imports.len() + buffer_imports.len());
    for (idx, import) in texture_imports.iter().enumerate() {
        let handle = ImportedTextureHandle(idx as u32);
        let written_by_retained_pass = imported_texture_written(pass_info, handle);
        if matches!(import.final_access, TextureAccess::Present)
            && matches!(
                import.source,
                ImportSource::Frame(FrameTargetRole::ColorAttachment)
            )
            && !written_by_retained_pass
        {
            return Err(GraphBuildError::MissingImportedFinalWriter {
                label: import.label,
                final_access: "present",
            });
        }
        final_accesses.push(ImportedResourceFinalAccess {
            label: import.label,
            resource: ImportedScheduleResource::Texture(handle),
            final_access: ImportedFinalAccess::Texture(import.final_access.clone()),
            written_by_retained_pass,
        });
    }
    for (idx, import) in buffer_imports.iter().enumerate() {
        let handle = ImportedBufferHandle(idx as u32);
        final_accesses.push(ImportedResourceFinalAccess {
            label: import.label,
            resource: ImportedScheduleResource::Buffer(handle),
            final_access: ImportedFinalAccess::Buffer(import.final_access),
            written_by_retained_pass: imported_buffer_written(pass_info, handle),
        });
    }
    Ok(final_accesses)
}

/// Returns whether any retained pass writes an imported texture.
fn imported_texture_written(pass_info: &[CompiledPassInfo], handle: ImportedTextureHandle) -> bool {
    pass_info.iter().any(|pass| {
        pass.accesses.iter().any(|access| {
            access.writes()
                && matches!(
                    access.resource,
                    ResourceHandle::Texture(TextureResourceHandle::Imported(h)) if h == handle
                )
        })
    })
}

/// Returns whether any retained pass writes an imported buffer.
fn imported_buffer_written(pass_info: &[CompiledPassInfo], handle: ImportedBufferHandle) -> bool {
    pass_info.iter().any(|pass| {
        pass.accesses.iter().any(|access| {
            access.writes()
                && matches!(
                    access.resource,
                    ResourceHandle::Buffer(BufferResourceHandle::Imported(h)) if h == handle
                )
        })
    })
}

/// Finds adjacent raster passes whose attachment templates are merge-compatible.
fn plan_render_pass_merge_groups(
    steps: &[ScheduleStep],
    pass_info: &[CompiledPassInfo],
) -> Vec<RenderPassMergeGroup> {
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < steps.len() {
        let mut end = start + 1;
        while end < steps.len()
            && render_passes_are_merge_compatible(
                pass_info.get(steps[end - 1].pass_idx),
                pass_info.get(steps[end].pass_idx),
            )
        {
            end += 1;
        }
        if end - start > 1 {
            groups.push(RenderPassMergeGroup {
                start_step: start,
                end_step: end,
            });
        }
        start = end;
    }
    groups
}

/// Returns whether two compiled pass infos can share a merge group.
fn render_passes_are_merge_compatible(
    first: Option<&CompiledPassInfo>,
    second: Option<&CompiledPassInfo>,
) -> bool {
    let Some(first) = first else {
        return false;
    };
    let Some(second) = second else {
        return false;
    };
    if first
        .workload_flags
        .contains(PassWorkloadFlags::NEVER_MERGE)
        || second
            .workload_flags
            .contains(PassWorkloadFlags::NEVER_MERGE)
    {
        return false;
    }
    let Some(first_template) = &first.raster_template else {
        return false;
    };
    let Some(second_template) = &second.raster_template else {
        return false;
    };
    if !merge_hints_allow_group(first.merge_hint, second.merge_hint) {
        return false;
    }
    render_templates_are_merge_compatible(first_template, second_template)
}

/// Returns whether adjacent pass merge hints are compatible with one merge group.
fn merge_hints_allow_group(first: PassMergeHint, second: PassMergeHint) -> bool {
    first == second || first.attachment_reuse || second.attachment_reuse
}

/// Returns whether two raster templates target the same attachments.
fn render_templates_are_merge_compatible(
    first: &RenderPassTemplate,
    second: &RenderPassTemplate,
) -> bool {
    if first.multiview_mask != second.multiview_mask
        || first.color_attachments.len() != second.color_attachments.len()
    {
        return false;
    }
    if !first
        .color_attachments
        .iter()
        .zip(&second.color_attachments)
        .all(|(a, b)| a.target == b.target && a.resolve_to == b.resolve_to)
    {
        return false;
    }
    let first_depth = first
        .depth_stencil_attachment
        .as_ref()
        .map(|depth| depth.target);
    let second_depth = second
        .depth_stencil_attachment
        .as_ref()
        .map(|depth| depth.target);
    first_depth == second_depth
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
                    ImportSource::Frame(FrameTargetRole::ColorAttachment)
                )
            })
        })
    })
}
