//! Render graph builder: setup collection, dependency synthesis, culling, and alias planning.

mod decl;
mod edges;
mod imports;
mod lifetime;
mod pass_info;
mod raster_merge;
mod schedule_plan;
mod store_ops;
mod topo;
mod validate;

#[cfg(test)]
mod tests;

use decl::{GroupEntry, PassEntry, SetupEntry};
use edges::{add_blackboard_edges, add_group_edges, add_resource_edges, explicit_edges};
use imports::{compile_imported_final_accesses, needs_surface_acquire};
use lifetime::{compile_buffers, compile_textures};
use pass_info::compile_pass_info;
use schedule_plan::{FrameScheduleBuildInput, build_frame_schedule};
use store_ops::{TransientAttachmentStoreStats, optimize_transient_attachment_stores};
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
    BlackboardSeedDecl, ComputePass, EncoderPass, GroupScope, PassBuilder, PassNode, PassPhase,
    PassWorkloadFlags, RasterPass,
};
use super::resources::{
    BufferHandle, ImportedBufferDecl, ImportedBufferHandle, ImportedTextureDecl,
    ImportedTextureHandle, SubresourceHandle, TextureHandle, TransientArrayLayers,
    TransientBufferDesc, TransientSubresourceDesc, TransientTextureDesc,
};
use super::schedule::{FrameSchedule, ScheduleHudSnapshot};
use super::validation::GraphValidationReport;
use crate::render_graph::RenderGraphValidationMode;

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
    compile_skipped_pass_count: usize,
    validation_mode: RenderGraphValidationMode,
    default_frame_group: GroupId,
    default_per_view_group: GroupId,
}

/// Inputs used to assemble compile diagnostics after scheduling.
struct CompileStatsInput<'a> {
    /// Number of passes registered before culling.
    registered_pass_count: usize,
    /// Number of topological waves before retention.
    topo_levels: usize,
    /// Number of passes culled as dead graph work.
    culled_count: usize,
    /// Number of passes intentionally skipped before setup.
    compile_skipped_pass_count: usize,
    /// Number of declared transient texture handles.
    transient_texture_count: usize,
    /// Number of physical transient texture slots.
    transient_texture_slots: usize,
    /// Number of transient texture lifetime lanes.
    transient_texture_lanes: usize,
    /// Number of declared transient buffer handles.
    transient_buffer_count: usize,
    /// Number of physical transient buffer slots.
    transient_buffer_slots: usize,
    /// Number of transient buffer lifetime lanes.
    transient_buffer_lanes: usize,
    /// Number of imported texture declarations.
    imported_texture_count: usize,
    /// Number of imported buffer declarations.
    imported_buffer_count: usize,
    /// Number of build-time validation diagnostics.
    validation_diagnostics: usize,
    /// Attachment store and resolve diagnostics.
    store_stats: TransientAttachmentStoreStats,
    /// Final retained schedule.
    schedule: &'a FrameSchedule,
    /// Retained pass metadata.
    pass_info: &'a [CompiledPassInfo],
}

/// Builds public compile diagnostics from retained graph metadata.
fn build_compile_stats(input: CompileStatsInput<'_>) -> CompileStats {
    CompileStats {
        registered_pass_count: input.registered_pass_count,
        pass_count: input.pass_info.len(),
        topo_levels: input.topo_levels,
        culled_count: input.culled_count,
        compile_skipped_pass_count: input.compile_skipped_pass_count,
        transient_texture_count: input.transient_texture_count,
        transient_texture_slots: input.transient_texture_slots,
        transient_texture_lanes: input.transient_texture_lanes,
        transient_buffer_count: input.transient_buffer_count,
        transient_buffer_slots: input.transient_buffer_slots,
        transient_buffer_lanes: input.transient_buffer_lanes,
        imported_texture_count: input.imported_texture_count,
        imported_buffer_count: input.imported_buffer_count,
        validation_diagnostics: input.validation_diagnostics,
        dependency_edge_count: input.schedule.dependency_edges.len(),
        render_pass_merge_groups: input.schedule.render_pass_merge_groups.len(),
        render_pass_materialization_groups: input
            .schedule
            .render_pass_materialization_plan
            .groups
            .len(),
        async_compute_capable_pass_count: input
            .pass_info
            .iter()
            .filter(|info| {
                info.workload_flags
                    .contains(PassWorkloadFlags::ASYNC_COMPUTE_CAPABLE)
            })
            .count(),
        parallel_recording_unit_count: input.schedule.recording_plan.parallel_unit_count(),
        parallel_recording_batch_count: input.schedule.recording_plan.parallel_batch_count(),
        attachment_resolve_count: input.store_stats.attachment_resolve_count,
        transient_attachment_store_count: input.store_stats.store_count,
        transient_attachment_discard_count: input.store_stats.discard_count,
        estimated_bandwidth_bytes: input.store_stats.estimated_bandwidth_bytes,
    }
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
            compile_skipped_pass_count: 0,
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

    /// Records a pass intentionally omitted before graph build.
    pub fn note_skipped_pass(&mut self) {
        self.compile_skipped_pass_count = self.compile_skipped_pass_count.saturating_add(1);
    }

    /// Compiles setup declarations into an immutable graph.
    pub fn build(mut self) -> Result<CompiledRenderGraph, GraphBuildError> {
        self.validate_subresource_decls()?;
        let n = self.passes.len();
        if n == 0 {
            return Ok(self.empty_graph());
        }

        let mut setups = self.collect_setup()?;
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
        let store_stats =
            optimize_transient_attachment_stores(&mut setups, &self.subresources, &retained_ord);
        let (compiled_textures, texture_slots, texture_lifetime_lanes) =
            compile_textures(&self.textures, &self.subresources, &setups, &retained_ord);
        let (compiled_buffers, buffer_slots, buffer_lifetime_lanes) =
            compile_buffers(&self.buffers, &setups, &retained_ord);
        let pass_info = compile_pass_info(&setups, &ordered);
        let imported_final_accesses =
            compile_imported_final_accesses(&self.imports_tex, &self.imports_buf, &pass_info)?;
        let needs_surface_acquire = needs_surface_acquire(&pass_info, &self.imports_tex);

        let ordered_passes = take_ordered_passes(self.passes, &ordered)?;

        // Build FrameSchedule: single source of truth for pass ordering and scheduler policy.
        let schedule = build_frame_schedule(FrameScheduleBuildInput {
            ordered_passes: &ordered_passes,
            ordered: &ordered,
            wave_by_node: &wave_by_node,
            compiled_textures: &compiled_textures,
            compiled_buffers: &compiled_buffers,
            imported_final_accesses,
            pass_info: &pass_info,
            edges: &edges,
        })?;
        let schedule_hud = ScheduleHudSnapshot::from_schedule(&schedule);
        let compile_stats = build_compile_stats(CompileStatsInput {
            registered_pass_count: n,
            topo_levels,
            culled_count,
            compile_skipped_pass_count: self.compile_skipped_pass_count,
            transient_texture_count: self.textures.len(),
            transient_texture_slots: texture_slots,
            transient_texture_lanes: texture_lifetime_lanes.len(),
            transient_buffer_count: self.buffers.len(),
            transient_buffer_slots: buffer_slots,
            transient_buffer_lanes: buffer_lifetime_lanes.len(),
            imported_texture_count: self.imports_tex.len(),
            imported_buffer_count: self.imports_buf.len(),
            validation_diagnostics: validation_report.len(),
            store_stats,
            schedule: &schedule,
            pass_info: &pass_info,
        });

        Ok(CompiledRenderGraph {
            passes: ordered_passes,
            needs_surface_acquire,
            compile_stats,
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
                registered_pass_count: 0,
                transient_texture_count: self.textures.len(),
                transient_buffer_count: self.buffers.len(),
                imported_texture_count: self.imports_tex.len(),
                imported_buffer_count: self.imports_buf.len(),
                compile_skipped_pass_count: self.compile_skipped_pass_count,
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
            let profiling_label = entry.pass.profiling_label().into_owned();
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
                profiling_label,
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

fn take_ordered_passes(
    passes: Vec<PassEntry>,
    ordered: &[usize],
) -> Result<Vec<PassNode>, GraphBuildError> {
    let mut pass_take: Vec<Option<PassNode>> =
        passes.into_iter().map(|entry| Some(entry.pass)).collect();
    let mut ordered_passes: Vec<PassNode> = Vec::with_capacity(ordered.len());
    for idx in ordered {
        let Some(pass) = pass_take[*idx].take() else {
            return Err(GraphBuildError::PassOwnershipInvariant {
                message: "pass index taken more than once during build",
            });
        };
        ordered_passes.push(pass);
    }
    Ok(ordered_passes)
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}
