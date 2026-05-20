//! Compiled-graph resource metadata (transient lifetimes, pass info, compile stats).

#[cfg(test)]
use super::super::pass::PassKind;
use super::super::pass::{
    BlackboardAccessDecl, PassMergeHint, PassParameterSchema, PassWorkloadFlags, RenderPassTemplate,
};
use super::super::resources::{ResourceAccess, TransientBufferDesc, TransientTextureDesc};

/// Statistics emitted when building a [`super::CompiledRenderGraph`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileStats {
    /// Number of passes in the flattened schedule.
    pub pass_count: usize,
    /// Number of Kahn sweep **waves** (parallel layers) in the build-time DAG sort.
    ///
    /// Runtime execution consumes the retained wave ranges in
    /// [`super::super::schedule::FrameSchedule`] while preserving deterministic pass order inside
    /// each wave. The value is exposed in the debug HUD with pass count.
    pub topo_levels: usize,
    /// Number of passes culled because their writes could not reach an import/export.
    pub culled_count: usize,
    /// Number of declared transient texture handles.
    pub transient_texture_count: usize,
    /// Number of physical transient texture slots after lifetime aliasing.
    pub transient_texture_slots: usize,
    /// Number of lifetime lanes used by transient textures.
    pub transient_texture_lanes: usize,
    /// Number of declared transient buffer handles.
    pub transient_buffer_count: usize,
    /// Number of physical transient buffer slots after lifetime aliasing.
    pub transient_buffer_slots: usize,
    /// Number of lifetime lanes used by transient buffers.
    pub transient_buffer_lanes: usize,
    /// Number of imported texture declarations.
    pub imported_texture_count: usize,
    /// Number of imported buffer declarations.
    pub imported_buffer_count: usize,
    /// Number of graph validation diagnostics emitted at build time.
    pub validation_diagnostics: usize,
    /// Number of conservative render-pass merge groups detected by the scheduler.
    pub render_pass_merge_groups: usize,
    /// Number of render-pass groups planned for materialized recording.
    pub render_pass_materialization_groups: usize,
}

/// Inclusive pass-index lifetime for one transient resource in the retained schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLifetime {
    /// First retained pass index that touches the resource.
    pub first_pass: usize,
    /// Last retained pass index that touches the resource.
    pub last_pass: usize,
}

impl ResourceLifetime {
    /// Returns true when two lifetimes do not overlap.
    pub fn disjoint(self, other: Self) -> bool {
        self.last_pass < other.first_pass || other.last_pass < self.first_pass
    }
}

/// One resource lifetime segment assigned to a physical transient lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceLifetimeSegment {
    /// Graph resource label.
    pub label: &'static str,
    /// Logical resource index in the graph declaration list.
    pub resource_index: usize,
    /// Inclusive retained-pass lifetime.
    pub lifetime: ResourceLifetime,
}

/// Physical transient-resource lane with every logical resource that aliases into it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceLifetimeLane {
    /// Physical slot index used by the transient pool.
    pub physical_slot: usize,
    /// Logical resources assigned to this lane, sorted by lifetime.
    pub segments: Vec<ResourceLifetimeSegment>,
}

/// Compiled metadata for a transient texture handle.
#[derive(Clone, Debug)]
pub struct CompiledTextureResource {
    /// Original descriptor.
    pub desc: TransientTextureDesc,
    /// Usage union across retained pass declarations.
    pub usage: wgpu::TextureUsages,
    /// Retained-schedule lifetime.
    pub lifetime: Option<ResourceLifetime>,
    /// Physical alias slot assigned by the compiler.
    pub physical_slot: usize,
}

/// Compiled metadata for a transient buffer handle.
#[derive(Clone, Debug)]
pub struct CompiledBufferResource {
    /// Original descriptor.
    pub desc: TransientBufferDesc,
    /// Usage union across retained pass declarations.
    pub usage: wgpu::BufferUsages,
    /// Retained-schedule lifetime.
    pub lifetime: Option<ResourceLifetime>,
    /// Physical alias slot assigned by the compiler.
    pub physical_slot: usize,
}

/// Compiled setup metadata for one retained pass.
#[derive(Clone, Debug)]
pub struct CompiledPassInfo {
    /// Pass name.
    pub name: String,
    /// Command kind.
    #[cfg(test)]
    pub kind: PassKind,
    /// Scheduler-visible workload and execution policy flags.
    pub workload_flags: PassWorkloadFlags,
    /// Declared accesses.
    pub(crate) accesses: Vec<ResourceAccess>,
    /// Declared blackboard access metadata.
    pub(crate) blackboard_accesses: Vec<BlackboardAccessDecl>,
    /// Pass-parameter schema for graph diagnostics.
    pub parameter_schema: Option<PassParameterSchema>,
    /// Optional multiview mask for raster passes.
    #[cfg(test)]
    pub multiview_mask: Option<std::num::NonZeroU32>,
    /// Render-pass attachment template for graph-managed raster passes.
    pub raster_template: Option<RenderPassTemplate>,
    /// Backend merge hint declared at setup time. See [`PassMergeHint`].
    ///
    /// Scheduler v1 consumes this while detecting conservative render-pass merge groups. The wgpu
    /// executor materializes compatible groups when runtime state stays merge-safe.
    pub merge_hint: PassMergeHint,
}
