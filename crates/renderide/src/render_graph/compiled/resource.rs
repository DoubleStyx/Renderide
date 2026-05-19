//! Compiled-graph resource metadata (transient lifetimes, pass info, compile stats).

#[cfg(test)]
use super::super::pass::PassKind;
use super::super::pass::{PassMergeHint, PassWorkloadFlags, RenderPassTemplate};
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
    /// Number of declared transient buffer handles.
    pub transient_buffer_count: usize,
    /// Number of physical transient buffer slots after lifetime aliasing.
    pub transient_buffer_slots: usize,
    /// Number of imported texture declarations.
    pub imported_texture_count: usize,
    /// Number of imported buffer declarations.
    pub imported_buffer_count: usize,
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
    /// Optional multiview mask for raster passes.
    #[cfg(test)]
    pub multiview_mask: Option<std::num::NonZeroU32>,
    /// Render-pass attachment template for graph-managed raster passes.
    pub raster_template: Option<RenderPassTemplate>,
    /// Backend merge hint declared at setup time. See [`PassMergeHint`].
    ///
    /// Scheduler v1 consumes this while detecting conservative render-pass merge groups. The wgpu
    /// executor still opens distinct render passes for the retained pass list.
    pub merge_hint: PassMergeHint,
}
