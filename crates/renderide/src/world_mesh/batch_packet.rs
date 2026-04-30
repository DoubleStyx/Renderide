//! Per-batch resolved draw packet feeding the world-mesh forward raster pass.

use std::sync::Arc;

use crate::materials::{MaterialPassDesc, MaterialPipelineSet, RasterFrontFace};
use crate::world_mesh::draw_prep::PipelineVariantKey;

/// One resolved per-batch draw packet covering a contiguous range of sorted draws with the same
/// [`crate::world_mesh::MaterialDrawBatchKey`].
///
/// Populated by the prepare pass (parallel rayon fan-out) so the recording loop can drive
/// pipeline and bind-group state entirely from this table — no LRU lookups during `RenderPass`.
#[derive(Clone)]
pub struct MaterialBatchPacket {
    /// First draw index (into the sorted draw list) covered by this entry.
    pub first_draw_idx: usize,
    /// Last draw index (inclusive) covered by this entry.
    pub last_draw_idx: usize,
    /// Exact pipeline variant requested for this batch.
    pub(crate) pipeline_key: PipelineVariantKey,
    /// Front-face winding used by the resolved pipeline set.
    pub front_face: RasterFrontFace,
    /// Resolved `@group(1)` bind group for this batch's material, or `None` for the empty fallback.
    pub bind_group: Option<Arc<wgpu::BindGroup>>,
    /// Resolved pipeline set for this batch, or `None` when the pipeline is unavailable (skip draws).
    pub pipelines: Option<MaterialPipelineSet>,
    /// Material pass descriptors parallel to `pipelines` (zero-alloc static reference).
    pub declared_passes: &'static [MaterialPassDesc],
}
