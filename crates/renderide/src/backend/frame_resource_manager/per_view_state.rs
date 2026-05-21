//! Per-view frame state, per-draw scratch, and packed-light data types.

use std::sync::Arc;

use crate::mesh_deform::PaddedPerDrawUniforms;
use crate::passes::MaterialBatchBoundary;

use super::super::frame_gpu::PerViewSceneSnapshots;
use super::super::light_gpu::GpuLight;

/// CPU-side packed lights for one render view.
#[derive(Default)]
pub(super) struct PreparedViewLights {
    /// Packed GPU light rows for this view.
    pub(super) lights: Vec<GpuLight>,
    /// Whether this view has at least one negative light contribution.
    pub(super) signed_scene_color_required: bool,
}

/// Per-view `@group(0)` frame uniform buffer, light buffer, and bind group.
///
/// The large cluster storage buffers (`cluster_light_counts` range rows,
/// `cluster_light_indices`) are shared across all views via [`super::FrameGpuResources::cluster_cache`]
/// and are safe to share because GPU in-order execution within a single submit ensures each view's
/// compute->raster pair retires before the next view's compute overwrites.
///
/// [`Self::cluster_params_buffer`] is intentionally **per-view**: it is written by
/// `ClusteredLightPass::record` via the graph upload sink, which accumulates writes from rayon
/// workers. Since insertion order into the sink is non-deterministic, a shared params buffer
/// would mean the last view to push wins -- corrupting every other view's cluster culling and
/// causing strobe flicker. Keeping params per-view eliminates the race at the cost of ~512 B
/// per view (completely negligible).
pub struct PerViewFrameState {
    /// Per-view `@group(0)` frame uniform buffer written by world-mesh frame planning each frame.
    pub frame_uniform_buffer: wgpu::Buffer,
    /// Per-view light storage buffer written during pre-record synchronization.
    pub lights_buffer: wgpu::Buffer,
    /// Per-view `@group(0)` bind group referencing [`Self::frame_uniform_buffer`],
    /// [`Self::lights_buffer`], shared cluster buffers, and view-local scene snapshots.
    pub frame_bind_group: Arc<wgpu::BindGroup>,
    /// Per-view `@group(0)` bind group whose scene-color bindings point at named grab snapshots.
    pub named_scene_color_frame_bind_group: Arc<wgpu::BindGroup>,
    /// Per-view uniform buffer for `ClusterParams` (camera matrix, projection, viewport, etc.).
    ///
    /// Sized `CLUSTER_PARAMS_UNIFORM_SIZE x eye_multiplier`. Must be per-view -- see struct doc.
    pub cluster_params_buffer: wgpu::Buffer,
    /// View-local depth/color snapshots sampled by embedded material helper passes.
    pub(super) scene_snapshots: PerViewSceneSnapshots,
    /// Shared cluster cache version at which [`Self::frame_bind_group`] was last built.
    pub(super) last_cluster_version: u64,
    /// Reflection-probe resource version at which [`Self::frame_bind_group`] was last built.
    pub(super) last_skybox_specular_version: u64,
    /// Stereo flag at which [`Self::cluster_params_buffer`] was last allocated.
    pub(super) last_stereo: bool,
}

/// Per-view CPU scratch used to pack `@group(2)` per-draw uniforms before upload.
#[derive(Default)]
pub struct PerViewPerDrawScratch {
    /// Packed per-draw uniforms before serializing into the byte slab.
    pub uniforms: Vec<PaddedPerDrawUniforms>,
    /// Serialized byte slab uploaded into per-draw storage.
    pub slab_bytes: Vec<u8>,
    /// Contiguous `(first_draw_idx, last_draw_idx)` runs of identical material batch keys for
    /// this view's sorted draw list. Cleared and refilled by world-mesh forward frame planning;
    /// held here so the boundary Vec does not reallocate as it grows across frames.
    pub material_batch_boundaries: Vec<MaterialBatchBoundary>,
}
