//! Command encoding for blendshape and skinning compute dispatches.
//!
//! [`record_mesh_deform`] is the single entry point per work item. Per-subsystem encoding
//! lives in [`blendshape`] (sparse scatter) and [`skinning`] (linear blend skinning); both
//! share [`MeshDeformEncodeGpu`] and the running cursor offsets carried in
//! [`MeshDeformRecordInputs`].

mod blendshape;
mod skinning;

use glam::Mat4;

use crate::gpu::GpuLimits;
use crate::mesh_deform::SkinCacheEntry;
use crate::scene::RenderSpaceId;

use super::snapshot::{
    MeshDeformSnapshot, deform_needs_blend_snapshot, deform_needs_skin_snapshot,
};

use blendshape::{BlendshapeCacheCtx, record_blendshape_deform};
use skinning::{SkinningDeformContext, record_skinning_deform};

/// GPU handles and scratch used while recording mesh deform compute on one encoder.
pub(super) struct MeshDeformEncodeGpu<'a> {
    /// Device for bind groups and pipelines.
    pub device: &'a wgpu::Device,
    /// Limits checked before dispatch.
    pub gpu_limits: &'a GpuLimits,
    /// Encoder receiving compute passes.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Preprocess pipelines (blendshape + skinning).
    pub pre: &'a crate::mesh_deform::MeshPreprocessPipelines,
    /// Scratch buffers and slab cursors backing.
    pub scratch: &'a mut crate::mesh_deform::MeshDeformScratch,
    /// Deferred [`wgpu::Queue::write_buffer`] sink shared with the rest of the frame.
    pub upload_batch: &'a crate::render_graph::frame_upload_batch::FrameUploadBatch,
    /// GPU profiler for per-dispatch pass-level timestamp queries; [`None`] when disabled.
    pub profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Scene, mesh snapshot, slab cursors, and GPU skin cache subranges for one deform work item.
pub(super) struct MeshDeformRecordInputs<'a, 'b> {
    /// Scene graph for bone palette resolution.
    pub scene: &'a crate::scene::SceneCoordinator,
    /// Active render space for the mesh.
    pub space_id: RenderSpaceId,
    /// GPU snapshot of mesh buffers and skinning metadata.
    pub mesh: &'a MeshDeformSnapshot,
    /// Per-bone scene transform indices (skinned meshes).
    pub bone_transform_indices: Option<&'a [i32]>,
    /// SMR node id for skinning fallbacks.
    pub smr_node_id: i32,
    /// Host render context (mono vs stereo clip).
    pub render_context: crate::shared::RenderingContext,
    /// Head / HMD output transform for palette construction.
    pub head_output_transform: Mat4,
    /// Blendshape weights (parallel to mesh blendshape count).
    pub blend_weights: &'a [f32],
    /// Running offset into the bone matrix slab.
    pub bone_cursor: &'b mut u64,
    /// Running offset into the blendshape scatter-param staging slab.
    pub blend_param_cursor: &'b mut u64,
    /// Running offset into the skin-dispatch uniform slab (256 B steps per dispatch).
    pub skin_dispatch_cursor: &'b mut u64,
    /// Resolved cache line for this instance's deform outputs.
    pub skin_cache_entry: &'a SkinCacheEntry,
    pub positions_arena: &'a wgpu::Buffer,
    pub normals_arena: &'a wgpu::Buffer,
    pub tangents_arena: &'a wgpu::Buffer,
    pub temp_arena: &'a wgpu::Buffer,
}

/// Compute dispatch counts emitted while recording one deform work item.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MeshDeformRecordStats {
    /// Compute passes opened.
    pub compute_passes: u64,
    /// Bind groups created.
    pub bind_groups_created: u64,
    /// Encoder copy operations recorded.
    pub copy_ops: u64,
    /// Sparse blendshape scatter dispatches.
    pub blend_dispatches: u64,
    /// Linear skinning dispatches.
    pub skin_dispatches: u64,
}

impl MeshDeformRecordStats {
    /// Adds `other` into this stats packet with saturating arithmetic.
    pub fn add(&mut self, other: Self) {
        self.compute_passes = self.compute_passes.saturating_add(other.compute_passes);
        self.bind_groups_created = self
            .bind_groups_created
            .saturating_add(other.bind_groups_created);
        self.copy_ops = self.copy_ops.saturating_add(other.copy_ops);
        self.blend_dispatches = self.blend_dispatches.saturating_add(other.blend_dispatches);
        self.skin_dispatches = self.skin_dispatches.saturating_add(other.skin_dispatches);
    }
}

/// Records blendshape and / or skinning compute for one deform work item.
pub(super) fn record_mesh_deform(
    mut gpu: MeshDeformEncodeGpu<'_>,
    inputs: MeshDeformRecordInputs<'_, '_>,
) -> MeshDeformRecordStats {
    profiling::scope!("mesh_deform::record");
    let Some(deform_guard) = validate_deform_preconditions(
        inputs.mesh,
        inputs.bone_transform_indices,
        inputs.blend_weights,
        gpu.gpu_limits,
    ) else {
        return MeshDeformRecordStats::default();
    };

    let blend_then_skin = deform_guard.needs_blend && deform_guard.needs_skin;
    let mut stats = MeshDeformRecordStats::default();

    if deform_guard.needs_blend {
        stats.add(record_blendshape_deform(
            &mut gpu,
            inputs.mesh,
            inputs.blend_weights,
            inputs.blend_param_cursor,
            BlendshapeCacheCtx {
                cache_entry: inputs.skin_cache_entry,
                positions_arena: inputs.positions_arena,
                normals_arena: inputs.normals_arena,
                tangents_arena: inputs.tangents_arena,
                temp_arena: inputs.temp_arena,
                blend_then_skin,
            },
        ));
    }

    if deform_guard.needs_skin
        && record_skinning_deform(
            &mut gpu,
            SkinningDeformContext {
                scene: inputs.scene,
                space_id: inputs.space_id,
                mesh: inputs.mesh,
                bone_transform_indices: inputs.bone_transform_indices,
                smr_node_id: inputs.smr_node_id,
                render_context: inputs.render_context,
                head_output_transform: inputs.head_output_transform,
                bone_cursor: inputs.bone_cursor,
                needs_blend: deform_guard.needs_blend,
                wg: deform_guard.skin_wg,
                cache_entry: inputs.skin_cache_entry,
                positions_arena: inputs.positions_arena,
                normals_arena: inputs.normals_arena,
                tangents_arena: inputs.tangents_arena,
                temp_arena: inputs.temp_arena,
                skin_dispatch_cursor: inputs.skin_dispatch_cursor,
            },
        )
    {
        stats.compute_passes = stats.compute_passes.saturating_add(1);
        stats.bind_groups_created = stats.bind_groups_created.saturating_add(1);
        stats.skin_dispatches = stats.skin_dispatches.saturating_add(1);
    }
    stats
}

/// Early-out state for [`record_mesh_deform`].
struct DeformValidate {
    needs_blend: bool,
    needs_skin: bool,
    /// Workgroups for skinning (`mesh_skinning.wgsl`), one thread per vertex.
    skin_wg: u32,
}

/// Returns `None` when there is no deform work or dispatch would exceed GPU limits.
fn validate_deform_preconditions(
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
    blend_weights: &[f32],
    gpu_limits: &GpuLimits,
) -> Option<DeformValidate> {
    mesh.positions_buffer.as_ref()?;
    let vc = mesh.vertex_count;
    if vc == 0 {
        return None;
    }
    let needs_blend = deform_needs_blend_snapshot(mesh, blend_weights);
    let needs_skin = deform_needs_skin_snapshot(mesh, bone_transform_indices);

    if !needs_blend && !needs_skin {
        return None;
    }

    let skin_wg = workgroup_count(vc);
    if needs_skin && !gpu_limits.compute_dispatch_fits(skin_wg, 1, 1) {
        logger::warn!(
            "mesh deform: skinning dispatch {}x1x1 exceeds max_compute_workgroups_per_dimension ({})",
            skin_wg,
            gpu_limits.max_compute_workgroups_per_dimension()
        );
        return None;
    }

    Some(DeformValidate {
        needs_blend,
        needs_skin,
        skin_wg,
    })
}

/// Workgroup count for a 64-thread compute (vertex / scatter chunk).
pub(super) fn workgroup_count(count: u32) -> u32 {
    (count.saturating_add(63)) / 64
}
