//! Command encoding for blendshape and skinning compute dispatches.
//!
//! [`record_mesh_deform`] plans one work item into a frame batch. Per-subsystem encoding
//! lives in [`blendshape`] (sparse scatter) and [`skinning`] (linear blend skinning); both
//! share [`MeshDeformEncodeGpu`] and the running cursor offsets carried in
//! [`MeshDeformRecordInputs`].

mod blendshape;
mod skinning;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use glam::Mat4;

use crate::frame_upload_batch::GraphUploadSink;
use crate::gpu::GpuLimits;
use crate::mesh_deform::{DeformSignature, EntryNeed, SkinCacheEntry};
use crate::scene::RenderSpaceId;
use crate::shared::SkinWeightMode;

use super::snapshot::{
    MeshDeformSnapshot, deform_needs_blend_snapshot, deform_needs_skin_snapshot,
};

use blendshape::{
    BlendshapeCacheCtx, BlendshapeDispatchJob, flush_blendshape_jobs, record_blendshape_deform,
};
use skinning::{
    SkinningDeformContext, SkinningDispatchJob, SkinningPaletteBuildContext, flush_skinning_jobs,
    prepare_skinning_palette_bytes, record_skinning_deform,
};

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
    /// Deferred graph upload sink shared with the rest of the frame.
    pub uploads: GraphUploadSink<'a>,
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
    /// Mesh-pool mutation generation observed when the snapshot was collected.
    pub mesh_pool_generation: u64,
    /// Per-bone scene transform indices (skinned meshes).
    pub bone_transform_indices: Option<&'a [i32]>,
    /// SMR node id for skinning fallbacks.
    pub smr_node_id: i32,
    /// Host render context (mono vs stereo clip).
    pub render_context: crate::shared::RenderingContext,
    /// Head / HMD output transform for palette construction.
    pub head_output_transform: Mat4,
    /// Optional CPU palette bytes prepared before serial command encoding.
    pub prepared_skinning_palette_bytes: Option<&'a [u8]>,
    /// Blendshape weights (parallel to mesh blendshape count).
    pub blend_weights: &'a [f32],
    /// Last cache-line signature, if any.
    pub previous_signature: Option<DeformSignature>,
    /// Host-owned skin influence mode.
    pub skin_weight_mode: SkinWeightMode,
    /// Running offset into the bone matrix slab.
    pub bone_cursor: &'b mut u64,
    /// Running offset into the blendshape scatter-param uniform slab.
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

/// Batched compute dispatch jobs for one mesh-deform frame.
#[derive(Default)]
pub(super) struct MeshDeformDispatchBatch {
    blendshape_jobs: Vec<BlendshapeDispatchJob>,
    skinning_jobs: Vec<SkinningDispatchJob>,
}

impl MeshDeformDispatchBatch {
    /// Creates an empty reusable batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears queued dispatch jobs while retaining capacity.
    pub fn clear(&mut self) {
        self.blendshape_jobs.clear();
        self.skinning_jobs.clear();
    }
}

/// Compute dispatch counts emitted while recording deform work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MeshDeformRecordStats {
    /// Compute passes opened.
    pub compute_passes: u64,
    /// Bind groups created.
    pub bind_groups_created: u64,
    /// Bind groups served from the mesh-deform cache.
    pub bind_group_cache_reuses: u64,
    /// Encoder copy operations recorded.
    pub copy_ops: u64,
    /// Sparse blendshape scatter dispatches.
    pub blend_dispatches: u64,
    /// Linear skinning dispatches.
    pub skin_dispatches: u64,
    /// Work items skipped because their deform inputs matched the cache line signature.
    pub stable_skips: u64,
}

impl MeshDeformRecordStats {
    /// Adds `other` into this stats packet with saturating arithmetic.
    pub fn add(&mut self, other: Self) {
        self.compute_passes = self.compute_passes.saturating_add(other.compute_passes);
        self.bind_groups_created = self
            .bind_groups_created
            .saturating_add(other.bind_groups_created);
        self.bind_group_cache_reuses = self
            .bind_group_cache_reuses
            .saturating_add(other.bind_group_cache_reuses);
        self.copy_ops = self.copy_ops.saturating_add(other.copy_ops);
        self.blend_dispatches = self.blend_dispatches.saturating_add(other.blend_dispatches);
        self.skin_dispatches = self.skin_dispatches.saturating_add(other.skin_dispatches);
        self.stable_skips = self.stable_skips.saturating_add(other.stable_skips);
    }
}

/// Result of planning one deform work item.
pub(super) struct MeshDeformRecordResult {
    /// Stats emitted while planning this item.
    pub stats: MeshDeformRecordStats,
    /// Signature to store on the cache entry after successful planning.
    pub signature_to_store: Option<DeformSignature>,
}

/// Records blendshape and / or skinning compute for one deform work item.
pub(super) fn record_mesh_deform(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    inputs: MeshDeformRecordInputs<'_, '_>,
    batch: &mut MeshDeformDispatchBatch,
) -> MeshDeformRecordResult {
    profiling::scope!("mesh_deform::record");
    let Some(deform_guard) = validate_deform_preconditions(
        inputs.mesh,
        inputs.bone_transform_indices,
        inputs.blend_weights,
        gpu.gpu_limits,
    ) else {
        return MeshDeformRecordResult {
            stats: MeshDeformRecordStats::default(),
            signature_to_store: None,
        };
    };

    let blend_then_skin = deform_guard.needs_blend && deform_guard.needs_skin;
    let mut stats = MeshDeformRecordStats::default();
    let prepared_palette_len = if deform_guard.needs_skin {
        let Some(palette_len) = prepared_skinning_palette_len(gpu, &inputs) else {
            return MeshDeformRecordResult {
                stats,
                signature_to_store: None,
            };
        };
        Some(palette_len)
    } else {
        None
    };
    let palette_bytes_for_signature = inputs
        .prepared_skinning_palette_bytes
        .filter(|bytes| !bytes.is_empty())
        .unwrap_or(gpu.scratch.bone_palette_bytes.as_slice());
    let signature = build_deform_signature(DeformSignatureInputs {
        mesh: inputs.mesh,
        mesh_pool_generation: inputs.mesh_pool_generation,
        blend_weights: inputs.blend_weights,
        bone_transform_indices: inputs.bone_transform_indices,
        render_context: inputs.render_context,
        entry_need: deform_guard.entry_need,
        skin_weight_mode: inputs.skin_weight_mode,
        prepared_palette_bytes: prepared_palette_len.map(|_| palette_bytes_for_signature),
    });
    if inputs.previous_signature == Some(signature) {
        stats.stable_skips = stats.stable_skips.saturating_add(1);
        return MeshDeformRecordResult {
            stats,
            signature_to_store: None,
        };
    }

    if deform_guard.needs_blend {
        stats.add(record_blendshape_deform(
            gpu,
            inputs.mesh,
            inputs.blend_weights,
            inputs.blend_param_cursor,
            &mut batch.blendshape_jobs,
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

    if deform_guard.needs_skin {
        let Some(prepared_palette_len) = prepared_palette_len else {
            return MeshDeformRecordResult {
                stats,
                signature_to_store: None,
            };
        };
        copy_preplanned_skinning_palette(gpu, inputs.prepared_skinning_palette_bytes);
        stats.add(record_skinning_deform(
            gpu,
            SkinningDeformContext {
                mesh: inputs.mesh,
                bone_cursor: inputs.bone_cursor,
                needs_blend: deform_guard.needs_blend,
                wg: deform_guard.skin_wg,
                cache_entry: inputs.skin_cache_entry,
                positions_arena: inputs.positions_arena,
                normals_arena: inputs.normals_arena,
                tangents_arena: inputs.tangents_arena,
                temp_arena: inputs.temp_arena,
                skin_dispatch_cursor: inputs.skin_dispatch_cursor,
                prepared_palette_len,
                skin_weight_mode: inputs.skin_weight_mode,
            },
            &mut batch.skinning_jobs,
        ));
    }
    let planned_any = stats.copy_ops > 0 || stats.blend_dispatches > 0 || stats.skin_dispatches > 0;
    MeshDeformRecordResult {
        stats,
        signature_to_store: planned_any.then_some(signature),
    }
}

/// Returns the byte length of the prepared skinning palette for one record item.
fn prepared_skinning_palette_len(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    inputs: &MeshDeformRecordInputs<'_, '_>,
) -> Option<u64> {
    if let Some(bytes) = inputs.prepared_skinning_palette_bytes
        && !bytes.is_empty()
    {
        return Some(bytes.len() as u64);
    }
    prepare_skinning_palette_bytes(
        gpu,
        SkinningPaletteBuildContext {
            scene: inputs.scene,
            space_id: inputs.space_id,
            mesh: inputs.mesh,
            bone_transform_indices: inputs.bone_transform_indices,
            smr_node_id: inputs.smr_node_id,
            render_context: inputs.render_context,
            head_output_transform: inputs.head_output_transform,
        },
    )
}

/// Copies a preplanned palette into scratch before the serial upload path consumes it.
fn copy_preplanned_skinning_palette(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    prepared_skinning_palette_bytes: Option<&[u8]>,
) {
    if let Some(bytes) = prepared_skinning_palette_bytes
        && !bytes.is_empty()
    {
        gpu.scratch.bone_palette_bytes.clear();
        gpu.scratch.bone_palette_bytes.extend_from_slice(bytes);
    }
}

/// Flushes all planned mesh-deform jobs in coarse compute passes.
pub(super) fn flush_mesh_deform_batch(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    batch: &mut MeshDeformDispatchBatch,
) -> MeshDeformRecordStats {
    profiling::scope!("mesh_deform::flush_batch");
    let mut stats = MeshDeformRecordStats::default();
    stats.add(flush_blendshape_jobs(gpu, &batch.blendshape_jobs));
    stats.add(flush_skinning_jobs(gpu, &batch.skinning_jobs));
    batch.clear();
    stats
}

/// Early-out state for [`record_mesh_deform`].
struct DeformValidate {
    needs_blend: bool,
    needs_skin: bool,
    entry_need: EntryNeed,
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
    let tangent_stream_ready = mesh.tangent_buffer.is_some();
    let entry_need = EntryNeed {
        needs_blend,
        needs_skin,
        needs_blend_normals: needs_blend
            && mesh.blendshape_has_normal_deltas
            && mesh.normals_buffer.is_some(),
        needs_tangents: tangent_stream_ready
            && (needs_skin || (needs_blend && mesh.blendshape_has_tangent_deltas)),
        needs_blend_tangents: needs_blend
            && tangent_stream_ready
            && mesh.blendshape_has_tangent_deltas,
    };

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
        entry_need,
        skin_wg,
    })
}

struct DeformSignatureInputs<'a> {
    mesh: &'a MeshDeformSnapshot,
    mesh_pool_generation: u64,
    blend_weights: &'a [f32],
    bone_transform_indices: Option<&'a [i32]>,
    render_context: crate::shared::RenderingContext,
    entry_need: EntryNeed,
    skin_weight_mode: SkinWeightMode,
    prepared_palette_bytes: Option<&'a [u8]>,
}

fn build_deform_signature(inputs: DeformSignatureInputs<'_>) -> DeformSignature {
    let mut hasher = DefaultHasher::new();
    inputs.mesh.asset_id.hash(&mut hasher);
    inputs.mesh_pool_generation.hash(&mut hasher);
    inputs.mesh.vertex_count.hash(&mut hasher);
    inputs.mesh.num_blendshapes.hash(&mut hasher);
    inputs.entry_need.needs_blend.hash(&mut hasher);
    inputs.entry_need.needs_skin.hash(&mut hasher);
    inputs.entry_need.needs_blend_normals.hash(&mut hasher);
    inputs.entry_need.needs_tangents.hash(&mut hasher);
    inputs.entry_need.needs_blend_tangents.hash(&mut hasher);
    (inputs.render_context as u8).hash(&mut hasher);
    if inputs.entry_need.needs_skin {
        (inputs.skin_weight_mode as i32).hash(&mut hasher);
    }
    let weight_count = inputs.mesh.num_blendshapes as usize;
    weight_count.hash(&mut hasher);
    for weight in inputs.blend_weights.iter().take(weight_count) {
        weight.to_bits().hash(&mut hasher);
    }
    if inputs.blend_weights.len() < weight_count {
        0usize.hash(&mut hasher);
    }
    if let Some(indices) = inputs.bone_transform_indices {
        indices.hash(&mut hasher);
    }
    if let Some(bytes) = inputs.prepared_palette_bytes {
        bytes.hash(&mut hasher);
    }
    DeformSignature {
        hash: hasher.finish(),
    }
}

/// Workgroup count for a 64-thread compute (vertex / scatter chunk).
pub(super) fn workgroup_count(count: u32) -> u32 {
    (count.saturating_add(63)) / 64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_entry_need() -> EntryNeed {
        EntryNeed {
            needs_blend: true,
            needs_skin: true,
            needs_blend_normals: true,
            needs_tangents: false,
            needs_blend_tangents: false,
        }
    }

    fn test_snapshot() -> MeshDeformSnapshot {
        MeshDeformSnapshot {
            asset_id: 7,
            vertex_count: 3,
            num_blendshapes: 2,
            has_skeleton: true,
            positions_buffer: None,
            normals_buffer: None,
            tangent_buffer: None,
            blendshape_sparse_buffer: None,
            blendshape_frame_ranges: Vec::new(),
            blendshape_shape_frame_spans: Vec::new(),
            bone_indices_buffer: None,
            bone_weights_vec4_buffer: None,
            bone_influence_offsets_buffer: None,
            bone_influences_buffer: None,
            skinning_bind_matrices: Vec::new(),
            blendshape_has_position_deltas: true,
            blendshape_has_normal_deltas: true,
            blendshape_has_tangent_deltas: false,
        }
    }

    fn test_signature(
        mesh: &MeshDeformSnapshot,
        blend_weights: &[f32],
        skin_weight_mode: SkinWeightMode,
        prepared_palette_bytes: Option<&[u8]>,
    ) -> DeformSignature {
        build_deform_signature(DeformSignatureInputs {
            mesh,
            mesh_pool_generation: 11,
            blend_weights,
            bone_transform_indices: Some(&[1, 2]),
            render_context: crate::shared::RenderingContext::UserView,
            entry_need: test_entry_need(),
            skin_weight_mode,
            prepared_palette_bytes,
        })
    }

    #[test]
    fn deform_signature_changes_with_blend_weight_bits() {
        let mesh = test_snapshot();
        let a = test_signature(
            &mesh,
            &[0.25, 0.0],
            SkinWeightMode::Unlimited,
            Some(&[1, 2, 3, 4]),
        );
        let b = test_signature(
            &mesh,
            &[0.5, 0.0],
            SkinWeightMode::Unlimited,
            Some(&[1, 2, 3, 4]),
        );

        assert_ne!(a, b);
    }

    #[test]
    fn deform_signature_changes_with_resolved_palette_bytes() {
        let mesh = test_snapshot();
        let a = test_signature(
            &mesh,
            &[0.25, 0.0],
            SkinWeightMode::Unlimited,
            Some(&[1, 2, 3, 4]),
        );
        let b = test_signature(
            &mesh,
            &[0.25, 0.0],
            SkinWeightMode::Unlimited,
            Some(&[1, 2, 3, 5]),
        );

        assert_ne!(a, b);
    }

    #[test]
    fn deform_signature_changes_with_skin_weight_mode() {
        let mesh = test_snapshot();
        let a = test_signature(
            &mesh,
            &[0.25, 0.0],
            SkinWeightMode::FourBones,
            Some(&[1, 2, 3, 4]),
        );
        let b = test_signature(
            &mesh,
            &[0.25, 0.0],
            SkinWeightMode::Unlimited,
            Some(&[1, 2, 3, 4]),
        );

        assert_ne!(a, b);
    }
}
