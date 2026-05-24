//! Sparse blendshape scatter compute encoding.
//!
//! Records one bind-pose copy followed by one or more scatter dispatches per weighted shape
//! frame, using dynamic offsets into [`crate::mesh_deform::MeshDeformScratch`]'s param slab.

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::assets::mesh::{
    BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS, BLENDSHAPE_POSITION_SPARSE_ENTRY_WORDS,
    BlendshapeFrameRange, select_blendshape_frame_coefficients,
};
use crate::mesh_deform::{
    BlendshapeBindGroupKey, SkinCacheEntry, buffer_identity, plan_blendshape_scatter_chunks,
};

use super::super::snapshot::MeshDeformSnapshot;
use super::{MeshDeformEncodeGpu, MeshDeformRecordStats, workgroup_count};

/// Arena subranges for blendshape scatter / copy destination.
pub(super) struct BlendshapeCacheCtx<'a> {
    /// Instance line from [`crate::mesh_deform::GpuSkinCache`].
    pub cache_entry: &'a SkinCacheEntry,
    pub positions_arena: &'a wgpu::Buffer,
    pub normals_arena: &'a wgpu::Buffer,
    pub tangents_arena: &'a wgpu::Buffer,
    pub temp_arena: &'a wgpu::Buffer,
    /// When true, blend output is written to the temp arena for the skinning pass.
    pub blend_then_skin: bool,
}

const BLENDSHAPE_CHANNEL_POSITION: u32 = 0;
const BLENDSHAPE_CHANNEL_NORMAL: u32 = 1;
const BLENDSHAPE_CHANNEL_TANGENT: u32 = 2;

/// One blendshape scatter dispatch inside the frame batch.
pub(super) struct BlendshapeDispatchJob {
    bind_group: Arc<wgpu::BindGroup>,
    params_offset: u32,
    wg: u32,
}

/// Resolved destination buffers and base element offsets for one blendshape dispatch batch.
struct BlendshapeDestinations<'a> {
    positions_buffer: &'a wgpu::Buffer,
    normals_buffer: Option<&'a wgpu::Buffer>,
    tangents_buffer: Option<&'a wgpu::Buffer>,
    base_pos_e: u32,
    base_nrm_e: u32,
    base_tan_e: u32,
    copy_normals: bool,
    copy_tangents: bool,
    apply_normals: bool,
    apply_tangents: bool,
}

/// Reserved uniform-slab range for packed blendshape scatter params.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlendshapeParamReservation {
    /// Byte offset inside [`crate::mesh_deform::MeshDeformScratch::blendshape_params`].
    offset: u64,
    /// Number of bytes occupied by this dispatch's packed scatter params.
    byte_len: u64,
    /// Cursor value to use for the next dispatch's reservation.
    next_cursor: u64,
}

/// Reserves a non-overlapping dynamic-uniform slot in the frame-global params slab.
fn reserve_blendshape_param_range(
    cursor: u64,
    byte_len: u64,
) -> Option<BlendshapeParamReservation> {
    if byte_len == 0 {
        return None;
    }
    let aligned_len = byte_len.checked_add(255)? & !255;
    cursor.checked_add(byte_len)?;
    let next_cursor = cursor.checked_add(aligned_len)?;
    Some(BlendshapeParamReservation {
        offset: cursor,
        byte_len,
        next_cursor,
    })
}

/// Sparse blendshape scatter: copy bind poses -> cache range, then one scatter dispatch per weighted shape chunk.
pub(super) fn record_blendshape_deform(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    mesh: &MeshDeformSnapshot,
    blend_weights: &[f32],
    blend_param_cursor: &mut u64,
    jobs: &mut Vec<BlendshapeDispatchJob>,
    ctx: BlendshapeCacheCtx<'_>,
) -> MeshDeformRecordStats {
    profiling::scope!("mesh_deform::record_blendshape");
    let mut stats = MeshDeformRecordStats::default();
    let BlendshapeCacheCtx {
        cache_entry,
        positions_arena,
        normals_arena,
        tangents_arena,
        temp_arena,
        blend_then_skin,
    } = ctx;
    let Some(ref positions) = mesh.positions_buffer else {
        return stats;
    };
    let Some(ref sparse) = mesh.blendshape_sparse_buffer else {
        return stats;
    };
    let shape_count = mesh.num_blendshapes;
    if shape_count == 0 {
        return stats;
    }
    if mesh.blendshape_shape_frame_spans.len() != shape_count as usize {
        logger::warn!(
            "mesh deform: blendshape_shape_frame_spans len {} != num_blendshapes {}",
            mesh.blendshape_shape_frame_spans.len(),
            shape_count
        );
        return stats;
    }

    let Some(destinations) = resolve_blendshape_destinations(
        mesh,
        cache_entry,
        positions_arena,
        normals_arena,
        tangents_arena,
        temp_arena,
        blend_then_skin,
    ) else {
        return stats;
    };

    stats.copy_ops = stats.copy_ops.saturating_add(copy_base_blendshape_streams(
        gpu.encoder,
        gpu.profiler,
        mesh,
        positions.as_ref(),
        &destinations,
    ));

    let max_wg = gpu.gpu_limits.max_compute_workgroups_per_dimension();
    if !pack_blendshape_scatter_params(gpu, mesh, blend_weights, &destinations, max_wg) {
        return stats;
    }

    if gpu.scratch.packed_scatter_params.is_empty() {
        return stats;
    }

    stats.add(blendshape_record_scatter_compute_passes(
        gpu,
        &destinations,
        sparse.as_ref(),
        blend_param_cursor,
        jobs,
    ));
    stats
}

fn resolve_blendshape_destinations<'a>(
    mesh: &MeshDeformSnapshot,
    cache_entry: &'a SkinCacheEntry,
    positions_arena: &'a wgpu::Buffer,
    normals_arena: &'a wgpu::Buffer,
    tangents_arena: &'a wgpu::Buffer,
    temp_arena: &'a wgpu::Buffer,
    blend_then_skin: bool,
) -> Option<BlendshapeDestinations<'a>> {
    let (positions_buffer, pos_range) = if blend_then_skin {
        (temp_arena, cache_entry.temp.as_ref()?)
    } else {
        (positions_arena, &cache_entry.positions)
    };
    let normals_range = if blend_then_skin {
        cache_entry.temp_normals.as_ref()
    } else {
        cache_entry.normals.as_ref()
    };
    let tangents_range = if blend_then_skin {
        cache_entry.temp_tangents.as_ref()
    } else {
        cache_entry.tangents.as_ref()
    };
    let copy_normals = normals_range.is_some();
    let copy_tangents = tangents_range.is_some() && mesh.tangent_buffer.is_some();
    let apply_normals = mesh.blendshape_has_normal_deltas && copy_normals;
    let apply_tangents = mesh.blendshape_has_tangent_deltas && copy_tangents;
    let normals_buffer = normals_range.map(|_| {
        if blend_then_skin {
            temp_arena
        } else {
            normals_arena
        }
    });
    let tangents_buffer = tangents_range.map(|_| {
        if blend_then_skin {
            temp_arena
        } else {
            tangents_arena
        }
    });
    Some(BlendshapeDestinations {
        positions_buffer,
        normals_buffer,
        tangents_buffer,
        base_pos_e: pos_range.first_element_index(16),
        base_nrm_e: normals_range.map_or(0, |range| range.first_element_index(16)),
        base_tan_e: tangents_range.map_or(0, |range| range.first_element_index(16)),
        copy_normals,
        copy_tangents,
        apply_normals,
        apply_tangents,
    })
}

fn copy_base_blendshape_streams(
    encoder: &mut wgpu::CommandEncoder,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
    mesh: &MeshDeformSnapshot,
    positions: &wgpu::Buffer,
    destinations: &BlendshapeDestinations<'_>,
) -> u64 {
    let mut copy_ops = 1u64;
    let copy_len = u64::from(mesh.vertex_count).saturating_mul(16).max(16);
    let copy_scope = crate::profiling::GpuEncoderScope::begin(
        profiler,
        "mesh_deform::blendshape_base_copies",
        encoder,
    );
    encoder.copy_buffer_to_buffer(
        positions,
        0,
        destinations.positions_buffer,
        u64::from(destinations.base_pos_e).saturating_mul(16),
        copy_len,
    );
    if destinations.copy_normals
        && let (Some(normals), Some(dst_normals)) =
            (mesh.normals_buffer.as_ref(), destinations.normals_buffer)
    {
        encoder.copy_buffer_to_buffer(
            normals.as_ref(),
            0,
            dst_normals,
            u64::from(destinations.base_nrm_e).saturating_mul(16),
            copy_len,
        );
        copy_ops = copy_ops.saturating_add(1);
    }
    if destinations.copy_tangents
        && let (Some(tangents), Some(dst_tangents)) =
            (mesh.tangent_buffer.as_ref(), destinations.tangents_buffer)
    {
        encoder.copy_buffer_to_buffer(
            tangents.as_ref(),
            0,
            dst_tangents,
            u64::from(destinations.base_tan_e).saturating_mul(16),
            copy_len,
        );
        copy_ops = copy_ops.saturating_add(1);
    }
    copy_scope.end(encoder);
    copy_ops
}

/// Records compute passes that scatter blendshape deltas using packed params and per-dispatch
/// workgroups stored in [`crate::mesh_deform::MeshDeformScratch::packed_scatter_params`] /
/// [`crate::mesh_deform::MeshDeformScratch::scatter_dispatch_wgs`].
fn blendshape_record_scatter_compute_passes(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    destinations: &BlendshapeDestinations<'_>,
    sparse: &wgpu::Buffer,
    blend_param_cursor: &mut u64,
    jobs: &mut Vec<BlendshapeDispatchJob>,
) -> MeshDeformRecordStats {
    let mut stats = MeshDeformRecordStats::default();
    let dispatch_count = gpu
        .scratch
        .scatter_dispatch_wgs
        .len()
        .min(gpu.scratch.scatter_dispatch_targets.len());
    for i in 0..dispatch_count {
        let scatter_wg = gpu.scratch.scatter_dispatch_wgs[i];
        let target = gpu.scratch.scatter_dispatch_targets[i];
        let Some(param_reservation) = reserve_blendshape_param_range(*blend_param_cursor, 32)
        else {
            return stats;
        };
        let Some(params_offset) = dynamic_uniform_offset(param_reservation.offset) else {
            return stats;
        };
        let src_off = i.saturating_mul(32);
        let src_end = src_off.saturating_add(32);
        let Some(params) = gpu.scratch.packed_scatter_params.get(src_off..src_end) else {
            return stats;
        };
        let mut params_bytes = [0u8; 32];
        params_bytes.copy_from_slice(params);
        gpu.scratch.ensure_blendshape_param_byte_capacity(
            gpu.device,
            param_reservation
                .offset
                .saturating_add(param_reservation.byte_len),
        );
        gpu.uploads.write_buffer(
            &gpu.scratch.blendshape_params,
            param_reservation.offset,
            &params_bytes,
        );
        *blend_param_cursor = param_reservation.next_cursor;

        let output = match target {
            BLENDSHAPE_CHANNEL_POSITION => destinations.positions_buffer,
            BLENDSHAPE_CHANNEL_NORMAL => destinations
                .normals_buffer
                .unwrap_or(destinations.positions_buffer),
            BLENDSHAPE_CHANNEL_TANGENT => destinations
                .tangents_buffer
                .unwrap_or(destinations.positions_buffer),
            _ => destinations.positions_buffer,
        };

        let (bind_group, reused) = blendshape_bind_group(gpu, sparse, output);
        if reused {
            stats.bind_group_cache_reuses = stats.bind_group_cache_reuses.saturating_add(1);
        } else {
            stats.bind_groups_created = stats.bind_groups_created.saturating_add(1);
        }
        jobs.push(BlendshapeDispatchJob {
            bind_group,
            params_offset,
            wg: scatter_wg,
        });
        stats.blend_dispatches = stats.blend_dispatches.saturating_add(1);
    }

    stats
}

fn dynamic_uniform_offset(offset: u64) -> Option<u32> {
    let offset = u32::try_from(offset).ok();
    if offset.is_none() {
        logger::warn!("mesh deform: blendshape param offset exceeded WebGPU dynamic-offset range");
    }
    offset
}

fn blendshape_bind_group(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    sparse: &wgpu::Buffer,
    output: &wgpu::Buffer,
) -> (Arc<wgpu::BindGroup>, bool) {
    let key = BlendshapeBindGroupKey {
        scratch_generation: gpu.scratch.resource_generation(),
        sparse_buffer: buffer_identity(sparse),
        output_buffer: buffer_identity(output),
    };
    if let Some(bind_group) = gpu.scratch.blendshape_bind_group(key) {
        return (bind_group, true);
    }
    let size = NonZeroU64::new(32);
    let bind_group = Arc::new(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blendshape_scatter_bg"),
        layout: &gpu.pre.blendshape_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &gpu.scratch.blendshape_params,
                    offset: 0,
                    size,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sparse.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    }));
    crate::profiling::note_resource_churn!(BindGroup, "mesh_deform::blendshape_bind_group");
    gpu.scratch
        .insert_blendshape_bind_group(key, Arc::clone(&bind_group));
    (bind_group, false)
}

/// Dispatches all queued blendshape jobs in a single compute pass.
pub(super) fn flush_blendshape_jobs(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    jobs: &[BlendshapeDispatchJob],
) -> MeshDeformRecordStats {
    let mut stats = MeshDeformRecordStats::default();
    if jobs.is_empty() {
        return stats;
    }
    let pass_query = gpu
        .profiler
        .map(|p| p.begin_pass_query("blendshape_scatter_batch", gpu.encoder));
    let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
    {
        let mut cpass = gpu
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blendshape_scatter_batch"),
                timestamp_writes,
            });
        cpass.set_pipeline(&gpu.pre.blendshape_pipeline);
        for job in jobs {
            cpass.set_bind_group(0, job.bind_group.as_ref(), &[job.params_offset]);
            cpass.dispatch_workgroups(job.wg, 1, 1);
        }
    }
    stats.compute_passes = 1;
    if let (Some(p), Some(q)) = (gpu.profiler, pass_query) {
        p.end_query(gpu.encoder, q);
    }
    stats
}

/// Builds the packed scatter `Params` and per-dispatch workgroup counts into
/// [`crate::mesh_deform::MeshDeformScratch::packed_scatter_params`] /
/// [`crate::mesh_deform::MeshDeformScratch::scatter_dispatch_wgs`]. Returns `false` when a dispatch
/// would exceed `max_compute_workgroups_per_dimension` (in which case the caller bails).
fn pack_blendshape_scatter_params(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    mesh: &MeshDeformSnapshot,
    blend_weights: &[f32],
    destinations: &BlendshapeDestinations<'_>,
    max_wg: u32,
) -> bool {
    let vc = mesh.vertex_count;
    let shape_count = mesh.num_blendshapes;
    gpu.scratch.packed_scatter_params.clear();
    gpu.scratch.scatter_dispatch_wgs.clear();
    gpu.scratch.scatter_dispatch_targets.clear();

    for s in 0..shape_count {
        let w = blend_weights.get(s as usize).copied().unwrap_or(0.0);
        for coefficient in select_blendshape_frame_coefficients(
            s,
            w,
            &mesh.blendshape_shape_frame_spans,
            &mesh.blendshape_frame_ranges,
        )
        .into_iter()
        .flatten()
        {
            let Some(range) = mesh
                .blendshape_frame_ranges
                .get(coefficient.frame_range_index)
            else {
                continue;
            };
            for channel in blendshape_frame_channels(range, destinations) {
                if !append_blendshape_channel_dispatches(
                    gpu,
                    vc,
                    channel,
                    coefficient.effective_weight,
                    max_wg,
                ) {
                    return false;
                }
            }
        }
    }
    true
}

#[derive(Clone, Copy)]
struct BlendshapeChannelDispatch {
    channel: u32,
    first_word: u32,
    entry_count: u32,
    entry_words: u32,
    base_dst_e: u32,
}

fn blendshape_frame_channels(
    range: &BlendshapeFrameRange,
    destinations: &BlendshapeDestinations<'_>,
) -> [Option<BlendshapeChannelDispatch>; 3] {
    [
        (range.position_count != 0).then_some(BlendshapeChannelDispatch {
            channel: BLENDSHAPE_CHANNEL_POSITION,
            first_word: range.position_first_word,
            entry_count: range.position_count,
            entry_words: BLENDSHAPE_POSITION_SPARSE_ENTRY_WORDS,
            base_dst_e: destinations.base_pos_e,
        }),
        (destinations.apply_normals && range.normal_count != 0).then_some(
            BlendshapeChannelDispatch {
                channel: BLENDSHAPE_CHANNEL_NORMAL,
                first_word: range.normal_first_word,
                entry_count: range.normal_count,
                entry_words: BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS,
                base_dst_e: destinations.base_nrm_e,
            },
        ),
        (destinations.apply_tangents && range.tangent_count != 0).then_some(
            BlendshapeChannelDispatch {
                channel: BLENDSHAPE_CHANNEL_TANGENT,
                first_word: range.tangent_first_word,
                entry_count: range.tangent_count,
                entry_words: BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS,
                base_dst_e: destinations.base_tan_e,
            },
        ),
    ]
}

fn append_blendshape_channel_dispatches(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    vertex_count: u32,
    channel: Option<BlendshapeChannelDispatch>,
    effective_weight: f32,
    max_wg: u32,
) -> bool {
    let Some(channel) = channel else {
        return true;
    };
    for (entry_offset, sparse_count) in
        plan_blendshape_scatter_chunks(0, channel.entry_count, max_wg)
    {
        let wg = workgroup_count(sparse_count);
        if !gpu.gpu_limits.compute_dispatch_fits(wg, 1, 1) {
            logger::warn!(
                "mesh deform: blendshape scatter dispatch {}x1x1 exceeds max_compute_workgroups_per_dimension ({})",
                wg,
                max_wg
            );
            return false;
        }
        gpu.scratch
            .packed_scatter_params
            .extend_from_slice(&build_scatter_params(ScatterParamFields {
                vertex_count,
                sparse_base_word: channel
                    .first_word
                    .saturating_add(entry_offset.saturating_mul(channel.entry_words)),
                sparse_count,
                base_dst_e: channel.base_dst_e,
                channel: channel.channel,
                effective_weight,
            }));
        gpu.scratch.scatter_dispatch_wgs.push(wg);
        gpu.scratch.scatter_dispatch_targets.push(channel.channel);
    }
    true
}

/// CPU-side field layout for `mesh_blendshape.wgsl` `Params`.
struct ScatterParamFields {
    vertex_count: u32,
    sparse_base_word: u32,
    sparse_count: u32,
    base_dst_e: u32,
    channel: u32,
    effective_weight: f32,
}

/// `shaders/passes/compute/mesh_blendshape.wgsl` `Params` (32 bytes).
fn build_scatter_params(fields: ScatterParamFields) -> [u8; 32] {
    let mut o = [0u8; 32];
    o[0..4].copy_from_slice(&fields.vertex_count.to_le_bytes());
    o[4..8].copy_from_slice(&fields.sparse_base_word.to_le_bytes());
    o[8..12].copy_from_slice(&fields.sparse_count.to_le_bytes());
    o[12..16].copy_from_slice(&fields.base_dst_e.to_le_bytes());
    o[16..20].copy_from_slice(&fields.channel.to_le_bytes());
    o[20..24].copy_from_slice(&fields.effective_weight.to_le_bytes());
    o
}

#[cfg(test)]
mod tests {
    use super::reserve_blendshape_param_range;

    #[test]
    fn blendshape_param_reservations_do_not_overlap_across_meshes() {
        let first = reserve_blendshape_param_range(0, 32).expect("first reservation");
        let second =
            reserve_blendshape_param_range(first.next_cursor, 32).expect("second reservation");

        assert_eq!(first.offset, 0);
        assert_eq!(first.next_cursor, 256);
        assert_eq!(second.offset, 256);
        assert_eq!(second.next_cursor, 512);
        assert!(first.offset + first.byte_len <= second.offset);
    }

    #[test]
    fn blendshape_param_reservation_rejects_empty_or_overflowing_ranges() {
        assert!(reserve_blendshape_param_range(0, 0).is_none());
        assert!(reserve_blendshape_param_range(u64::MAX, 32).is_none());
        assert!(reserve_blendshape_param_range(u64::MAX - 127, 1).is_none());
    }
}
