//! Sparse blendshape scatter compute encoding.
//!
//! Records one bind-pose copy followed by one or more scatter dispatches per weighted shape
//! frame, using packed `Params` arrays staged in [`crate::mesh_deform::MeshDeformScratch`].

use crate::assets::mesh::{
    BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS, BLENDSHAPE_POSITION_SPARSE_ENTRY_WORDS,
    BlendshapeFrameRange, select_blendshape_frame_coefficients,
};
use crate::mesh_deform::SkinCacheEntry;
use crate::mesh_deform::plan_blendshape_scatter_chunks;

use super::super::snapshot::MeshDeformSnapshot;
use super::{MeshDeformEncodeGpu, workgroup_count};

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

/// Reserved staging range for packed blendshape scatter params.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlendshapeParamReservation {
    /// Byte offset inside [`crate::mesh_deform::MeshDeformScratch::blendshape_params_staging`].
    offset: u64,
    /// Number of bytes occupied by this mesh's packed scatter params.
    byte_len: u64,
    /// Cursor value to use for the next mesh's reservation.
    next_cursor: u64,
}

/// Reserves a non-overlapping packed-param range in the frame-global staging slab.
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
    ctx: BlendshapeCacheCtx<'_>,
) -> u64 {
    profiling::scope!("mesh_deform::record_blendshape");
    let BlendshapeCacheCtx {
        cache_entry,
        positions_arena,
        normals_arena,
        tangents_arena,
        temp_arena,
        blend_then_skin,
    } = ctx;
    let Some(ref positions) = mesh.positions_buffer else {
        return 0;
    };
    let Some(ref sparse) = mesh.blendshape_sparse_buffer else {
        return 0;
    };
    let shape_count = mesh.num_blendshapes;
    if shape_count == 0 {
        return 0;
    }
    if mesh.blendshape_shape_frame_spans.len() != shape_count as usize {
        logger::warn!(
            "mesh deform: blendshape_shape_frame_spans len {} != num_blendshapes {}",
            mesh.blendshape_shape_frame_spans.len(),
            shape_count
        );
        return 0;
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
        return 0;
    };

    copy_base_blendshape_streams(gpu.encoder, mesh, positions.as_ref(), &destinations);

    let max_wg = gpu.gpu_limits.max_compute_workgroups_per_dimension();
    if !pack_blendshape_scatter_params(gpu, mesh, blend_weights, &destinations, max_wg) {
        return 0;
    }

    if gpu.scratch.packed_scatter_params.is_empty() {
        return 0;
    }

    blendshape_record_scatter_compute_passes(
        gpu,
        &destinations,
        sparse.as_ref(),
        blend_param_cursor,
    )
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
    mesh: &MeshDeformSnapshot,
    positions: &wgpu::Buffer,
    destinations: &BlendshapeDestinations<'_>,
) {
    let copy_len = u64::from(mesh.vertex_count).saturating_mul(16).max(16);
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
    }
}

/// Records compute passes that scatter blendshape deltas using packed params and per-dispatch
/// workgroups stored in [`crate::mesh_deform::MeshDeformScratch::packed_scatter_params`] /
/// [`crate::mesh_deform::MeshDeformScratch::scatter_dispatch_wgs`].
fn blendshape_record_scatter_compute_passes(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    destinations: &BlendshapeDestinations<'_>,
    sparse: &wgpu::Buffer,
    blend_param_cursor: &mut u64,
) -> u64 {
    let Some(param_reservation) = reserve_blendshape_param_range(
        *blend_param_cursor,
        gpu.scratch.packed_scatter_params.len() as u64,
    ) else {
        return 0;
    };
    gpu.scratch.ensure_blendshape_params_staging(
        gpu.device,
        param_reservation
            .offset
            .saturating_add(param_reservation.byte_len),
    );
    gpu.upload_batch.write_buffer(
        &gpu.scratch.blendshape_params_staging,
        param_reservation.offset,
        &gpu.scratch.packed_scatter_params,
    );
    *blend_param_cursor = param_reservation.next_cursor;

    let dispatch_count = gpu.scratch.scatter_dispatch_wgs.len() as u64;
    for (i, (&scatter_wg, &target)) in gpu
        .scratch
        .scatter_dispatch_wgs
        .iter()
        .zip(gpu.scratch.scatter_dispatch_targets.iter())
        .enumerate()
    {
        let src_off = param_reservation
            .offset
            .saturating_add((i as u64).saturating_mul(32));
        gpu.encoder.copy_buffer_to_buffer(
            &gpu.scratch.blendshape_params_staging,
            src_off,
            &gpu.scratch.blendshape_params,
            0,
            32,
        );
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

        let blend_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blendshape_scatter_bg"),
            layout: &gpu.pre.blendshape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.scratch.blendshape_params.as_entire_binding(),
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
        });

        let pass_query = gpu
            .profiler
            .map(|p| p.begin_pass_query("blendshape_scatter", gpu.encoder));
        let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
        {
            let mut cpass = gpu
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("blendshape_scatter"),
                    timestamp_writes,
                });
            cpass.set_pipeline(&gpu.pre.blendshape_pipeline);
            cpass.set_bind_group(0, &blend_bg, &[]);
            cpass.dispatch_workgroups(scatter_wg, 1, 1);
        };
        if let (Some(p), Some(q)) = (gpu.profiler, pass_query) {
            p.end_query(gpu.encoder, q);
        }
    }

    dispatch_count
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
        let first = reserve_blendshape_param_range(0, 64).expect("first reservation");
        let second =
            reserve_blendshape_param_range(first.next_cursor, 96).expect("second reservation");

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
