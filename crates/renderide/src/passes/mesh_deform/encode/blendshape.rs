//! Sparse blendshape scatter compute encoding.
//!
//! Records one bind-pose copy followed by one or more scatter dispatches per weighted shape
//! frame, using packed `Params` arrays staged in [`crate::mesh_deform::MeshDeformScratch`].

use std::num::NonZeroU64;

use crate::assets::mesh::select_blendshape_frame_coefficients;
use crate::mesh_deform::SkinCacheEntry;
use crate::mesh_deform::advance_slab_cursor;
use crate::mesh_deform::plan_blendshape_scatter_chunks;

use super::super::snapshot::MeshDeformSnapshot;
use super::{MeshDeformEncodeGpu, workgroup_count};

/// Arena subranges for blendshape scatter / copy destination.
pub(super) struct BlendshapeCacheCtx<'a> {
    /// Instance line from [`crate::mesh_deform::GpuSkinCache`].
    pub cache_entry: &'a SkinCacheEntry,
    pub positions_arena: &'a wgpu::Buffer,
    pub temp_arena: &'a wgpu::Buffer,
    /// When true, blend output is written to the temp arena for the skinning pass.
    pub blend_then_skin: bool,
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

/// Sparse blendshape scatter: copy bind poses → cache range, then one scatter dispatch per weighted shape chunk.
pub(super) fn record_blendshape_deform(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    mesh: &MeshDeformSnapshot,
    blend_weights: &[f32],
    blend_weight_cursor: &mut u64,
    blend_param_cursor: &mut u64,
    ctx: BlendshapeCacheCtx<'_>,
) -> u64 {
    profiling::scope!("mesh_deform::record_blendshape");
    let BlendshapeCacheCtx {
        cache_entry,
        positions_arena,
        temp_arena,
        blend_then_skin,
    } = ctx;
    let Some(ref positions) = mesh.positions_buffer else {
        return 0;
    };
    let Some(ref sparse) = mesh.blendshape_sparse_buffer else {
        return 0;
    };
    let vc = mesh.vertex_count;
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

    let (dst_buf, dst_off, base_dst_e) = if blend_then_skin {
        let Some(t) = cache_entry.temp.as_ref() else {
            return 0;
        };
        (temp_arena, t.offset_bytes, t.first_element_index(16))
    } else {
        let p = &cache_entry.positions;
        (positions_arena, p.offset_bytes, p.first_element_index(16))
    };

    let copy_len = u64::from(vc).saturating_mul(16).max(16);
    gpu.encoder
        .copy_buffer_to_buffer(positions.as_ref(), 0, dst_buf, dst_off, copy_len);

    let weight_binding_len = stage_blendshape_weights(gpu, blend_weights, *blend_weight_cursor);

    let max_wg = gpu.gpu_limits.max_compute_workgroups_per_dimension();
    if !pack_blendshape_scatter_params(gpu, mesh, blend_weights, base_dst_e, max_wg) {
        return 0;
    }

    if gpu.scratch.packed_scatter_params.is_empty() {
        *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
        return 0;
    }

    blendshape_record_scatter_compute_passes(
        gpu,
        dst_buf,
        sparse.as_ref(),
        weight_binding_len,
        blend_weight_cursor,
        blend_param_cursor,
    )
}

/// Records compute passes that scatter blendshape deltas using packed params and per-dispatch
/// workgroups stored in [`crate::mesh_deform::MeshDeformScratch::packed_scatter_params`] /
/// [`crate::mesh_deform::MeshDeformScratch::scatter_dispatch_wgs`].
fn blendshape_record_scatter_compute_passes(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    dst_buf: &wgpu::Buffer,
    sparse: &wgpu::Buffer,
    weight_binding_len: u64,
    blend_weight_cursor: &mut u64,
    blend_param_cursor: &mut u64,
) -> u64 {
    let Some(param_reservation) = reserve_blendshape_param_range(
        *blend_param_cursor,
        gpu.scratch.packed_scatter_params.len() as u64,
    ) else {
        *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
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

    let Some(weight_size) = NonZeroU64::new(weight_binding_len) else {
        *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
        return 0;
    };

    let dispatch_count = gpu.scratch.scatter_dispatch_wgs.len() as u64;
    for (i, &scatter_wg) in gpu.scratch.scatter_dispatch_wgs.iter().enumerate() {
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu.scratch.blendshape_weights,
                        offset: *blend_weight_cursor,
                        size: Some(weight_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dst_buf.as_entire_binding(),
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

    *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
    dispatch_count
}

/// Fills [`crate::mesh_deform::MeshDeformScratch::blend_weight_bytes`] with the per-shape weights and
/// queues one upload of the bound subrange. Returns the binding length in bytes (always
/// `shape_count * 4`); the caller threads it through subsequent slab advances.
fn stage_blendshape_weights(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    blend_weights: &[f32],
    blend_weight_cursor: u64,
) -> u64 {
    let shape_count = blend_weights.len() as u32;
    gpu.scratch
        .ensure_shape_weight_capacity(gpu.device, shape_count);
    let weight_byte_len = (shape_count as usize) * 4;
    gpu.scratch.blend_weight_bytes.clear();
    gpu.scratch.blend_weight_bytes.resize(weight_byte_len, 0);
    for (s, w) in blend_weights.iter().enumerate() {
        gpu.scratch.blend_weight_bytes[s * 4..s * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }

    let weight_binding_len = weight_byte_len as u64;
    gpu.scratch.ensure_blend_weight_byte_capacity(
        gpu.device,
        blend_weight_cursor.saturating_add(weight_binding_len),
    );
    gpu.upload_batch.write_buffer(
        &gpu.scratch.blendshape_weights,
        blend_weight_cursor,
        &gpu.scratch.blend_weight_bytes,
    );
    weight_binding_len
}

/// Builds the packed scatter `Params` and per-dispatch workgroup counts into
/// [`crate::mesh_deform::MeshDeformScratch::packed_scatter_params`] /
/// [`crate::mesh_deform::MeshDeformScratch::scatter_dispatch_wgs`]. Returns `false` when a dispatch
/// would exceed `max_compute_workgroups_per_dimension` (in which case the caller bails).
fn pack_blendshape_scatter_params(
    gpu: &mut MeshDeformEncodeGpu<'_>,
    mesh: &MeshDeformSnapshot,
    blend_weights: &[f32],
    base_dst_e: u32,
    max_wg: u32,
) -> bool {
    let vc = mesh.vertex_count;
    let shape_count = mesh.num_blendshapes;
    gpu.scratch.packed_scatter_params.clear();
    gpu.scratch.scatter_dispatch_wgs.clear();

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
            if range.entry_count == 0 {
                continue;
            }
            for (sparse_base, sparse_count) in
                plan_blendshape_scatter_chunks(range.first_entry, range.entry_count, max_wg)
            {
                let wg = workgroup_count(sparse_count);
                if !gpu.gpu_limits.compute_dispatch_fits(wg, 1, 1) {
                    logger::warn!(
                        "mesh deform: blendshape scatter dispatch {}×1×1 exceeds max_compute_workgroups_per_dimension ({})",
                        wg,
                        max_wg
                    );
                    return false;
                }
                gpu.scratch
                    .packed_scatter_params
                    .extend_from_slice(&build_scatter_params(
                        vc,
                        sparse_base,
                        sparse_count,
                        base_dst_e,
                        coefficient.effective_weight,
                    ));
                gpu.scratch.scatter_dispatch_wgs.push(wg);
            }
        }
    }
    true
}

/// `shaders/passes/compute/mesh_blendshape.wgsl` `Params` (32 bytes).
fn build_scatter_params(
    vertex_count: u32,
    sparse_base: u32,
    sparse_count: u32,
    base_dst_e: u32,
    effective_weight: f32,
) -> [u8; 32] {
    let mut o = [0u8; 32];
    o[0..4].copy_from_slice(&vertex_count.to_le_bytes());
    o[4..8].copy_from_slice(&sparse_base.to_le_bytes());
    o[8..12].copy_from_slice(&sparse_count.to_le_bytes());
    o[12..16].copy_from_slice(&base_dst_e.to_le_bytes());
    o[16..20].copy_from_slice(&effective_weight.to_le_bytes());
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
