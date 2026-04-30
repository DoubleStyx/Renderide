//! Decoders that turn raw GPU staging bytes into [`HiZCpuSnapshot`].

use std::sync::Arc;

use super::pyramid::{mip_dimensions, total_float_count};
use super::snapshot::HiZCpuSnapshot;

/// Unpacks a **linear** row-major buffer (no row padding) into [`HiZCpuSnapshot`].
///
/// The `mips` `Vec<f32>` is moved into an [`Arc<[f32]>`] so downstream clones stay cheap.
pub fn hi_z_snapshot_from_linear_linear(
    base_width: u32,
    base_height: u32,
    mip_levels: u32,
    mips: Vec<f32>,
) -> Option<HiZCpuSnapshot> {
    profiling::scope!("hi_z::build_cpu_snapshot");
    let snap = HiZCpuSnapshot {
        base_width,
        base_height,
        mip_levels,
        mips: Arc::from(mips),
    };
    snap.validate()?;
    Some(snap)
}

/// Unpacks GPU readback with `bytes_per_row` alignment (256-byte aligned rows) into dense `mips`.
///
/// The output is pre-allocated to [`total_float_count`] and row reads use `chunks_exact(4)` so the
/// inner loop avoids per-byte bounds checks. Bytes are interpreted via [`f32::from_le_bytes`] to
/// stay correct on misaligned [`Vec<u8>`] buffers from `wgpu::BufferSlice::get_mapped_range`.
pub fn unpack_linear_rows_to_mips(
    base_width: u32,
    base_height: u32,
    mip_levels: u32,
    staging: &[u8],
) -> Option<Vec<f32>> {
    profiling::scope!("hi_z::unpack_linear_rows");
    let expected = total_float_count(base_width, base_height, mip_levels);
    let mut out: Vec<f32> = Vec::with_capacity(expected);
    let mut staging_off = 0usize;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_width, base_height, mip)?;
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as usize;
        let mip_bytes = row_pitch * h as usize;
        if staging_off + mip_bytes > staging.len() {
            return None;
        }
        let dense_row_bytes = (w as usize) * 4;
        for row in 0..h {
            let row_start = staging_off + row as usize * row_pitch;
            let row_bytes = staging.get(row_start..row_start + dense_row_bytes)?;
            out.extend(
                row_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        staging_off += mip_bytes;
    }
    if out.len() != expected {
        return None;
    }
    Some(out)
}
