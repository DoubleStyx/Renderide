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
/// The output is pre-sized to [`total_float_count`] and rows are copied as bulk byte ranges into
/// the pre-aligned `Vec<f32>` storage. When the row's source byte range happens to satisfy `f32`
/// alignment (the typical case for `wgpu::BufferSlice::get_mapped_range`), the copy goes through
/// [`bytemuck::cast_slice`]; otherwise a [`f32::from_le_bytes`] fallback handles unaligned bytes.
/// Both paths produce identical values on little-endian targets.
pub fn unpack_linear_rows_to_mips(
    base_width: u32,
    base_height: u32,
    mip_levels: u32,
    staging: &[u8],
) -> Option<Vec<f32>> {
    const _: () = assert!(
        cfg!(target_endian = "little"),
        "renderide assumes a little-endian target for GPU readback unpacking",
    );

    profiling::scope!("hi_z::unpack_linear_rows");
    let expected = total_float_count(base_width, base_height, mip_levels);
    let mut out: Vec<f32> = vec![0.0; expected];
    let mut staging_off = 0usize;
    let mut out_off = 0usize;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_width, base_height, mip)?;
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as usize;
        let mip_bytes = row_pitch * h as usize;
        if staging_off + mip_bytes > staging.len() {
            return None;
        }
        let dense_row_bytes = (w as usize) * 4;
        let dense_row_floats = w as usize;
        for row in 0..h {
            let row_start = staging_off + row as usize * row_pitch;
            let row_bytes = staging.get(row_start..row_start + dense_row_bytes)?;
            let dst = out.get_mut(out_off..out_off + dense_row_floats)?;
            match bytemuck::try_cast_slice::<u8, f32>(row_bytes) {
                Ok(src_floats) => dst.copy_from_slice(src_floats),
                Err(_) => {
                    for (slot, c) in dst.iter_mut().zip(row_bytes.chunks_exact(4)) {
                        *slot = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                }
            }
            out_off += dense_row_floats;
        }
        staging_off += mip_bytes;
    }
    if out.len() != expected {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_le_f32s(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn unpack_dense_single_mip_matches_input() {
        let w = 4u32;
        let h = 3u32;
        let row_pitch =
            wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as usize;
        let mut staging = vec![0u8; row_pitch * h as usize];
        let dense: Vec<f32> = (0..(w * h)).map(|i| i as f32 + 0.25).collect();
        for row in 0..h as usize {
            let row_bytes = pack_le_f32s(&dense[row * w as usize..(row + 1) * w as usize]);
            staging[row * row_pitch..row * row_pitch + row_bytes.len()]
                .copy_from_slice(&row_bytes);
        }
        let out = unpack_linear_rows_to_mips(w, h, 1, &staging).expect("unpack");
        assert_eq!(out, dense);
    }

    #[test]
    fn unpack_handles_multiple_mips() {
        let base_w = 4u32;
        let base_h = 4u32;
        let levels = 3u32;
        let mut staging = Vec::new();
        let mut expected: Vec<f32> = Vec::new();
        for m in 0..levels {
            let (w, h) = mip_dimensions(base_w, base_h, m).unwrap();
            let row_pitch =
                wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as usize;
            let mut mip = vec![0u8; row_pitch * h as usize];
            for row in 0..h {
                for col in 0..w {
                    let v = (m as f32) * 100.0 + (row as f32) * 10.0 + (col as f32);
                    let idx = row as usize * row_pitch + col as usize * 4;
                    mip[idx..idx + 4].copy_from_slice(&v.to_le_bytes());
                    expected.push(v);
                }
            }
            staging.extend_from_slice(&mip);
        }
        let out = unpack_linear_rows_to_mips(base_w, base_h, levels, &staging).expect("unpack");
        assert_eq!(out, expected);
    }

    #[test]
    fn unpack_returns_none_on_truncated_staging() {
        let staging = vec![0u8; 16];
        assert!(unpack_linear_rows_to_mips(64, 64, 4, &staging).is_none());
    }
}
