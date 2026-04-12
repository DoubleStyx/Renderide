//! Plan storage-buffer subranges for blendshape compute so each bind respects [`wgpu::Limits`].
//!
//! `max_storage_buffer_binding_size` limits bytes bound per storage entry; subrange offsets must be
//! multiples of `min_storage_buffer_offset_alignment` (typically 256).

use crate::assets::mesh::BLENDSHAPE_OFFSET_GPU_STRIDE;

/// Greatest common divisor (Euclidean algorithm).
pub(crate) fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Byte stride of one blendshape’s packed delta block in the mesh GPU buffer.
pub(crate) fn blendshape_shape_stride_bytes(vertex_count: u32) -> Option<u64> {
    u64::from(vertex_count).checked_mul(BLENDSHAPE_OFFSET_GPU_STRIDE as u64)
}

/// Plans `(shape_start, shapes_in_chunk)` dispatches so each chunk’s byte length is ≤ `max_binding`
/// and each non-zero start offset is a multiple of `min_offset_alignment`.
///
/// Returns [`None`] if even one shape does not fit, or chunk capacity collapses to zero after
/// alignment rounding (caller should disable blendshapes for that mesh).
pub fn plan_blendshape_bind_chunks(
    shape_count: u32,
    vertex_count: u32,
    max_binding: u64,
    min_offset_alignment: u32,
) -> Option<Vec<(u32, u32)>> {
    if shape_count == 0 || vertex_count == 0 {
        return None;
    }
    let align = u64::from(min_offset_alignment.max(1));
    let stride = blendshape_shape_stride_bytes(vertex_count)?;
    if stride > max_binding {
        return None;
    }
    let g = gcd_u64(stride, align);
    let align_shapes = u32::try_from(align / g).ok()?;
    if align_shapes == 0 {
        return None;
    }
    let mut max_chunk_shapes = u32::try_from(max_binding / stride).ok()?;
    max_chunk_shapes = (max_chunk_shapes / align_shapes) * align_shapes;
    if max_chunk_shapes == 0 {
        return None;
    }

    let mut out = Vec::new();
    let mut start = 0u32;
    while start < shape_count {
        let remaining = shape_count - start;
        let take = remaining.min(max_chunk_shapes);
        out.push((start, take));
        start = start.saturating_add(take);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_plan_respects_max_binding() {
        let vc = 1000u32;
        let stride = blendshape_shape_stride_bytes(vc).unwrap();
        let max_b = stride * 10;
        let plan = plan_blendshape_bind_chunks(25, vc, max_b, 256).expect("plan");
        let sum: u32 = plan.iter().map(|(_, n)| n).sum();
        assert_eq!(sum, 25);
        for (s, c) in &plan {
            assert!(u64::from(*c) * stride <= max_b);
            if *s > 0 {
                assert_eq!(u64::from(*s) * stride % 256, 0);
            }
        }
    }

    #[test]
    fn single_shape_when_fits() {
        let plan = plan_blendshape_bind_chunks(1, 4096, 256 * 1024 * 1024, 256).expect("plan");
        assert_eq!(plan, vec![(0, 1)]);
    }

    #[test]
    fn rejects_when_one_shape_exceeds_limit() {
        let stride = blendshape_shape_stride_bytes(1_000_000).unwrap();
        assert!(
            plan_blendshape_bind_chunks(2, 1_000_000, stride - 1, 256).is_none(),
            "one shape must not fit"
        );
    }
}
