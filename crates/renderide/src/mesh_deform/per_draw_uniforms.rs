//! Per-draw uniform packing for mesh forward passes (WebGPU dynamic uniform offset = 256 bytes).

use glam::Mat4;
use rayon::prelude::*;

use super::wgsl_mat3x3::WgslMat3x3;

/// Stride between consecutive draw slots in the uniform slab (`mat4`x3 + WGSL padding).
pub const PER_DRAW_UNIFORM_STRIDE: usize = 256;

/// Initial number of draw slots allocated for [`crate::backend::PerDrawResources`].
pub const INITIAL_PER_DRAW_UNIFORM_SLOTS: usize = 256;

/// Metadata flag stored in [`PaddedPerDrawUniforms::_pad`] when the bound position stream is already world-space.
pub const PER_DRAW_POSITION_STREAM_WORLD_SPACE_FLAG: f32 = 1.0;

/// Metadata flag offset inside [`PaddedPerDrawUniforms::_pad`].
const PER_DRAW_POSITION_STREAM_WORLD_SPACE_PAD_SLOT: usize = 0;
/// Packed reflection-probe atlas indices offset inside [`PaddedPerDrawUniforms::_pad`].
const PER_DRAW_REFLECTION_PROBE_INDICES_PAD_SLOT: usize = 1;

/// GPU layout: left/right view-projection, `model`, inverse-transpose normal matrix, padding to 256 bytes.
///
/// Matches composed `shaders/target/null_*.wgsl` (`PerDrawUniforms` at `@group(2)`).
///
/// **Contract:** [`Self::view_proj_left`] and [`Self::view_proj_right`] normally store
/// **projection x view** (PV) only. Vertex shaders compute `clip = view_proj * (model * local_pos)`;
/// premultiplying `model` into the view-projection would apply it twice for static meshes. The
/// null fallback's world-space-deformed path is the narrow exception: it stores `PV * inverse(model)`
/// so the shader can keep the real model matrix for checker anchoring without double-transforming
/// already-world-space vertices.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PaddedPerDrawUniforms {
    /// Column-major view-projection for the left eye (or the only view on desktop).
    ///
    /// Normally excludes object `model`; see [`Self`] for the null fallback exception.
    pub view_proj_left: [f32; 16],
    /// Column-major view-projection for the right eye (duplicated when single-view).
    ///
    /// Normally excludes object `model`; see [`Self`] for the null fallback exception.
    pub view_proj_right: [f32; 16],
    /// Column-major world matrix from the scene.
    ///
    /// This is identity for most skinned meshes with world-space positions, except the null fallback
    /// keeps the real model matrix and compensates in [`Self::view_proj_left`] / [`Self::view_proj_right`].
    pub model: [f32; 16],
    /// Inverse transpose of the upper 3x3 of [`Self::model`] for normal transforms.
    pub(super) normal_matrix: WgslMat3x3,
    /// Metadata plus padding to [`PER_DRAW_UNIFORM_STRIDE`] bytes.
    ///
    /// Slot 0 is [`PER_DRAW_POSITION_STREAM_WORLD_SPACE_FLAG`] when the vertex position stream is
    /// already world-space. Slots 1 stores a bit mask indicating if each local reflection-probe
    /// is of a lower importance as its predecessor, as well as the atlas index of
    /// the global fallback reflection probe.
    /// Slots 2 and 3 store the 4 local reflection-probe atlas indices.
    /// If we need to save on performance, or provide under 4 bytes of additional data
    /// in the per-draw uniforms, we can lower the system down to 2 local reflection probes
    /// and reuse Slot 3 for other purposes.
    pub _pad: [f32; 4],
}

impl PaddedPerDrawUniforms {
    /// Single-view path: duplicates PV `view_proj` into both eye slots.
    ///
    /// `view_proj` is the matrix left-multiplied with `model * position`; it is normally **PV only**
    /// except for the null fallback exception described on [`Self`].
    #[inline]
    pub fn new_single(view_proj: Mat4, model: Mat4) -> Self {
        let vp = view_proj.to_cols_array();
        Self {
            view_proj_left: vp,
            view_proj_right: vp,
            model: model.to_cols_array(),
            normal_matrix: WgslMat3x3::from_model_upper_3x3(model),
            _pad: [0.0; 4],
        }
    }

    /// Stereo path: separate per-eye PV (multiview or single-view shader using left only).
    ///
    /// Both arguments are normally **PV only** except for the null fallback exception described on
    /// [`Self`].
    #[inline]
    pub fn new_stereo(view_proj_left: Mat4, view_proj_right: Mat4, model: Mat4) -> Self {
        Self {
            view_proj_left: view_proj_left.to_cols_array(),
            view_proj_right: view_proj_right.to_cols_array(),
            model: model.to_cols_array(),
            normal_matrix: WgslMat3x3::from_model_upper_3x3(model),
            _pad: [0.0; 4],
        }
    }

    /// Returns a copy with the position-stream space metadata set for shaders that need it.
    #[inline]
    #[must_use]
    pub fn with_position_stream_world_space(mut self, enabled: bool) -> Self {
        self._pad[PER_DRAW_POSITION_STREAM_WORLD_SPACE_PAD_SLOT] = if enabled {
            PER_DRAW_POSITION_STREAM_WORLD_SPACE_FLAG
        } else {
            0.0
        };
        self
    }

    /// Returns a copy with packed reflection-probe selection metadata.
    #[inline]
    #[must_use]
    pub fn with_reflection_probe_selection(
        mut self,
        atlas_indices: [u16; 5],
        importance_mask: u8,
    ) -> Self {
        let packed = u32::from(importance_mask) | (u32::from(atlas_indices[0]) << 16);
        self._pad[PER_DRAW_REFLECTION_PROBE_INDICES_PAD_SLOT] = f32::from_bits(packed);
        let packed = u32::from(atlas_indices[1]) | (u32::from(atlas_indices[2]) << 16);
        self._pad[PER_DRAW_REFLECTION_PROBE_INDICES_PAD_SLOT + 1] = f32::from_bits(packed);
        let packed = u32::from(atlas_indices[3]) | (u32::from(atlas_indices[4]) << 16);
        self._pad[PER_DRAW_REFLECTION_PROBE_INDICES_PAD_SLOT + 2] = f32::from_bits(packed);
        self
    }

    /// Unpacks reflection-probe selection metadata.
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub fn reflection_probe_selection(&self) -> ([u16; 5], u8) {
        let mut atlas_indices = [0; 5];
        let packed = self._pad[PER_DRAW_REFLECTION_PROBE_INDICES_PAD_SLOT].to_bits();
        let importance_mask = (packed & 0xFFFF) as u8;
        atlas_indices[0] = (packed >> 16) as u16;
        let packed = self._pad[PER_DRAW_REFLECTION_PROBE_INDICES_PAD_SLOT + 1].to_bits();
        atlas_indices[1] = (packed & 0xFFFF) as u16;
        atlas_indices[2] = (packed >> 16) as u16;
        let packed = self._pad[PER_DRAW_REFLECTION_PROBE_INDICES_PAD_SLOT + 2].to_bits();
        atlas_indices[3] = (packed & 0xFFFF) as u16;
        atlas_indices[4] = (packed >> 16) as u16;
        (atlas_indices, importance_mask)
    }

    /// Whether the metadata says the bound vertex position stream is already in world space.
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub fn position_stream_world_space(&self) -> bool {
        self._pad[PER_DRAW_POSITION_STREAM_WORLD_SPACE_PAD_SLOT] > 0.5
    }
}

/// Slot count above which slab writes fan out to a rayon worker pool.
///
/// Each slot is a 256-byte copy. At 256 slots the slab is already 64 KiB, large enough for
/// memory-bandwidth fan-out to pay off on typical desktop CPUs.
const PER_DRAW_SLAB_PARALLEL_MIN: usize = 256;
const PER_DRAW_SLAB_PARALLEL_CHUNK_SLOTS: usize = 64;

/// Writes `count` consecutive [`PaddedPerDrawUniforms`] into `out` (must be `count * 256` bytes).
///
/// Parallelizes across rayon when `slots.len() >= PER_DRAW_SLAB_PARALLEL_MIN`. Each worker writes
/// into a disjoint 256-byte region of `out`, so there is no synchronization on the hot path.
pub fn write_per_draw_uniform_slab(slots: &[PaddedPerDrawUniforms], out: &mut [u8]) {
    let need = slots.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE);
    assert!(
        out.len() >= need,
        "slab buffer too small: need {need}, have {}",
        out.len()
    );
    profiling::scope!("mesh_deform::write_per_draw_uniform_slab");
    let dst = &mut out[..need];
    if slots.len() >= PER_DRAW_SLAB_PARALLEL_MIN {
        dst.par_chunks_mut(PER_DRAW_UNIFORM_STRIDE * PER_DRAW_SLAB_PARALLEL_CHUNK_SLOTS)
            .zip(slots.par_chunks(PER_DRAW_SLAB_PARALLEL_CHUNK_SLOTS))
            .for_each(|(slabs, slots)| {
                profiling::scope!("mesh_deform::write_per_draw_uniform_slab::worker");
                for (slab, slot) in slabs
                    .chunks_exact_mut(PER_DRAW_UNIFORM_STRIDE)
                    .zip(slots.iter())
                {
                    slab.copy_from_slice(bytemuck::bytes_of(slot));
                }
            });
    } else {
        for (slab, slot) in dst
            .chunks_exact_mut(PER_DRAW_UNIFORM_STRIDE)
            .zip(slots.iter())
        {
            slab.copy_from_slice(bytemuck::bytes_of(slot));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_size_is_256() {
        assert_eq!(size_of::<PaddedPerDrawUniforms>(), PER_DRAW_UNIFORM_STRIDE);
    }

    /// Forward pass WGSL uses `clip = view_proj * (model * local)`. Packing PVxmodel into
    /// `view_proj` would apply `model` twice for static meshes (regression guard).
    #[test]
    fn shader_clip_uses_pv_times_model_once() {
        let proj = Mat4::from_cols_array(&[
            1.2, 0.0, 0.0, 0.0, //
            0.0, 0.9, 0.0, 0.0, //
            0.0, 0.0, -1.01, -1.0, //
            0.0, 0.0, -0.1, 0.0,
        ]);
        let view = Mat4::from_translation(glam::Vec3::new(0.0, 1.0, -5.0));
        let model = Mat4::from_scale(glam::Vec3::new(2.0, 2.0, 2.0));
        let pv = proj * view;
        let local = glam::Vec4::new(0.25, 0.0, 0.0, 1.0);

        let clip_correct = pv * (model * local);
        let clip_double_model = (pv * model) * (model * local);

        let expected = proj * view * model * local;
        assert!(
            (clip_correct - expected).length() < 1e-5,
            "PV * (M * p) should match single MVP chain"
        );
        assert!(
            (clip_double_model - expected).length() > 0.01,
            "regression: premultiplying M into PV double-applies M"
        );
    }

    #[test]
    fn slab_roundtrip_bytes() {
        let vp = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let m = Mat4::from_scale(glam::Vec3::new(4.0, 5.0, 6.0));
        let slot = PaddedPerDrawUniforms::new_single(vp, m).with_position_stream_world_space(true);
        let mut buf = vec![0u8; PER_DRAW_UNIFORM_STRIDE * 2];
        write_per_draw_uniform_slab(
            &[
                slot,
                PaddedPerDrawUniforms::new_single(Mat4::IDENTITY, Mat4::IDENTITY),
            ],
            &mut buf,
        );
        let a: &PaddedPerDrawUniforms = bytemuck::from_bytes(&buf[0..PER_DRAW_UNIFORM_STRIDE]);
        assert_eq!(a.view_proj_left, vp.to_cols_array());
        assert_eq!(a.view_proj_right, vp.to_cols_array());
        assert_eq!(a.model, m.to_cols_array());
        assert_eq!(a.normal_matrix, WgslMat3x3::from_model_upper_3x3(m));
        assert!(a.position_stream_world_space());
        assert_eq!(
            a._pad[PER_DRAW_POSITION_STREAM_WORLD_SPACE_PAD_SLOT],
            PER_DRAW_POSITION_STREAM_WORLD_SPACE_FLAG
        );
        let b: &PaddedPerDrawUniforms =
            bytemuck::from_bytes(&buf[PER_DRAW_UNIFORM_STRIDE..PER_DRAW_UNIFORM_STRIDE * 2]);
        assert!(!b.position_stream_world_space());
    }

    #[test]
    fn reflection_probe_selection_packs_into_reserved_slots() {
        let slot = PaddedPerDrawUniforms::new_single(Mat4::IDENTITY, Mat4::IDENTITY)
            .with_position_stream_world_space(true)
            .with_reflection_probe_selection([17, 23, 0, 0, 11], 0b0011);

        assert!(slot.position_stream_world_space());
        assert_eq!(
            slot.reflection_probe_selection(),
            ([17, 23, 0, 0, 11], 0b0011)
        );
        assert_eq!(size_of::<PaddedPerDrawUniforms>(), PER_DRAW_UNIFORM_STRIDE);
    }

    #[test]
    fn slab_parallel_path_matches_serial_for_large_input() {
        let count = PER_DRAW_SLAB_PARALLEL_MIN + 17;
        let slots: Vec<PaddedPerDrawUniforms> = (0..count)
            .map(|i| {
                let m = Mat4::from_translation(glam::Vec3::new(i as f32, i as f32 * 0.5, 0.0));
                PaddedPerDrawUniforms::new_single(Mat4::IDENTITY, m)
                    .with_position_stream_world_space(i % 2 == 0)
            })
            .collect();

        let mut parallel = vec![0u8; PER_DRAW_UNIFORM_STRIDE * count];
        write_per_draw_uniform_slab(&slots, &mut parallel);

        let mut serial = vec![0u8; PER_DRAW_UNIFORM_STRIDE * count];
        for (slab, slot) in serial
            .chunks_exact_mut(PER_DRAW_UNIFORM_STRIDE)
            .zip(slots.iter())
        {
            slab.copy_from_slice(bytemuck::bytes_of(slot));
        }
        assert_eq!(parallel, serial);
    }
}
