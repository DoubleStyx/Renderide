//! Per-draw uniform packing for mesh forward passes (WebGPU dynamic uniform offset = 256 bytes).

use glam::Mat4;

/// Stride between consecutive draw slots in the uniform slab (`mat4`×3 + WGSL padding).
pub const PER_DRAW_UNIFORM_STRIDE: usize = 256;

/// Initial number of draw slots allocated for [`super::debug_draw::DebugDrawResources`].
pub const INITIAL_PER_DRAW_UNIFORM_SLOTS: usize = 256;

/// GPU layout: left/right view–projection, `model`, then padding to 256 bytes.
///
/// Matches composed `shaders/target/debug_world_normals_*.wgsl` (`PerDrawUniforms` at `@group(2)`).
///
/// **Contract:** [`Self::view_proj_left`] and [`Self::view_proj_right`] store **projection × view**
/// (PV) only. They must **not** include the mesh world matrix. Vertex shaders compute
/// `clip = view_proj * (model * local_pos)`; premultiplying `model` into the view–projection
/// would apply it twice for static meshes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PaddedPerDrawUniforms {
    /// Column-major PV for the left eye (or the only view on desktop); excludes object `model`.
    pub view_proj_left: [f32; 16],
    /// Column-major PV for the right eye (duplicated when single-view); excludes object `model`.
    pub view_proj_right: [f32; 16],
    /// Column-major world matrix from the scene (identity for skinned meshes with world-space positions).
    pub model: [f32; 16],
    /// Padding to [`PER_DRAW_UNIFORM_STRIDE`] bytes.
    pub _pad: [f32; 16],
}

impl PaddedPerDrawUniforms {
    /// Single-view path: duplicates PV `view_proj` into both eye slots.
    ///
    /// `view_proj` must be **PV only** (see [`Self`]).
    pub fn new_single(view_proj: Mat4, model: Mat4) -> Self {
        let vp = view_proj.to_cols_array();
        Self {
            view_proj_left: vp,
            view_proj_right: vp,
            model: model.to_cols_array(),
            _pad: [0.0; 16],
        }
    }

    /// Stereo path: separate per-eye PV (multiview or single-view shader using left only).
    ///
    /// Both arguments must be **PV only** (see [`Self`]).
    pub fn new_stereo(view_proj_left: Mat4, view_proj_right: Mat4, model: Mat4) -> Self {
        Self {
            view_proj_left: view_proj_left.to_cols_array(),
            view_proj_right: view_proj_right.to_cols_array(),
            model: model.to_cols_array(),
            _pad: [0.0; 16],
        }
    }
}

/// Writes `count` consecutive [`PaddedPerDrawUniforms`] into `out` (must be `count * 256` bytes).
pub fn write_per_draw_uniform_slab(slots: &[PaddedPerDrawUniforms], out: &mut [u8]) {
    let need = slots.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE);
    assert!(
        out.len() >= need,
        "slab buffer too small: need {need}, have {}",
        out.len()
    );
    for (i, slot) in slots.iter().enumerate() {
        let start = i * PER_DRAW_UNIFORM_STRIDE;
        let bytes: &[u8] = bytemuck::bytes_of(slot);
        out[start..start + bytes.len()].copy_from_slice(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_size_is_256() {
        assert_eq!(
            std::mem::size_of::<PaddedPerDrawUniforms>(),
            PER_DRAW_UNIFORM_STRIDE
        );
    }

    /// Forward pass WGSL uses `clip = view_proj * (model * local)`. Packing PV×model into
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
        let slot = PaddedPerDrawUniforms::new_single(vp, m);
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
    }
}
