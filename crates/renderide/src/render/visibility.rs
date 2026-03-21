//! CPU frustum visibility: rigid and skinned mesh culling.
//!
//! # Module structure
//!
//! - [`rigid`]: frustum culling for non-skinned meshes.  Uses the local
//!   [`RenderBoundingBox`] transformed by the draw's model matrix.
//! - [`skinned`]: frustum culling for bone-deformed meshes (new).  Builds a world-space
//!   AABB from the final bone matrices' translation columns and expands it by the mesh's
//!   largest local half-extent.
//!
//! Shared primitives (view-matrix helpers, homogeneous-clip AABB test, bound helpers)
//! live in this root module and are used by both submodules.
//!
//! # Using the culling functions
//!
//! ```rust,ignore
//! // Rigid draw: cull before submitting.
//! if !visibility::rigid_mesh_potentially_visible(&bounds, model_matrix, view_proj) {
//!     continue;
//! }
//! // Skinned draw: cull after computing bone_matrices.
//! if !visibility::skinned_mesh_potentially_visible(&bounds, &bone_matrices, view_proj) {
//!     continue;
//! }
//! ```

pub mod rigid;
pub mod skinned;

use glam::{Mat4, Vec3, Vec4};
use nalgebra::{Matrix4, Vector3};

use crate::scene::math::matrix_na_to_glam;
use crate::scene::render_transform_to_matrix;
use crate::shared::RenderBoundingBox;

use super::batch::SpaceDrawBatch;
use super::view::ViewParams;

// ─── Public re-exports ────────────────────────────────────────────────────────

pub use rigid::{
    RigidFrustumCullBoundsTag, RigidFrustumCullCache, RigidFrustumCullCacheKey,
    rigid_mesh_potentially_visible, rigid_mesh_potentially_visible_cached,
};
pub use skinned::{
    skinned_mesh_potentially_visible, skinned_mesh_potentially_visible_from_bone_origins,
};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Epsilon for homogeneous clip comparisons and behind-camera checks.
pub(crate) const CLIP_EPS: f32 = 1e-5;

/// Maximum absolute half-extent below which uploaded mesh bounds are treated as **untrusted**
/// for frustum culling.
///
/// Hosts may send zero extents when metadata is invalid (FrooxEngine `Mesh` invalid-bounds
/// handling), which collapses the culled volume to a single point at local origin.
pub(crate) const DEGENERATE_MESH_BOUNDS_EXTENT_EPS: f32 = 1e-8;

/// Below this max half-extent (world-upload units), a successful frustum cull is logged at
/// trace level as potentially suspicious metadata.
pub(crate) const SUSPICIOUS_MESH_BOUNDS_MAX_EXTENT: f32 = 1e-3;

// ─── View matrix helpers ──────────────────────────────────────────────────────

/// Clamps scale components to avoid degenerate view matrices.
fn filter_scale(scale: Vector3<f32>) -> Vector3<f32> {
    const MIN_SCALE: f32 = 1e-8;
    if scale.x.abs() < MIN_SCALE || scale.y.abs() < MIN_SCALE || scale.z.abs() < MIN_SCALE {
        Vector3::new(1.0, 1.0, 1.0)
    } else {
        scale
    }
}

/// Applies the Z-flip for coordinate system alignment (RH engine → Vulkan/WebGPU NDC).
fn apply_view_handedness_fix(view: Mat4) -> Mat4 {
    let z_flip = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0));
    z_flip * view
}

/// World-to-view matrix (`glam`) for a [`SpaceDrawBatch`], matching the mesh pass MVP setup.
///
/// Applies scale clamping and handedness fix. Use this for clustered light eye-space transforms
/// so they are consistent with rasterized geometry.
pub fn view_matrix_glam_for_batch(batch: &SpaceDrawBatch) -> Mat4 {
    let mut vt = batch.view_transform;
    vt.scale = filter_scale(vt.scale);
    apply_view_handedness_fix(render_transform_to_matrix(&vt).inverse())
}

/// View–projection matrix (`glam`) for a [`SpaceDrawBatch`], matching the mesh pass MVP setup.
///
/// Uses the batch's `view_transform`, and for overlay batches optionally the
/// `overlay_projection_override` instead of the primary `proj`.
pub fn view_proj_glam_for_batch(
    batch: &SpaceDrawBatch,
    proj: &Matrix4<f32>,
    overlay_projection_override: Option<&ViewParams>,
) -> Mat4 {
    let view_mat = view_matrix_glam_for_batch(batch);
    let proj_na = batch
        .is_overlay
        .then_some(overlay_projection_override)
        .flatten()
        .map(|v| v.to_projection_matrix())
        .unwrap_or(*proj);
    matrix_na_to_glam(&proj_na) * view_mat
}

// ─── Shared homogeneous-clip AABB test ────────────────────────────────────────

/// Returns `true` if the world AABB may intersect the view frustum (homogeneous clip volume).
///
/// Uses WebGPU / Vulkan clip rules before perspective divide: `|x| ≤ w`, `|y| ≤ w`, `0 ≤ z ≤ w`.
/// The AABB is culled only when it lies entirely outside one of those half-spaces (tested on
/// all eight corners). Matches reverse-Z projection (visible depth still in `[0, w]` in clip).
pub(crate) fn world_aabb_visible_in_homogeneous_clip(
    view_proj: Mat4,
    world_min: Vec3,
    world_max: Vec3,
) -> bool {
    let xs = [world_min.x, world_max.x];
    let ys = [world_min.y, world_max.y];
    let zs = [world_min.z, world_max.z];

    // One transform per corner; plane tests reuse clip-space corners (no repeated `view_proj` muls).
    let mut clip_corners = [Vec4::ZERO; 8];
    let mut i = 0usize;
    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                clip_corners[i] = view_proj * Vec4::new(x, y, z, 1.0);
                i += 1;
            }
        }
    }

    // Behind: all corners have non-positive w (entire box is on or behind the eye plane).
    if clip_corners.iter().all(|p| p.w <= CLIP_EPS) {
        return false;
    }

    // Left  (x + w >= 0), Right (w - x >= 0), Bottom (y + w >= 0),
    // Top   (w - y >= 0), Near  (z >= 0),      Far    (z <= w).
    if clip_corners.iter().all(|p| p.x + p.w < -CLIP_EPS) {
        return false;
    }
    if clip_corners.iter().all(|p| p.w - p.x < -CLIP_EPS) {
        return false;
    }
    if clip_corners.iter().all(|p| p.y + p.w < -CLIP_EPS) {
        return false;
    }
    if clip_corners.iter().all(|p| p.w - p.y < -CLIP_EPS) {
        return false;
    }
    if clip_corners.iter().all(|p| p.z < -CLIP_EPS) {
        return false;
    }
    if clip_corners.iter().all(|p| p.z - p.w > CLIP_EPS) {
        return false;
    }

    true
}

// ─── Bound helpers ────────────────────────────────────────────────────────────

/// Returns `true` when bounds are degenerate: non-finite extents or all half-extents below
/// [`DEGENERATE_MESH_BOUNDS_EXTENT_EPS`].
///
/// In those cases frustum culling must not run: the volume collapses to a point (or is
/// undefined) and is not a reliable proxy for triangle coverage.
pub(crate) fn mesh_bounds_degenerate_for_cull(bounds: &RenderBoundingBox) -> bool {
    let e = bounds.extents;
    if !(e.x.is_finite() && e.y.is_finite() && e.z.is_finite()) {
        return true;
    }
    let m = e.x.abs().max(e.y.abs()).max(e.z.abs());
    m < DEGENERATE_MESH_BOUNDS_EXTENT_EPS
}

/// Largest absolute half-extent along any axis; `0` if extents are non-finite.
pub(crate) fn mesh_bounds_max_half_extent(bounds: &RenderBoundingBox) -> f32 {
    let e = bounds.extents;
    if !(e.x.is_finite() && e.y.is_finite() && e.z.is_finite()) {
        return 0.0;
    }
    e.x.abs().max(e.y.abs()).max(e.z.abs())
}

// ─── Tests: view-matrix helpers ───────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::batch::SpaceDrawBatch;
    use crate::render::pass::reverse_z_projection;
    use crate::scene::render_transform_to_matrix;
    use crate::shared::RenderTransform;
    use nalgebra::{Matrix4 as NaMatrix4, Point3, Quaternion, Vector3 as NaVector3};

    fn perspective_proj(aspect: f32) -> Matrix4<f32> {
        reverse_z_projection(aspect, 60f32.to_radians(), 0.1, 100.0)
    }

    /// [`view_proj_glam_for_batch`] must match `P * z_flip * V` for the same camera pose.
    #[test]
    fn view_proj_glam_for_batch_matches_z_flipped_look_at() {
        let view_na = NaMatrix4::look_at_rh(
            &Point3::new(0.0, 0.0, 5.0),
            &Point3::new(0.0, 0.0, 0.0),
            &NaVector3::new(0.0, 1.0, 0.0),
        );
        let proj = perspective_proj(1.0);
        let v_glam = matrix_na_to_glam(&view_na);
        let cam_glam = v_glam.inverse();
        let (_scale, rotation, translation) = cam_glam.to_scale_rotation_translation();
        let view_transform = RenderTransform {
            position: NaVector3::new(translation.x, translation.y, translation.z),
            scale: NaVector3::new(1.0, 1.0, 1.0),
            rotation: Quaternion::new(rotation.w, rotation.x, rotation.y, rotation.z),
        };
        let cam_from_rt = render_transform_to_matrix(&view_transform);
        let diff_cam = (cam_glam - cam_from_rt).to_cols_array();
        let max_cam = diff_cam.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_cam < 1e-4,
            "RenderTransform round-trip from look-at camera max abs {}",
            max_cam
        );

        let batch = SpaceDrawBatch {
            space_id: 0,
            is_overlay: false,
            view_transform,
            draws: vec![],
        };
        let vp_batch = view_proj_glam_for_batch(&batch, &proj, None);
        let z_flip = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0));
        let vp_ref = matrix_na_to_glam(&proj) * z_flip * v_glam;
        let diff = (vp_batch - vp_ref).to_cols_array();
        let max_abs = diff.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "view_proj batch vs P*z_flip*V max abs diff {}",
            max_abs
        );

        let v_batch = view_matrix_glam_for_batch(&batch);
        let v_expected = z_flip * v_glam;
        let dv = (v_batch - v_expected).to_cols_array();
        let max_v = dv.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_v < 1e-3,
            "view_matrix_glam_for_batch vs z_flip*V max abs diff {}",
            max_v
        );

        let p_glam = matrix_na_to_glam(&proj);
        let vp_from_parts = p_glam * v_batch;
        let dvp = (vp_batch - vp_from_parts).to_cols_array();
        let max_vp = dvp.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_vp < 1e-3,
            "view_proj should equal P * view_matrix max abs diff {}",
            max_vp
        );
    }

    /// Reference implementation: repeated `view_proj` multiplies per plane (legacy behavior).
    fn world_aabb_visible_in_homogeneous_clip_reference(
        view_proj: Mat4,
        world_min: Vec3,
        world_max: Vec3,
    ) -> bool {
        let xs = [world_min.x, world_max.x];
        let ys = [world_min.y, world_max.y];
        let zs = [world_min.z, world_max.z];
        let mut all_w_nonpositive = true;
        'behind: for &x in &xs {
            for &y in &ys {
                for &z in &zs {
                    let clip = view_proj * Vec4::new(x, y, z, 1.0);
                    if clip.w > CLIP_EPS {
                        all_w_nonpositive = false;
                        break 'behind;
                    }
                }
            }
        }
        if all_w_nonpositive {
            return false;
        }
        fn corners_all<F>(
            xs: &[f32; 2],
            ys: &[f32; 2],
            zs: &[f32; 2],
            view_proj: Mat4,
            pred: F,
        ) -> bool
        where
            F: Fn(Vec4) -> bool,
        {
            xs.iter().all(|&x| {
                ys.iter().all(|&y| {
                    zs.iter()
                        .all(|&z| pred(view_proj * Vec4::new(x, y, z, 1.0)))
                })
            })
        }
        if corners_all(&xs, &ys, &zs, view_proj, |p| p.x + p.w < -CLIP_EPS) {
            return false;
        }
        if corners_all(&xs, &ys, &zs, view_proj, |p| p.w - p.x < -CLIP_EPS) {
            return false;
        }
        if corners_all(&xs, &ys, &zs, view_proj, |p| p.y + p.w < -CLIP_EPS) {
            return false;
        }
        if corners_all(&xs, &ys, &zs, view_proj, |p| p.w - p.y < -CLIP_EPS) {
            return false;
        }
        if corners_all(&xs, &ys, &zs, view_proj, |p| p.z < -CLIP_EPS) {
            return false;
        }
        if corners_all(&xs, &ys, &zs, view_proj, |p| p.z - p.w > CLIP_EPS) {
            return false;
        }
        true
    }

    #[test]
    fn homogeneous_clip_aabb_matches_reference_random_boxes() {
        let mut t = 0xC0FFEEu32;
        let mut rnd = || {
            t = t.wrapping_mul(1664525).wrapping_add(1013904223);
            (t as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        let proj = perspective_proj(1.0);
        let view = NaMatrix4::look_at_rh(
            &Point3::new(0.0, 0.0, 5.0),
            &Point3::new(0.0, 0.0, 0.0),
            &NaVector3::new(0.0, 1.0, 0.0),
        );
        let vp = matrix_na_to_glam(&(proj * view));
        for _ in 0..300 {
            let p0 = Vec3::new(rnd() * 3.0, rnd() * 3.0, rnd() * 3.0);
            let p1 = Vec3::new(rnd() * 3.0, rnd() * 3.0, rnd() * 3.0);
            let mn = p0.min(p1);
            let mx = p0.max(p1);
            let a = world_aabb_visible_in_homogeneous_clip(vp, mn, mx);
            let b = world_aabb_visible_in_homogeneous_clip_reference(vp, mn, mx);
            assert_eq!(a, b, "min={:?} max={:?}", mn, mx);
        }
    }
}
