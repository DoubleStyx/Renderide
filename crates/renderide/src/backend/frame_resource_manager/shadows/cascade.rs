//! Camera-fitted directional shadow cascade projections.
//!
//! Each cascade fits the bounding sphere of a camera-frustum depth slice, texel-snapped so it does
//! not shimmer as the camera moves.

use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::camera::{HostCameraFrame, WorldProjectionSet, view_matrix_for_world_mesh_render_space};
use crate::scene::SceneSpaceRead;

/// Blend between uniform and logarithmic cascade splits (0 uniform, 1 logarithmic).
const CASCADE_SPLIT_LAMBDA: f32 = 0.75;
/// Extra light-space depth pulled toward the light to capture occluders above each slice, as a
/// multiple of the cascade bounding radius.
const CASCADE_CASTER_PULLBACK: f32 = 1.0;
/// Bounding-radius quantization steps across one cascade radius.
///
/// The radius is derived from live frustum corners, so it moves whenever the camera's clip planes,
/// field of view, or aspect move, and a resized cascade resamples the whole shadow map. Quantizing
/// relative to the radius keeps a cascade one fixed size across the small per-frame changes that
/// dominate normal movement, matching how a stable fit only ever steps in texels. An absolute step
/// cannot do this: 1/16 of a world unit is below the noise floor of a 50 unit cascade.
const CASCADE_RADIUS_STEPS: f32 = 32.0;

/// Smallest absolute bounding-radius step, for cascades small enough that the relative step vanishes.
const CASCADE_RADIUS_MIN_STEP: f32 = 1.0 / 16.0;

/// World-space camera frustum used to fit directional shadow cascades to what the camera sees.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShadowCameraFit {
    /// Near-plane world corners (bl, br, tr, tl).
    near_corners: [Vec3; 4],
    /// Far-plane world corners (bl, br, tr, tl).
    far_corners: [Vec3; 4],
    /// View-space near distance in world units.
    near: f32,
    /// View-space far distance in world units.
    far: f32,
}

impl ShadowCameraFit {
    /// Builds a fit from the scene main-camera world-to-clip, the same matrix the forward and cull
    /// paths use. Returns [`None`] when the projection is not invertible.
    pub(crate) fn from_scene_camera<S>(
        scene: &S,
        viewport_px: (u32, u32),
        host_camera: &HostCameraFrame,
    ) -> Option<Self>
    where
        S: SceneSpaceRead + ?Sized,
    {
        let projections = WorldProjectionSet::from_scene_host(scene, viewport_px, host_camera);
        let view = host_camera.explicit_world_to_view().unwrap_or_else(|| {
            scene.active_main_space().map_or(Mat4::IDENTITY, |space| {
                view_matrix_for_world_mesh_render_space(scene, space)
            })
        });
        Self::from_world_to_clip(
            projections.world_proj * view,
            projections.clip.near,
            projections.clip.far,
        )
    }

    /// Builds a fit from a world-to-clip matrix and its positive view-space clip distances.
    ///
    /// `near`/`far` are the view-space distances that map to the reverse-Z near (ndc z = 1) and far
    /// (ndc z = 0) planes.
    fn from_world_to_clip(world_to_clip: Mat4, near: f32, far: f32) -> Option<Self> {
        let inv = world_to_clip.inverse();
        let near_corners = [
            unproject(inv, -1.0, -1.0, 1.0),
            unproject(inv, 1.0, -1.0, 1.0),
            unproject(inv, 1.0, 1.0, 1.0),
            unproject(inv, -1.0, 1.0, 1.0),
        ];
        let far_corners = [
            unproject(inv, -1.0, -1.0, 0.0),
            unproject(inv, 1.0, -1.0, 0.0),
            unproject(inv, 1.0, 1.0, 0.0),
            unproject(inv, -1.0, 1.0, 0.0),
        ];
        if near_corners
            .iter()
            .chain(far_corners.iter())
            .any(|c| !c.is_finite())
        {
            return None;
        }
        let near = near.max(1e-3);
        let far = far.max(near + 1e-3);
        Some(Self {
            near_corners,
            far_corners,
            near,
            far,
        })
    }

    /// View-space near distance in world units.
    pub(crate) fn near(&self) -> f32 {
        self.near
    }

    /// View-space far distance in world units.
    pub(crate) fn far(&self) -> f32 {
        self.far
    }

    /// Corners of the sub-frustum spanning view-space depths `[d0, d1]`.
    ///
    /// Position along a frustum edge is affine in view depth, so a slice corner is a lerp of the
    /// near and far corner.
    fn slice_corners(&self, d0: f32, d1: f32) -> [Vec3; 8] {
        let span = (self.far - self.near).max(1e-6);
        let f0 = ((d0 - self.near) / span).clamp(0.0, 1.0);
        let f1 = ((d1 - self.near) / span).clamp(0.0, 1.0);
        let mut corners = [Vec3::ZERO; 8];
        for i in 0..4 {
            let edge = self.far_corners[i] - self.near_corners[i];
            corners[i] = self.near_corners[i] + edge * f0;
            corners[i + 4] = self.near_corners[i] + edge * f1;
        }
        corners
    }
}

#[inline]
fn unproject(clip_to_world: Mat4, x: f32, y: f32, z: f32) -> Vec3 {
    clip_to_world.project_point3(Vec3::new(x, y, z))
}

/// One cascade's split range in view-space world units.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CascadeSplit {
    /// Nearest view-space distance covered by the cascade.
    pub(crate) near: f32,
    /// Farthest view-space distance covered by the cascade.
    pub(crate) far: f32,
}

/// Split range for cascade `index` of `count` over `[near, far]`, blending uniform and logarithmic
/// splits by [`CASCADE_SPLIT_LAMBDA`].
pub(crate) fn cascade_split(near: f32, far: f32, index: u32, count: u32) -> CascadeSplit {
    let count = count.max(1);
    let near = near.max(1e-3);
    let far = far.max(near + 1e-3);
    let ratio = (far / near).max(1.0);
    let split_at = |i: u32| -> f32 {
        let p = i as f32 / count as f32;
        let log = near * ratio.powf(p);
        let uniform = near + (far - near) * p;
        (uniform + (log - uniform) * CASCADE_SPLIT_LAMBDA).clamp(near, far)
    };
    CascadeSplit {
        near: split_at(index.min(count)),
        far: split_at((index + 1).min(count)),
    }
}

/// Rounds a cascade bounding radius up to a step sized for its own magnitude.
///
/// The step is a power of two near `radius / CASCADE_RADIUS_STEPS`, so it holds constant across a
/// wide band of radii. A step derived directly from the radius would scale with its own input and
/// barely quantize at all, leaving the cascade tracking every wobble it was meant to absorb.
fn quantize_cascade_radius(radius: f32) -> f32 {
    let radius = radius.max(1e-3);
    let step = (radius / CASCADE_RADIUS_STEPS)
        .max(CASCADE_RADIUS_MIN_STEP)
        .log2()
        .floor()
        .exp2();
    ((radius / step).ceil() * step).max(1e-3)
}

/// Builds a camera-fitted, texel-snapped orthographic cascade projection (world to shadow clip).
pub(crate) fn directional_cascade_view_proj(
    direction: Vec3,
    up: Vec3,
    fit: &ShadowCameraFit,
    split: CascadeSplit,
    resolution: u32,
) -> Mat4 {
    let corners = fit.slice_corners(split.near, split.far);
    let mut center = Vec3::ZERO;
    for c in &corners {
        center += *c;
    }
    center /= corners.len() as f32;
    let mut radius = 0.0f32;
    for c in &corners {
        radius = radius.max(center.distance(*c));
    }
    radius = quantize_cascade_radius(radius);

    let pullback = radius * CASCADE_CASTER_PULLBACK;
    let eye = center - direction * (radius + pullback);
    let view = Mat4::look_at_rh(eye, center, up);
    let mut proj = Mat4::orthographic_rh(
        -radius,
        radius,
        -radius,
        radius,
        0.0,
        2.0 * radius + pullback,
    );

    // Texel snap: quantize the projected world origin to the texel grid. Ortho leaves w = 1, so
    // clip xy need no perspective divide.
    let res = resolution.max(1) as f32;
    let origin = proj * view * Vec4::new(0.0, 0.0, 0.0, 1.0);
    let origin_xy = Vec2::new(origin.x, origin.y) * (res * 0.5);
    let offset = (origin_xy.round() - origin_xy) * (2.0 / res);
    proj.w_axis.x += offset.x;
    proj.w_axis.y += offset.y;
    proj * view
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3, Vec4};

    use super::{ShadowCameraFit, cascade_split, directional_cascade_view_proj};
    use crate::camera::{apply_view_handedness_fix, reverse_z_perspective};

    fn look_down_neg_z_fit(near: f32, far: f32) -> ShadowCameraFit {
        // Camera at the origin looking down -Z with a 60 degree vertical fov, matching the engine's
        // reverse-Z perspective and handedness fix.
        let proj = reverse_z_perspective(16.0 / 9.0, 60f32.to_radians(), near, far);
        let view = apply_view_handedness_fix(Mat4::IDENTITY);
        ShadowCameraFit::from_world_to_clip(proj * view, near, far).expect("finite fit")
    }

    #[test]
    fn cascade_splits_cover_range_monotonically() {
        let near = 0.1;
        let far = 200.0;
        let count = 4;
        let mut prev_far = near;
        for index in 0..count {
            let split = cascade_split(near, far, index, count);
            assert!(split.far > split.near, "cascade {index} is degenerate");
            assert!(
                (split.near - prev_far).abs() < 1e-3,
                "cascade {index} leaves a gap"
            );
            prev_far = split.far;
        }
        assert!(
            (prev_far - far).abs() < 1e-2,
            "last cascade must reach the far plane"
        );
    }

    #[test]
    fn nearer_cascade_fits_tighter_than_farther_cascade() {
        let fit = look_down_neg_z_fit(0.1, 200.0);
        let count = 4;
        let near_split = cascade_split(fit.near(), 175.0, 0, count);
        let far_split = cascade_split(fit.near(), 175.0, count - 1, count);
        let near_corners = fit.slice_corners(near_split.near, near_split.far);
        let far_corners = fit.slice_corners(far_split.near, far_split.far);
        let radius = |corners: [Vec3; 8]| {
            let center = corners.iter().copied().sum::<Vec3>() / 8.0;
            corners
                .iter()
                .fold(0.0f32, |acc, c| acc.max(center.distance(*c)))
        };
        assert!(radius(near_corners) < radius(far_corners));
    }

    #[test]
    fn near_slice_center_projects_inside_the_cascade() {
        let fit = look_down_neg_z_fit(0.1, 200.0);
        let split = cascade_split(fit.near(), 175.0, 0, 4);
        // Sun straight down, up along +Z (parallel-safe).
        let view_proj =
            directional_cascade_view_proj(Vec3::new(0.0, -1.0, 0.0), Vec3::Z, &fit, split, 2048);
        // A point in the middle of the near slice must land inside the shadow map bounds.
        let mid_depth = 0.5 * (split.near + split.far);
        let sample = fit.slice_corners(mid_depth, mid_depth);
        let center = sample.iter().copied().sum::<Vec3>() / 8.0;
        let clip = view_proj * Vec4::new(center.x, center.y, center.z, 1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(
            ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0,
            "xy out of bounds: {ndc:?}"
        );
        assert!(ndc.z >= 0.0 && ndc.z <= 1.0, "depth out of bounds: {ndc:?}");
    }

    /// The camera's clip planes move slightly frame to frame. A cascade that resizes with them
    /// resamples the whole shadow map, which is the shimmer people report while moving, and it
    /// changes the shadow view signature so the cached atlas layer re-renders every frame.
    #[test]
    fn clip_plane_wobble_collapses_to_one_cascade_size() {
        let mut sizes = Vec::new();
        for step in 0..24 {
            let jitter = step as f32 * 0.02;
            let fit = look_down_neg_z_fit(0.1 + jitter * 0.01, 200.0 + jitter);
            let split = cascade_split(fit.near(), 175.0, 0, 4);
            let proj = directional_cascade_view_proj(
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::Z,
                &fit,
                split,
                2048,
            );
            sizes.push(proj.x_axis.x);
        }
        sizes.sort_by(f32::total_cmp);
        sizes.dedup();

        assert!(
            sizes.len() <= 2,
            "cascade resized {} times across sub-unit clip wobble: {sizes:?}",
            sizes.len()
        );
    }

    #[test]
    fn radius_quantization_collapses_nearby_radii_onto_one_step() {
        // Step near 64/32 = 2, so this whole band shares one cascade size.
        let base = super::quantize_cascade_radius(64.5);

        assert!(base >= 64.5 && base <= 66.0, "base: {base}");
        assert_eq!(base, super::quantize_cascade_radius(64.5));
        assert_eq!(base, super::quantize_cascade_radius(65.0));
        assert_eq!(base, super::quantize_cascade_radius(65.9));
        // Small cascades keep a usable floor instead of collapsing to nothing.
        assert!(super::quantize_cascade_radius(2.0) >= 2.0);
        assert!(super::quantize_cascade_radius(0.0) > 0.0);
    }

    #[test]
    fn degenerate_projection_yields_no_fit() {
        assert!(ShadowCameraFit::from_world_to_clip(Mat4::ZERO, 0.1, 100.0).is_none());
    }
}
