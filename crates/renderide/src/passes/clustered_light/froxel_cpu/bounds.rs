use glam::{Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::cpu_parallelism::{current_reference_worker_count, reference_worker_count};
use crate::gpu::GpuLight;
use crate::world_mesh::cluster::{ClusterFrameParams, TILE_SIZE, sanitize_cluster_clip_planes};

use super::types::FroxelLayout;

/// Point light tag in [`GpuLight::light_type`].
pub(super) const LIGHT_TYPE_POINT: u32 = 0;
/// Directional light tag in [`GpuLight::light_type`].
pub(super) const LIGHT_TYPE_DIRECTIONAL: u32 = 1;
/// Spot light tag in [`GpuLight::light_type`].
pub(super) const LIGHT_TYPE_SPOT: u32 = 2;
/// Cluster AABB padding used by the clustered-light compute shader.
const CLUSTER_BOUNDARY_EPSILON: f32 = 0.00001;
/// Largest half-angle cosine used by spotlight culling, equivalent to a 0.5 degree half-angle.
const SPOT_CULL_MIN_COS_HALF: f32 = 0.999_961_9;
/// Half-angle cosine below which spotlights use range-sphere culling to avoid wide-cone misses.
const SPOT_CULL_WIDE_COS_HALF: f32 = 0.5;
/// Small distance pad for cone/sphere boundary comparisons.
const SPOT_CULL_DISTANCE_EPSILON: f32 = 0.00001;
/// Froxel spheres assigned to one Rayon leaf when building spotlight culling bounds.
const FROXEL_SPHERE_PARALLEL_CHUNK_FROXELS: usize = 512;
/// Froxel count at which spotlight culling sphere generation may use Rayon.
pub(super) const FROXEL_SPHERE_PARALLEL_MIN_FROXELS: usize =
    FROXEL_SPHERE_PARALLEL_CHUNK_FROXELS * 2;

#[derive(Clone, Copy, Debug)]
pub(super) struct FroxelBounds {
    /// First X froxel.
    pub(super) x0: u32,
    /// Last X froxel.
    pub(super) x1: u32,
    /// First Y froxel.
    pub(super) y0: u32,
    /// Last Y froxel.
    pub(super) y1: u32,
    /// First Z froxel.
    pub(super) z0: u32,
    /// Last Z froxel.
    pub(super) z1: u32,
}

/// View-space AABB for one froxel.
#[derive(Clone, Copy, Debug)]
struct FroxelAabb {
    /// Minimum view-space corner.
    min: Vec3,
    /// Maximum view-space corner.
    max: Vec3,
}

impl FroxelAabb {
    /// Returns a sphere that fully contains this AABB.
    fn bounding_sphere(self) -> FroxelSphere {
        let center = (self.min + self.max) * 0.5;
        FroxelSphere {
            center,
            radius: (self.max - center).length(),
        }
    }
}

/// Bounding sphere that fully contains one froxel.
#[derive(Clone, Copy, Debug)]
pub(super) struct FroxelSphere {
    /// Sphere center in view space.
    pub(super) center: Vec3,
    /// Sphere radius in view-space units.
    pub(super) radius: f32,
}

/// Flat per-eye froxel sphere cache used by spotlight fine culling.
#[derive(Clone, Debug, Default)]
pub(super) struct EyeFroxelSpheres {
    spheres: Vec<FroxelSphere>,
    clusters_per_eye: usize,
}

impl EyeFroxelSpheres {
    fn empty() -> Self {
        Self::default()
    }

    fn new(spheres: Vec<FroxelSphere>, clusters_per_eye: usize) -> Self {
        Self {
            spheres,
            clusters_per_eye,
        }
    }

    fn eye(&self, eye_idx: usize) -> &[FroxelSphere] {
        if self.clusters_per_eye == 0 {
            return &[];
        }
        let Some(start) = eye_idx.checked_mul(self.clusters_per_eye) else {
            return &[];
        };
        let Some(end) = start.checked_add(self.clusters_per_eye) else {
            return &[];
        };
        self.spheres.get(start..end).unwrap_or(&[])
    }
}

/// Conservative assignment data for one point or spot light.
#[derive(Clone, Copy, Debug)]
pub(super) struct BoundedLight {
    /// Inclusive froxel range touched by the light's broad range sphere.
    pub(super) bounds: FroxelBounds,
    /// Spotlight-specific fine culling data.
    pub(super) spot: Option<SpotCull>,
}

/// View-space spotlight cone used for per-froxel fine culling.
#[derive(Clone, Copy, Debug)]
pub(super) struct SpotCull {
    /// Cone apex in view space.
    pub(super) apex: Vec3,
    /// Normalized cone axis in view space.
    pub(super) axis: Vec3,
    /// Cosine of the spotlight half-angle.
    pub(super) cos_half: f32,
    /// Light range in view-space units.
    pub(super) range: f32,
}

impl SpotCull {
    /// Returns whether this spotlight can affect a froxel bounding sphere.
    pub(super) fn intersects_froxel_sphere(self, sphere: FroxelSphere) -> bool {
        if !sphere_sphere_intersect(self.apex, self.range, sphere.center, sphere.radius) {
            return false;
        }

        let clamped_cos_half = self.cos_half.clamp(0.0, SPOT_CULL_MIN_COS_HALF);
        if clamped_cos_half <= SPOT_CULL_WIDE_COS_HALF {
            return true;
        }

        let sin_half = (1.0 - clamped_cos_half * clamped_cos_half).max(0.0).sqrt();
        let offset = sphere.center - self.apex;
        let axis_dist = offset.dot(self.axis);
        if axis_dist < -sphere.radius || axis_dist > self.range + sphere.radius {
            return false;
        }

        let lateral_len = (offset.length_squared() - axis_dist * axis_dist)
            .max(0.0)
            .sqrt();
        let closest_cone_distance = clamped_cos_half * lateral_len - axis_dist * sin_half;
        closest_cone_distance <= sphere.radius + SPOT_CULL_DISTANCE_EPSILON
    }
}

/// Computes conservative froxel bounds for point and spot lights.
pub(super) fn light_froxel_bounds(
    light: &GpuLight,
    view: Mat4,
    proj: Mat4,
    view_scale: f32,
    layout: FroxelLayout,
    params: ClusterFrameParams,
) -> Option<BoundedLight> {
    let center = transform_point(view, Vec3::from_array(light.position));
    let radius = (light.range * view_scale).max(0.0);
    if radius <= 0.0 || !radius.is_finite() {
        return None;
    }

    let spot = if light.light_type == LIGHT_TYPE_SPOT {
        let axis = transform_vector(view, Vec3::from_array(light.direction))
            .try_normalize()
            .unwrap_or(Vec3::Z);
        Some(SpotCull {
            apex: center,
            axis,
            cos_half: light.spot_cos_half_angle,
            range: radius,
        })
    } else if light.light_type != LIGHT_TYPE_POINT {
        return None;
    } else {
        None
    };

    let (near, far) = params.sanitized_clip_planes();
    let raw_nearest_depth = -(center.z + radius);
    let raw_farthest_depth = -(center.z - radius);
    if raw_farthest_depth < near || raw_nearest_depth > far {
        return None;
    }

    let nearest_depth = raw_nearest_depth.clamp(near, far);
    let farthest_depth = raw_farthest_depth.clamp(near, far);
    let z0 = cluster_z_from_depth(nearest_depth, near, far, layout.cluster_count_z);
    let z1 = cluster_z_from_depth(farthest_depth, near, far, layout.cluster_count_z);
    let (x0, x1, y0, y1) = projected_sphere_xy_bounds(center, radius, proj, near, far, layout)?;

    Some(BoundedLight {
        bounds: FroxelBounds {
            x0,
            x1,
            y0,
            y1,
            z0: z0.min(z1),
            z1: z0.max(z1),
        },
        spot,
    })
}

/// Returns whether two spheres overlap.
fn sphere_sphere_intersect(a_center: Vec3, a_radius: f32, b_center: Vec3, b_radius: f32) -> bool {
    let radius = a_radius + b_radius;
    a_center.distance_squared(b_center) <= radius * radius
}

/// Builds froxel bounding spheres for every eye.
pub(super) fn build_eye_froxel_spheres(
    lights: &[GpuLight],
    eye_params: &[ClusterFrameParams],
    layouts: &[FroxelLayout],
) -> Option<EyeFroxelSpheres> {
    if !lights
        .iter()
        .any(|light| light.light_type == LIGHT_TYPE_SPOT)
    {
        return Some(EyeFroxelSpheres::empty());
    }

    profiling::scope!("clustered_light::build_eye_froxel_spheres");
    let Some(&first_layout) = layouts.first() else {
        return Some(EyeFroxelSpheres::empty());
    };
    let clusters_per_eye = first_layout.cluster_count()?;
    let total_spheres = clusters_per_eye.checked_mul(eye_params.len())?;
    let mut all_spheres = Vec::with_capacity(total_spheres);
    for (params, &layout) in eye_params.iter().zip(layouts.iter()) {
        if layout.cluster_count()? != clusters_per_eye {
            return None;
        }
        all_spheres.extend(froxel_bounding_spheres(*params, layout)?);
    }
    Some(EyeFroxelSpheres::new(all_spheres, clusters_per_eye))
}

/// Returns one eye's froxel spheres, or an empty slice when no spotlights need them.
pub(super) fn eye_froxel_spheres(
    froxel_spheres_by_eye: &EyeFroxelSpheres,
    eye_idx: usize,
) -> &[FroxelSphere] {
    froxel_spheres_by_eye.eye(eye_idx)
}

/// Builds bounding spheres for every froxel in one eye.
fn froxel_bounding_spheres(
    params: ClusterFrameParams,
    layout: FroxelLayout,
) -> Option<Vec<FroxelSphere>> {
    let inv_proj = params.proj.inverse();
    if !inv_proj.is_finite() {
        return None;
    }

    let cluster_count = layout.cluster_count()?;
    let depth_params = cluster_depth_params(params, layout);
    if should_parallelize_froxel_spheres(cluster_count) {
        return froxel_bounding_spheres_parallel(
            params,
            inv_proj,
            layout,
            &depth_params,
            cluster_count,
        );
    }
    froxel_bounding_spheres_serial(params, inv_proj, layout, &depth_params, cluster_count)
}

fn froxel_bounding_spheres_serial(
    params: ClusterFrameParams,
    inv_proj: Mat4,
    layout: FroxelLayout,
    depth_params: &[ClusterDepthParams],
    cluster_count: usize,
) -> Option<Vec<FroxelSphere>> {
    let mut spheres = Vec::with_capacity(cluster_count);
    for cluster_idx in 0..cluster_count {
        let (x, y, depth_params) = cluster_coords_from_linear(layout, depth_params, cluster_idx)?;
        spheres.push(cluster_aabb(params, inv_proj, layout, x, y, depth_params)?.bounding_sphere());
    }
    Some(spheres)
}

fn froxel_bounding_spheres_parallel(
    params: ClusterFrameParams,
    inv_proj: Mat4,
    layout: FroxelLayout,
    depth_params: &[ClusterDepthParams],
    cluster_count: usize,
) -> Option<Vec<FroxelSphere>> {
    profiling::scope!("clustered_light::build_eye_froxel_spheres_parallel");
    (0..cluster_count)
        .into_par_iter()
        .with_min_len(FROXEL_SPHERE_PARALLEL_CHUNK_FROXELS)
        .map(|cluster_idx| {
            let (x, y, depth_params) =
                cluster_coords_from_linear(layout, depth_params, cluster_idx)?;
            cluster_aabb(params, inv_proj, layout, x, y, depth_params)
                .map(FroxelAabb::bounding_sphere)
        })
        .collect()
}

fn cluster_depth_params(
    params: ClusterFrameParams,
    layout: FroxelLayout,
) -> Vec<ClusterDepthParams> {
    (0..layout.cluster_count_z)
        .map(|z| {
            let depth_bounds = cluster_z_depth_bounds(
                z,
                layout.cluster_count_z,
                params.near_clip,
                params.far_clip,
            );
            ClusterDepthParams {
                cluster_z: z,
                near_depth: depth_bounds.0,
                far_depth: depth_bounds.1,
            }
        })
        .collect()
}

fn cluster_coords_from_linear(
    layout: FroxelLayout,
    depth_params: &[ClusterDepthParams],
    cluster_idx: usize,
) -> Option<(u32, u32, ClusterDepthParams)> {
    let cluster_count_x = layout.cluster_count_x as usize;
    let cluster_count_y = layout.cluster_count_y as usize;
    let xy_count = cluster_count_x.saturating_mul(cluster_count_y).max(1);
    let z = cluster_idx / xy_count;
    let xy = cluster_idx % xy_count;
    let y = xy / cluster_count_x.max(1);
    let x = xy % cluster_count_x.max(1);
    let depth_params = depth_params.get(z).copied()?;
    Some((x as u32, y as u32, depth_params))
}

/// Returns whether spotlight froxel sphere generation should fan out over Rayon.
#[inline]
pub(super) fn should_parallelize_froxel_spheres(froxel_count: usize) -> bool {
    should_parallelize_froxel_spheres_with_workers(froxel_count, current_reference_worker_count())
}

/// Returns whether spotlight froxel sphere generation should fan out for a known worker count.
#[inline]
pub(super) const fn should_parallelize_froxel_spheres_with_workers(
    froxel_count: usize,
    worker_count: usize,
) -> bool {
    reference_worker_count(worker_count) > 1 && froxel_count >= FROXEL_SPHERE_PARALLEL_MIN_FROXELS
}

/// Cached logarithmic clustered-depth bounds for one Z slice.
#[derive(Clone, Copy, Debug)]
struct ClusterDepthParams {
    cluster_z: u32,
    near_depth: f32,
    far_depth: f32,
}

/// Computes the view-space AABB for one froxel.
fn cluster_aabb(
    params: ClusterFrameParams,
    inv_proj: Mat4,
    layout: FroxelLayout,
    cluster_x: u32,
    cluster_y: u32,
    depth_params: ClusterDepthParams,
) -> Option<FroxelAabb> {
    let ClusterDepthParams {
        cluster_z,
        near_depth,
        far_depth,
    } = depth_params;
    let tile_near = -near_depth;
    let tile_far = -far_depth;

    let w = layout.viewport_width as f32;
    let h = layout.viewport_height as f32;
    let px_min = cluster_x.saturating_mul(TILE_SIZE) as f32;
    let px_max = cluster_x
        .saturating_add(1)
        .saturating_mul(TILE_SIZE)
        .min(layout.viewport_width) as f32;
    let py_min = cluster_y.saturating_mul(TILE_SIZE) as f32;
    let py_max = cluster_y
        .saturating_add(1)
        .saturating_mul(TILE_SIZE)
        .min(layout.viewport_height) as f32;
    let ndc_left = 2.0 * px_min / w - 1.0;
    let ndc_right = 2.0 * px_max / w - 1.0;
    let ndc_top = 1.0 - 2.0 * py_min / h;
    let ndc_bottom = 1.0 - 2.0 * py_max / h;

    let ndc_bl = Vec2::new(ndc_left, ndc_bottom);
    let ndc_br = Vec2::new(ndc_right, ndc_bottom);
    let ndc_tl = Vec2::new(ndc_left, ndc_top);
    let ndc_tr = Vec2::new(ndc_right, ndc_top);

    let points = [
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_bl, tile_near)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_br, tile_near)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tl, tile_near)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tr, tile_near)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_bl, tile_far)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_br, tile_far)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tl, tile_far)?,
        view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tr, tile_far)?,
    ];
    let (mut min_v, mut max_v) = min_max_points(&points);

    if cluster_z == 0 {
        let camera_clip = params.proj * Vec4::new(0.0, 0.0, 0.0, 1.0);
        if camera_clip.w.abs() > 1e-8 {
            let zero_points = [
                view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_bl, 0.0)?,
                view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_br, 0.0)?,
                view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tl, 0.0)?,
                view_point_at_ndc_xy_and_z(params.proj, inv_proj, ndc_tr, 0.0)?,
            ];
            let (zero_min, zero_max) = min_max_points(&zero_points);
            min_v = min_v.min(zero_min);
            max_v = max_v.max(zero_max);
        } else {
            min_v = min_v.min(Vec3::ZERO);
            max_v = max_v.max(Vec3::ZERO);
        }
    }

    let extent = max_v - min_v;
    let max_extent = extent.x.max(extent.y).max(extent.z).max(1.0);
    let pad = max_extent * CLUSTER_BOUNDARY_EPSILON;
    Some(FroxelAabb {
        min: min_v - Vec3::splat(pad),
        max: max_v + Vec3::splat(pad),
    })
}

/// Returns the component-wise min and max for a fixed set of view-space points.
fn min_max_points<const N: usize>(points: &[Vec3; N]) -> (Vec3, Vec3) {
    let mut min_v = Vec3::splat(f32::INFINITY);
    let mut max_v = Vec3::splat(f32::NEG_INFINITY);
    for &point in points {
        min_v = min_v.min(point);
        max_v = max_v.max(point);
    }
    (min_v, max_v)
}

/// Computes logarithmic clustered-depth bounds for one Z slice.
fn cluster_z_depth_bounds(
    cluster_z: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
) -> (f32, f32) {
    let z_count = cluster_count_z.max(1);
    let z = cluster_z.min(z_count - 1);
    let (near_safe, far_safe) = sanitize_cluster_clip_planes(near_clip, far_clip);
    let ratio = (far_safe / near_safe).max(1.0 + f32::EPSILON);
    let zf = z as f32;
    let num_z = z_count as f32;
    (
        near_safe * ratio.powf(zf / num_z),
        near_safe * ratio.powf((zf + 1.0) / num_z),
    )
}

/// Reconstructs a view-space point from NDC X/Y and view-space Z.
fn view_point_at_ndc_xy_and_z(
    proj: Mat4,
    inv_proj: Mat4,
    ndc_xy: Vec2,
    view_z: f32,
) -> Option<Vec3> {
    ndc_z_from_view_z(proj, view_z)
        .and_then(|ndc_z| ndc_to_view(inv_proj, Vec3::new(ndc_xy.x, ndc_xy.y, ndc_z)))
}

/// Reconstructs a view-space point from NDC coordinates.
fn ndc_to_view(inv_proj: Mat4, ndc: Vec3) -> Option<Vec3> {
    let clip = inv_proj * Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
    let point = if clip.w.abs() <= 1e-8 {
        clip.truncate()
    } else {
        clip.truncate() / clip.w
    };
    point.is_finite().then_some(point)
}

/// Projects view-space Z to NDC Z using the frame projection.
fn ndc_z_from_view_z(proj: Mat4, view_z: f32) -> Option<f32> {
    let clip = proj * Vec4::new(0.0, 0.0, view_z, 1.0);
    if clip.w.abs() <= 1e-8 || !clip.w.is_finite() {
        return None;
    }
    let ndc_z = clip.z / clip.w;
    ndc_z.is_finite().then_some(ndc_z)
}

/// Transforms a world-space point by `matrix`.
fn transform_point(matrix: Mat4, point: Vec3) -> Vec3 {
    (matrix * point.extend(1.0)).truncate()
}

/// Transforms a world-space vector by `matrix`.
fn transform_vector(matrix: Mat4, vector: Vec3) -> Vec3 {
    (matrix * vector.extend(0.0)).truncate()
}

/// Maps positive depth to a logarithmic clustered Z slice.
fn cluster_z_from_depth(depth: f32, near_clip: f32, far_clip: f32, cluster_count_z: u32) -> u32 {
    let z_count = cluster_count_z.max(1);
    let (near_safe, far_safe) = sanitize_cluster_clip_planes(near_clip, far_clip);
    let ratio = (far_safe / near_safe).max(1.0 + f32::EPSILON);
    let z = (depth.clamp(near_safe, far_safe) / near_safe).log(ratio) * z_count as f32;
    z.clamp(0.0, z_count.saturating_sub(1) as f32) as u32
}

/// Computes conservative screen-space froxel bounds for a view-space sphere.
fn projected_sphere_xy_bounds(
    center: Vec3,
    radius: f32,
    proj: Mat4,
    near: f32,
    far: f32,
    layout: FroxelLayout,
) -> Option<(u32, u32, u32, u32)> {
    let near_z = (center.z + radius).min(-near).max(-far);
    let far_z = (center.z - radius).min(-near).max(-far);
    let mut ndc_min = Vec2::splat(f32::INFINITY);
    let mut ndc_max = Vec2::splat(f32::NEG_INFINITY);
    for z in [near_z, far_z] {
        for x_sign in [-1.0, 1.0] {
            for y_sign in [-1.0, 1.0] {
                let p = Vec3::new(center.x + radius * x_sign, center.y + radius * y_sign, z);
                let ndc = project_view_point(proj, p)?;
                ndc_min = ndc_min.min(ndc);
                ndc_max = ndc_max.max(ndc);
            }
        }
    }
    let x0 = ndc_x_to_cluster(ndc_min.x, layout);
    let x1 = ndc_x_to_cluster(ndc_max.x, layout);
    let y0 = ndc_y_to_cluster(ndc_max.y, layout);
    let y1 = ndc_y_to_cluster(ndc_min.y, layout);
    Some((x0.min(x1), x0.max(x1), y0.min(y1), y0.max(y1)))
}

/// Projects a view-space point into normalized device coordinates.
fn project_view_point(proj: Mat4, point: Vec3) -> Option<Vec2> {
    let clip = proj * Vec4::new(point.x, point.y, point.z, 1.0);
    if clip.w.abs() <= 1e-8 || !clip.w.is_finite() {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    (ndc.x.is_finite() && ndc.y.is_finite()).then(|| ndc.truncate())
}

/// Converts NDC X to a froxel coordinate.
fn ndc_x_to_cluster(ndc_x: f32, layout: FroxelLayout) -> u32 {
    let px = ((ndc_x.clamp(-1.0, 1.0) + 1.0) * 0.5 * layout.viewport_width as f32).floor();
    (px as u32 / TILE_SIZE).min(layout.cluster_count_x - 1)
}

/// Converts NDC Y to a froxel coordinate with top-left screen origin.
fn ndc_y_to_cluster(ndc_y: f32, layout: FroxelLayout) -> u32 {
    let py = ((1.0 - ndc_y.clamp(-1.0, 1.0)) * 0.5 * layout.viewport_height as f32).floor();
    (py as u32 / TILE_SIZE).min(layout.cluster_count_y - 1)
}
