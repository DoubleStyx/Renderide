//! Camera-portal view matrix construction.

use glam::{Mat4, Vec3, Vec4};

use crate::camera::{
    CameraClipPlanes, CameraProjectionKind, EyeView, HostCameraFrame, Viewport,
    clamp_desktop_fov_degrees, reverse_z_perspective,
};
use crate::shared::CameraPortalState;

const MATRIX_DETERMINANT_EPSILON: f32 = 1e-12;
const NORMAL_LENGTH_SQUARED_EPSILON: f32 = 1e-12;

/// Source camera data used to build a camera-portal render view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPortalSourceView {
    /// Current source world-to-view matrix.
    pub view: Mat4,
    /// Current source view-to-clip projection.
    pub proj: Mat4,
    /// Current source world-space camera position.
    pub world_position: Vec3,
    /// Effective source clip planes.
    pub clip: CameraClipPlanes,
    /// Source projection kind.
    pub projection_kind: CameraProjectionKind,
    /// Viewport used to build the source projection.
    pub viewport: Viewport,
    /// Source vertical field of view in degrees when the projection can be rebuilt symmetrically.
    pub fov_degrees: f32,
    /// Whether this source can rebuild its projection from FOV/aspect/clip values.
    pub can_rebuild_symmetric_perspective: bool,
}

impl CameraPortalSourceView {
    /// Builds source camera data from a resolved eye view and clip-plane pair.
    #[inline]
    pub const fn new(
        eye: EyeView,
        clip: CameraClipPlanes,
        projection_kind: CameraProjectionKind,
    ) -> Self {
        Self {
            view: eye.view,
            proj: eye.proj,
            world_position: eye.world_position,
            clip,
            projection_kind,
            viewport: Viewport::new(1, 1),
            fov_degrees: 60.0,
            can_rebuild_symmetric_perspective: false,
        }
    }

    /// Builds source data for the main symmetric perspective desktop view.
    #[inline]
    pub const fn symmetric_perspective(
        eye: EyeView,
        clip: CameraClipPlanes,
        viewport: Viewport,
        fov_degrees: f32,
    ) -> Self {
        Self {
            view: eye.view,
            proj: eye.proj,
            world_position: eye.world_position,
            clip,
            projection_kind: CameraProjectionKind::Perspective,
            viewport,
            fov_degrees,
            can_rebuild_symmetric_perspective: true,
        }
    }
}

/// Renderable-space transform for the portal surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPortalSurface {
    /// World matrix of the mesh renderer that owns the portal surface.
    pub world_matrix: Mat4,
}

impl CameraPortalSurface {
    /// Builds surface data from a resolved mesh-renderer world matrix.
    #[inline]
    pub const fn new(world_matrix: Mat4) -> Self {
        Self { world_matrix }
    }
}

/// Resolved camera-portal mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraPortalMode {
    /// Reflect the source camera over the surface plane.
    Mirror,
    /// Transform the source camera through the supplied portal transform.
    Portal,
}

/// Builds a [`HostCameraFrame`] for one camera portal render.
pub fn host_camera_frame_for_camera_portal(
    base: &HostCameraFrame,
    state: &CameraPortalState,
    source: CameraPortalSourceView,
    surface: CameraPortalSurface,
    mode: CameraPortalMode,
    has_far_clip_override: bool,
) -> Option<HostCameraFrame> {
    let clip = camera_portal_clip(
        source.clip,
        state.override_far_clip_value,
        has_far_clip_override,
    );
    let source_proj = camera_portal_projection(source, clip, has_far_clip_override);
    let (view, world_position, clip_plane_position, clip_plane_normal) = match mode {
        CameraPortalMode::Mirror => {
            let plane = mirror_plane_from_surface(surface.world_matrix, state)?;
            let reflection = reflection_matrix(plane.normal, plane.position, state.plane_offset);
            let view = source.view * reflection;
            let world_position = reflection.transform_point3(source.world_position);
            (view, world_position, plane.position, plane.normal)
        }
        CameraPortalMode::Portal => {
            let normal = normalize_vec3(-state.portal_plane_normal)?;
            (
                source.view * state.portal_transform,
                state
                    .portal_transform
                    .transform_point3(source.world_position),
                state.portal_plane_position,
                normal,
            )
        }
    };
    if !matrix_is_valid_camera_basis(view) || !world_position.is_finite() {
        return None;
    }
    let camera_plane = camera_space_plane(
        view,
        clip_plane_position,
        clip_plane_normal,
        state.plane_offset,
    )?;
    let proj = oblique_reverse_z_projection(source_proj, camera_plane)?;
    if !matrix_is_valid_camera_basis(proj) {
        return None;
    }
    let explicit_view = EyeView::new(view, proj, proj * view, world_position);
    Some(HostCameraFrame {
        frame_index: base.frame_index,
        clip,
        desktop_fov_degrees: base.desktop_fov_degrees,
        vr_active: false,
        output_device: base.output_device,
        projection_kind: source.projection_kind,
        primary_ortho_task: None,
        stereo: None,
        head_output_transform: base.head_output_transform,
        explicit_view: Some(explicit_view),
        eye_world_position: Some(world_position),
        suppress_occlusion_temporal: true,
    })
}

#[derive(Clone, Copy)]
struct PortalPlane {
    position: Vec3,
    normal: Vec3,
}

fn camera_portal_clip(
    source: CameraClipPlanes,
    override_far_clip_value: f32,
    has_far_clip_override: bool,
) -> CameraClipPlanes {
    if !has_far_clip_override {
        return source;
    }
    let far = if override_far_clip_value.is_finite() {
        override_far_clip_value.max(source.near + 1e-4)
    } else {
        source.far
    };
    CameraClipPlanes::new(source.near, far)
}

fn camera_portal_projection(
    source: CameraPortalSourceView,
    clip: CameraClipPlanes,
    has_far_clip_override: bool,
) -> Mat4 {
    if has_far_clip_override
        && source.can_rebuild_symmetric_perspective
        && source.projection_kind == CameraProjectionKind::Perspective
    {
        reverse_z_perspective(
            source.viewport.aspect(),
            clamp_desktop_fov_degrees(source.fov_degrees).to_radians(),
            clip.near,
            clip.far,
        )
    } else {
        source.proj
    }
}

fn mirror_plane_from_surface(
    surface_world_matrix: Mat4,
    state: &CameraPortalState,
) -> Option<PortalPlane> {
    if !matrix_is_finite(surface_world_matrix) {
        return None;
    }
    let (_, rotation, position) = surface_world_matrix.to_scale_rotation_translation();
    let normal = normalize_vec3(rotation * state.plane_normal)?;
    Some(PortalPlane { position, normal })
}

fn reflection_matrix(normal: Vec3, position: Vec3, offset: f32) -> Mat4 {
    let d = -normal.dot(position) - offset;
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;
    Mat4::from_cols(
        Vec4::new(1.0 - 2.0 * x * x, -2.0 * y * x, -2.0 * z * x, 0.0),
        Vec4::new(-2.0 * x * y, 1.0 - 2.0 * y * y, -2.0 * z * y, 0.0),
        Vec4::new(-2.0 * x * z, -2.0 * y * z, 1.0 - 2.0 * z * z, 0.0),
        Vec4::new(-2.0 * d * x, -2.0 * d * y, -2.0 * d * z, 1.0),
    )
}

fn camera_space_plane(view: Mat4, position: Vec3, normal: Vec3, offset: f32) -> Option<Vec4> {
    let offset_position = position + normal * offset;
    let camera_position = view.transform_point3(offset_position);
    let camera_normal = normalize_vec3(view.transform_vector3(normal))?;
    Some(Vec4::new(
        camera_normal.x,
        camera_normal.y,
        camera_normal.z,
        -camera_position.dot(camera_normal),
    ))
}

fn oblique_reverse_z_projection(projection: Mat4, camera_plane: Vec4) -> Option<Mat4> {
    if !matrix_is_valid_camera_basis(projection) || !camera_plane.is_finite() {
        return None;
    }
    let inverse_projection = projection.inverse();
    if !matrix_is_finite(inverse_projection) {
        return None;
    }
    let far_corner = Vec4::new(
        sign_for_oblique_corner(camera_plane.x),
        sign_for_oblique_corner(camera_plane.y),
        0.0,
        1.0,
    );
    let q = inverse_projection * far_corner;
    let row_w = matrix_row(projection, 3);
    let numerator = row_w.dot(q);
    let denominator = camera_plane.dot(q);
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() < 1e-8 {
        return None;
    }
    let scaled_plane = camera_plane * (numerator / denominator);
    let new_row_z = row_w - scaled_plane;
    let oblique = set_matrix_row(projection, 2, new_row_z);
    matrix_is_valid_camera_basis(oblique).then_some(oblique)
}

fn sign_for_oblique_corner(value: f32) -> f32 {
    if value < 0.0 { -1.0 } else { 1.0 }
}

fn normalize_vec3(value: Vec3) -> Option<Vec3> {
    let len_sq = value.length_squared();
    (len_sq.is_finite() && len_sq > NORMAL_LENGTH_SQUARED_EPSILON).then(|| value / len_sq.sqrt())
}

fn matrix_is_valid_camera_basis(matrix: Mat4) -> bool {
    matrix_is_finite(matrix)
        && matrix.determinant().is_finite()
        && matrix.determinant().abs() > MATRIX_DETERMINANT_EPSILON
}

fn matrix_is_finite(matrix: Mat4) -> bool {
    matrix.to_cols_array().into_iter().all(f32::is_finite)
}

fn matrix_row(matrix: Mat4, row: usize) -> Vec4 {
    match row {
        0 => Vec4::new(
            matrix.x_axis.x,
            matrix.y_axis.x,
            matrix.z_axis.x,
            matrix.w_axis.x,
        ),
        1 => Vec4::new(
            matrix.x_axis.y,
            matrix.y_axis.y,
            matrix.z_axis.y,
            matrix.w_axis.y,
        ),
        2 => Vec4::new(
            matrix.x_axis.z,
            matrix.y_axis.z,
            matrix.z_axis.z,
            matrix.w_axis.z,
        ),
        3 => Vec4::new(
            matrix.x_axis.w,
            matrix.y_axis.w,
            matrix.z_axis.w,
            matrix.w_axis.w,
        ),
        _ => Vec4::ZERO,
    }
}

fn set_matrix_row(mut matrix: Mat4, row: usize, value: Vec4) -> Mat4 {
    match row {
        0 => {
            matrix.x_axis.x = value.x;
            matrix.y_axis.x = value.y;
            matrix.z_axis.x = value.z;
            matrix.w_axis.x = value.w;
        }
        1 => {
            matrix.x_axis.y = value.x;
            matrix.y_axis.y = value.y;
            matrix.z_axis.y = value.z;
            matrix.w_axis.y = value.w;
        }
        2 => {
            matrix.x_axis.z = value.x;
            matrix.y_axis.z = value.y;
            matrix.z_axis.z = value.z;
            matrix.w_axis.z = value.w;
        }
        3 => {
            matrix.x_axis.w = value.x;
            matrix.y_axis.w = value.y;
            matrix.z_axis.w = value.z;
            matrix.w_axis.w = value.w;
        }
        _ => {}
    }
    matrix
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3, Vec4};

    use super::{
        CameraPortalMode, CameraPortalSourceView, CameraPortalSurface,
        host_camera_frame_for_camera_portal, oblique_reverse_z_projection, reflection_matrix,
    };
    use crate::camera::{
        CameraClipPlanes, EyeView, HostCameraFrame, Viewport, reverse_z_perspective,
    };
    use crate::shared::CameraPortalState;

    fn source_eye(position: Vec3, clip: CameraClipPlanes) -> EyeView {
        let view = Mat4::from_translation(-position);
        let proj = reverse_z_perspective(
            Viewport::new(1280, 720).aspect(),
            60f32.to_radians(),
            clip.near,
            clip.far,
        );
        EyeView::new(view, proj, proj * view, position)
    }

    #[test]
    fn reflection_matrix_maps_points_across_plane() {
        let reflection = reflection_matrix(Vec3::Z, Vec3::ZERO, 0.0);

        let point = reflection.transform_point3(Vec3::new(1.0, 2.0, 3.0));

        assert!(point.abs_diff_eq(Vec3::new(1.0, 2.0, -3.0), 1e-6));
        assert!((reflection.determinant() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn oblique_projection_keeps_existing_near_plane_stable() {
        let near = 0.05;
        let far = 2000.0;
        let proj = reverse_z_perspective(16.0 / 9.0, 60f32.to_radians(), near, far);
        let near_plane = Vec4::new(0.0, 0.0, -1.0, -near);

        let oblique = oblique_reverse_z_projection(proj, near_plane).expect("oblique projection");

        assert!(oblique.abs_diff_eq(proj, 1e-5));
    }

    #[test]
    fn mirror_frame_is_matrix_only_and_finite_after_rotation_reset() {
        let clip = CameraClipPlanes::new(0.05, 500.0);
        let eye = source_eye(Vec3::new(0.0, 0.0, 5.0), clip);
        let source = CameraPortalSourceView::symmetric_perspective(
            eye,
            clip,
            Viewport::new(1280, 720),
            60.0,
        );
        let state = CameraPortalState {
            plane_normal: Vec3::Z,
            plane_offset: 0.001,
            ..CameraPortalState::default()
        };
        let rotated_surface = CameraPortalSurface::new(Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::from_rotation_y(0.75),
            Vec3::ZERO,
        ));
        let reset_surface = CameraPortalSurface::new(Mat4::IDENTITY);

        let rotated = host_camera_frame_for_camera_portal(
            &HostCameraFrame::default(),
            &state,
            source,
            rotated_surface,
            CameraPortalMode::Mirror,
            false,
        )
        .expect("rotated mirror frame");
        let reset = host_camera_frame_for_camera_portal(
            &HostCameraFrame::default(),
            &state,
            source,
            reset_surface,
            CameraPortalMode::Mirror,
            false,
        )
        .expect("reset mirror frame");

        let rotated_eye = rotated.explicit_view.expect("rotated explicit view");
        let reset_eye = reset.explicit_view.expect("reset explicit view");
        assert!(rotated_eye.view.is_finite());
        assert!(rotated_eye.proj.is_finite());
        assert!(reset_eye.view.is_finite());
        assert!(reset_eye.proj.is_finite());
        assert!(
            reset_eye
                .world_position
                .abs_diff_eq(Vec3::new(0.0, 0.0, -4.998), 1e-4)
        );
    }

    #[test]
    fn mirror_frame_rejects_zero_normal() {
        let clip = CameraClipPlanes::new(0.05, 500.0);
        let eye = source_eye(Vec3::new(0.0, 0.0, 5.0), clip);
        let source = CameraPortalSourceView::symmetric_perspective(
            eye,
            clip,
            Viewport::new(1280, 720),
            60.0,
        );
        let state = CameraPortalState {
            plane_normal: Vec3::ZERO,
            ..CameraPortalState::default()
        };

        let frame = host_camera_frame_for_camera_portal(
            &HostCameraFrame::default(),
            &state,
            source,
            CameraPortalSurface::new(Mat4::IDENTITY),
            CameraPortalMode::Mirror,
            false,
        );

        assert!(frame.is_none());
    }
}
