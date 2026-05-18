//! Shared cubemap capture plumbing for offscreen 360-degree render tasks.
//!
//! Reflection-probe bakes and Camera360 photos both render the scene through six 90-degree
//! perspective faces. This module owns the common face order, basis math, GPU target allocation,
//! and one-shot face render submission; callers keep their own task validation, result packing,
//! and post-capture processing.

use std::sync::Arc;

use glam::{Mat4, Vec3};

use crate::backend::RenderBackend;
use crate::camera::{
    CameraClipPlanes, CameraPose, CameraProjectionKind, EyeView, HostCameraFrame, Viewport,
};
use crate::gpu::{CUBEMAP_ARRAY_LAYERS, GpuContext};
use crate::render_graph::{GraphExecuteError, OffscreenSampleCountPolicy};
use crate::runtime::frame::extract::{ExtractedFrame, PreparedViews};
use crate::runtime::frame::view_plan::{FrameViewPlan, OffscreenRtHandles};
use crate::scene::SceneCoordinator;
use crate::world_mesh::WorldMeshDrawCollectParallelism;

/// Number of faces in a cubemap.
pub(in crate::runtime) const CUBE_FACE_COUNT: usize = CUBEMAP_ARRAY_LAYERS as usize;

/// Canonical cubemap face order used by host bitmap cubes and renderer-owned captures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::runtime) enum CubeCaptureFace {
    /// Positive X face.
    PosX,
    /// Negative X face.
    NegX,
    /// Positive Y face.
    PosY,
    /// Negative Y face.
    NegY,
    /// Positive Z face.
    PosZ,
    /// Negative Z face.
    NegZ,
}

impl CubeCaptureFace {
    /// Every cubemap face in host bitmap-cube order.
    pub(in crate::runtime) const ALL: [Self; CUBE_FACE_COUNT] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];

    /// Bit mask with all faces marked complete.
    pub(in crate::runtime) const ALL_MASK: u8 = 0b0011_1111;

    /// Dense face index in host bitmap-cube order.
    pub(in crate::runtime) const fn index(self) -> usize {
        match self {
            Self::PosX => 0,
            Self::NegX => 1,
            Self::PosY => 2,
            Self::NegY => 3,
            Self::PosZ => 4,
            Self::NegZ => 5,
        }
    }

    /// Texture array layer for this face.
    pub(in crate::runtime) const fn layer(self) -> u32 {
        self.index() as u32
    }

    /// Small face index stored in one-shot view ids.
    pub(in crate::runtime) const fn view_id_face_index(self) -> u8 {
        self.index() as u8
    }

    /// Bit for time-sliced face completion tracking.
    pub(in crate::runtime) const fn bit(self) -> u8 {
        1 << self.index()
    }

    /// World-space orthonormal basis for this face.
    #[cfg(test)]
    pub(in crate::runtime) const fn basis(self) -> CubeCaptureFaceBasis {
        self.basis_for(CubeCaptureBasisMode::Canonical)
    }

    /// World-space orthonormal basis for this face under a capture mode.
    pub(in crate::runtime) const fn basis_for(
        self,
        mode: CubeCaptureBasisMode,
    ) -> CubeCaptureFaceBasis {
        match mode {
            CubeCaptureBasisMode::Canonical => self.canonical_basis(),
            CubeCaptureBasisMode::Camera360Copied => self.camera360_copied_basis(),
        }
    }

    /// Canonical world-space orthonormal basis for this face.
    const fn canonical_basis(self) -> CubeCaptureFaceBasis {
        match self {
            Self::PosX => CubeCaptureFaceBasis {
                forward: Vec3::X,
                right: Vec3::NEG_Z,
                up: Vec3::Y,
            },
            Self::NegX => CubeCaptureFaceBasis {
                forward: Vec3::NEG_X,
                right: Vec3::Z,
                up: Vec3::Y,
            },
            Self::PosY => CubeCaptureFaceBasis {
                forward: Vec3::Y,
                right: Vec3::X,
                up: Vec3::NEG_Z,
            },
            Self::NegY => CubeCaptureFaceBasis {
                forward: Vec3::NEG_Y,
                right: Vec3::X,
                up: Vec3::Z,
            },
            Self::PosZ => CubeCaptureFaceBasis {
                forward: Vec3::Z,
                right: Vec3::X,
                up: Vec3::Y,
            },
            Self::NegZ => CubeCaptureFaceBasis {
                forward: Vec3::NEG_Z,
                right: Vec3::NEG_X,
                up: Vec3::Y,
            },
        }
    }

    /// Camera360 copied-cubemap world-space orthonormal basis for this face.
    const fn camera360_copied_basis(self) -> CubeCaptureFaceBasis {
        match self {
            Self::PosX => CubeCaptureFaceBasis {
                forward: Vec3::NEG_X,
                right: Vec3::Z,
                up: Vec3::Y,
            },
            Self::NegX => CubeCaptureFaceBasis {
                forward: Vec3::X,
                right: Vec3::NEG_Z,
                up: Vec3::Y,
            },
            Self::PosY => CubeCaptureFaceBasis {
                forward: Vec3::NEG_Y,
                right: Vec3::NEG_X,
                up: Vec3::NEG_Z,
            },
            Self::NegY => CubeCaptureFaceBasis {
                forward: Vec3::Y,
                right: Vec3::NEG_X,
                up: Vec3::Z,
            },
            Self::PosZ => CubeCaptureFaceBasis {
                forward: Vec3::NEG_Z,
                right: Vec3::NEG_X,
                up: Vec3::Y,
            },
            Self::NegZ => CubeCaptureFaceBasis {
                forward: Vec3::Z,
                right: Vec3::X,
                up: Vec3::Y,
            },
        }
    }

    /// Direction through a normalized face UV coordinate.
    #[cfg(test)]
    pub(in crate::runtime) fn direction_for_uv(self, u: f32, v: f32) -> Vec3 {
        self.direction_for_uv_with_basis(CubeCaptureBasisMode::Canonical, u, v)
    }

    /// Direction through a normalized face UV coordinate under a capture mode.
    #[cfg(test)]
    pub(in crate::runtime) fn direction_for_uv_with_basis(
        self,
        mode: CubeCaptureBasisMode,
        u: f32,
        v: f32,
    ) -> Vec3 {
        let x = 2.0 * u - 1.0;
        let y = 1.0 - 2.0 * v;
        let basis = self.basis_for(mode);
        (basis.forward + basis.right * x + basis.up * y).normalize()
    }
}

/// Cubemap face orientation mode for captures that share the same face storage order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::runtime) enum CubeCaptureBasisMode {
    /// Canonical cubemap orientation used by reflection probes and renderer-owned cube assets.
    Canonical,
    /// Camera360 copied-cubemap orientation used before equirectangular projection.
    Camera360Copied,
}

/// World-space orthonormal basis for one cubemap face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::runtime) struct CubeCaptureFaceBasis {
    /// Center ray for the face.
    pub(in crate::runtime) forward: Vec3,
    /// Positive face-local X direction.
    pub(in crate::runtime) right: Vec3,
    /// Positive face-local Y direction.
    pub(in crate::runtime) up: Vec3,
}

/// Texture size and mip count for a cubemap capture target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::runtime) struct CubeCaptureExtent {
    /// Face edge length in texels.
    pub(in crate::runtime) face_size: u32,
    /// Number of mip levels allocated in the cubemap texture.
    pub(in crate::runtime) mip_levels: u32,
}

impl CubeCaptureExtent {
    /// Builds a cubemap capture extent.
    pub(in crate::runtime) const fn new(face_size: u32, mip_levels: u32) -> Self {
        Self {
            face_size,
            mip_levels,
        }
    }

    /// Square viewport used by each cubemap face render.
    pub(in crate::runtime) const fn viewport(self) -> (u32, u32) {
        (self.face_size, self.face_size)
    }
}

/// Error returned while allocating shared cubemap capture targets.
#[derive(Debug, thiserror::Error)]
pub(in crate::runtime) enum CubeCaptureTargetError {
    /// Face edge exceeds the current device's two-dimensional texture limit.
    #[error("cubemap capture size {size} exceeds max_texture_dimension_2d={max}")]
    SizeExceedsLimit {
        /// Requested face edge.
        size: u32,
        /// Device limit.
        max: u32,
    },
    /// The device cannot allocate the six array layers needed for cubemaps.
    #[error("cubemap capture requires 6 texture array layers but max_texture_array_layers={max}")]
    CubemapArrayLayersUnsupported {
        /// Device limit.
        max: u32,
    },
}

/// GPU textures and views backing a six-face offscreen cubemap capture.
pub(in crate::runtime) struct CubeCaptureTargets {
    /// Cubemap color texture.
    pub(in crate::runtime) cube_texture: Arc<wgpu::Texture>,
    /// Per-face render-attachment color views.
    pub(in crate::runtime) face_color_views: [Arc<wgpu::TextureView>; CUBE_FACE_COUNT],
    /// Per-face depth textures.
    pub(in crate::runtime) face_depth_textures: [Arc<wgpu::Texture>; CUBE_FACE_COUNT],
    /// Per-face render-attachment depth views.
    pub(in crate::runtime) face_depth_views: [Arc<wgpu::TextureView>; CUBE_FACE_COUNT],
    /// Color format used by the cubemap texture.
    pub(in crate::runtime) color_format: wgpu::TextureFormat,
    /// Capture extent.
    pub(in crate::runtime) extent: CubeCaptureExtent,
}

impl CubeCaptureTargets {
    /// Allocates textures and views for a six-face cubemap capture.
    pub(in crate::runtime) fn create(
        gpu: &GpuContext,
        extent: CubeCaptureExtent,
        color_format: wgpu::TextureFormat,
        color_usage: wgpu::TextureUsages,
        label_prefix: &'static str,
    ) -> Result<Self, CubeCaptureTargetError> {
        let max_dim = gpu.limits().max_texture_dimension_2d();
        if extent.face_size > max_dim {
            return Err(CubeCaptureTargetError::SizeExceedsLimit {
                size: extent.face_size,
                max: max_dim,
            });
        }
        if !gpu.limits().array_layers_fit(CUBEMAP_ARRAY_LAYERS) {
            return Err(CubeCaptureTargetError::CubemapArrayLayersUnsupported {
                max: gpu.limits().max_texture_array_layers(),
            });
        }
        let size = wgpu::Extent3d {
            width: extent.face_size,
            height: extent.face_size,
            depth_or_array_layers: CUBEMAP_ARRAY_LAYERS,
        };
        let cube_texture = Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some(label_prefix),
            size,
            mip_level_count: extent.mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: color_usage,
            view_formats: &[],
        }));
        let face_color_views = std::array::from_fn(|i| {
            Arc::new(cube_texture.create_view(&face_view_desc(
                label_prefix,
                i as u32,
                color_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            )))
        });
        crate::profiling::note_resource_churn!(TextureView, "runtime::cube_capture_color_views");

        let depth_format = crate::gpu::main_forward_depth_stencil_format(gpu.device().features());
        let depth_size = wgpu::Extent3d {
            width: extent.face_size,
            height: extent.face_size,
            depth_or_array_layers: 1,
        };
        let face_depth_textures = std::array::from_fn(|_i| {
            Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some(label_prefix),
                size: depth_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: depth_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }))
        });
        let face_depth_views = std::array::from_fn(|i| {
            Arc::new(
                face_depth_textures[i].create_view(&wgpu::TextureViewDescriptor {
                    label: Some(label_prefix),
                    format: Some(depth_format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    aspect: wgpu::TextureAspect::All,
                    ..Default::default()
                }),
            )
        });
        crate::profiling::note_resource_churn!(TextureView, "runtime::cube_capture_depth_views");

        Ok(Self {
            cube_texture,
            face_color_views,
            face_depth_textures,
            face_depth_views,
            color_format,
            extent,
        })
    }

    /// Builds offscreen render-target handles for a single cubemap face.
    pub(in crate::runtime) fn to_offscreen_handles(
        &self,
        face: CubeCaptureFace,
        sample_count_policy: OffscreenSampleCountPolicy,
    ) -> OffscreenRtHandles {
        OffscreenRtHandles {
            rt_id: -1,
            color_texture: Arc::clone(&self.cube_texture),
            color_view: Arc::clone(&self.face_color_views[face.index()]),
            depth_texture: Arc::clone(&self.face_depth_textures[face.index()]),
            depth_view: Arc::clone(&self.face_depth_views[face.index()]),
            color_format: self.color_format,
            sample_count_policy,
            copy_to_color: None,
        }
    }

    /// Creates a full cubemap sample view over mip 0.
    pub(in crate::runtime) fn cube_sample_view(
        &self,
        label: &'static str,
    ) -> Arc<wgpu::TextureView> {
        Arc::new(self.cube_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            format: Some(self.color_format),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBEMAP_ARRAY_LAYERS),
        }))
    }

    /// Creates a two-dimensional array sample view over every cubemap face and mip.
    pub(in crate::runtime) fn array_sample_view(
        &self,
        label: &'static str,
    ) -> Arc<wgpu::TextureView> {
        Arc::new(self.cube_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            format: Some(self.color_format),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(self.extent.mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(CUBEMAP_ARRAY_LAYERS),
        }))
    }
}

/// Builds a texture-view descriptor exposing one cubemap face as a 2D render attachment.
pub(in crate::runtime) fn face_view_desc(
    label: &'static str,
    face: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor {
        label: Some(label),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        usage: Some(usage),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: face,
        array_layer_count: Some(1),
    }
}

/// Builds a world matrix oriented toward one cubemap face under a capture mode.
pub(in crate::runtime) fn cube_face_world_matrix_for_basis(
    position: Vec3,
    face: CubeCaptureFace,
    mode: CubeCaptureBasisMode,
) -> Mat4 {
    let basis = face.basis_for(mode);
    Mat4::from_cols(
        basis.right.extend(0.0),
        basis.up.extend(0.0),
        basis.forward.extend(0.0),
        position.extend(1.0),
    )
}

/// Builds a host camera frame for rendering one cubemap face.
pub(in crate::runtime) fn host_camera_frame_for_cube_face(
    base: &HostCameraFrame,
    clip: CameraClipPlanes,
    viewport_px: (u32, u32),
    position: Vec3,
    face: CubeCaptureFace,
) -> HostCameraFrame {
    host_camera_frame_for_cube_face_with_basis(
        base,
        clip,
        viewport_px,
        position,
        face,
        CubeCaptureBasisMode::Canonical,
    )
}

/// Builds a host camera frame for rendering one cubemap face under a capture mode.
pub(in crate::runtime) fn host_camera_frame_for_cube_face_with_basis(
    base: &HostCameraFrame,
    clip: CameraClipPlanes,
    viewport_px: (u32, u32),
    position: Vec3,
    face: CubeCaptureFace,
    mode: CubeCaptureBasisMode,
) -> HostCameraFrame {
    let world_matrix = cube_face_world_matrix_for_basis(position, face, mode);
    let pose = CameraPose::from_world_matrix(world_matrix);
    let viewport = Viewport::from_tuple(viewport_px);
    HostCameraFrame {
        frame_index: base.frame_index,
        clip,
        desktop_fov_degrees: 90.0,
        vr_active: false,
        output_device: base.output_device,
        projection_kind: CameraProjectionKind::Perspective,
        primary_ortho_task: None,
        stereo: None,
        head_output_transform: base.head_output_transform,
        explicit_view: Some(EyeView::from_pose_projection(
            pose,
            if viewport.is_empty() {
                Mat4::IDENTITY
            } else {
                EyeView::perspective_from_pose(pose, viewport, 90.0, clip).proj
            },
        )),
        eye_world_position: Some(position),
        suppress_occlusion_temporal: true,
    }
}

/// Renders a batch of cubemap face plans through the normal frame extraction and graph path.
pub(in crate::runtime) fn render_cube_capture_faces_offscreen(
    gpu: &mut GpuContext,
    backend: &mut RenderBackend,
    scene: &SceneCoordinator,
    plans: Vec<FrameViewPlan<'static>>,
) -> Result<(), GraphExecuteError> {
    profiling::scope!("cube_capture::offscreen_render");
    let prepared_views = PreparedViews::new(plans, None);
    backend.prepare_lights_for_views(
        scene,
        prepared_views
            .plans()
            .iter()
            .map(FrameViewPlan::light_view_desc),
    );
    let view_perms = prepared_views
        .plans()
        .iter()
        .map(|plan| (plan.render_context(), plan.shader_permutation()))
        .collect::<Vec<_>>();
    let shared =
        backend.extract_frame_shared(scene, WorldMeshDrawCollectParallelism::Full, &view_perms);
    let submit_frame = ExtractedFrame::new(prepared_views, shared)
        .prepare_draws()
        .into_submit_frame();
    submit_frame.execute(gpu, scene, backend)
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use hashbrown::HashSet;

    use super::*;

    fn matrix_direction_for_uv(face: CubeCaptureFace, u: f32, v: f32) -> Vec3 {
        matrix_direction_for_uv_with_basis(face, CubeCaptureBasisMode::Canonical, u, v)
    }

    fn matrix_direction_for_uv_with_basis(
        face: CubeCaptureFace,
        mode: CubeCaptureBasisMode,
        u: f32,
        v: f32,
    ) -> Vec3 {
        let x = 2.0 * u - 1.0;
        let y = 1.0 - 2.0 * v;
        cube_face_world_matrix_for_basis(Vec3::ZERO, face, mode)
            .transform_vector3(Vec3::new(x, y, 1.0))
            .normalize()
    }

    #[test]
    fn cubemap_face_directions_match_bitmap_cube_order() {
        let samples = [
            (CubeCaptureFace::PosX, Vec3::new(1.0, 1.0, 1.0)),
            (CubeCaptureFace::NegX, Vec3::new(-1.0, 1.0, -1.0)),
            (CubeCaptureFace::PosY, Vec3::new(-1.0, 1.0, -1.0)),
            (CubeCaptureFace::NegY, Vec3::new(-1.0, -1.0, 1.0)),
            (CubeCaptureFace::PosZ, Vec3::new(-1.0, 1.0, 1.0)),
            (CubeCaptureFace::NegZ, Vec3::new(1.0, 1.0, -1.0)),
        ];
        for (face, expected) in samples {
            let actual = face.direction_for_uv(0.0, 0.0);
            assert!((actual - expected.normalize()).length() < 1e-6);
        }
    }

    #[test]
    fn cubemap_face_indices_layers_bits_are_unique() {
        let mut bits = 0u8;
        let mut indices = HashSet::new();
        for (expected, face) in CubeCaptureFace::ALL.iter().copied().enumerate() {
            assert_eq!(face.index(), expected);
            assert_eq!(face.layer(), expected as u32);
            assert_eq!(face.view_id_face_index(), expected as u8);
            assert!(indices.insert(face.index()));
            bits |= face.bit();
        }

        assert_eq!(bits, CubeCaptureFace::ALL_MASK);
    }

    #[test]
    fn cubemap_face_basis_vectors_are_orthonormal() {
        for mode in [
            CubeCaptureBasisMode::Canonical,
            CubeCaptureBasisMode::Camera360Copied,
        ] {
            for face in CubeCaptureFace::ALL {
                let basis = face.basis_for(mode);
                assert!((basis.forward.length() - 1.0).abs() < 1e-6);
                assert!((basis.right.length() - 1.0).abs() < 1e-6);
                assert!((basis.up.length() - 1.0).abs() < 1e-6);
                assert!(basis.forward.dot(basis.right).abs() < 1e-6);
                assert!(basis.forward.dot(basis.up).abs() < 1e-6);
                assert!(basis.right.dot(basis.up).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn camera360_copied_face_basis_matches_custom_capture_rotations() {
        let expected = [
            (CubeCaptureFace::PosX, Vec3::NEG_X, Vec3::Z, Vec3::Y),
            (CubeCaptureFace::NegX, Vec3::X, Vec3::NEG_Z, Vec3::Y),
            (CubeCaptureFace::PosY, Vec3::NEG_Y, Vec3::NEG_X, Vec3::NEG_Z),
            (CubeCaptureFace::NegY, Vec3::Y, Vec3::NEG_X, Vec3::Z),
            (CubeCaptureFace::PosZ, Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
            (CubeCaptureFace::NegZ, Vec3::Z, Vec3::X, Vec3::Y),
        ];

        for (face, forward, right, up) in expected {
            let basis = face.basis_for(CubeCaptureBasisMode::Camera360Copied);
            assert_eq!(basis.forward, forward);
            assert_eq!(basis.right, right);
            assert_eq!(basis.up, up);
        }
    }

    #[test]
    fn camera360_copied_top_bottom_faces_are_not_canonical_y_faces() {
        for face in [CubeCaptureFace::PosY, CubeCaptureFace::NegY] {
            let canonical = face.basis_for(CubeCaptureBasisMode::Canonical);
            let camera360 = face.basis_for(CubeCaptureBasisMode::Camera360Copied);

            assert_eq!(camera360.right, Vec3::NEG_X);
            assert_ne!(camera360.forward, canonical.forward);
            assert_ne!(camera360.right, canonical.right);
            assert_eq!(camera360.up, canonical.up);
        }
    }

    #[test]
    fn cubemap_face_world_matrices_match_bitmap_cube_directions() {
        for face in CubeCaptureFace::ALL {
            assert!(
                (matrix_direction_for_uv(face, 0.5, 0.5) - face.basis().forward).length() < 1e-6
            );
            for (u, v) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                let actual = matrix_direction_for_uv(face, u, v);
                let expected = face.direction_for_uv(u, v);
                assert!(
                    (actual - expected).length() < 1e-6,
                    "{face:?} uv=({u}, {v}) actual={actual:?} expected={expected:?}"
                );
            }
        }
    }

    #[test]
    fn camera360_copied_world_matrices_match_copied_directions() {
        let mode = CubeCaptureBasisMode::Camera360Copied;
        for face in CubeCaptureFace::ALL {
            assert!(
                (matrix_direction_for_uv_with_basis(face, mode, 0.5, 0.5)
                    - face.basis_for(mode).forward)
                    .length()
                    < 1e-6
            );
            for (u, v) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                let actual = matrix_direction_for_uv_with_basis(face, mode, u, v);
                let expected = face.direction_for_uv_with_basis(mode, u, v);
                assert!(
                    (actual - expected).length() < 1e-6,
                    "{face:?} uv=({u}, {v}) actual={actual:?} expected={expected:?}"
                );
            }
        }
    }
}
