use std::ops::Range;
use std::sync::Arc;

use glam::{Quat, Vec3, Vec4};
use thiserror::Error;

use crate::color_space::srgb_channel_to_linear;
use crate::shared::{BillboardAlignment, MeshAlignment, MotionVectorMode, TrailTextureMode};

/// CPU metadata retained for a resident PhotonDust point render buffer.
#[derive(Clone, Debug)]
pub(crate) struct PointRenderBufferAsset {
    /// Host point render-buffer asset id.
    pub(crate) asset_id: i32,
    /// Number of particles decoded from the latest upload.
    pub(crate) count: usize,
    /// Texture-sheet frame grid copied from the upload.
    pub(crate) frame_grid_size: glam::IVec2,
    /// CPU point data retained for mesh-particle renderers.
    pub(crate) points: Arc<[PointParticle]>,
}

/// CPU metadata retained for a resident PhotonDust trail render buffer.
#[derive(Clone, Debug)]
pub(crate) struct TrailRenderBufferAsset {
    /// Host trail render-buffer asset id.
    pub(crate) asset_id: i32,
    /// Number of logical trails decoded from the latest upload.
    pub(crate) trails_count: usize,
    /// Number of trail point slots decoded from the latest upload.
    pub(crate) trail_point_count: usize,
}

/// Renderer-side particle draw family carried through world-mesh draw preparation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub(crate) enum ParticleDrawKind {
    /// Ordinary non-particle draw.
    #[default]
    None = 0,
    /// Point render buffer drawn through Billboard/Unlit.
    Billboard = 1,
    /// Source mesh instanced once per point particle.
    Mesh = 2,
    /// Trail render buffer drawn as generated ribbon geometry.
    Trail = 3,
}

/// Renderer-internal particle state attached to one prepared draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParticleDrawParams {
    /// Particle renderer family.
    pub(crate) kind: ParticleDrawKind,
    /// Billboard or mesh alignment discriminant, matching the generated wire enum value.
    pub(crate) alignment: u32,
    /// Minimum billboard screen-size clamp requested by the host.
    pub(crate) min_screen_size: f32,
    /// Maximum billboard screen-size clamp requested by the host.
    pub(crate) max_screen_size: f32,
    /// Motion-vector mode requested by the host.
    pub(crate) motion_vector_mode: u32,
    /// Per-particle tint for mesh-particle draw paths.
    pub(crate) color: Vec4,
    /// Texture-sheet frame index when the source point buffer supplied one.
    pub(crate) frame_index: u32,
    /// Trail texture-coordinate generation mode.
    pub(crate) trail_texture_mode: u32,
    /// Whether the host requested trail lighting data.
    pub(crate) trail_generate_lighting_data: bool,
}

impl Default for ParticleDrawParams {
    fn default() -> Self {
        Self {
            kind: ParticleDrawKind::None,
            alignment: 0,
            min_screen_size: 0.0,
            max_screen_size: 0.0,
            motion_vector_mode: MotionVectorMode::default() as u32,
            color: Vec4::ONE,
            frame_index: u32::MAX,
            trail_texture_mode: TrailTextureMode::default() as u32,
            trail_generate_lighting_data: false,
        }
    }
}

impl ParticleDrawParams {
    /// Builds draw metadata for a BillboardRenderBufferRenderer row.
    pub(crate) fn billboard(
        alignment: BillboardAlignment,
        min_screen_size: f32,
        max_screen_size: f32,
        motion_vector_mode: MotionVectorMode,
    ) -> Self {
        Self {
            kind: ParticleDrawKind::Billboard,
            alignment: alignment as u32,
            min_screen_size,
            max_screen_size,
            motion_vector_mode: motion_vector_mode as u32,
            ..Self::default()
        }
    }

    /// Builds draw metadata for one MeshRenderBufferRenderer particle instance.
    pub(crate) fn mesh(alignment: MeshAlignment, color: Vec4, frame_index: Option<u16>) -> Self {
        Self {
            kind: ParticleDrawKind::Mesh,
            alignment: alignment as u32,
            color,
            frame_index: frame_index.map(u32::from).unwrap_or(u32::MAX),
            ..Self::default()
        }
    }

    /// Builds draw metadata for a TrailsRenderBufferRenderer row.
    pub(crate) fn trail(
        texture_mode: TrailTextureMode,
        motion_vector_mode: MotionVectorMode,
        generate_lighting_data: bool,
    ) -> Self {
        Self {
            kind: ParticleDrawKind::Trail,
            motion_vector_mode: motion_vector_mode as u32,
            trail_texture_mode: texture_mode as u32,
            trail_generate_lighting_data: generate_lighting_data,
            ..Self::default()
        }
    }

    /// Encodes the metadata into WGSL `vec4<f32>` rows.
    pub(crate) fn to_uniform_rows(self) -> [[f32; 4]; 3] {
        [
            [
                self.kind as u32 as f32,
                self.alignment as f32,
                self.min_screen_size,
                self.max_screen_size,
            ],
            self.color.to_array(),
            [
                self.motion_vector_mode as f32,
                self.frame_index as f32,
                self.trail_texture_mode as f32,
                if self.trail_generate_lighting_data {
                    1.0
                } else {
                    0.0
                },
            ],
        ]
    }
}

/// Error raised while validating or generating a PhotonDust render-buffer mesh.
#[derive(Debug, Error)]
pub(crate) enum ParticleRenderBufferError {
    /// The host sent a negative count for a required row array.
    #[error("{kind} render buffer {asset_id}: negative {field} {value}")]
    NegativeCount {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
        /// Field that carried the invalid count.
        field: &'static str,
        /// Invalid value.
        value: i32,
    },
    /// A required payload offset was negative.
    #[error("{kind} render buffer {asset_id}: missing required {field} offset")]
    MissingOffset {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
        /// Missing field name.
        field: &'static str,
    },
    /// A payload byte range overflowed or fell outside the shared-memory copy.
    #[error(
        "{kind} render buffer {asset_id}: {field} byte range offset={offset} len={len} exceeds raw len {raw_len}"
    )]
    RangeOutOfBounds {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
        /// Field being read.
        field: &'static str,
        /// Requested byte offset.
        offset: i32,
        /// Requested byte length.
        len: usize,
        /// Available raw bytes.
        raw_len: usize,
    },
    /// The generated mesh id cannot fit into the renderer's signed asset id space.
    #[error("{kind} render buffer {asset_id}: generated mesh id overflow")]
    GeneratedIdOverflow {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
    },
    /// The generated vertex or index count exceeded supported limits.
    #[error("{kind} render buffer {asset_id}: generated mesh is too large")]
    MeshTooLarge {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
    },
    /// Mesh layout validation failed for generated geometry.
    #[error("{kind} render buffer {asset_id}: generated mesh layout is invalid")]
    InvalidMeshLayout {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
    },
    /// GPU upload failed for generated geometry.
    #[error("{kind} render buffer {asset_id}: generated mesh GPU upload failed")]
    GpuUploadFailed {
        /// Render-buffer family.
        kind: &'static str,
        /// Source asset id.
        asset_id: i32,
    },
}

pub(super) fn nonnegative_count(
    kind: &'static str,
    asset_id: i32,
    field: &'static str,
    value: i32,
) -> Result<usize, ParticleRenderBufferError> {
    if value < 0 {
        return Err(ParticleRenderBufferError::NegativeCount {
            kind,
            asset_id,
            field,
            value,
        });
    }
    Ok(value as usize)
}

pub(super) fn checked_range(
    kind: &'static str,
    asset_id: i32,
    raw_len: usize,
    field: &'static str,
    offset: i32,
    count: usize,
    stride: usize,
) -> Result<Range<usize>, ParticleRenderBufferError> {
    if offset < 0 {
        return Err(ParticleRenderBufferError::MissingOffset {
            kind,
            asset_id,
            field,
        });
    }
    let len = count
        .checked_mul(stride)
        .ok_or(ParticleRenderBufferError::RangeOutOfBounds {
            kind,
            asset_id,
            field,
            offset,
            len: usize::MAX,
            raw_len,
        })?;
    let start = offset as usize;
    let end = start
        .checked_add(len)
        .ok_or(ParticleRenderBufferError::RangeOutOfBounds {
            kind,
            asset_id,
            field,
            offset,
            len,
            raw_len,
        })?;
    if end > raw_len {
        return Err(ParticleRenderBufferError::RangeOutOfBounds {
            kind,
            asset_id,
            field,
            offset,
            len,
            raw_len,
        });
    }
    Ok(start..end)
}

pub(super) fn checked_optional_range(
    kind: &'static str,
    asset_id: i32,
    raw_len: usize,
    field: &'static str,
    offset: i32,
    count: usize,
    stride: usize,
) -> Result<Option<Range<usize>>, ParticleRenderBufferError> {
    if offset < 0 {
        return Ok(None);
    }
    checked_range(kind, asset_id, raw_len, field, offset, count, stride).map(Some)
}

pub(super) fn read_pod_at<T: bytemuck::Pod>(raw: &[u8], range: &Range<usize>, index: usize) -> T {
    let stride = size_of::<T>();
    let start = range.start + index * stride;
    bytemuck::pod_read_unaligned(&raw[start..start + stride])
}

/// Converts one PhotonDust LDR sRGB vertex-color channel into renderer-linear space.
fn photondust_srgb_ldr_channel_to_linear(value: f32) -> f32 {
    if value > -1.0 && value < 1.0 {
        srgb_channel_to_linear(value)
    } else {
        value
    }
}

/// Converts PhotonDust sRGB particle color into renderer-linear vertex color.
pub(super) fn photondust_particle_color_to_linear(color: Vec4) -> Vec4 {
    Vec4::new(
        photondust_srgb_ldr_channel_to_linear(color.x),
        photondust_srgb_ldr_channel_to_linear(color.y),
        photondust_srgb_ldr_channel_to_linear(color.z),
        color.w,
    )
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PointParticle {
    /// Particle center in the render-buffer renderer's local space.
    pub(crate) position: Vec3,
    /// Particle rotation in the render-buffer renderer's local space.
    pub(crate) rotation: Quat,
    /// Particle size from PhotonDust.
    pub(crate) size: Vec3,
    /// Particle color converted from PhotonDust sRGB to linear vertex color.
    pub(crate) color: Vec4,
    /// Optional texture-sheet frame index.
    pub(crate) frame_index: Option<u16>,
}
