//! Shared renderer contracts used across asset, material, and draw systems.

use std::sync::Arc;

use glam::Vec4;

use crate::shared::{BillboardAlignment, MeshAlignment, MotionVectorMode, TrailTextureMode};

/// Bytes per sparse position entry on the GPU: `vertex_index: u32` + `delta.xyz: f32`.
pub const BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE: usize = 16;

/// Bytes per sparse packed normal or tangent entry: `vertex_index: u32` + three snorm16 channels.
#[cfg(test)]
pub const BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_SIZE: usize = 12;

/// Number of `u32` words per sparse position entry in the GPU buffer.
pub const BLENDSHAPE_POSITION_SPARSE_ENTRY_WORDS: u32 = 4;

/// Number of `u32` words per sparse packed normal or tangent entry in the GPU buffer.
pub const BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS: u32 = 3;

/// Minimum storage buffer size used when a mesh has blendshapes but zero sparse bytes.
pub const BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES: u64 = 16;

/// Maximum number of local reflection probes packed into one draw.
pub const MAX_LOCAL_REFLECTION_PROBES: usize = 4;

/// GPU-ready channel-sparse blendshape deltas and CPU scatter ranges.
pub struct BlendshapeGpuPack {
    /// Tightly packed `u32` words containing position, normal, and tangent sparse sections.
    pub sparse_deltas: Vec<u8>,
    /// Per-frame sparse ranges sorted by shape and frame weight.
    pub frame_ranges: Vec<BlendshapeFrameRange>,
    /// Per-shape spans into [`Self::frame_ranges`].
    pub shape_frame_spans: Vec<BlendshapeFrameSpan>,
    /// Logical blendshape slot count (`max(blendshape_index) + 1`).
    pub num_blendshapes: i32,
    /// Whether any sparse row carries a nonzero position delta.
    pub has_position_deltas: bool,
    /// Whether any sparse row carries a nonzero normal delta.
    pub has_normal_deltas: bool,
    /// Whether any sparse row carries a nonzero tangent delta.
    pub has_tangent_deltas: bool,
    /// Whether any packed normal or tangent component was clamped to the supported delta range.
    pub clamped_packed_deltas: bool,
}

/// Sparse range and metadata for one Unity blendshape frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendshapeFrameRange {
    /// Logical blendshape index from [`crate::shared::BlendshapeBufferDescriptor::blendshape_index`].
    pub shape_index: u32,
    /// Host frame index from [`crate::shared::BlendshapeBufferDescriptor::frame_index`].
    pub frame_index: i32,
    /// Unity frame weight from [`crate::shared::BlendshapeBufferDescriptor::frame_weight`].
    pub frame_weight: f32,
    /// First `u32` word of this frame's position entries in [`BlendshapeGpuPack::sparse_deltas`].
    pub position_first_word: u32,
    /// Number of sparse position entries in this frame.
    pub position_count: u32,
    /// First `u32` word of this frame's packed normal entries in [`BlendshapeGpuPack::sparse_deltas`].
    pub normal_first_word: u32,
    /// Number of sparse packed normal entries in this frame.
    pub normal_count: u32,
    /// First `u32` word of this frame's packed tangent entries in [`BlendshapeGpuPack::sparse_deltas`].
    pub tangent_first_word: u32,
    /// Number of sparse packed tangent entries in this frame.
    pub tangent_count: u32,
}

/// Span of frame rows belonging to one logical blendshape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlendshapeFrameSpan {
    /// First row in [`BlendshapeGpuPack::frame_ranges`].
    pub first_frame: u32,
    /// Number of rows for this logical shape.
    pub frame_count: u32,
}

/// Returns `false` when sparse payloads cannot exist on the device or be bound as one storage read.
pub fn blendshape_sparse_buffers_fit_device(
    pack: &BlendshapeGpuPack,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> bool {
    let sparse_len = pack
        .sparse_deltas
        .len()
        .max(BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE);
    let sparse_u64 = sparse_len as u64;
    if sparse_u64 > max_buffer_size {
        return false;
    }
    if sparse_u64 > max_storage_buffer_binding_size {
        return false;
    }
    true
}

/// Lazy tangent upload behavior required by an embedded material.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EmbeddedTangentFallbackMode {
    /// Preserve host-authored tangent data and use a stable default when the mesh has no tangent attribute.
    #[default]
    PreserveHostOrDefault,
    /// Generate MikkTSpace tangents when the mesh has no tangent attribute but has enough geometry data.
    GenerateMissing,
}

impl EmbeddedTangentFallbackMode {
    /// `true` when lazy mesh upload should generate tangents for tangentless triangle meshes.
    pub fn generate_missing(self) -> bool {
        matches!(self, Self::GenerateMissing)
    }
}

/// Raster pipeline identity for mesh draws: one embedded WGSL target per shader, or the null fallback.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RasterPipelineKind {
    /// Composed WGSL package stem (for example `ui_textunlit_default`).
    EmbeddedStem(Arc<str>),
    /// Object-space black/grey checkerboard fallback when the host shader has no embedded target.
    Null,
}

/// Primitive topology selected per submesh for material pipeline and draw-batch keys.
///
/// `wgpu::PrimitiveTopology` does not derive `Ord`/`PartialOrd`, so draw and pipeline keys use
/// this enum instead of embedding the wgpu type directly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum RasterPrimitiveTopology {
    /// Each vertex is a point sprite; no shared topology.
    PointList,
    /// Each triple of vertices forms an independent triangle.
    #[default]
    TriangleList,
}

impl RasterPrimitiveTopology {
    /// Lowers the renderer's topology tag into the wgpu primitive setting used at pipeline build.
    #[must_use]
    #[inline]
    pub fn to_wgpu(self) -> wgpu::PrimitiveTopology {
        match self {
            Self::PointList => wgpu::PrimitiveTopology::PointList,
            Self::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        }
    }
}

impl From<crate::shared::SubmeshTopology> for RasterPrimitiveTopology {
    #[inline]
    fn from(t: crate::shared::SubmeshTopology) -> Self {
        match t {
            crate::shared::SubmeshTopology::Points => Self::PointList,
            crate::shared::SubmeshTopology::Triangles => Self::TriangleList,
        }
    }
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
