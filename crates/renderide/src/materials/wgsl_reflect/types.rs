//! Public reflected layout types and [`ReflectError`] for WGSL material reflection.

use hashbrown::HashMap;

use thiserror::Error;

/// Scalar shape of a named uniform struct member (for CPU packing from host properties).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedUniformScalarKind {
    /// Single `f32`.
    F32,
    /// `vec4<f32>` (or equivalent 16-byte float vector).
    Vec4,
    /// Single `u32` (e.g. shader `flags`).
    U32,
    /// Not mapped automatically (padding or unsupported type).
    Unsupported,
}

/// Byte layout of one field inside a `@group(1)` `var<uniform>` struct (from naga struct member offsets).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReflectedUniformField {
    /// Byte offset within the uniform block (WGSL struct layout).
    pub offset: u32,
    /// Size in bytes (`Layouter` type size).
    pub size: u32,
    /// Host packing strategy for this member.
    pub kind: ReflectedUniformScalarKind,
}

/// Uniform block at `@group(1)` (typically `@binding(0)`) used for material constants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReflectedMaterialUniformBlock {
    /// WGSL binding index for this uniform buffer (expected `0` for current materials).
    pub binding: u32,
    /// Total uniform block size in bytes (including tail padding).
    pub total_size: u32,
    /// Struct member name -> layout (only members with names; excludes padding-only slots if unnamed).
    pub fields: HashMap<String, ReflectedUniformField>,
}

/// Vertex attribute format reflected from `vs_main` input arguments.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReflectedVertexInputFormat {
    /// `vec2<f32>`.
    Float32x2,
    /// `vec4<f32>`.
    Float32x4,
    /// Any currently unsupported vertex input shape.
    Unsupported,
}

/// One reflected `vs_main` vertex input location and its shader-visible format.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReflectedVertexInput {
    /// Shader input location.
    pub location: u32,
    /// Shader-visible attribute format.
    pub format: ReflectedVertexInputFormat,
}

/// Result of `reflect_raster_material_wgsl` in the parent `wgsl_reflect` module.
#[derive(Clone, Debug)]
pub struct ReflectedRasterLayout {
    /// Stable hash of material + per-draw bind group layout shapes (tests, diagnostics, future cache versioning).
    pub layout_fingerprint: u64,
    /// `@group(1)` entries sorted by binding index.
    pub material_entries: Vec<wgpu::BindGroupLayoutEntry>,
    /// `@group(2)` entries sorted by binding index.
    pub per_draw_entries: Vec<wgpu::BindGroupLayoutEntry>,
    /// First `var<uniform>` in `@group(1)` with a struct body, if any (for CPU packing without hand-written `#[repr(C)]` structs).
    pub material_uniform: Option<ReflectedMaterialUniformBlock>,
    /// `@group(1)` `@binding` -> WGSL global identifier (matches Unity host property names where applicable).
    pub material_group1_names: HashMap<u32, String>,
    /// Exact `vs_main` vertex input locations and formats, sorted by location.
    pub vs_vertex_inputs: Vec<ReflectedVertexInput>,
    /// Highest `@location` index on `vs_main` vertex inputs (excluding builtins); `>= 2` implies a UV stream at `location(2)`.
    pub vs_max_vertex_location: Option<u32>,
    /// `true` when the shader declares a scene-depth snapshot binding at `@group(0)`.
    pub uses_scene_depth_snapshot: bool,
    /// `true` when the shader declares a scene-color snapshot binding at `@group(0)`.
    pub uses_scene_color_snapshot: bool,
    /// `true` when the material uniform block declares intersection tint (e.g. `_IntersectColor`), used for a second forward subpass.
    ///
    /// Derived from reflection only (no shader stem string checks in the render graph).
    pub requires_intersection_pass: bool,
}

impl ReflectedRasterLayout {
    /// Returns the unified scene-snapshot requirement flags for this layout.
    pub fn snapshot_requirements(&self) -> super::super::SnapshotRequirements {
        super::super::SnapshotRequirements {
            uses_scene_color: self.uses_scene_color_snapshot,
            uses_scene_depth: self.uses_scene_depth_snapshot,
            requires_intersection_pass: self.requires_intersection_pass,
        }
    }
}

/// Errors from `reflect_raster_material_wgsl` in the parent `wgsl_reflect` module.
#[derive(Debug, Error)]
pub enum ReflectError {
    /// Naga failed to parse the composed WGSL source.
    #[error("WGSL parse: {0}")]
    Parse(String),
    /// Naga validation failed after parse.
    #[error("WGSL validate: {0}")]
    Validate(String),
    /// Layouter could not compute buffer/struct sizes.
    #[error("layout computation: {0}")]
    Layout(String),
    /// `@group(0)` sizes did not match [`FrameGpuUniforms`](crate::gpu::frame_globals::FrameGpuUniforms), [`GpuLight`](crate::backend::GpuLight), or cluster buffers.
    #[error(
        "group(0) must have uniform binding 0 size {expected_frame}, storage binding 1 stride {expected_light}, bindings 2-3 u32 stride {expected_cluster_u32}; got b0={got0:?} b1={got1:?} b2={got2:?} b3={got3:?}"
    )]
    FrameGroupMismatch {
        /// Expected `FrameGpuUniforms` uniform size in bytes.
        expected_frame: u32,
        /// Expected `GpuLight` struct stride in the lights storage buffer.
        expected_light: u32,
        /// Expected `u32` stride for cluster count / index buffers.
        expected_cluster_u32: u32,
        /// Observed binding 0 size, if any.
        got0: Option<u32>,
        /// Observed binding 1 stride, if any.
        got1: Option<u32>,
        /// Observed binding 2 stride, if any.
        got2: Option<u32>,
        /// Observed binding 3 stride, if any.
        got3: Option<u32>,
    },
    /// A global resource at the given group/binding is not supported for raster materials.
    #[error("unsupported global resource at group {group} binding {binding}: {reason}")]
    UnsupportedBinding {
        /// Bind group index (`0`-`2` for materials).
        group: u32,
        /// Binding index within the group.
        binding: u32,
        /// Human-readable reason (type, access, or shape).
        reason: String,
    },
    /// Bind group index outside `0..=2`.
    #[error("invalid bind group index {0} (only 0, 1, 2 are allowed for raster materials)")]
    InvalidBindGroup(u32),
    /// Composed embedded shader stem has no WGSL payload (build/embed mismatch).
    #[error("embedded composed WGSL missing for material stem `{0}`")]
    EmbeddedTargetMissing(&'static str),
    /// A bind group layout has more entries than the device allows.
    #[error("group {group} has {count} bindings (device max_bindings_per_bind_group={max})")]
    ExceedsBindingsPerGroup {
        /// Bind group index.
        group: u32,
        /// Reflected entry count.
        count: u32,
        /// Device cap.
        max: u32,
    },
    /// A shader stage has more samplers than the device allows.
    #[error("stage has {count} samplers (device max_samplers_per_shader_stage={max})")]
    ExceedsSamplersPerStage {
        /// Reflected sampler count for the stage.
        count: u32,
        /// Device cap.
        max: u32,
    },
    /// A shader stage has more sampled textures than the device allows.
    #[error(
        "stage has {count} sampled textures (device max_sampled_textures_per_shader_stage={max})"
    )]
    ExceedsSampledTexturesPerStage {
        /// Reflected sampled texture count for the stage.
        count: u32,
        /// Device cap.
        max: u32,
    },
    /// A uniform buffer entry's `min_binding_size` exceeds device caps.
    #[error(
        "uniform binding at group {group} binding {binding} requires {size} bytes (device max_uniform_buffer_binding_size={max})"
    )]
    UniformBindingExceedsLimit {
        /// Group index.
        group: u32,
        /// Binding index.
        binding: u32,
        /// Required min binding size in bytes.
        size: u64,
        /// Device cap.
        max: u64,
    },
    /// A storage buffer entry's `min_binding_size` exceeds device caps.
    #[error(
        "storage binding at group {group} binding {binding} requires {size} bytes (device max_storage_buffer_binding_size={max})"
    )]
    StorageBindingExceedsLimit {
        /// Group index.
        group: u32,
        /// Binding index.
        binding: u32,
        /// Required min binding size in bytes.
        size: u64,
        /// Device cap.
        max: u64,
    },
    /// Vertex layout has more buffers or attributes than the device allows.
    #[error(
        "vertex layout has {buffers} buffers / {attributes} attributes (device caps: max_vertex_buffers={max_buffers}, max_vertex_attributes={max_attributes})"
    )]
    VertexLayoutExceedsLimit {
        /// Number of vertex buffers.
        buffers: u32,
        /// Number of vertex attributes (across all buffers).
        attributes: u32,
        /// Device cap.
        max_buffers: u32,
        /// Device cap.
        max_attributes: u32,
    },
}
