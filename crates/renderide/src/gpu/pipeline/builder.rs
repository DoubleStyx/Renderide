//! Shared pipeline construction helpers.
//!
//! Centralises the boilerplate that every builtin pipeline repeats:
//! primitive state, depth-stencil state, color target descriptors, and the
//! standard uniform-ring bind group layout / bind group.
//!
//! # Why so much repetition in the pipeline files?
//! Without offline shader compilation or runtime pipeline assembly there is one
//! [`wgpu::RenderPipeline`] per (vertex-format × shader-variant × depth-mode × output-targets)
//! combination. These helpers reduce per-pipeline boilerplate but cannot eliminate it entirely
//! until pipeline descriptors can be composed at runtime from shared fragments.

use super::core::{MAX_INSTANCE_RUN, UNIFORM_ALIGNMENT};
use super::mrt::{MRT_NORMAL_FORMAT, MRT_POSITION_FORMAT};

// ─── Primitive state ──────────────────────────────────────────────────────────

/// Standard CW-front, triangle-list, back-cull, fill primitive state.
///
/// Shared by all builtin pipelines. Override only when a pipeline needs
/// two-sided rendering or a different topology.
pub(crate) fn standard_primitive_state() -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Cw,
        cull_mode: Some(wgpu::Face::Back),
        unclipped_depth: false,
        polygon_mode: wgpu::PolygonMode::Fill,
        conservative: false,
    }
}

// ─── Depth-stencil states ─────────────────────────────────────────────────────

/// Depth-stencil for an opaque forward pass: `GreaterEqual` compare, depth write on.
pub(crate) fn depth_stencil_opaque() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::GreaterEqual,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Depth-stencil for screen-space overlays: `Always` compare, depth write off.
///
/// Available for overlay pipelines that completely ignore scene depth.
/// Currently unused because stencil-phase overlay pipelines carry custom stencil state
/// that must be constructed inline. Will become used once those pipelines are refactored.
#[allow(dead_code)]
pub(crate) fn depth_stencil_no_depth() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Depth-stencil that optionally disables depth testing.
///
/// Used by debug pipelines (`NormalDebug`, `UvDebug`) which can be placed in the
/// overlay pass as screen-space UI without occluding by scene depth.
pub(crate) fn depth_stencil_debug(disable_depth_test: bool) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: !disable_depth_test,
        depth_compare: if disable_depth_test {
            wgpu::CompareFunction::Always
        } else {
            wgpu::CompareFunction::GreaterEqual
        },
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

// ─── Color target states ──────────────────────────────────────────────────────

/// Single alpha-blending color target for the given surface format.
pub(crate) fn standard_color_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

/// Three-target MRT color states: surface color, position (`Rgba16Float`), normal (`Rgba16Float`).
///
/// Matches the G-buffer layout expected by [`super::mrt`] pipelines and the RTAO compute pass.
pub(crate) fn mrt_color_targets(
    surface_format: wgpu::TextureFormat,
) -> [Option<wgpu::ColorTargetState>; 3] {
    [
        Some(standard_color_target(surface_format)),
        Some(wgpu::ColorTargetState {
            format: MRT_POSITION_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: MRT_NORMAL_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ]
}

// ─── Bind group layout / bind group helpers ────────────────────────────────────

/// Bind group layout for a dynamic uniform ring buffer (vertex-visible only).
///
/// Used by all non-skinned debug pipelines and PBR (group 0). The binding is a
/// uniform buffer with dynamic offset sized for `MAX_INSTANCE_RUN × UNIFORM_ALIGNMENT` bytes.
pub(crate) fn uniform_ring_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(
                    (MAX_INSTANCE_RUN as u64) * UNIFORM_ALIGNMENT,
                ),
            },
            count: None,
        }],
    })
}

/// Bind group that connects a uniform ring buffer to the layout created by
/// [`uniform_ring_bind_group_layout`].
pub(crate) fn uniform_ring_bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new((MAX_INSTANCE_RUN as u64) * UNIFORM_ALIGNMENT),
            }),
        }],
    })
}

// ─── Vertex attribute constants ────────────────────────────────────────────────

/// Vertex attributes for `VertexPosNormal` (position: `Float32x3` @ 0, normal: `Float32x3` @ 12).
pub(crate) const POS_NORMAL_ATTRIBS: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 12,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x3,
    },
];

/// Vertex attributes for `VertexWithUv` (position: `Float32x3` @ 0, uv: `Float32x2` @ 12).
pub(crate) const POS_UV_ATTRIBS: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 12,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x2,
    },
];

/// Vertex attributes for `VertexSkinned`
/// (position @ 0, normal @ 12, tangent @ 24, bone_indices @ 36, bone_weights @ 52).
pub(crate) const SKINNED_ATTRIBS: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 12,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 24,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        offset: 36,
        shader_location: 3,
        format: wgpu::VertexFormat::Sint32x4,
    },
    wgpu::VertexAttribute {
        offset: 52,
        shader_location: 4,
        format: wgpu::VertexFormat::Float32x4,
    },
];
