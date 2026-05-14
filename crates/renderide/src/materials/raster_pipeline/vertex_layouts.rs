//! Static `VertexBufferLayout` table for the mesh-forward raster pipeline.
//!
//! Mesh-forward shaders consume compact vertex streams (position, normal, UV0, color, tangent,
//! UV1, UV2, UV3) and optionally one packed wide-UV stream for UV0-UV7. The compact table is
//! data-driven: each location is declared once with its stride and component format;
//! [`mesh_forward_vertex_buffer_layout`] renders one layout for the compact buffer-slot order
//! selected by pipeline reflection.

/// Bytes per vertex in the packed wide-UV stream.
pub(super) const WIDE_UV_VERTEX_STRIDE: u64 = 8 * 16;

/// Per-stream descriptor used to materialise the mesh-forward vertex layout table.
struct VertexStreamDescriptor {
    /// Vertex shader input location.
    location: u32,
    /// Stride in bytes of one vertex in this stream.
    stride: u64,
    /// Vertex attribute format.
    format: wgpu::VertexFormat,
}

const MESH_FORWARD_STREAMS: [VertexStreamDescriptor; 8] = [
    VertexStreamDescriptor {
        location: 0,
        stride: 16,
        format: wgpu::VertexFormat::Float32x4,
    },
    VertexStreamDescriptor {
        location: 1,
        stride: 16,
        format: wgpu::VertexFormat::Float32x4,
    },
    VertexStreamDescriptor {
        location: 2,
        stride: 8,
        format: wgpu::VertexFormat::Float32x2,
    },
    VertexStreamDescriptor {
        location: 3,
        stride: 16,
        format: wgpu::VertexFormat::Float32x4,
    },
    VertexStreamDescriptor {
        location: 4,
        stride: 16,
        format: wgpu::VertexFormat::Float32x4,
    },
    VertexStreamDescriptor {
        location: 5,
        stride: 8,
        format: wgpu::VertexFormat::Float32x2,
    },
    VertexStreamDescriptor {
        location: 6,
        stride: 8,
        format: wgpu::VertexFormat::Float32x2,
    },
    VertexStreamDescriptor {
        location: 7,
        stride: 8,
        format: wgpu::VertexFormat::Float32x2,
    },
];

const MESH_FORWARD_ATTRIBUTES: [[wgpu::VertexAttribute; 1]; 8] = {
    let mut out = [[wgpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x4,
    }]; 8];
    let mut i = 0;
    while i < MESH_FORWARD_STREAMS.len() {
        out[i] = [wgpu::VertexAttribute {
            offset: 0,
            shader_location: MESH_FORWARD_STREAMS[i].location,
            format: MESH_FORWARD_STREAMS[i].format,
        }];
        i += 1;
    }
    out
};

/// Returns the mesh-forward vertex buffer layout for one shader input location.
pub(super) fn mesh_forward_vertex_buffer_layout(
    location: usize,
) -> wgpu::VertexBufferLayout<'static> {
    layout_at(location)
}

/// Returns a vertex buffer layout for the packed wide-UV stream.
pub(super) fn mesh_forward_wide_uv_vertex_buffer_layout(
    attributes: &[wgpu::VertexAttribute],
) -> wgpu::VertexBufferLayout<'_> {
    wgpu::VertexBufferLayout {
        array_stride: WIDE_UV_VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes,
    }
}

const fn layout_at(index: usize) -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: MESH_FORWARD_STREAMS[index].stride,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &MESH_FORWARD_ATTRIBUTES[index],
    }
}
