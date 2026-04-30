//! Static `VertexBufferLayout` table for the mesh-forward raster pipeline.
//!
//! Mesh-forward shaders consume eight independent vertex streams (position, normal, uv0, color,
//! tangent, uv1, uv2, uv3). The table is data-driven: each location is declared once with its
//! stride and component format; [`mesh_forward_vertex_buffer_layouts`] renders that table into
//! the eight `wgpu::VertexBufferLayout<'static>` values the pipeline builder needs.

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

/// Returns the mesh-forward vertex buffer layout table.
pub(super) fn mesh_forward_vertex_buffer_layouts() -> [wgpu::VertexBufferLayout<'static>; 8] {
    [
        layout_at(0),
        layout_at(1),
        layout_at(2),
        layout_at(3),
        layout_at(4),
        layout_at(5),
        layout_at(6),
        layout_at(7),
    ]
}

/// Returns the position + normal vertex buffer layouts shared by geometry-only mesh passes.
pub(crate) fn mesh_forward_position_normal_vertex_buffer_layouts()
-> [wgpu::VertexBufferLayout<'static>; 2] {
    [layout_at(0), layout_at(1)]
}

const fn layout_at(index: usize) -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: MESH_FORWARD_STREAMS[index].stride,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &MESH_FORWARD_ATTRIBUTES[index],
    }
}
