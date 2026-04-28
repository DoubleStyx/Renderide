//! Resource access declarations used by render-graph ordering and validation.

use super::handles::{BufferResourceHandle, ResourceHandle, TextureResourceHandle};

/// Read/write intent for storage resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageAccess {
    /// Read-only storage access.
    ReadOnly,
    /// Write-only storage access.
    WriteOnly,
    /// Read/write storage access.
    ReadWrite,
}

impl StorageAccess {
    /// Returns whether this storage access writes.
    pub(crate) fn writes(self) -> bool {
        matches!(self, Self::WriteOnly | Self::ReadWrite)
    }

    /// Returns whether this storage access reads.
    pub(crate) fn reads(self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReadWrite)
    }
}

/// Declared texture access for one pass.
#[derive(Clone, Debug, PartialEq)]
pub enum TextureAccess {
    /// Color attachment access. `resolve_to` may point to a transient or imported texture.
    ColorAttachment {
        /// Attachment load operation.
        load: wgpu::LoadOp<wgpu::Color>,
        /// Attachment store operation.
        store: wgpu::StoreOp,
        /// Optional resolve target for multisampled color attachments.
        resolve_to: Option<TextureResourceHandle>,
    },
    /// Depth/stencil attachment access.
    DepthAttachment {
        /// Depth load/store operations.
        depth: wgpu::Operations<f32>,
        /// Optional stencil load/store operations.
        stencil: Option<wgpu::Operations<u32>>,
    },
    /// Sampled texture binding.
    Sampled {
        /// Shader stages that sample the texture.
        stages: wgpu::ShaderStages,
    },
    /// Storage texture binding.
    Storage {
        /// Shader stages that access the storage texture.
        stages: wgpu::ShaderStages,
        /// Storage access mode.
        access: StorageAccess,
    },
    /// Copy source.
    CopySrc,
    /// Copy destination.
    CopyDst,
    /// Imported texture is finalized for presentation.
    Present,
}

impl TextureAccess {
    /// Minimum texture usage required by this access.
    pub fn usage(&self) -> wgpu::TextureUsages {
        match self {
            Self::ColorAttachment { .. } | Self::DepthAttachment { .. } | Self::Present => {
                wgpu::TextureUsages::RENDER_ATTACHMENT
            }
            Self::Sampled { .. } => wgpu::TextureUsages::TEXTURE_BINDING,
            Self::Storage { .. } => wgpu::TextureUsages::STORAGE_BINDING,
            Self::CopySrc => wgpu::TextureUsages::COPY_SRC,
            Self::CopyDst => wgpu::TextureUsages::COPY_DST,
        }
    }

    /// Returns whether this access reads prior resource contents.
    pub(crate) fn reads(&self) -> bool {
        match self {
            Self::Sampled { .. } | Self::CopySrc => true,
            Self::Storage { access, .. } => access.reads(),
            Self::DepthAttachment { depth, stencil } => {
                matches!(depth.load, wgpu::LoadOp::Load)
                    || stencil
                        .as_ref()
                        .is_some_and(|ops| matches!(ops.load, wgpu::LoadOp::Load))
            }
            Self::ColorAttachment { load, .. } => matches!(load, wgpu::LoadOp::Load),
            Self::CopyDst | Self::Present => false,
        }
    }

    /// Returns whether this access writes resource contents.
    pub(crate) fn writes(&self) -> bool {
        match self {
            Self::Sampled { .. } | Self::CopySrc => false,
            Self::Storage { access, .. } => access.writes(),
            Self::ColorAttachment { .. }
            | Self::DepthAttachment { .. }
            | Self::CopyDst
            | Self::Present => true,
        }
    }

    /// Returns whether this access is a raster attachment.
    pub(crate) fn is_attachment(&self) -> bool {
        matches!(
            self,
            Self::ColorAttachment { .. } | Self::DepthAttachment { .. }
        )
    }
}

/// Declared buffer access for one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    /// Uniform buffer binding.
    Uniform {
        /// Shader stages that bind the uniform buffer.
        stages: wgpu::ShaderStages,
        /// Whether the binding uses a dynamic offset.
        dynamic_offset: bool,
    },
    /// Storage buffer binding.
    Storage {
        /// Shader stages that access the storage buffer.
        stages: wgpu::ShaderStages,
        /// Storage access mode.
        access: StorageAccess,
    },
    /// Index buffer binding.
    Index,
    /// Vertex buffer binding.
    Vertex,
    /// Indirect draw/dispatch buffer.
    Indirect,
    /// Copy source.
    CopySrc,
    /// Copy destination.
    CopyDst,
}

impl BufferAccess {
    /// Minimum buffer usage required by this access.
    pub fn usage(self) -> wgpu::BufferUsages {
        match self {
            Self::Uniform { .. } => wgpu::BufferUsages::UNIFORM,
            Self::Storage { .. } => wgpu::BufferUsages::STORAGE,
            Self::Index => wgpu::BufferUsages::INDEX,
            Self::Vertex => wgpu::BufferUsages::VERTEX,
            Self::Indirect => wgpu::BufferUsages::INDIRECT,
            Self::CopySrc => wgpu::BufferUsages::COPY_SRC,
            Self::CopyDst => wgpu::BufferUsages::COPY_DST,
        }
    }

    /// Returns whether this access reads prior buffer contents.
    pub(crate) fn reads(self) -> bool {
        match self {
            Self::Uniform { .. } | Self::Index | Self::Vertex | Self::Indirect | Self::CopySrc => {
                true
            }
            Self::Storage { access, .. } => access.reads(),
            Self::CopyDst => false,
        }
    }

    /// Returns whether this access writes buffer contents.
    pub(crate) fn writes(self) -> bool {
        match self {
            Self::Storage { access, .. } => access.writes(),
            Self::CopyDst => true,
            Self::Uniform { .. } | Self::Index | Self::Vertex | Self::Indirect | Self::CopySrc => {
                false
            }
        }
    }
}

/// One declared access by one pass.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResourceAccess {
    /// Resource key.
    pub(crate) resource: ResourceHandle,
    /// Access metadata.
    pub(crate) access: AccessKind,
    /// Whether the access reads.
    pub(crate) reads: bool,
    /// Whether the access writes.
    pub(crate) writes: bool,
}

impl ResourceAccess {
    /// Builds a texture access declaration.
    pub(crate) fn texture(
        handle: TextureResourceHandle,
        access: TextureAccess,
        reads: bool,
        writes: bool,
    ) -> Self {
        Self {
            resource: ResourceHandle::Texture(handle),
            access: AccessKind::Texture(access),
            reads,
            writes,
        }
    }

    /// Builds a buffer access declaration.
    pub(crate) fn buffer(
        handle: BufferResourceHandle,
        access: BufferAccess,
        reads: bool,
        writes: bool,
    ) -> Self {
        Self {
            resource: ResourceHandle::Buffer(handle),
            access: AccessKind::Buffer(access),
            reads,
            writes,
        }
    }

    /// Returns whether this declared access reads.
    pub(crate) fn reads(&self) -> bool {
        self.reads
    }

    /// Returns whether this declared access writes.
    pub(crate) fn writes(&self) -> bool {
        self.writes
    }

    /// Returns whether this declared access is a raster attachment.
    pub(crate) fn is_attachment(&self) -> bool {
        matches!(&self.access, AccessKind::Texture(access) if access.is_attachment())
    }

    /// Returns the minimum texture usage for this access, if it is a texture access.
    pub(crate) fn texture_usage(&self) -> Option<wgpu::TextureUsages> {
        match &self.access {
            AccessKind::Texture(access) => Some(access.usage()),
            AccessKind::Buffer(_) => None,
        }
    }

    /// Returns the minimum buffer usage for this access, if it is a buffer access.
    pub(crate) fn buffer_usage(&self) -> Option<wgpu::BufferUsages> {
        match self.access {
            AccessKind::Buffer(access) => Some(access.usage()),
            AccessKind::Texture(_) => None,
        }
    }
}

/// Access kind for dependency analysis.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AccessKind {
    /// Texture access metadata.
    Texture(TextureAccess),
    /// Buffer access metadata.
    Buffer(BufferAccess),
}
