//! Typed render-graph resources, access declarations, and import descriptors.

mod access;
mod handles;
mod imports;
mod transient;

pub(crate) use access::{AccessKind, ResourceAccess};
pub use access::{BufferAccess, StorageAccess, TextureAccess};
pub(crate) use handles::ResourceHandle;
pub use handles::{
    BufferHandle, BufferResourceHandle, ImportedBufferHandle, ImportedTextureHandle,
    SubresourceHandle, TextureAttachmentResolve, TextureAttachmentTarget, TextureHandle,
    TextureResourceHandle, TransientSubresourceDesc,
};
pub use imports::{
    BackendFrameBufferKind, BufferImportSource, FrameTargetRole, HistorySlotId, ImportSource,
    ImportedBufferDecl, ImportedTextureDecl,
};
pub use transient::{
    BufferSizePolicy, TransientArrayLayers, TransientBufferDesc, TransientExtent,
    TransientSampleCount, TransientTextureDesc, TransientTextureFormat,
};

#[cfg(test)]
mod tests;
