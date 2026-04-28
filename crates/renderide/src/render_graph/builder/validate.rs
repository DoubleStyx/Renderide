//! Setup-time handle validation for declared graph resources.

use super::super::error::SetupError;
use super::super::pass::PassSetup;
use super::super::resources::{
    AccessKind, BufferResourceHandle, ResourceHandle, TextureAccess, TextureResourceHandle,
};

pub(super) fn validate_handles(
    setup: &PassSetup,
    texture_count: usize,
    buffer_count: usize,
    subresource_count: usize,
    imported_texture_count: usize,
    imported_buffer_count: usize,
) -> Result<(), SetupError> {
    for access in &setup.accesses {
        validate_resource_handle(
            access.resource,
            texture_count,
            buffer_count,
            subresource_count,
            imported_texture_count,
            imported_buffer_count,
        )?;
        if let AccessKind::Texture(TextureAccess::ColorAttachment {
            resolve_to: Some(resolve_to),
            ..
        }) = &access.access
        {
            validate_resource_handle(
                ResourceHandle::Texture(*resolve_to),
                texture_count,
                buffer_count,
                subresource_count,
                imported_texture_count,
                imported_buffer_count,
            )?;
        }
    }
    Ok(())
}

fn validate_resource_handle(
    resource: ResourceHandle,
    texture_count: usize,
    buffer_count: usize,
    subresource_count: usize,
    imported_texture_count: usize,
    imported_buffer_count: usize,
) -> Result<(), SetupError> {
    match resource {
        ResourceHandle::Texture(TextureResourceHandle::Transient(h))
            if h.index() >= texture_count =>
        {
            Err(SetupError::UnknownTexture(h))
        }
        ResourceHandle::Buffer(BufferResourceHandle::Transient(h)) if h.index() >= buffer_count => {
            Err(SetupError::UnknownBuffer(h))
        }
        ResourceHandle::Texture(TextureResourceHandle::Imported(h))
            if h.index() >= imported_texture_count =>
        {
            Err(SetupError::UnknownImportedTexture(h))
        }
        ResourceHandle::Buffer(BufferResourceHandle::Imported(h))
            if h.index() >= imported_buffer_count =>
        {
            Err(SetupError::UnknownImportedBuffer(h))
        }
        ResourceHandle::TextureSubresource(h) if h.index() >= subresource_count => {
            Err(SetupError::UnknownSubresource(h))
        }
        _ => Ok(()),
    }
}
