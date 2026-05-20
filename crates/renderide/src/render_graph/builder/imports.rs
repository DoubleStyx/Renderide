//! Imported-resource final-access compilation.

use super::super::compiled::CompiledPassInfo;
use super::super::error::GraphBuildError;
use super::super::resources::{
    BufferResourceHandle, FrameTargetRole, ImportSource, ImportedBufferDecl, ImportedBufferHandle,
    ImportedTextureDecl, ImportedTextureHandle, ResourceHandle, TextureAccess,
    TextureResourceHandle,
};
use super::super::schedule::{
    ImportedFinalAccess, ImportedResourceFinalAccess, ImportedScheduleResource,
};

/// Builds the imported-resource final access plan and validates presentable frame targets.
pub(super) fn compile_imported_final_accesses(
    texture_imports: &[ImportedTextureDecl],
    buffer_imports: &[ImportedBufferDecl],
    pass_info: &[CompiledPassInfo],
) -> Result<Vec<ImportedResourceFinalAccess>, GraphBuildError> {
    let mut final_accesses = Vec::with_capacity(texture_imports.len() + buffer_imports.len());
    for (idx, import) in texture_imports.iter().enumerate() {
        let handle = ImportedTextureHandle(idx as u32);
        let written_by_retained_pass = imported_texture_written(pass_info, handle);
        if matches!(import.final_access, TextureAccess::Present)
            && matches!(
                import.source,
                ImportSource::Frame(FrameTargetRole::ColorAttachment)
            )
            && !written_by_retained_pass
        {
            return Err(GraphBuildError::MissingImportedFinalWriter {
                label: import.label,
                final_access: "present",
            });
        }
        final_accesses.push(ImportedResourceFinalAccess {
            label: import.label,
            resource: ImportedScheduleResource::Texture(handle),
            final_access: ImportedFinalAccess::Texture(import.final_access.clone()),
            written_by_retained_pass,
        });
    }
    for (idx, import) in buffer_imports.iter().enumerate() {
        let handle = ImportedBufferHandle(idx as u32);
        final_accesses.push(ImportedResourceFinalAccess {
            label: import.label,
            resource: ImportedScheduleResource::Buffer(handle),
            final_access: ImportedFinalAccess::Buffer(import.final_access),
            written_by_retained_pass: imported_buffer_written(pass_info, handle),
        });
    }
    Ok(final_accesses)
}

/// Returns whether retained passes need to acquire the frame surface.
pub(super) fn needs_surface_acquire(
    pass_info: &[CompiledPassInfo],
    imports: &[ImportedTextureDecl],
) -> bool {
    pass_info.iter().any(|pass| {
        pass.accesses.iter().any(|access| {
            if !access.writes() {
                return false;
            }
            let ResourceHandle::Texture(TextureResourceHandle::Imported(handle)) = access.resource
            else {
                return false;
            };
            imports.get(handle.index()).is_some_and(|decl| {
                matches!(
                    decl.source,
                    ImportSource::Frame(FrameTargetRole::ColorAttachment)
                )
            })
        })
    })
}

/// Returns whether any retained pass writes an imported texture.
fn imported_texture_written(pass_info: &[CompiledPassInfo], handle: ImportedTextureHandle) -> bool {
    pass_info.iter().any(|pass| {
        pass.accesses.iter().any(|access| {
            access.writes()
                && matches!(
                    access.resource,
                    ResourceHandle::Texture(TextureResourceHandle::Imported(h)) if h == handle
                )
        })
    })
}

/// Returns whether any retained pass writes an imported buffer.
fn imported_buffer_written(pass_info: &[CompiledPassInfo], handle: ImportedBufferHandle) -> bool {
    pass_info.iter().any(|pass| {
        pass.accesses.iter().any(|access| {
            access.writes()
                && matches!(
                    access.resource,
                    ResourceHandle::Buffer(BufferResourceHandle::Imported(h)) if h == handle
                )
        })
    })
}
