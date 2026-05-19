//! Render-graph pass-builder helpers for declaring texture reads and color attachments.

use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::{PassBuilder, PassMergeHint};
use crate::render_graph::resources::{
    ImportedTextureHandle, TextureAccess, TextureHandle, TextureResourceHandle,
};

/// Declares a transient texture read by a fragment shader.
pub(in crate::passes) fn read_fragment_sampled_texture(
    b: &mut PassBuilder<'_>,
    handle: TextureHandle,
) {
    b.read_texture_resource(
        handle,
        TextureAccess::Sampled {
            stages: wgpu::ShaderStages::FRAGMENT,
        },
    );
}

/// Declares a color attachment write with no resolve target.
pub(in crate::passes) fn color_attachment(
    b: &mut PassBuilder<'_>,
    handle: impl Into<TextureResourceHandle>,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let preserves_attachment = matches!(&load, wgpu::LoadOp::Load);
    if preserves_attachment {
        b.merge_hint(PassMergeHint {
            attachment_reuse: true,
            tile_memory_preferred: true,
        });
    }
    let mut r = b.raster();
    r.color(
        handle,
        wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
        },
        Option::<TextureHandle>::None,
    );
}

/// Declares an imported color attachment write with no resolve target.
pub(in crate::passes) fn imported_color_attachment(
    b: &mut PassBuilder<'_>,
    handle: ImportedTextureHandle,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let preserves_attachment = matches!(&load, wgpu::LoadOp::Load);
    if preserves_attachment {
        b.merge_hint(PassMergeHint {
            attachment_reuse: true,
            tile_memory_preferred: true,
        });
    }
    let mut r = b.raster();
    r.color(
        handle,
        wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
        },
        Option::<ImportedTextureHandle>::None,
    );
}

/// Builds a missing-frame-params error with pass-specific context.
pub(in crate::passes) fn missing_pass_resource(
    pass: &str,
    detail: impl std::fmt::Display,
) -> RenderPassError {
    RenderPassError::FrameParamsRequired {
        pass: format!("{pass} ({detail})"),
    }
}
