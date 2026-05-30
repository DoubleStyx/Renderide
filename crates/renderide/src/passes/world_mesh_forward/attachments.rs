//! Attachment helpers shared by world-mesh forward raster and encoder passes.

use crate::render_graph::pass::builder::RasterPassBuilder;
use crate::render_graph::resources::{ImportedTextureHandle, TextureHandle, TextureResourceHandle};

use super::{WorldMeshForwardGraphResources, WorldMeshForwardNormalGraphResources};

/// Concrete color/depth targets for an encoder-managed forward draw pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ForwardDrawAttachmentTargets {
    /// Color attachment selected for the current frame sample count.
    pub(super) color: TextureResourceHandle,
    /// Depth attachment selected for the current frame sample count.
    pub(super) depth: TextureResourceHandle,
}

struct ColorDepthAttachmentDesc {
    color: TextureHandle,
    msaa_color: Option<TextureHandle>,
    msaa_color_resolve_to: Option<TextureHandle>,
    depth: ImportedTextureHandle,
    msaa_depth: Option<TextureHandle>,
    color_ops: wgpu::Operations<wgpu::Color>,
    depth_ops: wgpu::Operations<f32>,
}

/// Declares a depth attachment that uses the imported single-sample target or the MSAA target.
pub(super) fn declare_forward_depth_attachment(
    r: &mut RasterPassBuilder<'_, '_>,
    depth: ImportedTextureHandle,
    msaa_depth: Option<TextureHandle>,
    depth_ops: wgpu::Operations<f32>,
) {
    if let Some(msaa_depth) = msaa_depth {
        r.frame_sampled_depth(depth, msaa_depth, depth_ops, None);
    } else {
        r.depth(depth, depth_ops, None);
    }
}

fn declare_color_depth_attachments(
    r: &mut RasterPassBuilder<'_, '_>,
    desc: ColorDepthAttachmentDesc,
) {
    let ColorDepthAttachmentDesc {
        color,
        msaa_color,
        msaa_color_resolve_to,
        depth,
        msaa_depth,
        color_ops,
        depth_ops,
    } = desc;
    if let (Some(msaa_color), Some(msaa_depth)) = (msaa_color, msaa_depth) {
        r.frame_sampled_color(color, msaa_color, color_ops, msaa_color_resolve_to);
        r.frame_sampled_depth(depth, msaa_depth, depth_ops, None);
    } else {
        r.color(color, color_ops, Option::<TextureHandle>::None);
        r.depth(depth, depth_ops, None);
    }
}

/// Declares world-mesh forward color/depth attachments.
pub(super) fn declare_forward_color_depth_attachments(
    r: &mut RasterPassBuilder<'_, '_>,
    resources: WorldMeshForwardGraphResources,
    color_ops: wgpu::Operations<wgpu::Color>,
    depth_ops: wgpu::Operations<f32>,
) {
    let msaa = resources.msaa;
    declare_color_depth_attachments(
        r,
        ColorDepthAttachmentDesc {
            color: resources.scene_color_hdr,
            msaa_color: msaa.map(|msaa| msaa.scene_color_hdr),
            msaa_color_resolve_to: None,
            depth: resources.depth,
            msaa_depth: msaa.map(|msaa| msaa.depth),
            color_ops,
            depth_ops,
        },
    );
}

/// Declares GTAO normal color/depth attachments.
pub(super) fn declare_normal_color_depth_attachments(
    r: &mut RasterPassBuilder<'_, '_>,
    resources: WorldMeshForwardNormalGraphResources,
    color_ops: wgpu::Operations<wgpu::Color>,
    depth_ops: wgpu::Operations<f32>,
) {
    declare_color_depth_attachments(
        r,
        ColorDepthAttachmentDesc {
            color: resources.normals,
            msaa_color: resources.normals_msaa,
            msaa_color_resolve_to: Some(resources.normals),
            depth: resources.depth,
            msaa_depth: resources.msaa_depth,
            color_ops,
            depth_ops,
        },
    );
}

/// Resolves concrete forward draw attachments for an encoder-managed render pass.
pub(super) fn forward_draw_attachment_targets(
    resources: WorldMeshForwardGraphResources,
    sample_count: u32,
) -> Option<ForwardDrawAttachmentTargets> {
    if sample_count > 1 {
        resources.msaa.map(|msaa| ForwardDrawAttachmentTargets {
            color: TextureResourceHandle::Transient(msaa.scene_color_hdr),
            depth: TextureResourceHandle::Transient(msaa.depth),
        })
    } else {
        Some(single_sample_forward_draw_attachment_targets(resources))
    }
}

fn single_sample_forward_draw_attachment_targets(
    resources: WorldMeshForwardGraphResources,
) -> ForwardDrawAttachmentTargets {
    ForwardDrawAttachmentTargets {
        color: TextureResourceHandle::Transient(resources.scene_color_hdr),
        depth: TextureResourceHandle::Imported(resources.depth),
    }
}

#[cfg(test)]
mod tests {
    use crate::render_graph::resources::{
        ImportedBufferHandle, ImportedTextureHandle, TextureHandle, TextureResourceHandle,
    };

    use super::super::ForwardMsaaResources;
    use super::{WorldMeshForwardGraphResources, forward_draw_attachment_targets};

    fn resources(msaa: Option<ForwardMsaaResources>) -> WorldMeshForwardGraphResources {
        WorldMeshForwardGraphResources {
            scene_color_hdr: TextureHandle(1),
            depth: ImportedTextureHandle(2),
            msaa,
            cluster_light_counts: ImportedBufferHandle(3),
            cluster_light_indices: ImportedBufferHandle(4),
            lights: ImportedBufferHandle(5),
            per_draw_slab: ImportedBufferHandle(6),
            frame_uniforms: ImportedBufferHandle(7),
        }
    }

    fn msaa_resources() -> ForwardMsaaResources {
        ForwardMsaaResources {
            scene_color_hdr: TextureHandle(8),
            depth: TextureHandle(9),
            depth_r32: TextureHandle(10),
        }
    }

    #[test]
    fn forward_draw_targets_use_single_sample_for_sample_count_one() {
        let targets =
            forward_draw_attachment_targets(resources(Some(msaa_resources())), 1).expect("targets");

        assert_eq!(
            targets.color,
            TextureResourceHandle::Transient(TextureHandle(1))
        );
        assert_eq!(
            targets.depth,
            TextureResourceHandle::Imported(ImportedTextureHandle(2))
        );
    }

    #[test]
    fn forward_draw_targets_use_msaa_for_multisampled_frames() {
        let targets =
            forward_draw_attachment_targets(resources(Some(msaa_resources())), 4).expect("targets");

        assert_eq!(
            targets.color,
            TextureResourceHandle::Transient(TextureHandle(8))
        );
        assert_eq!(
            targets.depth,
            TextureResourceHandle::Transient(TextureHandle(9))
        );
    }

    #[test]
    fn forward_draw_targets_reject_multisampled_frames_without_msaa_resources() {
        assert!(forward_draw_attachment_targets(resources(None), 4).is_none());
    }
}
