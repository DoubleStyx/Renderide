//! Transient attachment store-op reduction for retained raster passes.

use hashbrown::HashMap;

use super::decl::SetupEntry;
use crate::render_graph::pass::AttachmentStoreOp;
use crate::render_graph::resources::{
    ResourceHandle, SubresourceHandle, TextureAttachmentResolve, TextureAttachmentTarget,
    TextureHandle, TextureResourceHandle, TransientSubresourceDesc,
};

/// Store/discard and resolve diagnostics emitted by attachment store optimization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TransientAttachmentStoreStats {
    /// Retained attachment resolves.
    pub(super) attachment_resolve_count: usize,
    /// Retained transient attachment stores.
    pub(super) store_count: usize,
    /// Retained transient attachment discards.
    pub(super) discard_count: usize,
    /// Coarse one-pixel bandwidth estimate for retained attachment traffic.
    pub(super) estimated_bandwidth_bytes: u64,
}

/// Estimates one attachment pixel for diagnostics when runtime extents are unavailable.
const COLOR_ATTACHMENT_PIXEL_BYTES: u64 = 8;
/// Estimates one depth/stencil attachment pixel for diagnostics when runtime extents are unavailable.
const DEPTH_ATTACHMENT_PIXEL_BYTES: u64 = 4;

/// Rewrites final-use transient attachment stores to `Discard`.
pub(super) fn optimize_transient_attachment_stores(
    setups: &mut [SetupEntry],
    subresources: &[TransientSubresourceDesc],
    retained_ord: &HashMap<usize, usize>,
) -> TransientAttachmentStoreStats {
    let last_use_by_texture = last_retained_texture_uses(setups, subresources, retained_ord);
    for (pass_idx, entry) in setups.iter_mut().enumerate() {
        let Some(&ordinal) = retained_ord.get(&pass_idx) else {
            continue;
        };
        for color in &mut entry.setup.color_attachments {
            color.store = optimized_store_for_target(
                color.target,
                ordinal,
                &last_use_by_texture,
                color.store,
            );
        }
        if let Some(depth) = entry.setup.depth_stencil_attachment.as_mut() {
            depth.depth.store = optimized_store_for_target(
                depth.target,
                ordinal,
                &last_use_by_texture,
                depth.depth.store,
            );
        }
    }
    collect_store_stats(setups, retained_ord)
}

fn last_retained_texture_uses(
    setups: &[SetupEntry],
    subresources: &[TransientSubresourceDesc],
    retained_ord: &HashMap<usize, usize>,
) -> HashMap<TextureHandle, usize> {
    let mut last_use_by_texture = HashMap::new();
    for (pass_idx, entry) in setups.iter().enumerate() {
        let Some(&ordinal) = retained_ord.get(&pass_idx) else {
            continue;
        };
        for access in &entry.setup.accesses {
            if let Some(handle) = transient_texture_for_resource(access.resource, subresources) {
                last_use_by_texture
                    .entry(handle)
                    .and_modify(|last: &mut usize| *last = (*last).max(ordinal))
                    .or_insert(ordinal);
            }
        }
    }
    last_use_by_texture
}

fn transient_texture_for_resource(
    resource: ResourceHandle,
    subresources: &[TransientSubresourceDesc],
) -> Option<TextureHandle> {
    match resource {
        ResourceHandle::Texture(TextureResourceHandle::Transient(handle)) => Some(handle),
        ResourceHandle::Texture(TextureResourceHandle::Imported(_)) | ResourceHandle::Buffer(_) => {
            None
        }
        ResourceHandle::TextureSubresource(handle) => {
            transient_texture_for_subresource(handle, subresources)
        }
    }
}

fn transient_texture_for_subresource(
    handle: SubresourceHandle,
    subresources: &[TransientSubresourceDesc],
) -> Option<TextureHandle> {
    subresources.get(handle.index()).map(|desc| desc.parent)
}

fn optimized_store_for_target(
    target: TextureAttachmentTarget,
    ordinal: usize,
    last_use_by_texture: &HashMap<TextureHandle, usize>,
    current: AttachmentStoreOp,
) -> AttachmentStoreOp {
    match target {
        TextureAttachmentTarget::Resource(handle) => AttachmentStoreOp::static_op(
            optimized_store_for_resource(handle, ordinal, last_use_by_texture, current.resolve(1)),
        ),
        TextureAttachmentTarget::FrameSampled {
            single_sample,
            multisampled,
        } => AttachmentStoreOp::frame_sampled(
            optimized_store_for_resource(
                single_sample,
                ordinal,
                last_use_by_texture,
                current.resolve(1),
            ),
            optimized_store_for_resource(
                multisampled,
                ordinal,
                last_use_by_texture,
                current.resolve(2),
            ),
        ),
    }
}

fn optimized_store_for_resource(
    handle: TextureResourceHandle,
    ordinal: usize,
    last_use_by_texture: &HashMap<TextureHandle, usize>,
    current: wgpu::StoreOp,
) -> wgpu::StoreOp {
    match handle {
        TextureResourceHandle::Transient(handle)
            if last_use_by_texture
                .get(&handle)
                .is_some_and(|&last| last == ordinal) =>
        {
            wgpu::StoreOp::Discard
        }
        TextureResourceHandle::Transient(_) | TextureResourceHandle::Imported(_) => current,
    }
}

fn collect_store_stats(
    setups: &[SetupEntry],
    retained_ord: &HashMap<usize, usize>,
) -> TransientAttachmentStoreStats {
    let mut stats = TransientAttachmentStoreStats::default();
    for (pass_idx, entry) in setups.iter().enumerate() {
        if !retained_ord.contains_key(&pass_idx) {
            continue;
        }
        for color in &entry.setup.color_attachments {
            stats.attachment_resolve_count = stats
                .attachment_resolve_count
                .saturating_add(usize::from(color.resolve_to.is_some()));
            add_target_store_stats(
                &mut stats,
                color.target,
                color.store,
                COLOR_ATTACHMENT_PIXEL_BYTES,
            );
            if color.resolve_to.is_some_and(resolve_target_is_transient) {
                stats.estimated_bandwidth_bytes = stats
                    .estimated_bandwidth_bytes
                    .saturating_add(COLOR_ATTACHMENT_PIXEL_BYTES);
            }
        }
        if let Some(depth) = entry.setup.depth_stencil_attachment.as_ref() {
            add_target_store_stats(
                &mut stats,
                depth.target,
                depth.depth.store,
                DEPTH_ATTACHMENT_PIXEL_BYTES,
            );
        }
    }
    stats
}

fn add_target_store_stats(
    stats: &mut TransientAttachmentStoreStats,
    target: TextureAttachmentTarget,
    store: AttachmentStoreOp,
    pixel_bytes: u64,
) {
    match target {
        TextureAttachmentTarget::Resource(TextureResourceHandle::Transient(_)) => {
            add_store_stat(stats, store.resolve(1), pixel_bytes);
        }
        TextureAttachmentTarget::Resource(TextureResourceHandle::Imported(_)) => {}
        TextureAttachmentTarget::FrameSampled {
            single_sample,
            multisampled,
        } => {
            if matches!(single_sample, TextureResourceHandle::Transient(_)) {
                add_store_stat(stats, store.resolve(1), pixel_bytes);
            }
            if single_sample != multisampled
                && matches!(multisampled, TextureResourceHandle::Transient(_))
            {
                add_store_stat(stats, store.resolve(2), pixel_bytes);
            }
        }
    }
}

fn resolve_target_is_transient(target: TextureAttachmentResolve) -> bool {
    match target {
        TextureAttachmentResolve::Always(handle)
        | TextureAttachmentResolve::FrameMultisampled(handle) => {
            matches!(handle, TextureResourceHandle::Transient(_))
        }
    }
}

fn add_store_stat(
    stats: &mut TransientAttachmentStoreStats,
    store: wgpu::StoreOp,
    pixel_bytes: u64,
) {
    match store {
        wgpu::StoreOp::Store => {
            stats.store_count = stats.store_count.saturating_add(1);
            stats.estimated_bandwidth_bytes =
                stats.estimated_bandwidth_bytes.saturating_add(pixel_bytes);
        }
        wgpu::StoreOp::Discard => {
            stats.discard_count = stats.discard_count.saturating_add(1);
        }
    }
}
