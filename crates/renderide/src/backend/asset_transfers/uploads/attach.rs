//! GPU attach: flush pending texture allocations and replay queued IPC payloads.

use std::sync::Arc;

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::MaterialSystem;
use crate::shared::MeshUploadData;

use super::super::{AssetTransferQueue, drain_asset_tasks_unbounded};
use super::allocations::{
    flush_pending_cubemap_allocations, flush_pending_render_texture_allocations,
    flush_pending_texture_allocations, flush_pending_texture3d_allocations,
};
use super::cubemap::try_cubemap_upload_with_device;
use super::mesh::try_process_mesh_upload;
use super::texture2d::try_texture_upload_with_device;
use super::texture3d::try_texture3d_upload_with_device;
use super::video_texture::attach_flush_pending_video_textures;

/// After GPU [`crate::backend::RenderBackend::attach`], allocate textures for pending
/// formats and replay queued mesh/texture payloads when shared memory is available, then
/// drain the asset integrator synchronously (no per-frame budget). When `ipc` is available,
/// completions emitted during replay use the same host acknowledgement path as live uploads.
pub fn attach_flush_pending_asset_uploads(
    queue: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    device: &Arc<wgpu::Device>,
    shm: Option<&mut SharedMemoryAccessor>,
    ipc: Option<&mut DualQueueIpc>,
) {
    flush_pending_texture_allocations(queue, device);
    flush_pending_texture3d_allocations(queue, device);
    flush_pending_cubemap_allocations(queue, device);
    flush_pending_render_texture_allocations(queue, device);
    attach_flush_pending_video_textures(queue);
    let pending_tex: Vec<_> = queue.pending.pending_texture_uploads.drain(..).collect();
    let pending_tex3d: Vec<_> = queue.pending.pending_texture3d_uploads.drain(..).collect();
    let pending_cube: Vec<_> = queue.pending.pending_cubemap_uploads.drain(..).collect();
    let pending_mesh: Vec<MeshUploadData> = queue.pending.pending_mesh_uploads.drain(..).collect();
    let mut ipc = ipc;
    if let Some(shm) = shm {
        for data in pending_tex {
            let ipc_ref = ipc.as_deref_mut();
            try_texture_upload_with_device(queue, data, shm, ipc_ref, false);
        }
        for data in pending_tex3d {
            let ipc_ref = ipc.as_deref_mut();
            try_texture3d_upload_with_device(queue, data, shm, ipc_ref, false);
        }
        for data in pending_cube {
            let ipc_ref = ipc.as_deref_mut();
            try_cubemap_upload_with_device(queue, data, shm, ipc_ref, false);
        }
        for data in pending_mesh {
            let ipc_ref = ipc.as_deref_mut();
            try_process_mesh_upload(queue, data, Some(&mut *shm), ipc_ref);
        }
        drain_asset_tasks_unbounded(queue, materials, shm, &mut ipc);
    } else {
        for data in pending_tex {
            queue.pending.pending_texture_uploads.push_back(data);
        }
        for data in pending_tex3d {
            queue.pending.pending_texture3d_uploads.push_back(data);
        }
        for data in pending_cube {
            queue.pending.pending_cubemap_uploads.push_back(data);
        }
        for data in pending_mesh {
            queue.pending.pending_mesh_uploads.push_back(data);
        }
    }
}
