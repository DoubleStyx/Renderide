//! Mesh and Texture2D upload queues, shared-memory ingestion, and CPU-side format tables for uploads.

use std::sync::Arc;

use crate::assets::mesh::try_upload_mesh_from_raw;
use crate::assets::texture::write_texture2d_mips;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::resources::GpuTexture2d;
use crate::shared::{
    MeshUnload, MeshUploadData, MeshUploadResult, RendererCommand, SetTexture2DData,
    SetTexture2DFormat, SetTexture2DProperties, SetTexture2DResult, TextureUpdateResultType,
    UnloadTexture2D,
};

use super::RenderBackend;

/// Max queued [`MeshUploadData`] when GPU is not ready yet (host data stays in shared memory).
pub const MAX_PENDING_MESH_UPLOADS: usize = 256;

/// Max queued texture data commands when GPU or format is not ready.
pub const MAX_PENDING_TEXTURE_UPLOADS: usize = 256;

/// After GPU [`RenderBackend::attach`](super::RenderBackend::attach), allocate textures for pending
/// formats and replay queued mesh/texture payloads when shared memory is available.
pub(super) fn attach_flush_pending_asset_uploads(
    backend: &mut RenderBackend,
    device: &Arc<wgpu::Device>,
    shm: Option<&mut SharedMemoryAccessor>,
) {
    flush_pending_texture_allocations(backend, device);
    let pending_tex: Vec<SetTexture2DData> = backend.pending_texture_uploads.drain(..).collect();
    let pending_mesh: Vec<MeshUploadData> = backend.pending_mesh_uploads.drain(..).collect();
    if let Some(shm) = shm {
        for data in pending_tex {
            try_texture_upload_with_device(backend, data, shm, None);
        }
        for data in pending_mesh {
            try_mesh_upload_with_device(backend, device, data, shm, None);
        }
    } else {
        for data in pending_tex {
            backend.pending_texture_uploads.push_back(data);
        }
        for data in pending_mesh {
            backend.pending_mesh_uploads.push_back(data);
        }
    }
}

pub(super) fn flush_pending_texture_allocations(
    backend: &mut RenderBackend,
    device: &Arc<wgpu::Device>,
) {
    let ids: Vec<i32> = backend.texture_formats.keys().copied().collect();
    for id in ids {
        if backend.texture_pool.get_texture(id).is_some() {
            continue;
        }
        let Some(fmt) = backend.texture_formats.get(&id).cloned() else {
            continue;
        };
        let props = backend.texture_properties.get(&id);
        let Some(tex) = GpuTexture2d::new_from_format(device.as_ref(), &fmt, props) else {
            logger::warn!("texture {id}: failed to allocate GPU texture on attach");
            continue;
        };
        let _ = backend.texture_pool.insert_texture(tex);
    }
}

fn send_texture_2d_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    update: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    ipc.send_background(RendererCommand::set_texture_2d_result(SetTexture2DResult {
        asset_id,
        r#type: TextureUpdateResultType(update),
        instance_changed,
    }));
}

/// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
pub(super) fn on_set_texture_2d_format(
    backend: &mut RenderBackend,
    f: SetTexture2DFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    backend.texture_formats.insert(id, f.clone());
    let props = backend.texture_properties.get(&id);
    let Some(device) = backend.gpu_device.clone() else {
        send_texture_2d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            backend.texture_pool.get_texture(id).is_none(),
        );
        return;
    };
    let Some(tex) = GpuTexture2d::new_from_format(device.as_ref(), &f, props) else {
        logger::warn!("texture {id}: SetTexture2DFormat rejected (bad size or device)");
        return;
    };
    let existed_before = backend.texture_pool.insert_texture(tex);
    send_texture_2d_result(
        ipc,
        id,
        TextureUpdateResultType::FORMAT_SET,
        !existed_before,
    );
    logger::info!(
        "texture {} format {:?} {}×{} mips={} (resident_bytes≈{})",
        id,
        f.format,
        f.width,
        f.height,
        f.mipmap_count,
        backend.texture_pool.accounting().texture_resident_bytes()
    );
}

/// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
pub(super) fn on_set_texture_2d_properties(
    backend: &mut RenderBackend,
    p: SetTexture2DProperties,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = p.asset_id;
    backend.texture_properties.insert(id, p.clone());
    if let Some(t) = backend.texture_pool.get_texture_mut(id) {
        t.apply_properties(&p);
    }
    send_texture_2d_result(ipc, id, TextureUpdateResultType::PROPERTIES_SET, false);
}

/// Handle [`SetTexture2DData`](crate::shared::SetTexture2DData). Pass shared memory when available
/// so mips can be read from the host buffer; if GPU or texture is not ready, data is queued.
pub(super) fn on_set_texture_2d_data(
    backend: &mut RenderBackend,
    d: SetTexture2DData,
    shm: Option<&mut SharedMemoryAccessor>,
    ipc: Option<&mut DualQueueIpc>,
) {
    if d.data.length <= 0 {
        return;
    }
    if !backend.texture_formats.contains_key(&d.asset_id) {
        logger::warn!(
            "texture {}: SetTexture2DData before format; ignored",
            d.asset_id
        );
        return;
    }
    if backend.gpu_device.is_none() || backend.gpu_queue.is_none() {
        if backend.pending_texture_uploads.len() >= MAX_PENDING_TEXTURE_UPLOADS {
            logger::warn!(
                "texture {}: pending texture upload queue full; dropping",
                d.asset_id
            );
            return;
        }
        backend.pending_texture_uploads.push_back(d);
        return;
    }
    let Some(ref device) = backend.gpu_device.clone() else {
        return;
    };
    if backend.texture_pool.get_texture(d.asset_id).is_none() {
        flush_pending_texture_allocations(backend, device);
    }
    if backend.texture_pool.get_texture(d.asset_id).is_none() {
        if backend.pending_texture_uploads.len() >= MAX_PENDING_TEXTURE_UPLOADS {
            logger::warn!(
                "texture {}: no GPU texture and pending full; dropping data",
                d.asset_id
            );
            return;
        }
        backend.pending_texture_uploads.push_back(d);
        return;
    }
    let Some(shm) = shm else {
        logger::warn!(
            "texture {}: SetTexture2DData needs shared memory for upload",
            d.asset_id
        );
        return;
    };
    try_texture_upload_with_device(backend, d, shm, ipc);
}

/// Upload texture mips from shared memory and optionally notify the host on the background queue.
pub(super) fn try_texture_upload_with_device(
    backend: &mut RenderBackend,
    data: SetTexture2DData,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = data.asset_id;
    let Some(fmt) = backend.texture_formats.get(&id).cloned() else {
        logger::warn!("texture {id}: missing format");
        return;
    };
    let (tex_arc, wgpu_fmt) = match backend.texture_pool.get_texture(id) {
        Some(t) => (t.texture.clone(), t.wgpu_format),
        None => {
            logger::warn!("texture {id}: missing GPU texture");
            return;
        }
    };
    let Some(queue_arc) = backend.gpu_queue.as_ref() else {
        return;
    };
    let upload_out = shm.with_read_bytes(&data.data, |raw| {
        let q = queue_arc.lock().unwrap_or_else(|e| e.into_inner());
        Some(write_texture2d_mips(
            &q,
            tex_arc.as_ref(),
            &fmt,
            wgpu_fmt,
            &data,
            raw,
        ))
    });
    match upload_out {
        Some(Ok(())) => {
            if let Some(t) = backend.texture_pool.get_texture_mut(id) {
                let uploaded_mips = data.mip_map_sizes.len() as u32;
                let start = data.start_mip_level.max(0) as u32;
                let end_exclusive = start.saturating_add(uploaded_mips).min(t.mip_levels_total);
                t.mip_levels_resident = t.mip_levels_resident.max(end_exclusive);
            }
            send_texture_2d_result(ipc, id, TextureUpdateResultType::DATA_UPLOAD, false);
            logger::trace!("texture {id}: data upload ok");
        }
        Some(Err(e)) => {
            logger::warn!("texture {id}: upload failed: {e}");
        }
        None => {
            logger::warn!("texture {id}: shared memory slice missing");
        }
    }
}

/// Remove a texture asset from CPU tables and the pool.
pub(super) fn on_unload_texture_2d(backend: &mut RenderBackend, u: UnloadTexture2D) {
    let id = u.asset_id;
    backend.texture_formats.remove(&id);
    backend.texture_properties.remove(&id);
    if backend.texture_pool.remove_texture(id) {
        logger::info!(
            "texture {id} unloaded (mesh≈{} tex≈{} total≈{})",
            backend.mesh_pool.accounting().mesh_resident_bytes(),
            backend.texture_pool.accounting().texture_resident_bytes(),
            backend.mesh_pool.accounting().total_resident_bytes()
        );
    }
}

/// Ingest mesh bytes from shared memory; notifies host when `ipc` is set.
pub(super) fn try_process_mesh_upload(
    backend: &mut RenderBackend,
    data: MeshUploadData,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
) {
    if data.buffer.length <= 0 {
        return;
    }
    let Some(device) = backend.gpu_device.clone() else {
        if backend.pending_mesh_uploads.len() >= MAX_PENDING_MESH_UPLOADS {
            logger::warn!(
                "mesh upload pending queue full; dropping asset {}",
                data.asset_id
            );
            return;
        }
        backend.pending_mesh_uploads.push_back(data);
        return;
    };
    try_mesh_upload_with_device(backend, &device, data, shm, ipc);
}

pub(super) fn try_mesh_upload_with_device(
    backend: &mut RenderBackend,
    device: &Arc<wgpu::Device>,
    data: MeshUploadData,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
) {
    let upload_result = shm.with_read_bytes(&data.buffer, |raw| {
        try_upload_mesh_from_raw(device.as_ref(), raw, &data)
    });
    let Some(mesh) = upload_result else {
        logger::warn!("mesh {}: upload failed or rejected", data.asset_id);
        return;
    };
    let existed_before = backend.mesh_pool.insert_mesh(mesh);
    if let Some(ipc) = ipc {
        ipc.send_background(RendererCommand::mesh_upload_result(MeshUploadResult {
            asset_id: data.asset_id,
            instance_changed: !existed_before,
        }));
    }
    logger::info!(
        "mesh {} uploaded (replaced={} resident_bytes≈{})",
        data.asset_id,
        existed_before,
        backend.mesh_pool.accounting().total_resident_bytes()
    );
}

/// Remove a mesh from the pool.
pub(super) fn on_mesh_unload(backend: &mut RenderBackend, u: MeshUnload) {
    if backend.mesh_pool.remove_mesh(u.asset_id) {
        logger::info!(
            "mesh {} unloaded (resident_bytes≈{})",
            u.asset_id,
            backend.mesh_pool.accounting().total_resident_bytes()
        );
    }
}
