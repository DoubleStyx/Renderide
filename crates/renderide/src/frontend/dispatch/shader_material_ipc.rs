//! Shader routing and material batch IPC handlers.

use crossbeam_channel::{Receiver, TryRecvError, bounded};

use crate::assets::{ResolvedShaderUpload, resolve_shader_upload};
use crate::backend::RenderBackend;
use crate::frontend::RendererFrontend;
use crate::shared::{
    MaterialsUpdateBatch, RendererCommand, ShaderUnload, ShaderUpload, ShaderUploadResult,
};

/// In-flight `resolve_shader_upload` work dispatched to the rayon pool by
/// [`on_shader_upload`].
///
/// The host sends the upload; resolving its AssetBundle shader asset name can read up to 32 MiB from disk,
/// which must not block the IPC poll on the main thread. The resolver runs on a rayon worker,
/// and [`drain_pending_shader_resolutions`] applies the result on a subsequent tick by
/// calling [`RenderBackend::register_shader_route`] and sending `ShaderUploadResult`.
pub(crate) struct PendingShaderResolution {
    /// Host `asset_id` echoed back in the eventual `ShaderUploadResult`.
    asset_id: i32,
    /// Receives the resolved pipeline from the rayon worker (bounded(1)).
    rx: Receiver<ResolvedShaderUpload>,
}

/// Spawns disk-bound shader upload resolution on the rayon pool so the main thread stays free.
///
/// The IPC acknowledgement (`ShaderUploadResult`) and pipeline route registration run only after
/// the resolver completes, via [`drain_pending_shader_resolutions`] on a later tick, preserving
/// the lock-step invariant that the host sees routing ready before it sends dependent materials.
pub(crate) fn on_shader_upload(pending: &mut Vec<PendingShaderResolution>, upload: ShaderUpload) {
    let asset_id = upload.asset_id;
    let (tx, rx) = bounded::<ResolvedShaderUpload>(1);
    rayon::spawn(move || {
        let resolved = resolve_shader_upload(&upload);
        let _ = tx.send(resolved);
    });
    pending.push(PendingShaderResolution { asset_id, rx });
}

/// Applies completed rayon-side [`resolve_shader_upload`] results: registers the pipeline route
/// and sends the `ShaderUploadResult` IPC acknowledgement.
///
/// Called at the top of [`crate::runtime::RendererRuntime::poll_ipc`] so resolutions landing
/// between ticks are applied before this tick's IPC batch (which may reference the routes) is
/// dispatched.
pub(crate) fn drain_pending_shader_resolutions(
    pending: &mut Vec<PendingShaderResolution>,
    backend: &mut RenderBackend,
    frontend: &mut RendererFrontend,
) {
    profiling::scope!("ipc::drain_pending_shader_resolutions");
    let mut completed: Vec<(i32, ResolvedShaderUpload)> = Vec::new();
    pending.retain_mut(|p| match p.rx.try_recv() {
        Ok(resolved) => {
            completed.push((p.asset_id, resolved));
            false
        }
        Err(TryRecvError::Empty) => true,
        Err(TryRecvError::Disconnected) => {
            logger::error!(
                "shader_upload: resolver task disconnected for asset_id={}",
                p.asset_id
            );
            false
        }
    });
    for (asset_id, resolved) in completed {
        logger::info!(
            "shader_upload: asset_id={} shader_asset_name={:?} raster_pipeline={:?}",
            asset_id,
            resolved.shader_asset_name.as_deref(),
            resolved.pipeline,
        );
        let shader_asset_name = resolved.shader_asset_name.clone();
        backend.register_shader_route(asset_id, resolved.pipeline, shader_asset_name);
        if let Some(ipc) = frontend.ipc_mut() {
            let _ = ipc.send_background(RendererCommand::ShaderUploadResult(ShaderUploadResult {
                asset_id,
                instance_changed: true,
            }));
        }
    }
}

pub(crate) fn on_shader_unload(backend: &mut RenderBackend, unload: ShaderUnload) {
    let id = unload.asset_id;
    backend.unregister_shader_route(id);
}

pub(crate) fn on_materials_update_batch(
    frontend: &mut RendererFrontend,
    backend: &mut RenderBackend,
    batch: MaterialsUpdateBatch,
) {
    if frontend.shared_memory().is_none() {
        if !backend.enqueue_materials_batch_no_shm(batch) {
            // already logged
        }
        return;
    }
    let (shm, ipc) = frontend.transport_pair_mut();
    let (Some(shm), Some(ipc)) = (shm, ipc) else {
        return;
    };
    backend.apply_materials_update_batch(batch, shm, ipc);
}
