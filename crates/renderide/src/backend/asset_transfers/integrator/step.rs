//! Per-task cooperative step dispatch for asset uploads.

use std::sync::Arc;

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::{MaterialSystem, RasterPipelineKind};
use crate::particles::{build_point_render_buffer_upload, build_trail_render_buffer_upload};
use crate::shared::{
    MaterialsUpdateBatch, PointRenderBufferConsumed, PointRenderBufferUpload, RendererCommand,
    ShaderUploadResult, TrailRenderBufferConsumed, TrailRenderBufferUpload,
};

use super::super::AssetTransferQueue;
use super::super::cubemap_task::CubemapUploadTask;
use super::super::mesh_task::MeshTaskGpu;
use super::super::mesh_task::MeshUploadTask;
use super::super::texture_task::TextureUploadTask;
use super::super::texture_task_common::TextureTaskGpu;
use super::super::texture3d_task::Texture3dUploadTask;
use super::gpu_context::AssetUploadGpuContext;

/// Shader-route registration plus host acknowledgement produced by the async shader resolver.
#[derive(Debug)]
pub struct ShaderRouteTask {
    /// Host shader asset id.
    pub asset_id: i32,
    /// Resolved raster pipeline.
    pub pipeline: RasterPipelineKind,
    /// Resolved AssetBundle shader asset name, when available.
    pub shader_asset_name: Option<String>,
    /// Froox shader variant bitmask parsed from the serialized Shader name suffix.
    pub shader_variant_bits: Option<u32>,
}

/// One cooperative integration task.
#[derive(Debug)]
pub enum AssetTask {
    /// Renderer-main-thread material batch application.
    MaterialUpdate(MaterialsUpdateBatch),
    /// Renderer-main-thread shader route registration.
    ShaderRoute(ShaderRouteTask),
    /// Placeholder point render-buffer ingestion and acknowledgement.
    PointRenderBuffer(PointRenderBufferUpload),
    /// Placeholder trail render-buffer ingestion and acknowledgement.
    TrailRenderBuffer(TrailRenderBufferUpload),
    /// Host mesh payload integration.
    Mesh(MeshUploadTask),
    /// Host Texture2D mip integration.
    Texture(TextureUploadTask),
    /// Host Texture3D mip integration.
    Texture3d(Texture3dUploadTask),
    /// Host cubemap face/mip integration.
    Cubemap(CubemapUploadTask),
}

/// Whether a task needs another step call in a later drain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepResult {
    /// More work remains for this logical upload.
    Continue,
    /// Upload finished (success or logged failure; host callbacks sent when applicable).
    Done,
    /// Task is waiting for a background thread to finish; push to the back of the queue.
    YieldBackground,
}

/// Returns a stable tag for [`AssetTask`] variants, used as Tracy zone data.
#[cfg_attr(
    not(feature = "tracy"),
    expect(dead_code, reason = "tag only consumed by Tracy zones")
)]
fn asset_task_kind_tag(task: &AssetTask) -> &'static str {
    match task {
        AssetTask::MaterialUpdate(_) => "MaterialUpdate",
        AssetTask::ShaderRoute(_) => "ShaderRoute",
        AssetTask::PointRenderBuffer(_) => "PointRenderBuffer",
        AssetTask::TrailRenderBuffer(_) => "TrailRenderBuffer",
        AssetTask::Mesh(_) => "Mesh",
        AssetTask::Texture(_) => "Texture",
        AssetTask::Texture3d(_) => "Texture3d",
        AssetTask::Cubemap(_) => "Cubemap",
    }
}

/// Dispatches a single task step, opening a Tracy zone tagged with the variant name.
pub(super) fn step_asset_task(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut AssetTask,
) -> StepResult {
    profiling::scope!("asset::upload", asset_task_kind_tag(task));
    match task {
        AssetTask::MaterialUpdate(batch) => step_material_update_task(materials, shm, ipc, batch),
        AssetTask::ShaderRoute(route) => step_shader_route_task(materials, ipc, route),
        AssetTask::PointRenderBuffer(upload) => {
            step_point_render_buffer_task(asset, gpu, shm, ipc, upload)
        }
        AssetTask::TrailRenderBuffer(upload) => {
            step_trail_render_buffer_task(asset, gpu, shm, ipc, upload)
        }
        AssetTask::Mesh(task) => step_mesh_upload_task(asset, gpu, shm, ipc, task),
        AssetTask::Texture(task) => step_texture_upload_task(asset, gpu, shm, ipc, task),
        AssetTask::Texture3d(task) => step_texture3d_upload_task(asset, gpu, shm, ipc, task),
        AssetTask::Cubemap(task) => step_cubemap_upload_task(asset, gpu, shm, ipc, task),
    }
}

fn step_material_update_task(
    materials: &mut MaterialSystem,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    batch: &mut MaterialsUpdateBatch,
) -> StepResult {
    let batch = std::mem::take(batch);
    if let Some(ipc) = ipc.as_deref_mut() {
        materials.apply_materials_update_batch(batch, shm, ipc);
    } else {
        logger::warn!(
            "materials update batch {}: IPC unavailable during integration; applying without ack",
            batch.update_batch_id
        );
        materials.apply_materials_update_batch_no_ack(batch, shm);
    }
    StepResult::Done
}

fn step_shader_route_task(
    materials: &mut MaterialSystem,
    ipc: &mut Option<&mut DualQueueIpc>,
    route: &mut ShaderRouteTask,
) -> StepResult {
    let shader_asset_name = route.shader_asset_name.take();
    materials.register_shader_route(
        route.asset_id,
        route.pipeline.clone(),
        shader_asset_name,
        route.shader_variant_bits,
    );
    if let Some(ipc) = ipc.as_deref_mut() {
        let ack_queued =
            ipc.send_background_reliable(RendererCommand::ShaderUploadResult(ShaderUploadResult {
                asset_id: route.asset_id,
                instance_changed: true,
            }));
        if !ack_queued {
            logger::warn!(
                "shader route asset_id={}: failed to enqueue reliable ShaderUploadResult ack",
                route.asset_id
            );
        }
    }
    StepResult::Done
}

fn step_point_render_buffer_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    upload: &mut PointRenderBufferUpload,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    let upload = std::mem::take(upload);
    let asset_id = upload.asset_id;
    let raw_len = upload.buffer.length.max(0) as usize;
    let raw = copy_render_buffer_payload(shm, upload.buffer, "point", asset_id, raw_len);
    match raw.and_then(|raw| {
        build_point_render_buffer_upload(
            crate::assets::mesh::MeshGpuUploadContext {
                device: gpu.device.as_ref(),
                queue: gpu.queue.as_ref(),
                gpu_limits: gpu.gpu_limits.as_ref(),
                mapped_buffer_health: gpu.mapped_buffer_health.as_ref(),
                mapped_buffer_generation: gpu.mapped_buffer_health.generation(),
            },
            raw,
            &upload,
        )
        .map_err(|err| {
            logger::warn!("{err}");
            err
        })
        .ok()
    }) {
        Some(result) => {
            let stored_asset_id = result.asset.asset_id;
            let count = result.asset.count;
            let frame_grid_size = result.asset.frame_grid_size;
            asset
                .catalogs
                .point_render_buffers
                .insert(asset_id, result.asset);
            asset.pools.mesh_pool.insert(result.billboard_mesh);
            logger::trace!(
                "point render buffer {stored_asset_id}: uploaded billboard mesh for {count} particles frame_grid={frame_grid_size:?}"
            );
        }
        None => {
            asset.catalogs.point_render_buffers.remove(&asset_id);
            for mesh_id in crate::particles::point_render_buffer_generated_mesh_ids(asset_id) {
                asset.pools.mesh_pool.remove(mesh_id);
            }
        }
    }
    send_point_render_buffer_consumed(ipc, asset_id);
    StepResult::Done
}

fn step_trail_render_buffer_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    upload: &mut TrailRenderBufferUpload,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    let upload = std::mem::take(upload);
    let asset_id = upload.asset_id;
    let raw_len = upload.buffer.length.max(0) as usize;
    let raw = copy_render_buffer_payload(shm, upload.buffer, "trail", asset_id, raw_len);
    match raw.and_then(|raw| {
        build_trail_render_buffer_upload(
            crate::assets::mesh::MeshGpuUploadContext {
                device: gpu.device.as_ref(),
                queue: gpu.queue.as_ref(),
                gpu_limits: gpu.gpu_limits.as_ref(),
                mapped_buffer_health: gpu.mapped_buffer_health.as_ref(),
                mapped_buffer_generation: gpu.mapped_buffer_health.generation(),
            },
            raw,
            &upload,
        )
        .map_err(|err| {
            logger::warn!("{err}");
            err
        })
        .ok()
    }) {
        Some(result) => {
            let stored_asset_id = result.asset.asset_id;
            let trails_count = result.asset.trails_count;
            let trail_point_count = result.asset.trail_point_count;
            asset
                .catalogs
                .trail_render_buffers
                .insert(asset_id, result.asset);
            for mesh in result.meshes {
                asset.pools.mesh_pool.insert(mesh);
            }
            logger::trace!(
                "trail render buffer {stored_asset_id}: uploaded trail meshes trails={trails_count} points={trail_point_count}"
            );
        }
        None => {
            asset.catalogs.trail_render_buffers.remove(&asset_id);
            for mesh_id in crate::particles::trail_render_buffer_generated_mesh_ids(asset_id) {
                asset.pools.mesh_pool.remove(mesh_id);
            }
        }
    }
    send_trail_render_buffer_consumed(ipc, asset_id);
    StepResult::Done
}

fn copy_render_buffer_payload(
    shm: &mut SharedMemoryAccessor,
    buffer: crate::shared::buffer::SharedMemoryBufferDescriptor,
    kind: &'static str,
    asset_id: i32,
    raw_len: usize,
) -> Option<Arc<[u8]>> {
    if raw_len == 0 {
        return Some(Arc::from([]));
    }
    shm.with_read_bytes(&buffer, |raw| {
        if raw.len() < raw_len {
            logger::warn!(
                "{kind} render buffer {asset_id}: raw too short (need {raw_len}, got {})",
                raw.len()
            );
            return None;
        }
        Some(Arc::from(&raw[..raw_len]))
    })
}

fn send_point_render_buffer_consumed(ipc: &mut Option<&mut DualQueueIpc>, asset_id: i32) {
    if let Some(ipc) = ipc.as_deref_mut() {
        let ack_queued = ipc.send_background_reliable(RendererCommand::PointRenderBufferConsumed(
            PointRenderBufferConsumed { asset_id },
        ));
        if !ack_queued {
            logger::warn!(
                "point render buffer {asset_id}: failed to enqueue reliable consumed ack"
            );
        }
    }
}

fn send_trail_render_buffer_consumed(ipc: &mut Option<&mut DualQueueIpc>, asset_id: i32) {
    if let Some(ipc) = ipc.as_deref_mut() {
        let ack_queued = ipc.send_background_reliable(RendererCommand::TrailRenderBufferConsumed(
            TrailRenderBufferConsumed { asset_id },
        ));
        if !ack_queued {
            logger::warn!(
                "trail render buffer {asset_id}: failed to enqueue reliable consumed ack"
            );
        }
    }
}

fn step_mesh_upload_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut MeshUploadTask,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    task.step(
        asset,
        MeshTaskGpu {
            device: gpu.device,
            gpu_limits: gpu.gpu_limits,
            queue: gpu.queue,
            mapped_buffer_health: gpu.mapped_buffer_health,
        },
        shm,
        ipc,
    )
}

fn step_texture_upload_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut TextureUploadTask,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    task.step(
        asset,
        TextureTaskGpu {
            device: gpu.device,
            queue: gpu.queue.as_ref(),
            queue_access_gate: gpu.gpu_queue_access_gate,
            queue_access_mode: gpu.queue_access_mode,
        },
        shm,
        ipc,
    )
}

fn step_texture3d_upload_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut Texture3dUploadTask,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    task.step(
        asset,
        TextureTaskGpu {
            device: gpu.device,
            queue: gpu.queue.as_ref(),
            queue_access_gate: gpu.gpu_queue_access_gate,
            queue_access_mode: gpu.queue_access_mode,
        },
        shm,
        ipc,
    )
}

fn step_cubemap_upload_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut CubemapUploadTask,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    task.step(
        asset,
        TextureTaskGpu {
            device: gpu.device,
            queue: gpu.queue.as_ref(),
            queue_access_gate: gpu.gpu_queue_access_gate,
            queue_access_mode: gpu.queue_access_mode,
        },
        shm,
        ipc,
    )
}
