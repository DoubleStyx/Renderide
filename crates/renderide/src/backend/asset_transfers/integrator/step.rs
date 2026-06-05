//! Per-task cooperative step dispatch for asset uploads.

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::{MaterialSystem, RasterPipelineKind};
use crate::shared::{MaterialsUpdateBatch, RendererCommand, ShaderUploadResult};

use super::super::AssetTransferQueue;
use super::super::cubemap_task::CubemapUploadTask;
use super::super::mesh_task::MeshUploadTask;
use super::super::particle_task::{PointRenderBufferTask, TrailRenderBufferTask};
use super::super::reliable_ack::send_background_reliable;
use super::super::texture_task::TextureUploadTask;
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
    /// Point render-buffer ingestion, acknowledgement, and generated mesh build.
    PointRenderBuffer(PointRenderBufferTask),
    /// Trail render-buffer ingestion, acknowledgement, and generated mesh build.
    TrailRenderBuffer(TrailRenderBufferTask),
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
        AssetTask::PointRenderBuffer(task) => task.step(asset, particle_task_gpu(gpu), shm, ipc),
        AssetTask::TrailRenderBuffer(task) => task.step(asset, particle_task_gpu(gpu), shm, ipc),
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
    let asset_id = route.asset_id;
    let _ = send_background_reliable(
        ipc,
        RendererCommand::ShaderUploadResult(ShaderUploadResult {
            asset_id,
            instance_changed: true,
        }),
        || {
            format!(
                "shader route asset_id={asset_id}: failed to enqueue reliable ShaderUploadResult ack"
            )
        },
    );
    StepResult::Done
}

pub(super) fn particle_task_gpu<'context, 'handles: 'context>(
    gpu: Option<&'context AssetUploadGpuContext<'handles>>,
) -> Option<super::super::particle_task::ParticleTaskGpu<'context>> {
    gpu.map(AssetUploadGpuContext::particle_task_gpu)
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
    task.step(asset, gpu.mesh_task_gpu(), shm, ipc)
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
    task.step(asset, gpu.texture_task_gpu(), shm, ipc)
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
    task.step(asset, gpu.texture_task_gpu(), shm, ipc)
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
    task.step(asset, gpu.texture_task_gpu(), shm, ipc)
}
