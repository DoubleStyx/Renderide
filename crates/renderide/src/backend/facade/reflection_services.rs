//! Reflection-probe services owned behind the backend facade.

use crate::backend::AssetTransferQueue;
use crate::backend::frame_gpu::ReflectionProbeSpecularResources;
use crate::gpu::GpuContext;
use crate::ipc::SharedMemoryAccessor;
use crate::reflection_probes::ReflectionProbeSh2System;
use crate::reflection_probes::specular::{
    ReflectionProbeFrameSelection, ReflectionProbeSpecularMaintainParams,
    ReflectionProbeSpecularSystem, RuntimeReflectionProbeCapture,
};
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::{FrameSubmitData, RenderingContext};
use hashbrown::HashSet;

/// Nonblocking reflection-probe projection, bake, cache, and selection services.
pub(super) struct ReflectionProbeServices {
    /// Nonblocking reflection-probe SH2 GPU projection service.
    sh2: ReflectionProbeSh2System,
    /// Reflection-probe specular IBL bake/cache/selection system.
    specular: ReflectionProbeSpecularSystem,
}

impl ReflectionProbeServices {
    /// Creates empty reflection-probe services.
    pub(super) fn new() -> Self {
        Self {
            sh2: ReflectionProbeSh2System::new(),
            specular: ReflectionProbeSpecularSystem::new(),
        }
    }

    /// Starts SH2 projection pipeline builds early so first probe use does not discover them lazily.
    pub(super) fn pre_warm_sh2_projection_pipelines(
        &mut self,
        device: &std::sync::Arc<wgpu::Device>,
    ) {
        self.sh2.pre_warm_projection_pipelines(device);
    }

    /// Answers host SH2 task rows for the latest frame submit without blocking GPU readback.
    pub(super) fn answer_sh2_frame_submit_tasks(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        scene: &SceneCoordinator,
        asset_transfers: &AssetTransferQueue,
        data: &FrameSubmitData,
    ) {
        let captures = self.specular.capture_store();
        self.sh2
            .answer_frame_submit_tasks(shm, scene, asset_transfers, captures, data);
    }

    /// Advances nonblocking SH2 GPU jobs and schedules queued projection work.
    pub(super) fn maintain_sh2_jobs(
        &mut self,
        gpu: &mut GpuContext,
        asset_transfers: &AssetTransferQueue,
    ) {
        self.sh2.maintain_gpu_jobs(gpu, asset_transfers);
    }

    /// Advances reflection-probe specular IBL jobs and returns frame-global probe bindings.
    pub(super) fn maintain_specular_jobs(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        asset_transfers: &AssetTransferQueue,
        render_context: RenderingContext,
        reflection_probe_sh2_enabled: bool,
        max_local_reflection_probes: usize,
    ) -> Option<ReflectionProbeSpecularResources> {
        self.specular
            .maintain(ReflectionProbeSpecularMaintainParams {
                gpu,
                scene,
                assets: asset_transfers,
                render_context,
                sh2_system: &mut self.sh2,
                reflection_probe_sh2_enabled,
                max_local_reflection_probes,
            });
        self.specular.resources()
    }

    /// CPU selection snapshot used by draw collection.
    pub(super) fn selection(&self) -> &ReflectionProbeFrameSelection {
        self.specular.selection()
    }

    /// Registers a completed OnChanges runtime cubemap capture.
    pub(super) fn register_runtime_capture(&mut self, capture: RuntimeReflectionProbeCapture) {
        self.specular.register_runtime_capture(capture);
    }

    /// Purges reflection-probe resources tied to closed render spaces.
    pub(super) fn purge_render_space_resources(&mut self, spaces: &[RenderSpaceId]) {
        if spaces.is_empty() {
            return;
        }
        let space_set: HashSet<RenderSpaceId> = spaces.iter().copied().collect();
        let sh2 = self.sh2.purge_render_space_resources(&space_set);
        let specular = self.specular.purge_render_space_resources(&space_set);
        if sh2 > 0 || specular > 0 {
            logger::debug!(
                "reflection probe purge: sh2_entries={} specular_entries={}",
                sh2,
                specular
            );
        }
    }
}
