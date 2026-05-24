//! Nonblocking SH2 projection cache and GPU-job scheduler.

mod gpu_source;
mod pipeline_cache;
mod scheduling;
mod source_keys;
mod task_pass;

#[cfg(test)]
mod tests;

pub(super) use gpu_source::GpuSh2Source;
#[cfg(test)]
pub(super) use source_keys::SkyParamMode;
pub(super) use source_keys::{
    CubemapResidency, CubemapSourceMaterialIdentity, DEFAULT_SAMPLE_SIZE,
    MAX_PENDING_JOB_AGE_FRAMES, SH2_OUTPUT_BYTES, Sh2ProjectParams, Sh2SourceKey,
};

use std::collections::VecDeque;
use std::sync::Arc;

use glam::Vec3;
use hashbrown::{HashMap, HashSet};

use super::sh2_math::constant_color_sh2;
use super::source_resolution::Sh2ResolvedSource;
use crate::backend::AssetTransferQueue;
use crate::gpu::GpuContext;
use crate::ipc::SharedMemoryAccessor;
use crate::profiling;
use crate::reflection_probes::specular::RuntimeReflectionProbeCaptureStore;
use crate::scene::SceneCoordinator;
use crate::shared::{FrameSubmitData, RenderSH2};
use crate::skybox::specular::SkyboxIblSource;

use pipeline_cache::ProjectionPipelineCache;
use source_keys::sh2_key_matches_closed_spaces;
use task_pass::Sh2TaskSourceContext;

use super::readback_jobs::Sh2ReadbackJobs;

/// Maximum completed SH2 projections retained before pruning to recently touched sources.
const MAX_COMPLETED_SH2_CACHE_ENTRIES: usize = 512;

/// Nonblocking SH2 projection cache and GPU-job scheduler.
pub struct ReflectionProbeSh2System {
    /// Completed projection results keyed by source identity.
    completed: HashMap<Sh2SourceKey, RenderSH2>,
    /// In-flight GPU readback jobs keyed by source identity.
    readback_jobs: Sh2ReadbackJobs,
    /// Sources that failed recently.
    failed: HashSet<Sh2SourceKey>,
    /// Source payloads awaiting an in-flight slot.
    queued_sources: HashMap<Sh2SourceKey, GpuSh2Source>,
    /// FIFO ordering for [`Self::queued_sources`].
    queue_order: VecDeque<Sh2SourceKey>,
    /// Lazy-built SH2 projection pipelines.
    pipeline_cache: ProjectionPipelineCache,
    /// Source keys touched by the current task pass.
    touched_this_pass: HashSet<Sh2SourceKey>,
}

impl Default for ReflectionProbeSh2System {
    fn default() -> Self {
        Self::new()
    }
}

impl ReflectionProbeSh2System {
    /// Creates an empty SH2 system.
    pub fn new() -> Self {
        Self {
            completed: HashMap::new(),
            readback_jobs: Sh2ReadbackJobs::new(),
            failed: HashSet::new(),
            queued_sources: HashMap::new(),
            queue_order: VecDeque::new(),
            pipeline_cache: ProjectionPipelineCache::new(),
            touched_this_pass: HashSet::new(),
        }
    }

    /// Purges queued, pending, failed, and completed SH2 work tied to closed render spaces.
    pub(crate) fn purge_render_space_resources(
        &mut self,
        spaces: &HashSet<crate::scene::RenderSpaceId>,
    ) -> usize {
        if spaces.is_empty() {
            return 0;
        }
        profiling::scope!("reflection_probe_sh2::purge_render_space_resources");
        let before = self.completed.len()
            + self.failed.len()
            + self.queued_sources.len()
            + self.readback_jobs.len();
        self.completed
            .retain(|key, _| !sh2_key_matches_closed_spaces(key, spaces));
        self.failed
            .retain(|key| !sh2_key_matches_closed_spaces(key, spaces));
        self.queued_sources
            .retain(|key, _| !sh2_key_matches_closed_spaces(key, spaces));
        let queued_sources = &self.queued_sources;
        self.queue_order
            .retain(|key| queued_sources.contains_key(key));
        self.touched_this_pass
            .retain(|key| !sh2_key_matches_closed_spaces(key, spaces));
        self.readback_jobs
            .retain(|key| !sh2_key_matches_closed_spaces(key, spaces));
        let after = self.completed.len()
            + self.failed.len()
            + self.queued_sources.len()
            + self.readback_jobs.len();
        before.saturating_sub(after)
    }

    /// Answers every SH2 task row in a frame submit without blocking for GPU readback.
    pub fn answer_frame_submit_tasks(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        scene: &SceneCoordinator,
        assets: &AssetTransferQueue,
        captures: &RuntimeReflectionProbeCaptureStore,
        data: &FrameSubmitData,
    ) {
        profiling::scope!("reflection_probe_sh2::answer_frame_submit_tasks");
        self.touched_this_pass.clear();
        for update in &data.render_spaces {
            let Some(tasks) = update.reflection_probe_sh2_taks.as_ref() else {
                continue;
            };
            self.answer_task_buffer(
                shm,
                Sh2TaskSourceContext {
                    scene,
                    assets,
                    captures,
                    render_space_id: update.id,
                },
                tasks,
            );
        }
        self.prune_untouched_failures();
    }

    /// Advances GPU callbacks, maps completed buffers, and schedules queued work.
    pub fn maintain_gpu_jobs(&mut self, gpu: &mut GpuContext, assets: &AssetTransferQueue) {
        profiling::scope!("reflection_probe_sh2::maintain_gpu_jobs");
        let _ = gpu.device().poll(wgpu::PollType::Poll);
        let outcomes = self.readback_jobs.maintain();
        for (key, sh) in outcomes.completed {
            self.failed.remove(&key);
            self.completed.insert(key, sh);
        }
        for (key, reason) in outcomes.failed {
            logger::warn!("reflection_probe_sh2: GPU SH2 readback failed for {key:?}: {reason:?}");
            self.failed.insert(key);
        }
        self.pipeline_cache.drain_completed_builds();
        self.schedule_queued_sources(gpu, assets);
        self.prune_completed_cache_if_needed();
    }

    /// Starts background builds for all SH2 projection pipelines before the first probe asks for them.
    pub fn pre_warm_projection_pipelines(&mut self, device: &Arc<wgpu::Device>) {
        profiling::scope!("reflection_probe_sh2::pre_warm_projection_pipelines");
        self.pipeline_cache.pre_warm(device);
    }

    /// Ensures an SH2 projection exists for a renderer-owned reflection-probe IBL source.
    ///
    /// Returns [`Some`] only after the source has a completed CPU or GPU projection. GPU-backed
    /// sources are queued on cache misses and complete through [`Self::maintain_gpu_jobs`].
    pub(crate) fn ensure_ibl_source(
        &mut self,
        render_space_id: i32,
        source: &SkyboxIblSource,
    ) -> Option<RenderSH2> {
        let (key, source) = sh2_source_from_ibl_source(render_space_id, source);
        self.ensure_resolved_source(key, source)
    }

    pub(super) fn ensure_resolved_source(
        &mut self,
        key: Sh2SourceKey,
        source: Sh2ResolvedSource,
    ) -> Option<RenderSH2> {
        self.touched_this_pass.insert(key.clone());
        if let Some(sh) = self.completed.get(&key) {
            return Some(*sh);
        }
        if self.readback_jobs.contains_key(&key) || self.failed.contains(&key) {
            return None;
        }
        match source {
            Sh2ResolvedSource::Cpu(sh) => {
                let sh = *sh;
                self.completed.insert(key, sh);
                Some(sh)
            }
            Sh2ResolvedSource::Gpu(gpu_source) => {
                self.queue_source(key, gpu_source);
                None
            }
            Sh2ResolvedSource::Postpone => None,
        }
    }

    /// Queues a source for later GPU scheduling.
    fn queue_source(&mut self, key: Sh2SourceKey, source: GpuSh2Source) {
        if self.queued_sources.contains_key(&key) {
            return;
        }
        self.queue_order.push_back(key.clone());
        self.queued_sources.insert(key, source);
    }

    /// Drops failed keys that are no longer present in host task rows.
    fn prune_untouched_failures(&mut self) {
        self.failed
            .retain(|key| self.touched_this_pass.contains(key));
    }

    /// Bounds completed SH2 cache growth without dropping currently active sources.
    fn prune_completed_cache_if_needed(&mut self) {
        if self.completed.len() <= MAX_COMPLETED_SH2_CACHE_ENTRIES {
            return;
        }
        let before = self.completed.len();
        self.completed
            .retain(|key, _| self.touched_this_pass.contains(key));
        let removed = before.saturating_sub(self.completed.len());
        if removed > 0 {
            logger::debug!("reflection_probe_sh2: pruned {removed} completed SH2 cache entries");
        }
    }
}

fn sh2_source_from_ibl_source(
    render_space_id: i32,
    source: &SkyboxIblSource,
) -> (Sh2SourceKey, Sh2ResolvedSource) {
    match source {
        SkyboxIblSource::Cubemap(src) => {
            let identity = CubemapSourceMaterialIdentity {
                material_asset_id: src.material_asset_id,
                material_generation: src.material_generation,
                route_hash: src.route_hash,
            };
            let residency = CubemapResidency {
                allocation_generation: src.allocation_generation,
                size: src.face_size,
                resident_mips: src.mip_levels_resident,
                content_generation: src.content_generation,
                storage_v_inverted: src.storage_v_inverted,
            };
            (
                Sh2SourceKey::cubemap(
                    render_space_id,
                    identity,
                    src.asset_id,
                    residency,
                    src.clear_color,
                ),
                Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap {
                    asset_id: src.asset_id,
                    storage_v_inverted: src.storage_v_inverted,
                    clear_color: src.clear_color,
                }),
            )
        }
        SkyboxIblSource::SolidColor(src) => (
            Sh2SourceKey::ConstantColor {
                render_space_id,
                color_bits: src.color.map(|f| f.to_bits()),
            },
            Sh2ResolvedSource::Cpu(Box::new(constant_color_sh2(Vec3::new(
                src.color[0],
                src.color[1],
                src.color[2],
            )))),
        ),
        SkyboxIblSource::RuntimeCubemap(src) => (
            Sh2SourceKey::RuntimeCubemap {
                render_space_id,
                renderable_index: src.renderable_index,
                generation: src.generation,
                size: src.face_size,
                sample_size: DEFAULT_SAMPLE_SIZE,
                clear_color_key: src.clear_color.map(|c| c.to_array().map(|f| f.to_bits())),
            },
            Sh2ResolvedSource::Gpu(GpuSh2Source::RuntimeCubemap {
                clear_color: src.clear_color,
                texture: src.texture.clone(),
                view: src.view.clone(),
            }),
        ),
    }
}
