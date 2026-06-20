use std::sync::Arc;

use hashbrown::{HashMap, HashSet};

use crate::gpu::{FrameSubmitKind, GpuContext};
use crate::gpu::{GpuReflectionProbeMetadata, REFLECTION_PROBE_ATLAS_FORMAT};
use crate::reflection_probes::ReflectionProbeCubemapAssets;
use crate::scene::{RenderSpaceId, SceneCoordinator, reflection_probe_solid_color};
use crate::shared::{
    ReflectionProbeTimeSlicingMode, ReflectionProbeType, RenderSH2, RenderingContext,
};
use crate::skybox::ibl_cache::{
    IblBakePolicy, SkyboxIblCache, SkyboxIblKey, build_key, clamp_face_size, mip_extent,
    mip_levels_for_edge,
};
use crate::skybox::specular::SkyboxIblSource;
use crate::{profiling, reflection_probes::ReflectionProbeSh2System};

use super::atlas::{AtlasCopyJob, ReflectionProbeAtlas, max_atlas_slots};
use super::captures::{
    RuntimeReflectionProbeCapture, RuntimeReflectionProbeCaptureKey,
    RuntimeReflectionProbeCaptureStore,
};
use super::resources::ReflectionProbeSpecularResources;
use super::selection::{ReflectionProbeFrameSelection, SpatialProbe};
use super::source::{metadata_for_spatial, resolve_probe_source, spatial_probe_for_state};

/// Default destination face size for reflection-probe IBL bakes.
const DEFAULT_REFLECTION_PROBE_FACE_SIZE: u32 = 256;
/// First atlas slot is reserved as a non-sampled black fallback.
const FIRST_PROBE_ATLAS_SLOT: u16 = 1;

/// Inputs for advancing specular reflection-probe IBL and selection state.
pub(crate) struct ReflectionProbeSpecularMaintainParams<'a> {
    /// GPU context used for IBL jobs and atlas writes.
    pub(crate) gpu: &'a mut GpuContext,
    /// Scene snapshot containing render spaces and reflection-probe entries.
    pub(crate) scene: &'a SceneCoordinator,
    /// Asset queues and pools used to resolve uploaded cubemaps.
    pub(crate) assets: &'a dyn ReflectionProbeCubemapAssets,
    /// Render context used for reflection-probe world transform lookup.
    pub(crate) render_context: RenderingContext,
    /// SH2 projection service used when reflection-probe diffuse SH is enabled.
    pub(crate) sh2_system: &'a mut ReflectionProbeSh2System,
    /// Whether reflection probes should contribute SH2 indirect diffuse lighting.
    pub(crate) reflection_probe_sh2_enabled: bool,
    /// Maximum number of local reflection probes that can contribute to reflections on a single mesh.
    pub(crate) max_local_reflection_probes: usize,
}

/// Specular reflection-probe bake/cache/selection system.
pub struct ReflectionProbeSpecularSystem {
    ibl_cache: SkyboxIblCache,
    atlas: Option<ReflectionProbeAtlas>,
    resources: Option<ReflectionProbeSpecularResources>,
    selection: ReflectionProbeFrameSelection,
    captures: RuntimeReflectionProbeCaptureStore,
    space_cache: HashMap<RenderSpaceId, CachedSpace>,
    dirty_spaces: HashSet<RenderSpaceId>,
    collect_config: Option<ProbeCollectConfig>,
    sync_signature: Option<SpecularSyncSignature>,
    last_stats: MaintainStats,
    /// Last source that finished IBL and optional SH2 work for each probe.
    last_ready: HashMap<ProbeIdentity, LastReadyProbe>,
    /// Highest runtime capture generation with a completed final specular IBL cube.
    runtime_final_ready_generation: HashMap<ProbeIdentity, u64>,
    version: u64,
}

impl Default for ReflectionProbeSpecularSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ReflectionProbeSpecularSystem {
    /// Creates an empty reflection-probe specular system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ibl_cache: SkyboxIblCache::new(),
            atlas: None,
            resources: None,
            selection: ReflectionProbeFrameSelection::default(),
            captures: RuntimeReflectionProbeCaptureStore::default(),
            space_cache: HashMap::new(),
            dirty_spaces: HashSet::new(),
            collect_config: None,
            sync_signature: None,
            last_stats: MaintainStats::default(),
            last_ready: HashMap::new(),
            runtime_final_ready_generation: HashMap::new(),
            version: 1,
        }
    }

    /// Highest runtime capture generation whose final specular IBL cube is ready.
    pub(crate) fn final_ready_generation(
        &self,
        space_id: i32,
        renderable_index: i32,
    ) -> Option<u64> {
        self.runtime_final_ready_generation
            .get(&ProbeIdentity {
                space_id: RenderSpaceId(space_id),
                renderable_index,
            })
            .copied()
    }

    /// Registers a completed runtime cubemap capture for a dynamic reflection probe.
    pub(crate) fn register_runtime_capture(&mut self, capture: RuntimeReflectionProbeCapture) {
        self.dirty_spaces.insert(capture.key.space_id);
        self.sync_signature = None;
        self.captures.insert(capture);
    }

    /// Marks render spaces whose reflection-probe selection state may need refresh.
    pub(crate) fn mark_render_spaces_dirty<I>(&mut self, spaces: I)
    where
        I: IntoIterator<Item = RenderSpaceId>,
    {
        let mut any_dirty = false;
        for space in spaces {
            any_dirty |= self.dirty_spaces.insert(space);
        }
        if any_dirty {
            self.sync_signature = None;
        }
    }

    /// Runtime dynamic capture store used by SH2 task resolution.
    #[must_use]
    pub(crate) fn capture_store(&self) -> &RuntimeReflectionProbeCaptureStore {
        &self.captures
    }

    /// Purges reflection-probe GPU resources tied to closed render spaces.
    pub(crate) fn purge_render_space_resources(
        &mut self,
        spaces: &HashSet<RenderSpaceId>,
    ) -> usize {
        if spaces.is_empty() {
            return 0;
        }
        profiling::scope!("reflection_probes::specular::purge_render_space_resources");
        let captures = self.captures.purge_spaces(spaces);
        let last_ready_before = self.last_ready.len();
        self.last_ready
            .retain(|identity, _probe| !spaces.contains(&identity.space_id));
        let last_ready = last_ready_before.saturating_sub(self.last_ready.len());
        self.runtime_final_ready_generation
            .retain(|identity, _generation| !spaces.contains(&identity.space_id));
        let cache_before = self.space_cache.len();
        self.space_cache
            .retain(|space_id, _cache| !spaces.contains(space_id));
        let cached_spaces = cache_before.saturating_sub(self.space_cache.len());
        self.dirty_spaces
            .retain(|space_id| !spaces.contains(space_id));
        let ibl = self
            .ibl_cache
            .purge_where(|key| specular_ibl_key_matches_closed_spaces(key, spaces));
        let removed = captures
            .saturating_add(ibl)
            .saturating_add(last_ready)
            .saturating_add(cached_spaces);
        if removed > 0 {
            self.version = self.version.wrapping_add(1);
            self.sync_signature = None;
        }
        removed
    }

    /// Advances GPU bakes, updates the atlas, and rebuilds the CPU selection index.
    pub(crate) fn maintain(&mut self, mut params: ReflectionProbeSpecularMaintainParams<'_>) {
        profiling::scope!("reflection_probes::specular::maintain");
        let mut stats = MaintainStats::default();
        self.ibl_cache.maintain_gpu_jobs(params.gpu);
        let face_size = clamp_face_size(DEFAULT_REFLECTION_PROBE_FACE_SIZE, params.gpu.limits());
        self.refresh_collect_config(ProbeCollectConfig {
            face_size,
            render_context: params.render_context,
            reflection_probe_sh2_enabled: params.reflection_probe_sh2_enabled,
        });
        let mut collected = CollectedProbeResources::default();

        self.selection
            .set_max_local_reflection_probes(params.max_local_reflection_probes);
        self.collect_probe_resources(&mut params, face_size, &mut collected, &mut stats);
        self.captures.retain_active(&collected.active_capture_keys);
        self.last_ready
            .retain(|identity, _probe| collected.active_identities.contains(identity));
        self.runtime_final_ready_generation
            .retain(|identity, _generation| collected.active_identities.contains(identity));
        self.ibl_cache
            .prune_completed_except(&collected.active_keys);
        collected.ready.sort_unstable_by_key(|probe| {
            (probe.identity.space_id.0, probe.identity.renderable_index)
        });
        stats.ready_probes = collected.ready.len();
        stats.ibl_pending = self.ibl_cache.pending_len();
        stats.ibl_active_sliced = self.ibl_cache.active_sliced_len();
        stats.ibl_completed = self.ibl_cache.completed_len();
        self.sync_atlas_and_selection(
            params.gpu,
            face_size,
            params.max_local_reflection_probes,
            collected.ready,
            &mut stats,
        );
        plot_maintain_stats(&stats);
        self.last_stats = stats;
    }

    fn collect_probe_resources(
        &mut self,
        params: &mut ReflectionProbeSpecularMaintainParams<'_>,
        face_size: u32,
        collected: &mut CollectedProbeResources,
        stats: &mut MaintainStats,
    ) {
        profiling::scope!("reflection_probes::specular::collect");
        let mut active_spaces = HashSet::new();
        for space_id in params.scene.render_space_ids() {
            let Some(space) = params.scene.space(space_id) else {
                continue;
            };
            if !space.is_active() {
                continue;
            }
            active_spaces.insert(space_id);
            stats.active_spaces = stats.active_spaces.saturating_add(1);
            stats.scanned_probes = stats
                .scanned_probes
                .saturating_add(space.reflection_probes().len());

            let dirty = self.dirty_spaces.contains(&space_id);
            if !dirty {
                let summary =
                    self.collect_space_source_summary(params, space_id, space, face_size, stats);
                if let Some(cache) = self.space_cache.get(&space_id)
                    && cache.summary == summary
                {
                    stats.reused_spaces = stats.reused_spaces.saturating_add(1);
                    collected.extend_cached(cache);
                    continue;
                }
            }

            let cache = self.collect_space_probe_cache(params, space_id, space, face_size, stats);
            collected.extend_cached(&cache);
            self.space_cache.insert(space_id, cache);
            self.dirty_spaces.remove(&space_id);
        }
        self.space_cache
            .retain(|space_id, _cache| active_spaces.contains(space_id));
        self.dirty_spaces
            .retain(|space_id| active_spaces.contains(space_id));
    }

    fn collect_space_source_summary(
        &mut self,
        params: &mut ReflectionProbeSpecularMaintainParams<'_>,
        space_id: RenderSpaceId,
        space: crate::scene::RenderSpaceView<'_>,
        face_size: u32,
        stats: &mut MaintainStats,
    ) -> CachedSpaceSummary {
        profiling::scope!("reflection_probes::specular::collect_source_summary");
        let mut summary = CachedSpaceSummary::default();
        for probe in space.reflection_probes() {
            self.collect_probe_source_summary(
                params,
                space_id,
                probe,
                face_size,
                &mut summary,
                stats,
            );
        }
        summary.normalize();
        summary
    }

    fn collect_probe_source_summary(
        &mut self,
        params: &mut ReflectionProbeSpecularMaintainParams<'_>,
        space_id: RenderSpaceId,
        probe: &crate::scene::ReflectionProbeEntry,
        face_size: u32,
        summary: &mut CachedSpaceSummary,
        stats: &mut MaintainStats,
    ) {
        let identity = ProbeIdentity {
            space_id,
            renderable_index: probe.renderable_index,
        };
        if matches!(
            probe.state.r#type,
            ReflectionProbeType::OnChanges | ReflectionProbeType::Realtime
        ) && !reflection_probe_solid_color(probe.state)
        {
            summary
                .active_capture_keys
                .insert(RuntimeReflectionProbeCaptureKey {
                    space_id,
                    renderable_index: probe.renderable_index,
                });
        }
        let Some(source) = resolve_probe_source(space_id, probe, params.assets, &self.captures)
        else {
            return;
        };
        summary.active_identities.insert(identity);
        let runtime_generation = runtime_source_generation(&source);
        let key = build_key(&source, face_size);
        summary.active_keys.insert(key.clone());
        let sh2 = params
            .reflection_probe_sh2_enabled
            .then(|| params.sh2_system.ensure_ibl_source(space_id.0, &source))
            .flatten();
        let policy = ibl_policy_for_probe_source(probe.state.time_slicing_mode, &source);
        if self
            .ibl_cache
            .ensure_source_with_policy(params.gpu, key.clone(), source, policy)
        {
            stats.scheduled_ibl_bakes = stats.scheduled_ibl_bakes.saturating_add(1);
        }
        let current_ready = self
            .ibl_cache
            .completed_cube(&key)
            .filter(|_cube| !params.reflection_probe_sh2_enabled || sh2.is_some())
            .map(|cube| (key.clone(), cube.mip_levels, sh2.is_some()));
        if let Some((key, mip_levels, has_sh2)) = current_ready {
            if let Some(generation) = runtime_generation {
                self.runtime_final_ready_generation
                    .insert(identity, generation);
            }
            summary.ready.push(ReadyProbeSummary {
                identity,
                key,
                mip_levels,
                has_sh2,
            });
            return;
        }
        if let Some(fallback) = self.last_ready.get(&identity) {
            if params.reflection_probe_sh2_enabled && fallback.sh2.is_none() {
                return;
            }
            summary.active_keys.insert(fallback.key.clone());
            summary.ready.push(ReadyProbeSummary {
                identity,
                key: fallback.key.clone(),
                mip_levels: fallback.mip_levels,
                has_sh2: fallback.sh2.is_some(),
            });
        }
    }

    fn collect_space_probe_cache(
        &mut self,
        params: &mut ReflectionProbeSpecularMaintainParams<'_>,
        space_id: RenderSpaceId,
        space: crate::scene::RenderSpaceView<'_>,
        face_size: u32,
        stats: &mut MaintainStats,
    ) -> CachedSpace {
        profiling::scope!("reflection_probes::specular::collect_space");
        let mut cache = CachedSpace::default();
        for probe in space.reflection_probes() {
            self.collect_probe_resource(params, space_id, probe, face_size, &mut cache, stats);
        }
        cache.summary.normalize();
        cache
    }

    fn collect_probe_resource(
        &mut self,
        params: &mut ReflectionProbeSpecularMaintainParams<'_>,
        space_id: RenderSpaceId,
        probe: &crate::scene::ReflectionProbeEntry,
        face_size: u32,
        cache: &mut CachedSpace,
        stats: &mut MaintainStats,
    ) {
        let identity = ProbeIdentity {
            space_id,
            renderable_index: probe.renderable_index,
        };
        if matches!(
            probe.state.r#type,
            ReflectionProbeType::OnChanges | ReflectionProbeType::Realtime
        ) && !reflection_probe_solid_color(probe.state)
        {
            cache
                .summary
                .active_capture_keys
                .insert(RuntimeReflectionProbeCaptureKey {
                    space_id,
                    renderable_index: probe.renderable_index,
                });
        }
        let Some(source) = resolve_probe_source(space_id, probe, params.assets, &self.captures)
        else {
            return;
        };
        cache.summary.active_identities.insert(identity);
        let runtime_generation = runtime_source_generation(&source);
        let key = build_key(&source, face_size);
        cache.summary.active_keys.insert(key.clone());
        let sh2 = params
            .reflection_probe_sh2_enabled
            .then(|| params.sh2_system.ensure_ibl_source(space_id.0, &source))
            .flatten();
        let policy = ibl_policy_for_probe_source(probe.state.time_slicing_mode, &source);
        if self
            .ibl_cache
            .ensure_source_with_policy(params.gpu, key.clone(), source, policy)
        {
            stats.scheduled_ibl_bakes = stats.scheduled_ibl_bakes.saturating_add(1);
        }
        let Some(spatial) =
            spatial_probe_for_state(params.scene, space_id, probe, params.render_context, 0)
        else {
            return;
        };
        let current_ready = self
            .ibl_cache
            .completed_cube(&key)
            .filter(|_cube| !params.reflection_probe_sh2_enabled || sh2.is_some())
            .map(|cube| (cube.texture.clone(), cube.mip_levels));
        if let Some((texture, mip_levels)) = current_ready {
            if let Some(generation) = runtime_generation {
                self.runtime_final_ready_generation
                    .insert(identity, generation);
            }
            let mut metadata = metadata_for_spatial(&spatial, probe.state, sh2.as_ref());
            metadata.params[1] = mip_levels.saturating_sub(1) as f32;
            self.last_ready.insert(
                identity,
                LastReadyProbe {
                    key: key.clone(),
                    texture: texture.clone(),
                    mip_levels,
                    sh2,
                },
            );
            cache.summary.ready.push(ReadyProbeSummary {
                identity,
                key: key.clone(),
                mip_levels,
                has_sh2: sh2.is_some(),
            });
            cache.ready.push(ReadyProbe {
                identity,
                key,
                texture,
                mip_levels,
                metadata,
                spatial,
            });
            return;
        }
        if let Some(fallback) = self.last_ready.get(&identity).cloned() {
            if params.reflection_probe_sh2_enabled && fallback.sh2.is_none() {
                return;
            }
            cache.summary.active_keys.insert(fallback.key.clone());
            let mut metadata = metadata_for_spatial(&spatial, probe.state, fallback.sh2.as_ref());
            metadata.params[1] = fallback.mip_levels.saturating_sub(1) as f32;
            cache.summary.ready.push(ReadyProbeSummary {
                identity,
                key: fallback.key.clone(),
                mip_levels: fallback.mip_levels,
                has_sh2: fallback.sh2.is_some(),
            });
            cache.ready.push(ReadyProbe {
                identity,
                key: fallback.key,
                texture: fallback.texture,
                mip_levels: fallback.mip_levels,
                metadata,
                spatial,
            });
        }
    }

    fn refresh_collect_config(&mut self, config: ProbeCollectConfig) {
        if self.collect_config == Some(config) {
            return;
        }
        self.collect_config = Some(config);
        self.space_cache.clear();
        self.dirty_spaces.clear();
        self.sync_signature = None;
    }

    #[cfg(test)]
    fn last_stats(&self) -> MaintainStats {
        self.last_stats
    }

    /// Current frame-global GPU resources, if allocated.
    #[must_use]
    pub fn resources(&self) -> Option<ReflectionProbeSpecularResources> {
        self.resources.clone()
    }

    /// CPU selection snapshot used by draw collection.
    #[must_use]
    pub fn selection(&self) -> &ReflectionProbeFrameSelection {
        &self.selection
    }

    fn sync_atlas_and_selection(
        &mut self,
        gpu: &mut GpuContext,
        face_size: u32,
        max_local_reflection_probes: usize,
        mut ready: Vec<ReadyProbe>,
        stats: &mut MaintainStats,
    ) {
        profiling::scope!("reflection_probes::specular::sync_atlas_selection");
        let max_slots = max_atlas_slots(gpu.limits());
        if max_slots <= 1 {
            self.sync_signature = None;
            self.selection.rebuild_spatial(Vec::new());
            return;
        }
        let usable_slots = usize::from(max_slots.saturating_sub(FIRST_PROBE_ATLAS_SLOT));
        if ready.len() > usable_slots {
            logger::warn!(
                "reflection probes: {} ready probes exceed atlas capacity {}; truncating",
                ready.len(),
                usable_slots
            );
            ready.truncate(usable_slots);
        }
        let signature = SpecularSyncSignature::new(face_size, max_local_reflection_probes, &ready);
        if self.sync_signature.as_ref() == Some(&signature) && self.resources.is_some() {
            stats.atlas_capacity = self
                .atlas
                .as_ref()
                .map_or(0, |atlas| usize::from(atlas.capacity));
            stats.reused_atlas_selection = true;
            return;
        }
        let used_slots = ready.len();
        let required_slots = (used_slots + usize::from(FIRST_PROBE_ATLAS_SLOT)).max(1);
        self.ensure_atlas(gpu.device(), face_size, required_slots as u16);

        let Some(atlas) = self.atlas.as_mut() else {
            self.sync_signature = None;
            self.selection.rebuild_spatial(Vec::new());
            return;
        };
        stats.atlas_capacity = usize::from(atlas.capacity);
        let mip_levels = atlas.mip_levels;
        let mut metadata = vec![GpuReflectionProbeMetadata::default(); atlas.capacity as usize];
        let mut copy_jobs = Vec::new();
        let mut selectable = Vec::with_capacity(ready.len());
        for (i, mut probe) in ready.into_iter().enumerate() {
            let slot = FIRST_PROBE_ATLAS_SLOT + i as u16;
            if atlas.keys[slot as usize].as_ref() != Some(&probe.key) {
                atlas.keys[slot as usize] = Some(probe.key.clone());
                copy_jobs.push(AtlasCopyJob {
                    slot,
                    texture: probe.texture.clone(),
                    mip_levels: probe.mip_levels.min(mip_levels),
                });
            }
            probe.spatial.atlas_index = slot;
            metadata[slot as usize] = probe.metadata;
            selectable.push((probe.identity.space_id, probe.spatial));
        }
        stats.atlas_copy_jobs = copy_jobs.len();
        self.write_metadata(gpu.queue(), &metadata);
        self.encode_atlas_copies(gpu, face_size, mip_levels, copy_jobs);
        {
            profiling::scope!("reflection_probes::specular::rebuild_spatial_selection");
            self.selection.rebuild_spatial(selectable);
        }
        self.sync_signature = Some(signature);
    }

    fn ensure_atlas(&mut self, device: &wgpu::Device, face_size: u32, required_slots: u16) {
        profiling::scope!("reflection_probes::specular::ensure_atlas");
        let needs_new = self
            .atlas
            .as_ref()
            .is_none_or(|atlas| atlas.face_size != face_size || atlas.capacity < required_slots);
        if !needs_new {
            return;
        }
        let capacity = required_slots.max(2);
        let mip_levels = mip_levels_for_edge(face_size);
        let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reflection_probe_specular_atlas"),
            size: wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: u32::from(capacity) * 6,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: REFLECTION_PROBE_ATLAS_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("reflection_probe_specular_atlas_view"),
            format: Some(REFLECTION_PROBE_ATLAS_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(u32::from(capacity) * 6),
        }));
        crate::profiling::note_resource_churn!(
            TextureView,
            "reflection_probes::specular_atlas_view"
        );
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reflection_probe_specular_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: mip_levels.saturating_sub(1) as f32,
            ..Default::default()
        }));
        let metadata_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reflection_probe_specular_metadata"),
            size: (usize::from(capacity) * size_of::<GpuReflectionProbeMetadata>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        crate::profiling::note_resource_churn!(
            Buffer,
            "reflection_probes::specular_metadata_buffer"
        );
        self.version = self.version.wrapping_add(1).max(1);
        self.resources = Some(ReflectionProbeSpecularResources {
            array_view: view,
            sampler,
            metadata_buffer,
            version: self.version,
        });
        self.atlas = Some(ReflectionProbeAtlas {
            texture,
            face_size,
            mip_levels,
            capacity,
            keys: vec![None; usize::from(capacity)],
        });
    }

    fn write_metadata(&self, queue: &wgpu::Queue, metadata: &[GpuReflectionProbeMetadata]) {
        profiling::scope!("reflection_probes::specular::write_metadata");
        let Some(resources) = &self.resources else {
            return;
        };
        queue.write_buffer(
            resources.metadata_buffer.as_ref(),
            0,
            bytemuck::cast_slice(metadata),
        );
    }

    fn encode_atlas_copies(
        &self,
        gpu: &mut GpuContext,
        face_size: u32,
        atlas_mips: u32,
        copy_jobs: Vec<AtlasCopyJob>,
    ) {
        profiling::scope!("reflection_probes::specular::atlas_copies");
        if copy_jobs.is_empty() {
            return;
        }
        let Some(atlas) = &self.atlas else {
            return;
        };
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reflection_probe_atlas_copy"),
            });
        let mut profiler = gpu.take_gpu_profiler();
        let copy_query = profiler
            .as_ref()
            .map(|p| p.begin_query("reflection_probe_specular::atlas_copies", &mut encoder));
        for job in copy_jobs {
            let mips = job.mip_levels.min(atlas_mips);
            for mip in 0..mips {
                let extent = mip_extent(face_size, mip);
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: job.texture.as_ref(),
                        mip_level: mip,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: atlas.texture.as_ref(),
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: u32::from(job.slot) * 6,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: extent,
                        height: extent,
                        depth_or_array_layers: 6,
                    },
                );
            }
        }
        if let (Some(profiler), Some(query)) = (profiler.as_mut(), copy_query) {
            profiler.end_query(&mut encoder, query);
            profiler.resolve_queries(&mut encoder);
        }
        let command_buffer = {
            profiling::scope!("CommandEncoder::finish::reflection_probe_atlas_copy");
            encoder.finish()
        };
        gpu.restore_gpu_profiler(profiler);
        {
            profiling::scope!("reflection_probes::specular::atlas_copy_submit");
            gpu.submit_frame_batch(
                FrameSubmitKind::BackgroundGpuWork,
                vec![command_buffer],
                None,
                None,
            );
        }
    }
}

fn specular_ibl_key_matches_closed_spaces(
    key: &SkyboxIblKey,
    spaces: &HashSet<RenderSpaceId>,
) -> bool {
    match key {
        SkyboxIblKey::Cubemap { .. } | SkyboxIblKey::SolidColor { .. } => false,
        SkyboxIblKey::RuntimeCubemap {
            render_space_id, ..
        } => spaces.contains(&RenderSpaceId(*render_space_id)),
    }
}

fn runtime_source_generation(source: &SkyboxIblSource) -> Option<u64> {
    match source {
        SkyboxIblSource::RuntimeCubemap(src) => Some(src.generation),
        SkyboxIblSource::Cubemap(_) | SkyboxIblSource::SolidColor(_) => None,
    }
}

fn ibl_policy_for_probe_source(
    time_slicing_mode: ReflectionProbeTimeSlicingMode,
    source: &SkyboxIblSource,
) -> IblBakePolicy {
    if !matches!(source, SkyboxIblSource::RuntimeCubemap(_)) {
        return IblBakePolicy::Immediate;
    }
    match time_slicing_mode {
        ReflectionProbeTimeSlicingMode::NoTimeSlicing => IblBakePolicy::Immediate,
        ReflectionProbeTimeSlicingMode::AllFacesAtOnce
        | ReflectionProbeTimeSlicingMode::IndividualFaces => IblBakePolicy::UnityTimeSliced,
    }
}

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct ProbeIdentity {
    /// Render space that owns the probe.
    space_id: RenderSpaceId,
    /// Dense reflection-probe renderable index.
    renderable_index: i32,
}

/// Last known source that can be sampled immediately for one probe.
#[derive(Clone)]
struct LastReadyProbe {
    /// IBL cache key for the filtered source.
    key: SkyboxIblKey,
    /// Filtered source texture.
    texture: Arc<wgpu::Texture>,
    /// Number of resident mip levels in the filtered source.
    mip_levels: u32,
    /// Optional SH2 projection paired with this source.
    sh2: Option<RenderSH2>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MaintainStats {
    active_spaces: usize,
    scanned_probes: usize,
    ready_probes: usize,
    reused_spaces: usize,
    reused_atlas_selection: bool,
    scheduled_ibl_bakes: usize,
    atlas_copy_jobs: usize,
    atlas_capacity: usize,
    ibl_pending: usize,
    ibl_active_sliced: usize,
    ibl_completed: usize,
}

#[cfg(feature = "tracy")]
fn plot_maintain_stats(stats: &MaintainStats) {
    tracy_client::plot!(
        "reflection_probes::specular::active_spaces",
        stats.active_spaces as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::scanned_probes",
        stats.scanned_probes as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::ready_probes",
        stats.ready_probes as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::reused_spaces",
        stats.reused_spaces as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::reused_atlas_selection",
        if stats.reused_atlas_selection {
            1.0
        } else {
            0.0
        }
    );
    tracy_client::plot!(
        "reflection_probes::specular::scheduled_ibl_bakes",
        stats.scheduled_ibl_bakes as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::atlas_copy_jobs",
        stats.atlas_copy_jobs as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::atlas_capacity",
        stats.atlas_capacity as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::ibl_pending",
        stats.ibl_pending as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::ibl_active_sliced",
        stats.ibl_active_sliced as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::ibl_completed",
        stats.ibl_completed as f64
    );
}

#[cfg(not(feature = "tracy"))]
fn plot_maintain_stats(_stats: &MaintainStats) {}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ProbeCollectConfig {
    face_size: u32,
    render_context: RenderingContext,
    reflection_probe_sh2_enabled: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CachedSpaceSummary {
    active_keys: HashSet<SkyboxIblKey>,
    active_capture_keys: HashSet<RuntimeReflectionProbeCaptureKey>,
    active_identities: HashSet<ProbeIdentity>,
    ready: Vec<ReadyProbeSummary>,
}

impl CachedSpaceSummary {
    fn normalize(&mut self) {
        self.ready.sort_unstable_by(|a, b| {
            (a.identity.space_id.0, a.identity.renderable_index)
                .cmp(&(b.identity.space_id.0, b.identity.renderable_index))
                .then_with(|| a.mip_levels.cmp(&b.mip_levels))
                .then_with(|| a.has_sh2.cmp(&b.has_sh2))
        });
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReadyProbeSummary {
    identity: ProbeIdentity,
    key: SkyboxIblKey,
    mip_levels: u32,
    has_sh2: bool,
}

#[derive(Clone, Default)]
struct CachedSpace {
    summary: CachedSpaceSummary,
    ready: Vec<ReadyProbe>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SpecularSyncSignature {
    face_size: u32,
    max_local_reflection_probes: usize,
    ready: Vec<ReadyProbeSummary>,
}

impl SpecularSyncSignature {
    fn new(face_size: u32, max_local_reflection_probes: usize, ready: &[ReadyProbe]) -> Self {
        Self {
            face_size,
            max_local_reflection_probes,
            ready: ready
                .iter()
                .map(|probe| ReadyProbeSummary {
                    identity: probe.identity,
                    key: probe.key.clone(),
                    mip_levels: probe.mip_levels,
                    has_sh2: probe.metadata.params[3].to_bits()
                        == crate::gpu::REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL.to_bits(),
                })
                .collect(),
        }
    }
}

#[derive(Clone)]
struct ReadyProbe {
    identity: ProbeIdentity,
    key: SkyboxIblKey,
    texture: Arc<wgpu::Texture>,
    mip_levels: u32,
    metadata: GpuReflectionProbeMetadata,
    spatial: SpatialProbe,
}

#[derive(Default)]
struct CollectedProbeResources {
    active_keys: HashSet<SkyboxIblKey>,
    active_capture_keys: HashSet<RuntimeReflectionProbeCaptureKey>,
    active_identities: HashSet<ProbeIdentity>,
    ready: Vec<ReadyProbe>,
}

impl CollectedProbeResources {
    fn extend_cached(&mut self, cache: &CachedSpace) {
        self.active_keys
            .extend(cache.summary.active_keys.iter().cloned());
        self.active_capture_keys
            .extend(cache.summary.active_capture_keys.iter().copied());
        self.active_identities
            .extend(cache.summary.active_identities.iter().copied());
        self.ready.extend(cache.ready.iter().cloned());
    }
}
