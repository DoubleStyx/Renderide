use std::sync::Arc;

use hashbrown::{HashMap, HashSet};

use crate::gpu::{FrameSubmitKind, GpuContext, GpuRetainedResources};
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

use super::atlas::{
    AtlasCopyJob, AtlasProbeIdentity, AtlasResidentProbe, ReflectionProbeAtlas, max_atlas_slots,
};
use super::captures::{
    RuntimeReflectionProbeCapture, RuntimeReflectionProbeCaptureKey,
    RuntimeReflectionProbeCaptureStore,
};
use super::resources::ReflectionProbeSpecularResources;
use super::selection::{ReflectionProbeFrameSelection, SpatialProbe};
use super::source::{metadata_for_spatial, resolve_probe_source, spatial_probe_for_state};

/// Default destination face size for reflection-probe IBL bakes.
const DEFAULT_REFLECTION_PROBE_FACE_SIZE: u32 = 256;
/// Maximum steady atlas payload. Standalone bake outputs become transient after atlas upload.
const REFLECTION_PROBE_ATLAS_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
/// Lowest automatic reflection-probe face edge under extreme global probe demand.
const MIN_REFLECTION_PROBE_FACE_SIZE: u32 = 32;
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
    /// Whether this maintenance call may advance one sliced IBL bake.
    pub(crate) advance_sliced_ibl: bool,
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
    /// Per-probe rows currently committed to `atlas`, retained separately from pending face-size
    /// bakes so transforms/removals can keep updating while the old atlas remains active.
    active_ready: Vec<ReadyProbe>,
    /// Highest runtime capture generation with a completed final specular IBL cube.
    runtime_final_ready_generation: HashMap<ProbeIdentity, u64>,
    /// Session-stable face-size ceiling; demand may lower it but never trigger an automatic upscale.
    resident_face_size: Option<u32>,
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
            active_ready: Vec::new(),
            runtime_final_ready_generation: HashMap::new(),
            resident_face_size: None,
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
        self.active_ready
            .retain(|probe| !spaces.contains(&probe.identity.space_id));
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
        self.ibl_cache
            .maintain_gpu_jobs(params.gpu, params.advance_sliced_ibl);
        let max_slots = max_atlas_slots(params.gpu.limits());
        let max_ready_probes = usize::from(max_slots.saturating_sub(FIRST_PROBE_ATLAS_SLOT));
        let selected = selected_probe_identities(
            params.scene,
            params.assets,
            &self.captures,
            params.render_context,
            max_ready_probes,
        );
        stats.selected_probes = selected.probes.len();
        stats.selected_unique_sources = selected.unique_source_count;
        let budgeted_face_size = budgeted_reflection_probe_face_size(
            clamp_face_size(DEFAULT_REFLECTION_PROBE_FACE_SIZE, params.gpu.limits()),
            selected.unique_source_count,
        );
        let face_size = self
            .resident_face_size
            .map_or(budgeted_face_size, |current| {
                current.min(budgeted_face_size)
            });
        self.resident_face_size = Some(face_size);
        self.refresh_collect_config(ProbeCollectConfig {
            face_size,
            render_context: params.render_context,
            reflection_probe_sh2_enabled: params.reflection_probe_sh2_enabled,
        });
        let mut collected = CollectedProbeResources::default();

        self.selection
            .set_max_local_reflection_probes(params.max_local_reflection_probes);
        self.collect_probe_resources(
            &mut params,
            face_size,
            &selected.probes,
            &mut collected,
            &mut stats,
        );
        self.captures.retain_active(&collected.active_capture_keys);
        self.last_ready
            .retain(|identity, _probe| collected.active_identities.contains(identity));
        self.runtime_final_ready_generation
            .retain(|identity, _generation| collected.active_identities.contains(identity));
        self.ibl_cache.prune_except(&collected.active_keys);
        collected.ready.sort_unstable_by_key(|probe| {
            (probe.identity.space_id.0, probe.identity.renderable_index)
        });
        stats.ready_probes = collected.ready.len();
        stats.ibl_pending = self.ibl_cache.pending_len();
        stats.ibl_active_sliced = self.ibl_cache.active_sliced_len();
        stats.ibl_completed = self.ibl_cache.completed_len();
        stats.ibl_owned_completed = self.ibl_cache.owned_completed_len();
        stats.ibl_external_resident = self.ibl_cache.externally_resident_len();
        self.sync_atlas_and_selection(
            params.gpu,
            face_size,
            params.max_local_reflection_probes,
            collected.ready,
            &mut stats,
        );
        stats.ibl_completed = self.ibl_cache.completed_len();
        stats.ibl_owned_completed = self.ibl_cache.owned_completed_len();
        stats.ibl_external_resident = self.ibl_cache.externally_resident_len();
        plot_maintain_stats(&stats);
        self.last_stats = stats;
    }

    fn collect_probe_resources(
        &mut self,
        params: &mut ReflectionProbeSpecularMaintainParams<'_>,
        face_size: u32,
        selected_probes: &HashMap<ProbeIdentity, PreselectedProbe>,
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
                let summary = self.collect_space_source_summary(
                    params,
                    space_id,
                    space,
                    face_size,
                    selected_probes,
                    stats,
                );
                if let Some(cache) = self.space_cache.get(&space_id)
                    && cache.summary == summary
                {
                    stats.reused_spaces = stats.reused_spaces.saturating_add(1);
                    collected.extend_cached(cache);
                    continue;
                }
            }

            let cache = self.collect_space_probe_cache(
                params,
                space_id,
                space,
                face_size,
                selected_probes,
                stats,
            );
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
        selected_probes: &HashMap<ProbeIdentity, PreselectedProbe>,
        stats: &mut MaintainStats,
    ) -> CachedSpaceSummary {
        profiling::scope!("reflection_probes::specular::collect_source_summary");
        let mut summary = CachedSpaceSummary::default();
        for probe in space.reflection_probes() {
            let identity = ProbeIdentity {
                space_id,
                renderable_index: probe.renderable_index,
            };
            let Some(preselected) = selected_probes.get(&identity) else {
                continue;
            };
            self.collect_probe_source_summary(
                params,
                space_id,
                probe,
                face_size,
                preselected,
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
        preselected: &PreselectedProbe,
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
        let source = preselected.source.clone();
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
        let spatial_summary = SpatialProbeSummary::from(&preselected.spatial);
        let current_ready = self
            .ibl_cache
            .completed_mip_levels(&key)
            .filter(|_mip_levels| !params.reflection_probe_sh2_enabled || sh2.is_some())
            .map(|mip_levels| (key.clone(), mip_levels, sh2.is_some()));
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
                spatial: spatial_summary,
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
                spatial: spatial_summary,
            });
        }
    }

    fn collect_space_probe_cache(
        &mut self,
        params: &mut ReflectionProbeSpecularMaintainParams<'_>,
        space_id: RenderSpaceId,
        space: crate::scene::RenderSpaceView<'_>,
        face_size: u32,
        selected_probes: &HashMap<ProbeIdentity, PreselectedProbe>,
        stats: &mut MaintainStats,
    ) -> CachedSpace {
        profiling::scope!("reflection_probes::specular::collect_space");
        let mut cache = CachedSpace::default();
        for probe in space.reflection_probes() {
            let identity = ProbeIdentity {
                space_id,
                renderable_index: probe.renderable_index,
            };
            let Some(preselected) = selected_probes.get(&identity) else {
                continue;
            };
            self.collect_probe_resource(
                params,
                space_id,
                probe,
                face_size,
                preselected,
                &mut cache,
                stats,
            );
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
        preselected: &PreselectedProbe,
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
        let source = preselected.source.clone();
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
        let spatial = preselected.spatial.clone();
        let spatial_summary = SpatialProbeSummary::from(&spatial);
        let current_ready = self
            .ibl_cache
            .completed_mip_levels(&key)
            .filter(|_mip_levels| !params.reflection_probe_sh2_enabled || sh2.is_some());
        if let Some(mip_levels) = current_ready {
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
                    mip_levels,
                    sh2,
                },
            );
            cache.summary.ready.push(ReadyProbeSummary {
                identity,
                key: key.clone(),
                mip_levels,
                has_sh2: sh2.is_some(),
                spatial: spatial_summary,
            });
            cache.ready.push(ReadyProbe {
                identity,
                key,
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
                spatial: spatial_summary,
            });
            cache.ready.push(ReadyProbe {
                identity,
                key: fallback.key,
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
            self.active_ready.clear();
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
        if ready.is_empty() {
            let had_resources = self.atlas.is_some() || self.resources.is_some();
            let previous_face_size = self.atlas.as_ref().map(|atlas| atlas.face_size);
            if !self.active_ready.is_empty() {
                self.selection.rebuild_spatial(Vec::new());
            }
            self.active_ready.clear();
            self.atlas = None;
            self.resources = None;
            if previous_face_size.is_some_and(|previous| previous != face_size) {
                self.last_ready
                    .retain(|_identity, probe| probe.key.face_size() == face_size);
                self.ibl_cache
                    .purge_where(|key| key.face_size() != face_size);
                self.space_cache.clear();
            }
            if had_resources {
                self.version = self.version.wrapping_add(1).max(1);
            }
            self.sync_signature = None;
            return;
        }
        if self.atlas.as_ref().is_some_and(|atlas| {
            atlas_face_transition_pending(
                atlas.face_size,
                face_size,
                ready.iter().map(|probe| &probe.key),
            )
        }) {
            // A face-size transition changes every IBL cache key. Keep sampling the complete old
            // atlas while the replacement bakes finish; externally-resident old cubes deliberately
            // have no standalone texture to recover from after the atlas is dropped.
            if let Some(atlas) = self.atlas.as_ref() {
                stats.atlas_capacity = usize::from(atlas.capacity);
                stats.atlas_unique_keys = atlas.slots.iter().flatten().count();
                stats.atlas_face_size = atlas.face_size as usize;
                stats.atlas_payload_bytes = reflection_probe_cube_bytes(atlas.face_size)
                    .saturating_mul(u64::from(atlas.capacity));
            }
            stats.reused_atlas_selection = true;
            stats.atlas_transition_pending = true;
            self.refresh_active_atlas_selection_during_transition(gpu.queue(), ready);
            if self.active_ready.is_empty() {
                // No active row samples this atlas. Remove its external-only fallbacks before
                // target-face bakes continue.
                self.atlas = None;
                self.resources = None;
                self.last_ready
                    .retain(|_identity, probe| probe.key.face_size() == face_size);
                self.ibl_cache
                    .purge_where(|key| key.face_size() != face_size);
                self.space_cache.clear();
                self.version = self.version.wrapping_add(1).max(1);
                stats.atlas_capacity = 0;
                stats.atlas_unique_keys = 0;
                stats.atlas_face_size = 0;
                stats.atlas_payload_bytes = 0;
                stats.atlas_transition_pending = false;
            }
            self.sync_signature = None;
            return;
        }
        let signature = SpecularSyncSignature::new(face_size, max_local_reflection_probes, &ready);
        if self.sync_signature.as_ref() == Some(&signature) && self.resources.is_some() {
            if let Some(atlas) = self.atlas.as_ref() {
                stats.atlas_capacity = usize::from(atlas.capacity);
                stats.atlas_unique_keys = atlas.slots.iter().flatten().count();
                stats.atlas_face_size = atlas.face_size as usize;
                stats.atlas_payload_bytes = reflection_probe_cube_bytes(atlas.face_size)
                    .saturating_mul(u64::from(atlas.capacity));
            }
            stats.reused_atlas_selection = true;
            return;
        }
        let requests = deduplicate_atlas_requests(
            ready
                .iter()
                .map(AtlasProbeRequest::from)
                .collect::<Vec<_>>(),
        );
        let required_texture_slots = (requests.len() + usize::from(FIRST_PROBE_ATLAS_SLOT)).max(1);
        let required_metadata_slots = (ready.len() + usize::from(FIRST_PROBE_ATLAS_SLOT)).max(1);
        let mut resident_keys = HashMap::<SkyboxIblKey, u32>::with_capacity(ready.len());
        for probe in &ready {
            resident_keys
                .entry(probe.key.clone())
                .and_modify(|mips| *mips = (*mips).max(probe.mip_levels))
                .or_insert(probe.mip_levels);
        }
        let previous_atlas = self.atlas.clone();
        let previous_resources = self.resources.clone();
        let previous_version = self.version;
        if !self.ensure_atlas(
            gpu,
            face_size,
            required_texture_slots as u16,
            required_metadata_slots as u16,
            max_slots,
            &requests,
        ) {
            self.sync_signature = None;
            return;
        }

        let Some(atlas) = self.atlas.as_ref() else {
            self.sync_signature = None;
            self.selection.rebuild_spatial(Vec::new());
            return;
        };
        stats.atlas_capacity = usize::from(atlas.capacity);
        stats.atlas_unique_keys = requests.len();
        stats.atlas_face_size = face_size as usize;
        stats.atlas_payload_bytes =
            reflection_probe_cube_bytes(face_size).saturating_mul(u64::from(atlas.capacity));
        let mip_levels = atlas.mip_levels;
        let mut metadata =
            vec![GpuReflectionProbeMetadata::default(); atlas.metadata_capacity as usize];
        let placement_plan =
            plan_atlas_placements(&requests, &atlas.slots, usize::from(atlas.capacity));
        let mut copy_jobs = Vec::new();
        let mut texture_slots = HashMap::with_capacity(requests.len());
        for (request, placement) in requests.iter().zip(&placement_plan.placements) {
            let texture_slot = placement.slot;
            if let Some(source) = placement.copy_source {
                let (completed_cube, resident_slot) = match source {
                    PlannedAtlasCopySource::ResidentSlot(source_slot) => (None, source_slot),
                    PlannedAtlasCopySource::CompletedCube => {
                        let Some(cube) = self.ibl_cache.completed_cube(&request.key) else {
                            logger::warn!(
                                "reflection probes: resident key {:?} lost both its atlas slot and completed cube; rebuilding",
                                request.key
                            );
                            self.atlas = previous_atlas;
                            self.resources = previous_resources;
                            self.version = previous_version;
                            self.sync_signature = None;
                            return;
                        };
                        (Some(cube.texture.clone()), 0)
                    }
                };
                copy_jobs.push(AtlasCopyJob {
                    slot: texture_slot,
                    completed_cube,
                    resident_slot,
                    mip_levels: request.mip_levels.min(mip_levels),
                });
            }
            texture_slots.insert(request.texture_key.clone(), texture_slot);
        }
        let mut selectable = Vec::with_capacity(ready.len());
        let mut committed_ready = Vec::with_capacity(ready.len());
        for (probe_index, mut probe) in ready.into_iter().enumerate() {
            let metadata_slot = FIRST_PROBE_ATLAS_SLOT.saturating_add(probe_index as u16);
            let texture_slot = texture_slots
                .get(&AtlasTextureKey::from(&probe.key))
                .copied()
                .expect("every ready reflection-probe key received one texture slot");
            probe.spatial.atlas_index = metadata_slot;
            probe.metadata.position[3] = f32::from(texture_slot);
            metadata[metadata_slot as usize] = probe.metadata;
            selectable.push((probe.identity.space_id, probe.spatial.clone()));
            committed_ready.push(probe);
        }
        stats.atlas_copy_jobs = copy_jobs.len();
        if self
            .encode_atlas_copies(gpu, face_size, mip_levels, copy_jobs)
            .is_none()
        {
            self.atlas = previous_atlas;
            self.resources = previous_resources;
            self.version = previous_version;
            self.sync_signature = None;
            return;
        }
        self.atlas
            .as_mut()
            .expect("atlas existed while placement plan was built")
            .slots = placement_plan.slots;
        self.write_metadata(gpu.queue(), &metadata);
        for (key, mip_levels) in resident_keys {
            self.ibl_cache.mark_external_resident(key, mip_levels);
        }
        {
            profiling::scope!("reflection_probes::specular::rebuild_spatial_selection");
            self.selection.rebuild_spatial(selectable);
        }
        self.active_ready = committed_ready;
        self.sync_signature = Some(signature);
    }

    fn refresh_active_atlas_selection_during_transition(
        &mut self,
        queue: &wgpu::Queue,
        ready: Vec<ReadyProbe>,
    ) {
        profiling::scope!("reflection_probes::specular::refresh_active_transition");
        let Some(metadata_capacity) = self.atlas.as_ref().map(|atlas| atlas.metadata_capacity)
        else {
            return;
        };
        let mut current = ready
            .into_iter()
            .map(|probe| (probe.identity, probe))
            .collect::<HashMap<_, _>>();
        let mut metadata =
            vec![GpuReflectionProbeMetadata::default(); usize::from(metadata_capacity)];
        let mut selectable = Vec::with_capacity(self.active_ready.len());
        let mut refreshed = Vec::with_capacity(self.active_ready.len());
        for active in std::mem::take(&mut self.active_ready) {
            let Some(mut probe) = current.remove(&active.identity) else {
                continue;
            };
            let metadata_slot = active.spatial.atlas_index;
            if usize::from(metadata_slot) >= metadata.len() {
                continue;
            }
            probe.key = active.key;
            probe.mip_levels = active.mip_levels;
            probe.metadata.params[1] = active.mip_levels.saturating_sub(1) as f32;
            probe.metadata.position[3] = active.metadata.position[3];
            probe.spatial.atlas_index = metadata_slot;
            metadata[metadata_slot as usize] = probe.metadata;
            selectable.push((probe.identity.space_id, probe.spatial.clone()));
            refreshed.push(probe);
        }
        self.write_metadata(queue, &metadata);
        self.selection.rebuild_spatial(selectable);
        self.active_ready = refreshed;
    }

    fn ensure_atlas(
        &mut self,
        gpu: &GpuContext,
        face_size: u32,
        required_texture_slots: u16,
        required_metadata_slots: u16,
        max_slots: u16,
        requests: &[AtlasProbeRequest],
    ) -> bool {
        profiling::scope!("reflection_probes::specular::ensure_atlas");
        let target_capacity = atlas_growth_capacity(required_texture_slots, max_slots);
        let target_metadata_capacity = atlas_growth_capacity(required_metadata_slots, max_slots);
        let needs_new = self.atlas.as_ref().is_none_or(|atlas| {
            atlas.face_size != face_size
                || atlas_capacity_needs_reallocation(
                    atlas.capacity,
                    required_texture_slots,
                    target_capacity,
                )
                || atlas_capacity_needs_reallocation(
                    atlas.metadata_capacity,
                    required_metadata_slots,
                    target_metadata_capacity,
                )
        });
        if !needs_new {
            return true;
        }
        let capacity = target_capacity;
        let mip_levels = mip_levels_for_edge(face_size);
        let texture = Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
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
        let sampler = Arc::new(gpu.device().create_sampler(&wgpu::SamplerDescriptor {
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
        let metadata_buffer = Arc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("reflection_probe_specular_metadata"),
            size: (usize::from(target_metadata_capacity) * size_of::<GpuReflectionProbeMetadata>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        crate::profiling::note_resource_churn!(
            Buffer,
            "reflection_probes::specular_metadata_buffer"
        );
        let old_atlas = self.atlas.clone();
        let mut slots = vec![None; usize::from(capacity)];
        if let Some(old) = old_atlas.as_ref().filter(|old| old.face_size == face_size) {
            let accepted = if capacity >= old.capacity {
                let preserved = old.slots.len().min(slots.len());
                slots[..preserved].clone_from_slice(&old.slots[..preserved]);
                self.encode_atlas_resize_copy(gpu, old, texture.as_ref(), face_size, mip_levels)
            } else {
                let repack = plan_atlas_repack(requests, &old.slots, usize::from(capacity));
                slots = repack.slots;
                self.encode_atlas_repack_copy(
                    gpu,
                    old,
                    texture.as_ref(),
                    face_size,
                    mip_levels,
                    &repack.copies,
                )
            };
            if !accepted {
                return false;
            }
        }
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
            metadata_capacity: target_metadata_capacity,
            slots,
        });
        true
    }

    fn encode_atlas_resize_copy(
        &self,
        gpu: &GpuContext,
        old: &ReflectionProbeAtlas,
        destination: &wgpu::Texture,
        face_size: u32,
        mip_levels: u32,
    ) -> bool {
        profiling::scope!("reflection_probes::specular::atlas_resize_copy");
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reflection_probe_atlas_resize_copy"),
            });
        for mip in 0..mip_levels.min(old.mip_levels) {
            let extent = mip_extent(face_size, mip);
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: old.texture.as_ref(),
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: destination,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: extent,
                    height: extent,
                    depth_or_array_layers: u32::from(
                        old.capacity
                            .min((destination.depth_or_array_layers() / 6) as u16),
                    ) * 6,
                },
            );
        }
        let mut retained = GpuRetainedResources::new();
        retained.retain_texture(old.texture.as_ref().clone());
        retained.retain_texture(destination.clone());
        gpu.submit_frame_batch_with_retained_resources(
            FrameSubmitKind::BackgroundGpuWork,
            vec![encoder.finish()],
            None,
            None,
            Vec::new(),
            retained,
        )
        .is_some()
    }

    fn encode_atlas_repack_copy(
        &self,
        gpu: &GpuContext,
        old: &ReflectionProbeAtlas,
        destination: &wgpu::Texture,
        face_size: u32,
        mip_levels: u32,
        copies: &[AtlasRepackCopy],
    ) -> bool {
        if copies.is_empty() {
            return true;
        }
        profiling::scope!("reflection_probes::specular::atlas_repack_copy");
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reflection_probe_atlas_repack_copy"),
            });
        for copy in copies {
            for mip in 0..copy.mip_levels.min(mip_levels).min(old.mip_levels) {
                let extent = mip_extent(face_size, mip);
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: old.texture.as_ref(),
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: u32::from(copy.source_slot) * 6,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: destination,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: u32::from(copy.destination_slot) * 6,
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
        let mut retained = GpuRetainedResources::new();
        retained.retain_texture(old.texture.as_ref().clone());
        retained.retain_texture(destination.clone());
        gpu.submit_frame_batch_with_retained_resources(
            FrameSubmitKind::BackgroundGpuWork,
            vec![encoder.finish()],
            None,
            None,
            Vec::new(),
            retained,
        )
        .is_some()
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
    ) -> Option<()> {
        profiling::scope!("reflection_probes::specular::atlas_copies");
        if copy_jobs.is_empty() {
            return Some(());
        }
        let Some(atlas) = &self.atlas else {
            return None;
        };
        let mut retained = GpuRetainedResources::new();
        retained.retain_texture(atlas.texture.as_ref().clone());
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reflection_probe_atlas_copy"),
            });
        let mut profiler = gpu.take_gpu_profiler();
        let copy_query = profiler
            .as_ref()
            .map(|p| p.begin_query("reflection_probe_specular::atlas_copies", &mut encoder));
        for job in &copy_jobs {
            let mips = job.mip_levels.min(atlas_mips);
            let (source_texture, source_layer) = match &job.completed_cube {
                Some(texture) => {
                    retained.retain_texture(texture.as_ref().clone());
                    (texture.as_ref(), 0)
                }
                None => (atlas.texture.as_ref(), u32::from(job.resident_slot) * 6),
            };
            for mip in 0..mips {
                let extent = mip_extent(face_size, mip);
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: source_texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: source_layer,
                        },
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
            gpu.submit_frame_batch_with_retained_resources(
                FrameSubmitKind::BackgroundGpuWork,
                vec![command_buffer],
                None,
                None,
                Vec::new(),
                retained,
            )?;
        }
        Some(())
    }
}

#[derive(Clone)]
struct AtlasProbeRequest {
    identity: AtlasProbeIdentity,
    key: SkyboxIblKey,
    texture_key: AtlasTextureKey,
    mip_levels: u32,
}

impl From<&ReadyProbe> for AtlasProbeRequest {
    fn from(probe: &ReadyProbe) -> Self {
        Self {
            identity: probe.identity.into(),
            key: probe.key.clone(),
            texture_key: AtlasTextureKey::from(&probe.key),
            mip_levels: probe.mip_levels,
        }
    }
}

/// Pixel-producing identity of a filtered atlas cube. Per-probe/cache lifecycle fields that do
/// not change texels are intentionally excluded so metadata rows can share one texture slot.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum AtlasTextureKey {
    Cubemap {
        asset_id: i32,
        allocation_generation: u64,
        mip_levels_resident: u32,
        content_generation: u64,
        storage_v_inverted: bool,
        face_size: u32,
    },
    SolidColor {
        color_hash: u64,
        face_size: u32,
    },
    RuntimeCubemap {
        render_space_id: i32,
        renderable_index: i32,
        generation: u64,
        mip_levels: u32,
        storage_v_inverted: bool,
        face_size: u32,
    },
}

impl From<&SkyboxIblKey> for AtlasTextureKey {
    fn from(key: &SkyboxIblKey) -> Self {
        match *key {
            SkyboxIblKey::Cubemap {
                asset_id,
                allocation_generation,
                mip_levels_resident,
                content_generation,
                storage_v_inverted,
                face_size,
                ..
            } => Self::Cubemap {
                asset_id,
                allocation_generation,
                mip_levels_resident,
                content_generation,
                storage_v_inverted,
                face_size,
            },
            SkyboxIblKey::SolidColor {
                color_hash,
                face_size,
                ..
            } => Self::SolidColor {
                color_hash,
                face_size,
            },
            SkyboxIblKey::RuntimeCubemap {
                render_space_id,
                renderable_index,
                generation,
                mip_levels,
                storage_v_inverted,
                face_size,
            } => Self::RuntimeCubemap {
                render_space_id,
                renderable_index,
                generation,
                mip_levels,
                storage_v_inverted,
                face_size,
            },
        }
    }
}

#[derive(Clone, Copy)]
enum PlannedAtlasCopySource {
    CompletedCube,
    ResidentSlot(u16),
}

#[derive(Clone, Copy)]
struct AtlasPlacement {
    slot: u16,
    copy_source: Option<PlannedAtlasCopySource>,
}

struct AtlasPlacementPlan {
    slots: Vec<Option<AtlasResidentProbe>>,
    placements: Vec<AtlasPlacement>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AtlasRepackCopy {
    source_slot: u16,
    destination_slot: u16,
    mip_levels: u32,
}

struct AtlasRepackPlan {
    slots: Vec<Option<AtlasResidentProbe>>,
    copies: Vec<AtlasRepackCopy>,
}

fn deduplicate_atlas_requests(requests: Vec<AtlasProbeRequest>) -> Vec<AtlasProbeRequest> {
    let mut unique = Vec::<AtlasProbeRequest>::with_capacity(requests.len());
    let mut by_key = HashMap::<AtlasTextureKey, usize>::with_capacity(requests.len());
    for request in requests {
        if let Some(&index) = by_key.get(&request.texture_key) {
            unique[index].mip_levels = unique[index].mip_levels.max(request.mip_levels);
            continue;
        }
        by_key.insert(request.texture_key.clone(), unique.len());
        unique.push(request);
    }
    unique
}

fn plan_atlas_placements(
    requests: &[AtlasProbeRequest],
    previous_slots: &[Option<AtlasResidentProbe>],
    capacity: usize,
) -> AtlasPlacementPlan {
    debug_assert!(capacity >= requests.len().saturating_add(1));
    let mut slots = vec![None; capacity];
    let mut claimed = vec![false; capacity];
    let mut placements = vec![None; requests.len()];

    // Preserve exact identity/key matches first. Camera and scene changes can then update only
    // metadata without moving or recopying any cubemap texels.
    for (request_index, request) in requests.iter().enumerate() {
        let exact = previous_slots
            .iter()
            .enumerate()
            .skip(usize::from(FIRST_PROBE_ATLAS_SLOT))
            .take(capacity.saturating_sub(usize::from(FIRST_PROBE_ATLAS_SLOT)))
            .find(|(slot, resident)| {
                !claimed[*slot]
                    && resident.as_ref().is_some_and(|resident| {
                        resident.identity == request.identity
                            && AtlasTextureKey::from(&resident.key) == request.texture_key
                    })
            })
            .map(|(slot, _)| slot);
        if let Some(slot) = exact {
            claimed[slot] = true;
            slots[slot] = Some(resident_probe_for_request(request));
            placements[request_index] = Some(AtlasPlacement {
                slot: slot as u16,
                copy_source: None,
            });
        }
    }

    // If an identity changed but another old slot already contains the requested cube, adopt that
    // slot. This keeps externally-resident cubes usable without restoring a standalone texture.
    for (request_index, request) in requests.iter().enumerate() {
        if placements[request_index].is_some() {
            continue;
        }
        let matching_key = previous_slots
            .iter()
            .enumerate()
            .skip(usize::from(FIRST_PROBE_ATLAS_SLOT))
            .take(capacity.saturating_sub(usize::from(FIRST_PROBE_ATLAS_SLOT)))
            .find(|(slot, resident)| {
                !claimed[*slot]
                    && resident.as_ref().is_some_and(|resident| {
                        AtlasTextureKey::from(&resident.key) == request.texture_key
                    })
            })
            .map(|(slot, _)| slot);
        if let Some(slot) = matching_key {
            claimed[slot] = true;
            slots[slot] = Some(resident_probe_for_request(request));
            placements[request_index] = Some(AtlasPlacement {
                slot: slot as u16,
                copy_source: None,
            });
        }
    }

    // New identities take any remaining slot. Shared keys copy from the stable resident slot;
    // genuinely new keys copy once from the just-completed bake texture.
    for (request_index, request) in requests.iter().enumerate() {
        if placements[request_index].is_some() {
            continue;
        }
        let destination = (usize::from(FIRST_PROBE_ATLAS_SLOT)..capacity)
            .find(|slot| !claimed[*slot])
            .expect("reflection-probe atlas capacity covered every ready probe");
        let resident_source = previous_slots
            .iter()
            .enumerate()
            .skip(usize::from(FIRST_PROBE_ATLAS_SLOT))
            .find(|(_slot, resident)| {
                resident.as_ref().is_some_and(|resident| {
                    AtlasTextureKey::from(&resident.key) == request.texture_key
                })
            })
            .map(|(slot, _)| slot as u16);
        claimed[destination] = true;
        slots[destination] = Some(resident_probe_for_request(request));
        placements[request_index] = Some(AtlasPlacement {
            slot: destination as u16,
            copy_source: Some(resident_source.map_or(
                PlannedAtlasCopySource::CompletedCube,
                PlannedAtlasCopySource::ResidentSlot,
            )),
        });
    }

    AtlasPlacementPlan {
        slots,
        placements: placements
            .into_iter()
            .map(|placement| placement.expect("every atlas request received a placement"))
            .collect(),
    }
}

fn plan_atlas_repack(
    requests: &[AtlasProbeRequest],
    previous_slots: &[Option<AtlasResidentProbe>],
    capacity: usize,
) -> AtlasRepackPlan {
    debug_assert!(capacity >= requests.len().saturating_add(1));
    let mut slots = vec![None; capacity];
    let mut copies = Vec::with_capacity(requests.len());
    let mut destination = usize::from(FIRST_PROBE_ATLAS_SLOT);
    for request in requests {
        let source = previous_slots
            .iter()
            .enumerate()
            .skip(usize::from(FIRST_PROBE_ATLAS_SLOT))
            .find(|(_slot, resident)| {
                resident.as_ref().is_some_and(|resident| {
                    resident.identity == request.identity
                        && AtlasTextureKey::from(&resident.key) == request.texture_key
                })
            })
            .or_else(|| {
                previous_slots
                    .iter()
                    .enumerate()
                    .skip(usize::from(FIRST_PROBE_ATLAS_SLOT))
                    .find(|(_slot, resident)| {
                        resident.as_ref().is_some_and(|resident| {
                            AtlasTextureKey::from(&resident.key) == request.texture_key
                        })
                    })
            });
        let Some((source_slot, resident)) = source else {
            continue;
        };
        let resident = resident
            .as_ref()
            .expect("atlas repack source was selected from a resident slot");
        debug_assert!(destination < capacity);
        slots[destination] = Some(resident_probe_for_request(request));
        copies.push(AtlasRepackCopy {
            source_slot: source_slot as u16,
            destination_slot: destination as u16,
            mip_levels: request.mip_levels.min(resident.mip_levels),
        });
        destination += 1;
    }
    AtlasRepackPlan { slots, copies }
}

fn resident_probe_for_request(request: &AtlasProbeRequest) -> AtlasResidentProbe {
    AtlasResidentProbe {
        identity: request.identity,
        key: request.key.clone(),
        mip_levels: request.mip_levels,
    }
}

fn atlas_growth_capacity(required_slots: u16, max_slots: u16) -> u16 {
    let required = u32::from(required_slots.max(2));
    let maximum = u32::from(max_slots.max(required_slots));
    required.next_power_of_two().min(maximum).max(required) as u16
}

fn atlas_capacity_needs_reallocation(
    current_capacity: u16,
    required_slots: u16,
    target_capacity: u16,
) -> bool {
    current_capacity < required_slots || current_capacity >= target_capacity.saturating_mul(2)
}

fn atlas_face_transition_pending<'a>(
    current_face_size: u32,
    target_face_size: u32,
    ready_keys: impl IntoIterator<Item = &'a SkyboxIblKey>,
) -> bool {
    if current_face_size == target_face_size {
        return false;
    }
    for key in ready_keys {
        if key.face_size() != target_face_size {
            return true;
        }
    }
    false
}

struct SelectedProbeIdentities {
    probes: HashMap<ProbeIdentity, PreselectedProbe>,
    unique_source_count: usize,
}

struct PreselectedProbe {
    source: SkyboxIblSource,
    spatial: SpatialProbe,
}

fn selected_probe_identities(
    scene: &SceneCoordinator,
    assets: &dyn ReflectionProbeCubemapAssets,
    captures: &RuntimeReflectionProbeCaptureStore,
    render_context: RenderingContext,
    max_ready_probes: usize,
) -> SelectedProbeIdentities {
    profiling::scope!("reflection_probes::specular::preselect");
    let mut candidates = Vec::new();
    for space_id in scene.render_space_ids() {
        let Some(space) = scene.space(space_id) else {
            continue;
        };
        if !space.is_active() {
            continue;
        }
        for probe in space.reflection_probes() {
            let Some(source) = resolve_probe_source(space_id, probe, assets, captures) else {
                continue;
            };
            let Some(spatial) = spatial_probe_for_state(scene, space_id, probe, render_context, 0)
            else {
                continue;
            };
            candidates.push((
                ProbeIdentity {
                    space_id,
                    renderable_index: probe.renderable_index,
                },
                AtlasTextureKey::from(&build_key(&source, 1)),
                source,
                spatial,
            ));
        }
    }
    candidates.sort_unstable_by_key(|(identity, _key, _source, _spatial)| {
        (identity.space_id.0, identity.renderable_index)
    });
    candidates.truncate(max_ready_probes);
    let unique_source_count = candidates
        .iter()
        .map(|(_identity, key, _source, _spatial)| key)
        .collect::<HashSet<_>>()
        .len();
    SelectedProbeIdentities {
        probes: candidates
            .into_iter()
            .map(|(identity, _key, source, spatial)| {
                (identity, PreselectedProbe { source, spatial })
            })
            .collect(),
        unique_source_count,
    }
}

fn budgeted_reflection_probe_face_size(default_face_size: u32, unique_source_count: usize) -> u32 {
    let mut face_size = default_face_size.max(1);
    let minimum = MIN_REFLECTION_PROBE_FACE_SIZE.min(face_size);
    let slots = unique_source_count
        .saturating_add(usize::from(FIRST_PROBE_ATLAS_SLOT))
        .max(1) as u64;
    while face_size > minimum
        && reflection_probe_cube_bytes(face_size).saturating_mul(slots)
            > REFLECTION_PROBE_ATLAS_BUDGET_BYTES
    {
        face_size = (face_size / 2).max(minimum);
    }
    face_size
}

fn reflection_probe_cube_bytes(face_size: u32) -> u64 {
    let mut edge = face_size.max(1);
    let mut face_texels = 0_u64;
    loop {
        face_texels = face_texels.saturating_add(u64::from(edge) * u64::from(edge));
        if edge == 1 {
            break;
        }
        edge = (edge / 2).max(1);
    }
    // Six faces and eight bytes per RGBA16F texel.
    face_texels.saturating_mul(6 * 8)
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
    runtime_ibl_policy(time_slicing_mode)
}

fn runtime_ibl_policy(_time_slicing_mode: ReflectionProbeTimeSlicingMode) -> IblBakePolicy {
    IblBakePolicy::UnityTimeSliced
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

impl From<ProbeIdentity> for AtlasProbeIdentity {
    fn from(identity: ProbeIdentity) -> Self {
        Self {
            space_id: identity.space_id,
            renderable_index: identity.renderable_index,
        }
    }
}

/// Last known source that can be sampled immediately for one probe.
#[derive(Clone)]
struct LastReadyProbe {
    /// IBL cache key for the filtered source.
    key: SkyboxIblKey,
    /// Number of resident mip levels in the filtered source.
    mip_levels: u32,
    /// Optional SH2 projection paired with this source.
    sh2: Option<RenderSH2>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MaintainStats {
    active_spaces: usize,
    scanned_probes: usize,
    selected_probes: usize,
    selected_unique_sources: usize,
    ready_probes: usize,
    reused_spaces: usize,
    reused_atlas_selection: bool,
    scheduled_ibl_bakes: usize,
    atlas_copy_jobs: usize,
    atlas_capacity: usize,
    atlas_unique_keys: usize,
    atlas_face_size: usize,
    atlas_payload_bytes: u64,
    atlas_transition_pending: bool,
    ibl_pending: usize,
    ibl_active_sliced: usize,
    ibl_completed: usize,
    ibl_owned_completed: usize,
    ibl_external_resident: usize,
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
        "reflection_probes::specular::selected_probes",
        stats.selected_probes as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::selected_unique_sources",
        stats.selected_unique_sources as f64
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
        "reflection_probes::specular::atlas_unique_keys",
        stats.atlas_unique_keys as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::atlas_face_size",
        stats.atlas_face_size as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::atlas_payload_mib",
        stats.atlas_payload_bytes as f64 / (1024.0 * 1024.0)
    );
    tracy_client::plot!(
        "reflection_probes::specular::atlas_transition_pending",
        if stats.atlas_transition_pending {
            1.0
        } else {
            0.0
        }
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
    tracy_client::plot!(
        "reflection_probes::specular::ibl_owned_completed",
        stats.ibl_owned_completed as f64
    );
    tracy_client::plot!(
        "reflection_probes::specular::ibl_external_resident",
        stats.ibl_external_resident as f64
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
    spatial: SpatialProbeSummary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SpatialProbeSummary {
    renderable_index: i32,
    importance: i32,
    aabb_min: [u32; 3],
    aabb_max: [u32; 3],
    influence_aabb_min: [u32; 3],
    influence_aabb_max: [u32; 3],
    center: [u32; 3],
    volume: u32,
    skybox: bool,
}

impl From<&SpatialProbe> for SpatialProbeSummary {
    fn from(probe: &SpatialProbe) -> Self {
        let bits = |value: glam::Vec3A| value.to_array().map(f32::to_bits);
        Self {
            renderable_index: probe.renderable_index,
            importance: probe.importance,
            aabb_min: bits(probe.aabb_min),
            aabb_max: bits(probe.aabb_max),
            influence_aabb_min: bits(probe.influence_aabb_min),
            influence_aabb_max: bits(probe.influence_aabb_max),
            center: bits(probe.center),
            volume: probe.volume.to_bits(),
            skybox: probe.skybox,
        }
    }
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
                    spatial: SpatialProbeSummary::from(&probe.spatial),
                })
                .collect(),
        }
    }
}

#[derive(Clone)]
struct ReadyProbe {
    identity: ProbeIdentity,
    key: SkyboxIblKey,
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
