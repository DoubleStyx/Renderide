//! Retained instance-plan cache for world-mesh forward preparation.

use std::collections::VecDeque;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::cpu_parallelism::RENDER_COMMAND_CHUNK_DRAWS;
use crate::graph_inputs::OffscreenWriteTarget;
use crate::materials::{MaterialPipelineDesc, ShaderPermutation};
use crate::world_mesh::draw_prep::WorldMeshDrawItem;
use crate::world_mesh::{InstancePlan, fingerprint_world_mesh_draws};

use super::super::MaterialBatchPacket;
use super::super::state::WorldMeshForwardPipelineState;
use super::material_packet_submission_fingerprint;

const WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_CAPACITY: usize = 256;
const WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_DRAWS: usize = RENDER_COMMAND_CHUNK_DRAWS * 2;
const WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_LOW_PACKET_MIN_DRAWS: usize =
    RENDER_COMMAND_CHUNK_DRAWS * 8;
const WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_PACKETS: usize = 2;
const WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_THRASH_WINDOW_LOOKUPS: u32 = 16;
const WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_THRASH_MIN_HIT_RATE_PER_MILLE: u32 = 250;
const WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_THRASH_BYPASS_LOOKUPS: u32 = 16;

/// Runtime counters for the retained forward instance-plan cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WorldMeshForwardInstancePlanCacheStats {
    /// Retained instance plans currently resident in the cache.
    pub(crate) entries: usize,
    /// Cache lookups that reused an instance plan.
    pub(crate) hits: u64,
    /// Cache lookups that had to rebuild an instance plan.
    pub(crate) misses: u64,
    /// Eligible cache attempts skipped because the draw or packet count was too small.
    pub(crate) skipped_small: u64,
    /// Eligible cache attempts skipped while recent probes were missing too often.
    pub(crate) skipped_thrash: u64,
    /// Hit rate for cache probes, in hits per 1000 lookups.
    pub(crate) hit_rate_per_mille: u16,
    /// New instance plans inserted into the cache.
    pub(crate) insertions: u64,
    /// Entries evicted to keep the cache bounded.
    pub(crate) evictions: u64,
}

/// Bounded cache for per-view world-mesh forward instance plans.
#[derive(Debug, Default)]
pub(crate) struct WorldMeshForwardInstancePlanCache {
    inner: Mutex<WorldMeshForwardInstancePlanCacheInner>,
}

#[derive(Debug, Default)]
struct WorldMeshForwardInstancePlanCacheInner {
    entries: HashMap<WorldMeshForwardInstancePlanCacheKey, InstancePlan>,
    recency: VecDeque<WorldMeshForwardInstancePlanCacheKey>,
    stats: WorldMeshForwardInstancePlanCacheStats,
    thrash: InstancePlanCacheThrashWindow,
}

#[derive(Debug, Default)]
struct InstancePlanCacheThrashWindow {
    lookups: u32,
    hits: u32,
    bypass_remaining: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct WorldMeshForwardInstancePlanCacheKey {
    draw_fingerprint: u64,
    draw_count: usize,
    submission_fingerprint: u64,
    packet_count: usize,
    supports_base_instance: bool,
    shader_perm: ShaderPermutation,
    pass_desc: MaterialPipelineDesc,
    front_face_flip: bool,
    offscreen_write_target: OffscreenWriteTarget,
}

impl WorldMeshForwardInstancePlanCache {
    /// Returns whether the next lookup should probe the retained cache.
    pub(super) fn should_probe_cache(&self, draw_count: usize, packet_count: usize) -> bool {
        profiling::scope!("world_mesh::prepare_frame::instance_plan_cache_admit");
        if !Self::admits_inputs(draw_count, packet_count) {
            let mut inner = self.inner.lock();
            inner.stats.skipped_small = inner.stats.skipped_small.saturating_add(1);
            drop(inner);
            return false;
        }
        let mut inner = self.inner.lock();
        if inner.thrash.bypass_remaining == 0 {
            return true;
        }
        inner.thrash.bypass_remaining = inner.thrash.bypass_remaining.saturating_sub(1);
        inner.stats.skipped_thrash = inner.stats.skipped_thrash.saturating_add(1);
        false
    }

    /// Returns a cached instance plan for the view inputs or stores the plan produced by `build`.
    pub(super) fn get_or_build_plan(
        &self,
        draws: &[WorldMeshDrawItem],
        packets: &[MaterialBatchPacket],
        pipeline: &WorldMeshForwardPipelineState,
        supports_base_instance: bool,
        offscreen_write_target: OffscreenWriteTarget,
        build: impl FnOnce() -> InstancePlan,
    ) -> InstancePlan {
        let key = WorldMeshForwardInstancePlanCacheKey::new(
            draws,
            packets,
            pipeline,
            supports_base_instance,
            offscreen_write_target,
        );
        self.get_or_build(key, build)
    }

    /// Captures a point-in-time diagnostic snapshot of the instance-plan cache.
    pub(crate) fn stats(&self) -> WorldMeshForwardInstancePlanCacheStats {
        let inner = self.inner.lock();
        let mut stats = inner.stats;
        stats.entries = inner.entries.len();
        drop(inner);
        stats.hit_rate_per_mille = instance_plan_cache_hit_rate_per_mille(stats.hits, stats.misses);
        stats
    }

    fn admits_inputs(draw_count: usize, packet_count: usize) -> bool {
        draw_count >= WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_DRAWS
            && (packet_count >= WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_PACKETS
                || draw_count >= WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_LOW_PACKET_MIN_DRAWS)
    }

    fn get_or_build(
        &self,
        key: WorldMeshForwardInstancePlanCacheKey,
        build: impl FnOnce() -> InstancePlan,
    ) -> InstancePlan {
        if let Some(plan) = self.entry(&key) {
            return plan;
        }
        let plan = build();
        self.insert(key, plan.clone());
        plan
    }

    fn entry(&self, key: &WorldMeshForwardInstancePlanCacheKey) -> Option<InstancePlan> {
        let mut inner = self.inner.lock();
        let plan = inner.entries.get(key).cloned();
        if plan.is_some() {
            inner.stats.hits = inner.stats.hits.saturating_add(1);
            inner.recency.push_back(key.clone());
            inner.thrash.record_hit();
        } else {
            inner.stats.misses = inner.stats.misses.saturating_add(1);
            inner.thrash.record_miss();
        }
        if inner.thrash.should_enter_bypass() {
            inner.thrash.bypass_remaining =
                WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_THRASH_BYPASS_LOOKUPS;
        }
        plan
    }

    fn insert(&self, key: WorldMeshForwardInstancePlanCacheKey, plan: InstancePlan) {
        let mut inner = self.inner.lock();
        if let Some(entry) = inner.entries.get_mut(&key) {
            *entry = plan;
            inner.recency.push_back(key);
            drop(inner);
            return;
        }
        inner.entries.insert(key.clone(), plan);
        inner.recency.push_back(key);
        inner.stats.insertions = inner.stats.insertions.saturating_add(1);
        while inner.entries.len() > WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_CAPACITY {
            let Some(candidate) = inner.recency.pop_front() else {
                break;
            };
            if inner.entries.remove(&candidate).is_some() {
                inner.stats.evictions = inner.stats.evictions.saturating_add(1);
            }
        }
        drop(inner);
    }
}

impl WorldMeshForwardInstancePlanCacheKey {
    fn new(
        draws: &[WorldMeshDrawItem],
        packets: &[MaterialBatchPacket],
        pipeline: &WorldMeshForwardPipelineState,
        supports_base_instance: bool,
        offscreen_write_target: OffscreenWriteTarget,
    ) -> Self {
        Self {
            draw_fingerprint: fingerprint_world_mesh_draws(draws),
            draw_count: draws.len(),
            submission_fingerprint: material_packet_submission_fingerprint(packets),
            packet_count: packets.len(),
            supports_base_instance,
            shader_perm: pipeline.shader_perm,
            pass_desc: pipeline.pass_desc,
            front_face_flip: pipeline.front_face_flip,
            offscreen_write_target,
        }
    }
}

impl InstancePlanCacheThrashWindow {
    fn record_hit(&mut self) {
        self.lookups = self.lookups.saturating_add(1);
        self.hits = self.hits.saturating_add(1);
    }

    fn record_miss(&mut self) {
        self.lookups = self.lookups.saturating_add(1);
    }

    fn should_enter_bypass(&mut self) -> bool {
        if self.lookups < WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_THRASH_WINDOW_LOOKUPS {
            return false;
        }
        let hits = self.hits as u64;
        let misses = self.lookups.saturating_sub(self.hits) as u64;
        let should_bypass = instance_plan_cache_hit_rate_per_mille(hits, misses)
            < WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_THRASH_MIN_HIT_RATE_PER_MILLE as u16;
        self.lookups = 0;
        self.hits = 0;
        should_bypass
    }
}

fn instance_plan_cache_hit_rate_per_mille(hits: u64, misses: u64) -> u16 {
    let lookups = hits.saturating_add(misses);
    if lookups == 0 {
        return 0;
    }
    ((hits.saturating_mul(1000)) / lookups).min(1000) as u16
}

#[cfg(test)]
mod tests {
    use super::super::super::material_batch::{MaterialGroup1Binding, PipelineVariantKey};
    use super::*;
    use crate::materials::{MaterialPipelineDesc, RasterPipelineKind, RasterPrimitiveTopology};
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn test_packet(first: usize, last: usize) -> MaterialBatchPacket {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.primitive_topology = RasterPrimitiveTopology::TriangleList;
        MaterialBatchPacket {
            first_draw_idx: first,
            last_draw_idx: last,
            pipeline_key: PipelineVariantKey::for_draw_item(
                &item,
                MaterialPipelineDesc {
                    surface_format: wgpu::TextureFormat::Rgba16Float,
                    depth_stencil_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                    sample_count: 1,
                    multiview_mask: None,
                },
                ShaderPermutation(0),
            ),
            resolved_pipeline_kind: None,
            group1_binding: MaterialGroup1Binding::Empty,
            pipelines: None,
        }
    }

    fn test_packet_with_key(
        first: usize,
        last: usize,
        pipeline_key: PipelineVariantKey,
    ) -> MaterialBatchPacket {
        MaterialBatchPacket {
            first_draw_idx: first,
            last_draw_idx: last,
            pipeline_key,
            resolved_pipeline_kind: Some(RasterPipelineKind::Null),
            group1_binding: MaterialGroup1Binding::Empty,
            pipelines: None,
        }
    }

    fn pipeline_state() -> WorldMeshForwardPipelineState {
        WorldMeshForwardPipelineState {
            use_multiview: false,
            pass_desc: MaterialPipelineDesc {
                surface_format: wgpu::TextureFormat::Rgba16Float,
                depth_stencil_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                sample_count: 1,
                multiview_mask: None,
            },
            shader_perm: ShaderPermutation(0),
            front_face_flip: false,
        }
    }

    fn cache_draws(count: usize) -> Vec<WorldMeshDrawItem> {
        (0..count)
            .map(|index| {
                dummy_world_mesh_draw_item(DummyDrawItemSpec {
                    material_asset_id: 1,
                    property_block: None,
                    skinned: false,
                    sorting_order: 0,
                    mesh_asset_id: 1,
                    node_id: index as i32,
                    slot_index: 0,
                    collect_order: index,
                    alpha_blended: false,
                })
            })
            .collect()
    }

    fn cache_key(
        draws: &[WorldMeshDrawItem],
        packets: &[MaterialBatchPacket],
    ) -> WorldMeshForwardInstancePlanCacheKey {
        WorldMeshForwardInstancePlanCacheKey::new(
            draws,
            packets,
            &pipeline_state(),
            true,
            OffscreenWriteTarget::None,
        )
    }

    #[test]
    fn instance_plan_cache_bypasses_small_inputs() {
        let cache = WorldMeshForwardInstancePlanCache::default();

        assert!(!cache.should_probe_cache(
            WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_DRAWS - 1,
            WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_PACKETS,
        ));

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.skipped_small, 1);
    }

    #[test]
    fn instance_plan_cache_reuses_stable_keys() {
        let cache = WorldMeshForwardInstancePlanCache::default();
        let draws = cache_draws(WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_DRAWS);
        let packets = [test_packet(0, 63), test_packet(64, draws.len() - 1)];
        let key = cache_key(&draws, &packets);

        let first = cache.get_or_build(key.clone(), InstancePlan::default);
        let second = cache.get_or_build(key, || panic!("stable key should hit"));

        assert_eq!(first, second);
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate_per_mille, 500);
    }

    #[test]
    fn instance_plan_cache_misses_when_material_submission_changes() {
        let cache = WorldMeshForwardInstancePlanCache::default();
        let draws = cache_draws(WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_DRAWS);
        let mut distinct_key = test_packet(0, 63).pipeline_key;
        distinct_key.render_state.depth_write = Some(true);
        let first_packets = [test_packet(0, 63), test_packet(64, draws.len() - 1)];
        let second_packets = [
            test_packet_with_key(0, 63, distinct_key),
            test_packet(64, draws.len() - 1),
        ];

        let _ = cache.get_or_build(cache_key(&draws, &first_packets), InstancePlan::default);
        let _ = cache.get_or_build(cache_key(&draws, &second_packets), InstancePlan::default);

        assert_eq!(cache.stats().misses, 2);
    }

    #[test]
    fn instance_plan_cache_temporarily_bypasses_after_repeated_misses() {
        let cache = WorldMeshForwardInstancePlanCache::default();
        let packets = [
            test_packet(0, 63),
            test_packet(64, WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_DRAWS - 1),
        ];

        for seed in 0..WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_THRASH_WINDOW_LOOKUPS {
            let draws = cache_draws(WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_DRAWS)
                .into_iter()
                .map(|mut item| {
                    item.node_id += (seed as i32) * 10_000;
                    item
                })
                .collect::<Vec<_>>();
            let _ = cache.get_or_build(cache_key(&draws, &packets), InstancePlan::default);
        }

        assert!(!cache.should_probe_cache(
            WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_DRAWS,
            WORLD_MESH_FORWARD_INSTANCE_PLAN_CACHE_MIN_PACKETS,
        ));
        assert_eq!(cache.stats().skipped_thrash, 1);
    }
}
