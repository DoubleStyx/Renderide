//! Hierarchical depth (Hi-Z) occlusion culling subsystem.
//!
//! Owns GPU pyramid state per logical view ([`OcclusionViewId`]), CPU readback snapshots, and
//! temporal view/projection data used by [`crate::render_graph::passes::WorldMeshForwardOpaquePass`] and
//! [`crate::render_graph::passes::HiZBuildPass`].

use std::num::NonZeroUsize;
use std::sync::Arc;

use glam::Mat4;
use lru::LruCache;
use parking_lot::Mutex;

use crate::render_graph::occlusion::{
    encode_hi_z_build, HiZBuildRecord, HiZGpuState, HiZHistoryTarget,
};
use crate::render_graph::OcclusionViewId;
use crate::render_graph::{
    capture_hi_z_temporal, HiZCullData, HiZTemporalState, OutputDepthMode, WorldMeshCullProjParams,
};
use crate::scene::SceneCoordinator;

/// Maximum distinct host render-texture occlusion pyramids retained ([`OcclusionSystem::offscreen`] LRU).
const OFFSCREEN_HIZ_LRU_CAP: usize = 64;

const OFFSCREEN_HIZ_LRU_CAP_NZ: NonZeroUsize = {
    match NonZeroUsize::new(OFFSCREEN_HIZ_LRU_CAP) {
        Some(n) => n,
        None => panic!("OFFSCREEN_HIZ_LRU_CAP must be non-zero"),
    }
};

/// Depth source, layout, and logical view for [`OcclusionSystem::encode_hi_z_build_pass`].
pub(crate) struct HiZBuildInput<'a> {
    /// Depth attachment view (desktop 2D or multiview array) sampled for mip0.
    pub depth_view: &'a wgpu::TextureView,
    /// Registry-owned ping-pong history texture that receives the pyramid.
    pub history_texture: &'a wgpu::Texture,
    /// Registry-owned per-layer/per-mip views for [`Self::history_texture`].
    pub history_mip_views: &'a crate::backend::HistoryTextureMipViews,
    /// Full framebuffer extent in pixels (matches the depth attachment).
    pub extent: (u32, u32),
    /// Desktop single-view vs stereo depth array layout.
    pub mode: OutputDepthMode,
}

/// GPU pyramid, CPU readback ring, and temporal cull snapshots for Hi-Z occlusion.
pub struct OcclusionSystem {
    /// Main window / OpenXR multiview Hi-Z (desktop and stereo layouts).
    main: Arc<Mutex<HiZGpuState>>,
    /// Per host render-texture secondary camera pyramids (single-view desktop layout each), LRU-bounded.
    offscreen: Mutex<LruCache<i32, Arc<Mutex<HiZGpuState>>>>,
}

impl Default for OcclusionSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl OcclusionSystem {
    /// Creates an empty occlusion system with no pyramid data.
    pub fn new() -> Self {
        Self {
            main: Arc::new(Mutex::new(HiZGpuState::default())),
            offscreen: Mutex::new(LruCache::new(OFFSCREEN_HIZ_LRU_CAP_NZ)),
        }
    }

    /// Returns the mutex-wrapped Hi-Z slot for `view`, creating it when needed.
    pub(crate) fn ensure_hi_z_state(&self, view: OcclusionViewId) -> Arc<Mutex<HiZGpuState>> {
        match view {
            OcclusionViewId::Main => self.main.clone(),
            OcclusionViewId::OffscreenRenderTexture(id) => {
                let mut offscreen = self.offscreen.lock();
                if let Some(existing) = offscreen.get(&id) {
                    return existing.clone();
                }
                let slot = Arc::new(Mutex::new(HiZGpuState::default()));
                let _ = offscreen.put(id, slot.clone());
                slot
            }
        }
    }

    /// Returns the existing mutex-wrapped Hi-Z slot for `view` without creating one.
    fn hi_z_state_slot(&self, view: OcclusionViewId) -> Option<Arc<Mutex<HiZGpuState>>> {
        match view {
            OcclusionViewId::Main => Some(self.main.clone()),
            OcclusionViewId::OffscreenRenderTexture(id) => self.offscreen.lock().peek(&id).cloned(),
        }
    }

    /// Hi-Z occlusion data cloned from the **previous** frame's pyramid readback, matching `mode`.
    pub(crate) fn hi_z_cull_data(
        &self,
        mode: OutputDepthMode,
        view: OcclusionViewId,
    ) -> Option<HiZCullData> {
        let slot = self.hi_z_state_slot(view)?;
        let state = slot.lock();
        match view {
            OcclusionViewId::Main => match mode {
                OutputDepthMode::DesktopSingle => state
                    .desktop
                    .as_ref()
                    .map(|s| HiZCullData::Desktop(s.clone())),
                OutputDepthMode::StereoArray { .. } => {
                    state.stereo.as_ref().map(|s| HiZCullData::Stereo {
                        left: s.left.clone(),
                        right: s.right.clone(),
                    })
                }
            },
            OcclusionViewId::OffscreenRenderTexture(_) => state
                .desktop
                .as_ref()
                .map(|s| HiZCullData::Desktop(s.clone())),
        }
    }

    /// Records Hi-Z GPU work into `encoder` (staging copy included).
    pub(crate) fn encode_hi_z_build_pass(
        &self,
        record: HiZBuildRecord<'_>,
        state_slot: &Mutex<HiZGpuState>,
        input: HiZBuildInput<'_>,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) {
        profiling::scope!("hi_z::build");
        let mut state = state_slot.lock();
        encode_hi_z_build(
            record,
            input.depth_view,
            HiZHistoryTarget {
                texture: input.history_texture,
                mip_views: input.history_mip_views,
            },
            input.extent,
            input.mode,
            &mut state,
            profiler,
        );
    }

    /// Drains completed Hi-Z `map_async` readbacks into CPU snapshots for [`Self::hi_z_cull_data`]
    /// and promotes any submit-done readback slots into fresh `map_async` requests on the main thread.
    ///
    /// Non-blocking: uses at most one [`wgpu::Device::poll`]; if a read is not ready, prior
    /// snapshots are kept.
    ///
    /// The poll runs **before** any [`HiZGpuState`] lock so the
    /// [`wgpu::Queue::on_submitted_work_done`] callback installed by
    /// [`crate::render_graph::compiled::exec::CompiledRenderGraph::execute_multi_view`]
    /// (which itself locks the per-view [`HiZGpuState`]) can execute without re-entering
    /// a lock held by this function. That callback only marks the encoded readback slot as
    /// submit-done; the actual `map_async` runs here via
    /// [`crate::render_graph::occlusion::HiZGpuState::start_ready_maps`], so no
    /// wgpu call is issued from inside the device-poll callback (which would risk deadlocks
    /// with wgpu's internal queue-write locks — observed as a futex hang inside
    /// `queue.write_texture` during asset upload).
    pub fn hi_z_begin_frame_readback(&mut self, device: &wgpu::Device) {
        profiling::scope!("hi_z::readback_drain");
        let _ = device.poll(wgpu::PollType::Poll);
        {
            let mut main = self.main.lock();
            main.drain_completed_map_async();
            main.start_ready_maps();
        }
        let offscreen = self.offscreen.lock();
        for (_, slot) in offscreen.iter() {
            let mut state = slot.lock();
            state.drain_completed_map_async();
            state.start_ready_maps();
        }
    }

    /// View/projection snapshot from the **previous** world forward pass (for Hi-Z occlusion tests).
    pub(crate) fn hi_z_temporal_snapshot(&self, view: OcclusionViewId) -> Option<HiZTemporalState> {
        self.hi_z_state_slot(view)?.lock().temporal.clone()
    }

    /// Records per-space views and cull params from **this** frame for Hi-Z tests on the **next** frame.
    pub(crate) fn capture_hi_z_temporal_for_next_frame(
        &self,
        scene: &SceneCoordinator,
        prev_cull: WorldMeshCullProjParams,
        viewport_px: (u32, u32),
        state_slot: &Mutex<HiZGpuState>,
        explicit_world_to_view: Option<Mat4>,
    ) {
        profiling::scope!("hi_z::capture_temporal");
        let temporal = Some(capture_hi_z_temporal(
            scene,
            prev_cull,
            viewport_px,
            explicit_world_to_view,
        ));
        let mut state = state_slot.lock();
        state.temporal = temporal;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::OcclusionSystem;
    use crate::render_graph::OcclusionViewId;

    #[test]
    fn ensure_hi_z_state_reuses_slots_per_view() {
        let system = OcclusionSystem::new();
        let main_a = system.ensure_hi_z_state(OcclusionViewId::Main);
        let main_b = system.ensure_hi_z_state(OcclusionViewId::Main);
        let offscreen_a = system.ensure_hi_z_state(OcclusionViewId::OffscreenRenderTexture(17));
        let offscreen_b = system.ensure_hi_z_state(OcclusionViewId::OffscreenRenderTexture(17));

        assert!(Arc::ptr_eq(&main_a, &main_b));
        assert!(Arc::ptr_eq(&offscreen_a, &offscreen_b));
        assert!(!Arc::ptr_eq(&main_a, &offscreen_a));
    }

    #[test]
    fn ensure_hi_z_state_is_thread_safe_for_shared_view() {
        let system = Arc::new(OcclusionSystem::new());
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let system = Arc::clone(&system);
                std::thread::spawn(move || {
                    system.ensure_hi_z_state(OcclusionViewId::OffscreenRenderTexture(99))
                })
            })
            .collect();

        let first = threads
            .into_iter()
            .map(|thread| thread.join().expect("thread should finish"))
            .reduce(|first, next| {
                assert!(Arc::ptr_eq(&first, &next));
                first
            })
            .expect("at least one slot");

        let again = system.ensure_hi_z_state(OcclusionViewId::OffscreenRenderTexture(99));
        assert!(Arc::ptr_eq(&first, &again));
    }
}
