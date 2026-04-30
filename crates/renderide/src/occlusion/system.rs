//! Hierarchical depth (Hi-Z) occlusion culling subsystem.
//!
//! Owns GPU pyramid state per logical view ([`ViewId`]), CPU readback snapshots, and
//! temporal view/projection data used by [`crate::passes::WorldMeshForwardOpaquePass`] and
//! [`crate::passes::HiZBuildPass`].

use std::sync::Arc;

use glam::Mat4;
use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::camera::ViewId;
use crate::gpu::OutputDepthMode;
use crate::occlusion::HiZCullData;
use crate::occlusion::gpu::{HiZBuildRecord, HiZGpuState, HiZHistoryTarget, encode_hi_z_build};
use crate::scene::SceneCoordinator;
use crate::world_mesh::{HiZTemporalState, WorldMeshCullProjParams, capture_hi_z_temporal};

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
    /// Per logical secondary-camera view pyramids (single-view desktop layout each).
    offscreen: Mutex<HashMap<ViewId, Arc<Mutex<HiZGpuState>>>>,
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
            offscreen: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the mutex-wrapped Hi-Z slot for `view`, creating it when needed.
    pub(crate) fn ensure_hi_z_state(&self, view: ViewId) -> Arc<Mutex<HiZGpuState>> {
        match view {
            ViewId::Main => self.main.clone(),
            ViewId::SecondaryCamera(_) => {
                let mut offscreen = self.offscreen.lock();
                offscreen
                    .entry(view)
                    .or_insert_with(|| Arc::new(Mutex::new(HiZGpuState::default())))
                    .clone()
            }
        }
    }

    /// Returns the existing mutex-wrapped Hi-Z slot for `view` without creating one.
    fn hi_z_state_slot(&self, view: ViewId) -> Option<Arc<Mutex<HiZGpuState>>> {
        match view {
            ViewId::Main => Some(self.main.clone()),
            ViewId::SecondaryCamera(_) => self.offscreen.lock().get(&view).cloned(),
        }
    }

    /// Hi-Z occlusion data cloned from the **previous** frame's pyramid readback, matching `mode`.
    pub(crate) fn hi_z_cull_data(
        &self,
        mode: OutputDepthMode,
        view: ViewId,
    ) -> Option<HiZCullData> {
        let slot = self.hi_z_state_slot(view)?;
        let state = slot.lock();
        match view {
            ViewId::Main => match mode {
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
            ViewId::SecondaryCamera(_) => state
                .desktop
                .as_ref()
                .map(|s| HiZCullData::Desktop(s.clone())),
        }
    }

    /// Retires all Hi-Z state owned by `view`.
    ///
    /// Main-view state persists for the life of the renderer; secondary views are removed when
    /// the view graph changes.
    pub(crate) fn retire_view(&self, view: ViewId) -> bool {
        match view {
            ViewId::Main => false,
            ViewId::SecondaryCamera(_) => self.offscreen.lock().remove(&view).is_some(),
        }
    }

    /// Number of live secondary-view Hi-Z slots.
    #[cfg(test)]
    pub(crate) fn secondary_view_count(&self) -> usize {
        self.offscreen.lock().len()
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
    /// [`crate::occlusion::gpu::HiZGpuState::start_ready_maps`], so no
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
        };
        let offscreen = self.offscreen.lock();
        for slot in offscreen.values() {
            let mut state = slot.lock();
            state.drain_completed_map_async();
            state.start_ready_maps();
        }
    }

    /// View/projection snapshot from the **previous** world forward pass (for Hi-Z occlusion tests).
    pub(crate) fn hi_z_temporal_snapshot(&self, view: ViewId) -> Option<HiZTemporalState> {
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
    use crate::camera::ViewId;
    use crate::scene::RenderSpaceId;

    /// Builds a secondary-camera view id for occlusion tests.
    fn secondary_view(render_space_id: i32, renderable_index: i32) -> ViewId {
        ViewId::secondary_camera(RenderSpaceId(render_space_id), renderable_index)
    }

    #[test]
    fn ensure_hi_z_state_reuses_slots_per_view() {
        let system = OcclusionSystem::new();
        let main_a = system.ensure_hi_z_state(ViewId::Main);
        let main_b = system.ensure_hi_z_state(ViewId::Main);
        let offscreen_a = system.ensure_hi_z_state(secondary_view(17, 0));
        let offscreen_b = system.ensure_hi_z_state(secondary_view(17, 0));

        assert!(Arc::ptr_eq(&main_a, &main_b));
        assert!(Arc::ptr_eq(&offscreen_a, &offscreen_b));
        assert!(!Arc::ptr_eq(&main_a, &offscreen_a));
    }

    #[test]
    fn ensure_hi_z_state_is_thread_safe_for_shared_view() {
        let system = Arc::new(OcclusionSystem::new());

        let first = (0..8)
            .map(|_| {
                let system = Arc::clone(&system);
                std::thread::spawn(move || system.ensure_hi_z_state(secondary_view(99, 0)))
            })
            .map(|thread| thread.join().expect("thread should finish"))
            .reduce(|first, next| {
                assert!(Arc::ptr_eq(&first, &next));
                first
            })
            .expect("at least one slot");

        let again = system.ensure_hi_z_state(secondary_view(99, 0));
        assert!(Arc::ptr_eq(&first, &again));
    }

    /// Retiring one secondary view preserves other secondary view Hi-Z slots.
    #[test]
    fn retire_view_removes_only_target_slot() {
        let system = OcclusionSystem::new();
        let retired = secondary_view(99, 0);
        let surviving = secondary_view(99, 1);
        let _ = system.ensure_hi_z_state(retired);
        let _ = system.ensure_hi_z_state(surviving);

        assert_eq!(system.secondary_view_count(), 2);
        assert!(system.retire_view(retired));
        assert_eq!(system.secondary_view_count(), 1);
        assert!(system.hi_z_state_slot(retired).is_none());
        assert!(system.hi_z_state_slot(surviving).is_some());
    }
}
