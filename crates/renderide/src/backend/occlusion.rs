//! Hierarchical depth (Hi-Z) occlusion culling subsystem.
//!
//! Owns GPU pyramid state per logical view ([`OcclusionViewId`]), CPU readback snapshots, and
//! temporal view/projection data used by [`crate::render_graph::passes::WorldMeshForwardPass`] and
//! [`crate::render_graph::passes::HiZBuildPass`].

use std::collections::HashMap;

use crate::render_graph::occlusion::{encode_hi_z_build, HiZGpuState};
use crate::render_graph::OcclusionViewId;
use crate::render_graph::{
    capture_hi_z_temporal, HiZCullData, HiZTemporalState, OutputDepthMode, WorldMeshCullProjParams,
};
use crate::scene::SceneCoordinator;

/// GPU pyramid, CPU readback ring, and temporal cull snapshots for Hi-Z occlusion.
pub struct OcclusionSystem {
    /// Main window / OpenXR multiview Hi-Z (desktop and stereo layouts).
    main: HiZGpuState,
    /// Per host render-texture secondary camera pyramids (single-view desktop layout each).
    offscreen: HashMap<i32, HiZGpuState>,
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
            main: HiZGpuState::default(),
            offscreen: HashMap::new(),
        }
    }

    fn hi_z_state_mut(&mut self, view: OcclusionViewId) -> &mut HiZGpuState {
        match view {
            OcclusionViewId::Main => &mut self.main,
            OcclusionViewId::OffscreenRenderTexture(id) => self.offscreen.entry(id).or_default(),
        }
    }

    fn hi_z_state_ref(&self, view: OcclusionViewId) -> Option<&HiZGpuState> {
        match view {
            OcclusionViewId::Main => Some(&self.main),
            OcclusionViewId::OffscreenRenderTexture(id) => self.offscreen.get(&id),
        }
    }

    /// Hi-Z occlusion data cloned from the **previous** frame's pyramid readback, matching `mode`.
    pub(crate) fn hi_z_cull_data(
        &self,
        mode: OutputDepthMode,
        view: OcclusionViewId,
    ) -> Option<HiZCullData> {
        let state = self.hi_z_state_ref(view)?;
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_hi_z_build_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        extent: (u32, u32),
        mode: OutputDepthMode,
        view: OcclusionViewId,
    ) {
        let state = self.hi_z_state_mut(view);
        encode_hi_z_build(device, queue, encoder, depth_view, extent, mode, state);
    }

    /// Drains completed Hi-Z `map_async` readbacks into CPU snapshots for [`Self::hi_z_cull_data`].
    ///
    /// Non-blocking: uses at most one [`wgpu::Device::poll`]; if a read is not ready, prior
    /// snapshots are kept.
    pub fn hi_z_begin_frame_readback(&mut self, device: &wgpu::Device) {
        self.main.begin_frame_readback(device);
        for s in self.offscreen.values_mut() {
            s.begin_frame_readback(device);
        }
    }

    /// Call after each successful render-graph submit that recorded Hi-Z copies for the given view.
    pub(crate) fn hi_z_on_frame_submitted_for_view(
        &mut self,
        device: &wgpu::Device,
        view: OcclusionViewId,
    ) {
        self.hi_z_state_mut(view).on_frame_submitted(device);
    }

    /// View/projection snapshot from the **previous** world forward pass (for Hi-Z occlusion tests).
    pub(crate) fn hi_z_temporal_snapshot(&self, view: OcclusionViewId) -> Option<HiZTemporalState> {
        self.hi_z_state_ref(view)?.temporal.clone()
    }

    /// Records per-space views and cull params from **this** frame for Hi-Z tests on the **next** frame.
    pub(crate) fn capture_hi_z_temporal_for_next_frame(
        &mut self,
        scene: &SceneCoordinator,
        prev_cull: WorldMeshCullProjParams,
        viewport_px: (u32, u32),
        view: OcclusionViewId,
    ) {
        let temporal = Some(capture_hi_z_temporal(scene, prev_cull, viewport_px));
        let state = self.hi_z_state_mut(view);
        state.temporal = temporal;
    }
}
