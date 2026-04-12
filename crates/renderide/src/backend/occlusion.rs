//! Hierarchical depth (Hi-Z) occlusion culling subsystem.
//!
//! Owns the GPU pyramid state, CPU readback snapshots, and temporal view/projection data used by
//! [`crate::render_graph::passes::WorldMeshForwardPass`] for previous-frame occlusion tests and by
//! [`crate::render_graph::passes::HiZBuildPass`] for pyramid construction after the forward pass.

use crate::render_graph::occlusion::{encode_hi_z_build, HiZGpuState};
use crate::render_graph::{
    capture_hi_z_temporal, HiZCullData, HiZTemporalState, OutputDepthMode, WorldMeshCullProjParams,
};
use crate::scene::SceneCoordinator;

/// GPU pyramid, CPU readback ring, and temporal cull snapshots for Hi-Z occlusion.
pub struct OcclusionSystem {
    hi_z_gpu: HiZGpuState,
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
            hi_z_gpu: HiZGpuState::default(),
        }
    }

    /// Hi-Z occlusion data cloned from the **previous** frame's pyramid readback, matching `mode`.
    pub(crate) fn hi_z_cull_data(&self, mode: OutputDepthMode) -> Option<HiZCullData> {
        match mode {
            OutputDepthMode::DesktopSingle => self
                .hi_z_gpu
                .desktop
                .as_ref()
                .map(|s| HiZCullData::Desktop(s.clone())),
            OutputDepthMode::StereoArray { .. } => {
                self.hi_z_gpu.stereo.as_ref().map(|s| HiZCullData::Stereo {
                    left: s.left.clone(),
                    right: s.right.clone(),
                })
            }
        }
    }

    /// Records Hi-Z GPU work into `encoder` (staging copy included).
    pub(crate) fn encode_hi_z_build_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        extent: (u32, u32),
        mode: OutputDepthMode,
    ) {
        encode_hi_z_build(
            device,
            queue,
            encoder,
            depth_view,
            extent,
            mode,
            &mut self.hi_z_gpu,
        );
    }

    /// Drains completed Hi-Z `map_async` readbacks into CPU snapshots for [`Self::hi_z_cull_data`].
    ///
    /// Non-blocking: uses at most one [`wgpu::Device::poll`]; if a read is not ready, prior
    /// snapshots are kept.
    pub fn hi_z_begin_frame_readback(&mut self, device: &wgpu::Device) {
        self.hi_z_gpu.begin_frame_readback(device);
    }

    /// Call after each successful render-graph submit that recorded Hi-Z copies (ring slot).
    pub(crate) fn hi_z_on_frame_submitted(&mut self, device: &wgpu::Device) {
        self.hi_z_gpu.on_frame_submitted(device);
    }

    /// View/projection snapshot from the **previous** world forward pass (for Hi-Z occlusion tests).
    pub(crate) fn hi_z_temporal_snapshot(&self) -> Option<HiZTemporalState> {
        self.hi_z_gpu.temporal.clone()
    }

    /// Records per-space views and cull params from **this** frame for Hi-Z tests on the **next** frame.
    pub(crate) fn capture_hi_z_temporal_for_next_frame(
        &mut self,
        scene: &SceneCoordinator,
        prev_cull: WorldMeshCullProjParams,
        viewport_px: (u32, u32),
    ) {
        self.hi_z_gpu.temporal = Some(capture_hi_z_temporal(scene, prev_cull, viewport_px));
    }
}
