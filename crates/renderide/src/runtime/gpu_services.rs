//! GPU-facing per-tick services on [`super::RendererRuntime`].
//!
//! These helpers run once per tick (from the render entry point, the tick prologue, or the
//! app driver) and forward into a backend pool/cache concern that needs the GPU device or a
//! [`GpuContext`]. Keeping them on one file groups the runtime's cross-cutting GPU duties.

use crate::gpu::GpuContext;

use super::RendererRuntime;

impl RendererRuntime {
    /// Applies the renderer-wide MSAA setting to the mono forward path and returns the effective tier.
    pub(super) fn sync_master_msaa(&mut self, gpu: &mut GpuContext) -> u32 {
        profiling::scope!("render::sync_master_msaa");
        let requested_msaa = self.requested_master_msaa_count();
        let prev_msaa = gpu.msaa().swapchain_msaa_effective();
        gpu.msaa_mut().set_swapchain_msaa_requested(requested_msaa);
        let effective = gpu.msaa().swapchain_msaa_effective();
        self.transient_evict_stale_msaa_tiers_if_changed(prev_msaa, effective);
        effective.max(1)
    }

    /// Applies the renderer-wide MSAA setting to the stereo forward path and returns the effective tier.
    pub(super) fn sync_stereo_msaa_from_master(&mut self, gpu: &mut GpuContext) -> u32 {
        profiling::scope!("render::sync_stereo_msaa");
        let requested_msaa = self.requested_master_msaa_count();
        let prev_stereo = gpu.msaa().swapchain_msaa_effective_stereo();
        gpu.msaa_mut()
            .set_swapchain_msaa_requested_stereo(requested_msaa);
        let effective = gpu.msaa().swapchain_msaa_effective_stereo();
        self.transient_evict_stale_msaa_tiers_if_changed(prev_stereo, effective);
        effective.max(1)
    }

    fn requested_master_msaa_count(&self) -> u32 {
        self.config
            .settings
            .read()
            .map(|s| s.rendering.msaa.as_count())
            .unwrap_or(1)
    }

    /// Drops transient-pool GPU textures for free-list entries whose MSAA sample count no longer
    /// matches the effective swapchain tier (avoids VRAM retention when toggling MSAA).
    pub(super) fn transient_evict_stale_msaa_tiers_if_changed(
        &mut self,
        prev_effective: u32,
        new_effective: u32,
    ) {
        if prev_effective == new_effective {
            return;
        }
        let eff = new_effective.max(1);
        self.backend
            .transient_pool_mut()
            .evict_texture_keys_where(|k| k.sample_count > 1 && k.sample_count != eff);
    }

    /// Drains completed Hi-Z `map_async` readbacks into CPU snapshots (once per tick).
    ///
    /// Call at the top of the render-views phase so both the HMD and desktop paths share one drain.
    pub fn drain_hi_z_readback(&mut self, gpu: &mut GpuContext) {
        profiling::scope!("tick::drain_hi_z_readback");
        let mapped_buffer_recovery = gpu.begin_mapped_buffer_recovery_frame();
        if mapped_buffer_recovery.invalidated {
            self.backend.reset_mapped_buffer_recovery_state(
                mapped_buffer_recovery.generation,
                "tick begin",
            );
        }
        if mapped_buffer_recovery.avoid_mapped_buffers {
            return;
        }
        self.backend.hi_z_begin_frame_readback(gpu.device());
        if gpu.observe_mapped_buffer_invalidation_during_frame() {
            self.backend.reset_mapped_buffer_recovery_state(
                gpu.mapped_buffer_invalidation_generation(),
                "tick Hi-Z readback",
            );
        }
    }

    /// Advances nonblocking GPU services that feed host-visible async results.
    pub fn maintain_nonblocking_gpu_jobs(&mut self, gpu: &mut GpuContext) {
        profiling::scope!("tick::maintain_nonblocking_gpu_jobs");
        self.flush_reflection_probe_render_results();
        let advance_sliced_ibl = self.tick_state.take_reflection_probe_ibl_advance_for_tick();
        if advance_sliced_ibl {
            profiling::scope!("tick::poll_nonblocking_gpu_jobs");
            let _ = gpu.device().poll(wgpu::PollType::Poll);
        }
        self.backend.maintain_reflection_probe_specular_jobs(
            gpu,
            &self.scene,
            self.scene.active_main_render_context(),
            advance_sliced_ibl,
        );
        self.backend.maintain_reflection_probe_sh2_jobs(gpu);
    }
}
