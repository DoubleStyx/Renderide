//! Frame-driving half of [`super::RenderideApp`]: tick orchestration, XR/render dispatch, and
//! present/diagnostics coordination.

use std::sync::Arc;
use std::time::Instant;

use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use super::*;

impl RenderideApp {
    /// Ends the current frame tick and emits the Tracy frame mark once.
    fn finish_frame_tick(&mut self) {
        self.frame_tick_epilogue();
        crate::profiling::emit_frame_mark();
    }

    /// Handles the post-lock-step shutdown / fatal-exit checks that must run before view rendering.
    fn handle_frame_exit_requests(&mut self, event_loop: &ActiveEventLoop) -> bool {
        if let Some(bundle) = self.xr_session.as_ref() {
            if bundle.handles.xr_session.exit_requested() {
                logger::info!("OpenXR requested exit");
                self.exit_code = Some(0);
                event_loop.exit();
                return true;
            }
        }

        if self.runtime.shutdown_requested() {
            logger::info!("Renderer shutdown requested by host");
            self.exit_code = Some(0);
            event_loop.exit();
            return true;
        }

        if self.runtime.fatal_error() {
            logger::error!("Renderer fatal IPC error");
            self.exit_code = Some(4);
            event_loop.exit();
            return true;
        }

        false
    }

    fn frame_tick_prologue(&mut self, frame_start: Instant) {
        profiling::scope!("tick::prologue");
        tick_phase_trace("frame_tick_prologue");
        if let Some(prev_end) = self.previous_tick_end {
            let idle_ms = frame_start
                .saturating_duration_since(prev_end)
                .as_secs_f64()
                * 1000.0;
            crate::profiling::plot_event_loop_idle_ms(idle_ms);
        }
        self.record_frame_tick_start(frame_start);
        self.sync_log_level_from_settings();
        self.runtime.tick_frame_wall_clock_begin(frame_start);
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.begin_frame_timing(frame_start);
            if let Ok(settings) = self.runtime.settings().read() {
                gpu.set_present_mode(settings.rendering.vsync);
            }
        }
    }

    /// Phase: drain incoming IPC and apply host-driven window state (cursor/output) plus per-frame cursor lock.
    fn poll_ipc_and_window(&mut self) {
        profiling::scope!("tick::poll_ipc_and_window");
        tick_phase_trace("poll_ipc_and_window");
        self.runtime.poll_ipc();

        if let (Some(window), Some(output_state)) = (
            self.window.as_ref(),
            self.runtime.take_pending_output_state(),
        ) {
            if let Err(error) = apply_output_state_to_window(
                window.as_ref(),
                &output_state,
                &mut self.cursor_output_tracking,
            ) {
                logger::debug!("apply_output_state_to_window: {error:?}");
            }
        }

        if let Some(window) = self.window.as_ref() {
            if self.runtime.host_cursor_lock_requested() {
                let lock_pos = self
                    .runtime
                    .last_output_state()
                    .and_then(|state| state.lock_cursor_position);
                if let Err(error) = apply_per_frame_cursor_lock_when_locked(
                    window.as_ref(),
                    &mut self.input,
                    lock_pos,
                ) {
                    logger::trace!("apply_per_frame_cursor_lock_when_locked: {error:?}");
                }
            }
        }
    }

    /// Phase: OpenXR frame tick (view poses and sampling before lock-step). Updates cached head pose and controllers.
    ///
    /// Returns [`None`] when OpenXR is not active or the session does not produce a tick this frame.
    fn xr_begin_tick(&mut self) -> Option<OpenxrFrameTick> {
        profiling::scope!("tick::xr_begin_tick");
        tick_phase_trace("xr_begin_tick");
        let gpu_queue_access_gate = self
            .gpu
            .as_ref()
            .map(|gpu| gpu.gpu_queue_access_gate().clone())?;
        let xr_tick = self.xr_session.as_mut().and_then(|bundle| {
            frame_loop::begin_openxr_frame_tick(
                &mut bundle.handles,
                &mut self.runtime,
                &gpu_queue_access_gate,
            )
        });

        if let Some(ref tick) = xr_tick {
            crate::xr::OpenxrInput::log_stereo_view_order_once(&tick.views);
            if let Some(bundle) = &self.xr_session {
                if let Some(ref input) = bundle.handles.openxr_input {
                    if bundle.handles.xr_session.session_running() {
                        match input.sync_and_sample(
                            bundle.handles.xr_session.xr_vulkan_session(),
                            bundle.handles.xr_session.stage_space(),
                            tick.predicted_display_time,
                        ) {
                            Ok(controllers) => self.cached_openxr_controllers = controllers,
                            Err(error) => logger::trace!("OpenXR input sync: {error:?}"),
                        }
                    }
                }
            }
            self.cached_head_pose =
                crate::xr::headset_center_pose_from_stereo_views(tick.views.as_slice());
            if let (Some(v0), Some(v1), Some((ipc_p, ipc_q))) =
                (tick.views.first(), tick.views.get(1), self.cached_head_pose)
            {
                let rp0 = &v0.pose.position;
                let rp1 = &v1.pose.position;
                let render_center_x = (rp0.x + rp1.x) * 0.5;
                let render_center_y = (rp0.y + rp1.y) * 0.5;
                let render_center_z = (rp0.z + rp1.z) * 0.5;
                logger::trace!(
                    "HEAD POS | render(OpenXR RH): ({:.3},{:.3},{:.3}) | ipc->host(Unity LH): ({:.3},{:.3},{:.3}) | ipc_quat: ({:.4},{:.4},{:.4},{:.4})",
                    render_center_x, render_center_y, render_center_z,
                    ipc_p.x, ipc_p.y, ipc_p.z,
                    ipc_q.x, ipc_q.y, ipc_q.z, ipc_q.w,
                );
            }
        }

        xr_tick
    }

    /// Phase: lock-step begin-frame to host when [`RendererRuntime::should_send_begin_frame`].
    fn lock_step_exchange(&mut self) {
        profiling::scope!("tick::lock_step_exchange");
        tick_phase_trace("lock_step_exchange");
        if self.runtime.should_send_begin_frame() {
            let lock = self.runtime.host_cursor_lock_requested();
            let mut inputs = self.input.take_input_state(lock);
            crate::diagnostics::sanitize_input_state_for_imgui_host(
                &mut inputs,
                self.runtime.debug_hud_last_want_capture_mouse(),
                self.runtime.debug_hud_last_want_capture_keyboard(),
            );
            let synthesised_hands = synthesize_hand_states(&self.cached_openxr_controllers);
            if let Some(vr) = vr_inputs_for_session(
                self.session_output_device,
                self.cached_head_pose,
                &self.cached_openxr_controllers,
                synthesised_hands,
            ) {
                inputs.vr = Some(vr);
            }
            self.runtime.pre_frame(inputs);
        } else {
            profiling::scope!("lock_step::skipped");
        }
    }

    /// Phase: HMD multiview submission, secondary cameras to render textures,
    /// and debug HUD input/time for this frame.
    ///
    /// Returns [`None`] if no [`GpuContext`] is available (mirror epilogue-only path). Otherwise returns
    /// whether the HMD projection layer was submitted (`hmd_projection_ended`).
    fn render_views(
        &mut self,
        window: &Arc<Window>,
        xr_tick: Option<&OpenxrFrameTick>,
    ) -> Option<bool> {
        profiling::scope!("tick::render_views");
        tick_phase_trace("render_views");
        if let Some(gpu) = self.gpu.as_mut() {
            self.runtime.drain_hi_z_readback(gpu.device());
        }
        let hmd_projection_ended = match (self.gpu.as_mut(), self.xr_session.as_mut(), xr_tick) {
            (Some(gpu), Some(bundle), Some(tick)) => {
                frame_loop::try_hmd_multiview_submit(gpu, bundle, &mut self.runtime, tick)
            }
            _ => false,
        };

        let gpu = self.gpu.as_mut()?;

        if !hmd_projection_ended {
            use crate::xr::XrFrameRenderer;
            let result = if self.runtime.vr_active() {
                self.runtime.submit_secondary_only(gpu)
            } else {
                self.runtime.render_desktop_frame(gpu)
            };
            if let Err(error) = result {
                Self::handle_frame_graph_error(gpu, error);
            }
        }

        {
            let now = Instant::now();
            let ms = self
                .hud_frame_last
                .map(|time| now.duration_since(time).as_secs_f64() * 1000.0)
                .unwrap_or(16.67);
            self.hud_frame_last = Some(now);
            let hud_in =
                crate::diagnostics::DebugHudInput::from_winit(window.as_ref(), &mut self.input);
            self.runtime.set_debug_hud_frame_data(hud_in, ms);
        }

        Some(hmd_projection_ended)
    }

    /// Phase: VR mirror vs desktop world render, then OpenXR `end_frame_empty` when the HMD path did not submit.
    ///
    /// Call only after [`Self::render_views`] returned [`Some`], so a [`GpuContext`] is guaranteed.
    fn present_and_diagnostics(
        &mut self,
        xr_tick: Option<OpenxrFrameTick>,
        hmd_projection_ended: bool,
    ) {
        profiling::scope!("tick::present_and_diagnostics");
        tick_phase_trace("present_and_diagnostics");
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let gpu_queue_access_gate = gpu.gpu_queue_access_gate().clone();
        if self.runtime.vr_active() {
            if hmd_projection_ended {
                if let Some(bundle) = self.xr_session.as_mut() {
                    if let Err(error) = frame_loop::present_vr_mirror_blit(
                        gpu,
                        &mut bundle.mirror_blit,
                        |encoder, view, gpu| {
                            self.runtime
                                .encode_debug_hud_overlay_on_surface(gpu, encoder, view)
                        },
                    ) {
                        logger::debug!("VR mirror blit failed: {error:?}");
                        if let Err(present_error) = present_clear_frame_overlay_traced(
                            gpu,
                            SurfaceAcquireTrace::VrClear,
                            SurfaceSubmitTrace::VrClear,
                            |encoder, view, gpu| {
                                self.runtime
                                    .encode_debug_hud_overlay_on_surface(gpu, encoder, view)
                            },
                        ) {
                            logger::warn!(
                                "present_clear_frame after mirror blit: {present_error:?}"
                            );
                        }
                    }
                }
            } else if let Err(error) = present_clear_frame_overlay_traced(
                gpu,
                SurfaceAcquireTrace::VrClear,
                SurfaceSubmitTrace::VrClear,
                |encoder, view, gpu| {
                    self.runtime
                        .encode_debug_hud_overlay_on_surface(gpu, encoder, view)
                },
            ) {
                logger::debug!("VR mirror clear (no HMD frame): {error:?}");
            }
        }

        if let (Some(bundle), Some(tick)) = (self.xr_session.as_mut(), xr_tick) {
            if !hmd_projection_ended {
                profiling::scope!("xr::end_frame_if_open");
                if let Err(error) = bundle
                    .handles
                    .xr_session
                    .end_frame_if_open(tick.predicted_display_time, &gpu_queue_access_gate)
                {
                    logger::debug!("OpenXR end_frame_if_open: {error:?}");
                }
            }
        }
    }

    /// Ends GPU frame timing, refreshes debug HUD snapshots, and forwards the most recently
    /// completed GPU render time to the frontend for [`crate::shared::PerformanceState::render_time`].
    /// Pairs with [`Self::frame_tick_prologue`].
    fn frame_tick_epilogue(&mut self) {
        profiling::scope!("tick::epilogue");
        tick_phase_trace("frame_tick_epilogue");
        self.drain_driver_thread_error();
        self.end_frame_timing_and_hud_capture();
        let gpu_render_time_seconds = self
            .gpu
            .as_ref()
            .and_then(|gpu| gpu.last_completed_gpu_render_time_seconds());
        self.runtime
            .tick_frame_render_time_end(gpu_render_time_seconds);
        self.runtime.note_render_tick_complete();
        self.previous_tick_end = Some(Instant::now());
    }

    /// Logs any error captured on the driver thread since the last epilogue.
    ///
    /// Current wgpu 29's `Queue::submit` and `SurfaceTexture::present` are infallible, so
    /// this path is reserved for future wgpu versions that return fallible submits or
    /// explicit present errors. The check is cheap (one mutex acquire on an empty slot)
    /// and keeps the wiring in place for when it begins firing.
    fn drain_driver_thread_error(&mut self) {
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        if let Some(err) = gpu.take_driver_error() {
            logger::error!("{err}");
        }
    }

    /// One winit redraw; phase order is documented on this module ([`crate::app::renderide_app`]).
    pub(super) fn tick_frame(&mut self, event_loop: &ActiveEventLoop) {
        profiling::scope!("tick::frame");
        let frame_start = Instant::now();
        if let Some(heartbeat) = self.main_heartbeat.as_ref() {
            heartbeat.pet();
        }
        self.frame_tick_prologue(frame_start);
        self.poll_ipc_and_window();
        if self.check_external_shutdown(event_loop) {
            self.finish_frame_tick();
            return;
        }
        self.runtime.update_decoupling_activation(Instant::now());
        {
            profiling::scope!("tick::asset_integration");
            self.runtime.run_asset_integration();
        }
        if let Some(gpu) = self.gpu.as_ref() {
            self.runtime.maintain_nonblocking_gpu_jobs(gpu);
        }
        let xr_pause = self
            .main_heartbeat
            .as_ref()
            .map(|heartbeat| heartbeat.pause());
        let xr_tick = self.xr_begin_tick();
        drop(xr_pause);
        self.lock_step_exchange();

        if self.handle_frame_exit_requests(event_loop) {
            self.finish_frame_tick();
            return;
        }

        let Some(window) = self.window.clone() else {
            self.finish_frame_tick();
            return;
        };

        let Some(hmd_projection_ended) = self.render_views(&window, xr_tick.as_ref()) else {
            self.finish_frame_tick();
            return;
        };

        let _ = window;
        self.present_and_diagnostics(xr_tick, hmd_projection_ended);
        self.finish_frame_tick();
    }

    /// Finalizes [`GpuContext`] frame timing, drains GPU profiler results, and refreshes debug HUD snapshots for the tick.
    fn end_frame_timing_and_hud_capture(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.end_frame_timing();
            gpu.end_gpu_profiler_frame();
            self.runtime.capture_debug_hud_after_frame_end(gpu);
        }
    }

    fn handle_frame_graph_error(gpu: &mut GpuContext, error: GraphExecuteError) {
        match error {
            GraphExecuteError::NoFrameGraph => {
                if let Err(present_error) = present_clear_frame(gpu) {
                    logger::warn!("present fallback failed: {present_error:?}");
                    reconfigure_gpu_for_window(gpu);
                }
            }
            _ => {
                logger::warn!("frame graph failed: {error:?}");
                reconfigure_gpu_for_window(gpu);
            }
        }
    }
}
