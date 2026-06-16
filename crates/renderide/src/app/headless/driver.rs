//! Headless offscreen driver loop.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::crash_context::{self, RenderMode, TargetMode, TickPhase};
use crate::gpu::GpuContext;
use crate::ipc::HeadlessParams;
use crate::run_error::RunError;
use crate::runtime::{RendererRuntime, TickOutcome};
use crate::shared::InputState;

use super::super::bootstrap::{ExternalShutdownCoordinator, GpuStartupConfig};
use super::super::exit::RunExit;
use super::readback;
use super::schedule::{HeadlessSchedule, HeadlessTickKind};

/// Sleep granularity inside the headless tick loop.
const HEADLESS_TICK_SLEEP: Duration = Duration::from_millis(5);
/// Maximum time headless mode waits for cooperative resource shutdown.
const HEADLESS_GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// Runs the renderer in headless offscreen mode until shutdown or fatal IPC.
pub(crate) fn run_headless(
    runtime: &mut RendererRuntime,
    params: HeadlessParams,
    external_shutdown: Option<ExternalShutdownCoordinator>,
    gpu_config: GpuStartupConfig,
) -> Result<RunExit, RunError> {
    let mut driver = HeadlessDriver::new(runtime, params, external_shutdown, gpu_config)?;
    driver.run()
}

struct HeadlessDriver<'a> {
    runtime: &'a mut RendererRuntime,
    params: HeadlessParams,
    external_shutdown: Option<ExternalShutdownCoordinator>,
    gpu: GpuContext,
    schedule: HeadlessSchedule,
    frames_written: u64,
}

impl<'a> HeadlessDriver<'a> {
    fn new(
        runtime: &'a mut RendererRuntime,
        params: HeadlessParams,
        external_shutdown: Option<ExternalShutdownCoordinator>,
        gpu_config: GpuStartupConfig,
    ) -> Result<Self, RunError> {
        logger::info!(
            "Headless mode: output={} size={}x{} interval_ms={}",
            params.output_path.display(),
            params.width,
            params.height,
            params.interval_ms,
        );
        crash_context::set_target_mode(TargetMode::Headless);
        crash_context::set_render_mode(RenderMode::Headless);

        let gpu = pollster::block_on(GpuContext::new_headless(
            params.width,
            params.height,
            gpu_config.max_frame_latency,
            gpu_config.gpu_validation_layers,
            gpu_config.power_preference,
            gpu_config.graphics_api,
        ))
        .map_err(RunError::gpu)?;
        runtime.attach_gpu(&gpu);
        let schedule = HeadlessSchedule::new(params.interval_ms, Instant::now());

        Ok(Self {
            runtime,
            params,
            external_shutdown,
            gpu,
            schedule,
            frames_written: 0,
        })
    }

    fn run(&mut self) -> Result<RunExit, RunError> {
        loop {
            if self.external_shutdown_requested() {
                logger::info!("Headless: external shutdown requested, exiting");
                self.drain_graceful_shutdown();
                break;
            }

            let now = Instant::now();
            crash_context::set_tick_phase(TickPhase::Headless);
            self.runtime.tick_frame_wall_clock_begin(now);
            let tick_kind = self.schedule.tick_kind(now);
            let outcome = self.run_tick(tick_kind);
            let render_skipped = outcome.render_skipped;
            if let Some(exit) = self.handle_tick_outcome(outcome) {
                return Ok(exit);
            }
            self.after_tick(tick_kind, render_skipped);
        }

        Ok(RunExit::Clean)
    }

    fn external_shutdown_requested(&self) -> bool {
        self.external_shutdown
            .as_ref()
            .is_some_and(|coord| coord.requested.load(Ordering::Relaxed))
    }

    fn run_tick(&mut self, kind: HeadlessTickKind) -> TickOutcome {
        match kind {
            HeadlessTickKind::FullFrame => {
                profiling::scope!("headless::full_frame");
                let outcome = self
                    .runtime
                    .tick_one_frame(&mut self.gpu, InputState::default());
                self.schedule.complete_full_frame(Instant::now());
                outcome
            }
            HeadlessTickKind::LockstepOnly => {
                profiling::scope!("headless::lockstep_tick");
                self.runtime
                    .tick_one_frame_lockstep_only(Some(&mut self.gpu), InputState::default())
            }
        }
    }

    fn handle_tick_outcome(&mut self, outcome: TickOutcome) -> Option<RunExit> {
        if outcome.shutdown_requested {
            logger::info!("Headless: host shutdown requested, exiting");
            self.runtime
                .log_compact_renderer_summary("headless-host-shutdown");
            self.drain_graceful_shutdown();
            return Some(RunExit::Clean);
        }
        if outcome.fatal_error {
            logger::error!("Headless: fatal IPC error, exiting");
            self.runtime
                .log_compact_renderer_summary("headless-fatal-ipc");
            crate::profiling::flush_resource_churn_plots();
            crate::profiling::emit_frame_mark();
            return Some(RunExit::Code(4));
        }
        if let Some(err) = outcome.graph_error {
            let kind = crate::render_graph::graph_error_kind(&err);
            crash_context::set_last_graph_error(kind);
            logger::warn!("Headless: render graph error this tick: {err:?}");
        }
        None
    }

    fn drain_graceful_shutdown(&mut self) {
        profiling::scope!("headless::graceful_shutdown");
        self.gpu.wait_for_previous_present();
        self.runtime.begin_graceful_shutdown();

        let deadline = Instant::now() + HEADLESS_GRACEFUL_SHUTDOWN_TIMEOUT;
        while !self.runtime.graceful_shutdown_complete() {
            if Instant::now() >= deadline {
                logger::warn!(
                    "Headless graceful shutdown timed out after {}ms; exiting",
                    HEADLESS_GRACEFUL_SHUTDOWN_TIMEOUT.as_millis()
                );
                break;
            }
            std::thread::sleep(HEADLESS_TICK_SLEEP);
        }
    }

    fn after_tick(&mut self, kind: HeadlessTickKind, render_skipped: bool) {
        if kind == HeadlessTickKind::FullFrame && !render_skipped {
            self.maybe_write_frame_png();
        }
        self.runtime
            .tick_frame_render_time_end(self.gpu.last_completed_gpu_render_time_seconds());
        crate::profiling::flush_resource_churn_plots();
        crate::profiling::emit_frame_mark();
        std::thread::sleep(HEADLESS_TICK_SLEEP);
    }

    fn maybe_write_frame_png(&mut self) {
        if self.gpu.headless_color_texture().is_none() {
            return;
        }
        if let Err(e) =
            readback::readback_and_write_png_atomically(&self.gpu, &self.params.output_path)
        {
            logger::warn!("Headless PNG write failed: {e}");
        } else {
            self.frames_written = self.frames_written.saturating_add(1);
            logger::trace!("Headless wrote PNG #{}", self.frames_written);
        }
    }
}
