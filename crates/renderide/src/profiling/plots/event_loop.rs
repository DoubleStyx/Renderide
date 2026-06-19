//! Tracy plots for the app driver's frame-pacing and swapchain-acquire signals.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them. New signals belong in this file when they describe the winit event loop, the
//! window's focus/cap state, or the surface-acquire and present-backpressure state.

use std::time::Duration;

use super::tracy_plot::tracy_plot;

/// Records the FPS cap currently applied by the app driver's `about_to_wait` handler after
/// resolving foreground/background renderer settings and host desktop overrides while presentation
/// mode uses renderer-side pacing. Zero means uncapped, presentation-paced, or HMD-compositor-paced.
///
/// Call once per winit iteration so the Tracy plot sits adjacent to the frame-mark timeline and
/// the value-per-frame is an exact reading rather than an interpolation. Expands to nothing when
/// the `tracy` feature is off.
pub fn plot_fps_cap_active(cap: u32) {
    tracy_plot!("fps_cap_active", f64::from(cap));
}

/// Records winit keyboard focus (`1.0` focused, `0.0` unfocused) as a Tracy plot so
/// foreground/background cap switches in the app driver's `about_to_wait` handler are visible at a glance.
///
/// Intended to be plotted next to [`plot_fps_cap_active`]: a drop from `1.0` to `0.0` should line
/// up with the cap changing between foreground and background values (or vice versa) when
/// renderer-side pacing is active, which is the usual cause of a sudden frame-time change while
/// profiling.
///
/// Expands to nothing when the `tracy` feature is off.
pub fn plot_window_focused(focused: bool) {
    tracy_plot!("window_focused", if focused { 1.0 } else { 0.0 });
}

/// Records, in milliseconds, how long
/// the app driver's `about_to_wait` handler asked winit to park before the next
/// `RedrawRequested`. Emit the [`std::time::Duration`] between `now` and the
/// [`winit::event_loop::ControlFlow::WaitUntil`] deadline when the capped branch is taken, and
/// `0.0` when the handler returns with [`winit::event_loop::ControlFlow::Poll`].
///
/// The gap between Tracy frames that no [`profiling::scope`] can cover (because the main thread
/// is parked inside winit) shows up on this plot as a non-zero value, attributing the idle time
/// to the CPU-side frame-pacing cap rather than missing instrumentation. Expands to nothing when
/// the `tracy` feature is off.
pub fn plot_event_loop_wait_ms(ms: f64) {
    tracy_plot!("event_loop_wait_ms", ms);
}

/// Records the driver-thread submit backlog (`submits_pushed - submits_done`) as a Tracy
/// plot.
///
/// Call once per tick from the frame epilogue. A steady-state value of `0` or `1` is
/// healthy (one frame in flight on the driver matches the ring's nominal pipelining
/// depth); a sustained value at the ring capacity means the producer is back-pressured
/// by the driver and CPU/GPU pacing is bound by submit throughput. Useful next to
/// [`plot_event_loop_idle_ms`] when diagnosing why the main thread is sleeping.
///
/// Expands to nothing when the `tracy` feature is off.
pub fn plot_driver_submit_backlog(count: u64) {
    tracy_plot!("driver_submit_backlog", count as f64);
}

/// Records the number of surface-carrying driver batches that have been submitted but not yet
/// presented.
///
/// A non-zero value means the next surface acquire must wait for the driver thread to call
/// [`wgpu::SurfaceTexture::present`] before wgpu will allow another
/// [`wgpu::SurfaceTexture`] to be acquired. This is the visible-frame backpressure signal for
/// the offscreen desktop path, where the final blit owns the only surface texture.
pub fn plot_surface_in_flight_count(count: u64) {
    tracy_plot!("surface_acquire::in_flight", count as f64);
}

/// Records how long the main thread waited for the previous surface texture to be presented
/// before attempting a new [`wgpu::Surface::get_current_texture`] call.
///
/// This separates Renderide's explicit single-surface-texture barrier from the raw wgpu acquire
/// call below, so a Tracy spike can be attributed to driver-thread present catch-up rather than
/// being lumped into `get_current_texture`.
pub fn plot_surface_previous_present_wait_ms(wait: Duration) {
    tracy_plot!(
        "surface_acquire::previous_present_wait_ms",
        wait.as_secs_f64() * 1000.0
    );
}

/// Records the raw wall-clock duration spent inside [`wgpu::Surface::get_current_texture`].
///
/// This excludes Renderide's explicit previous-present barrier, making compositor/swapchain
/// acquire stalls visible independently from driver-ring present catch-up.
pub fn plot_surface_get_current_texture_ms(wait: Duration) {
    tracy_plot!(
        "surface_acquire::get_current_texture_ms",
        wait.as_secs_f64() * 1000.0
    );
}

/// Records, in milliseconds, the wall-clock gap between the end of the previous
/// app-driver redraw tick and the start of the current one.
///
/// Complements [`plot_event_loop_wait_ms`] (the *requested* wait) by showing the *actual* slept
/// duration -- divergence between the two points at additional blocking outside the pacing cap,
/// for example surface acquire or previous-present waits.
///
/// Expands to nothing when the `tracy` feature is off.
pub fn plot_event_loop_idle_ms(ms: f64) {
    tracy_plot!("event_loop_idle_ms", ms);
}

/// Records the result of a swapchain acquire attempt as one-hot Tracy plots.
///
/// These samples explain CPU frames that have a frame mark but no render-graph GPU markers: a
/// timeout or occluded surface intentionally skips graph recording for that tick, while a
/// reconfigure means the graph will resume on a later acquire.
pub fn plot_surface_acquire_outcome(acquired: bool, skipped: bool, reconfigured: bool) {
    tracy_plot!(
        "surface_acquire::acquired",
        if acquired { 1.0 } else { 0.0 }
    );
    tracy_plot!("surface_acquire::skipped", if skipped { 1.0 } else { 0.0 });
    tracy_plot!(
        "surface_acquire::reconfigured",
        if reconfigured { 1.0 } else { 0.0 }
    );
}
