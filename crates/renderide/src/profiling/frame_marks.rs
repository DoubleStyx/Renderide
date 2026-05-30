/// Starts the Tracy client (if the `tracy` feature is on) and registers the calling thread as
/// `"renderer-main"` in the active profiler.
///
/// Must be called exactly once, before any other `profiling::scope!` macro or
/// [`GpuProfilerHandle::try_new`] runs -- the `profiling` crate's tracy backend expects a running
/// `tracy_client::Client` on every span, so the client has to be live first.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn register_main_thread() {
    #[cfg(feature = "tracy")]
    {
        let _ = tracy_client::Client::start();
    }
    profiling::register_thread!("renderer-main");
}

/// Emits a frame mark to the active profiler, closing the current frame boundary.
///
/// Call exactly once per winit tick, at the very end of the app driver's redraw tick.
/// Without frame marks Tracy still records spans but the frame timeline and histogram are empty.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn emit_frame_mark() {
    profiling::finish_frame!();
}

/// Emits a secondary Tracy frame mark for command-buffer batches handed to the GPU driver thread.
///
/// The default Tracy frame remains the winit redraw tick. This secondary track marks actual GPU
/// submits so empty redraw ticks, swapchain acquire skips, and delayed GPU timestamp readback do
/// not make the pass timeline look like graph work vanished.
#[inline]
pub fn emit_render_submit_frame_mark() {
    #[cfg(feature = "tracy")]
    {
        if let Some(client) = tracy_client::Client::running() {
            client.secondary_frame_mark(tracy_client::frame_name!("render-submit"));
        }
    }
}
/// Returns a closure suitable for [`rayon::ThreadPoolBuilder::start_handler`].
///
/// Each Rayon worker thread registers itself as `"rayon-worker-{index}"` with the active profiler,
/// so it appears by name on the Tracy thread timeline. When the `tracy` feature is off this
/// returns a no-op closure with zero overhead.
pub fn rayon_thread_start_handler() -> impl Fn(usize) + Send + Sync + 'static {
    move |_thread_index| {
        profiling::register_thread!(&format!("rayon-worker-{_thread_index}"));
    }
}
