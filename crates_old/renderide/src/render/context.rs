//! Render context: current rendering context and frame phase.
//!
//! Separates "user view" (main window) from "render to asset" (offscreen camera tasks).
//! In RenderingManager.cs, ProcessRenderTasks runs after WaitForEndOfFrame and switches
//! context to RenderToAsset before processing each CameraRenderTask.

use std::cell::Cell;

use crate::shared::RenderingContext;

/// Phase of the current frame. Used to structure the app's redraw flow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FramePhase {
    /// Session update and command processing. No rendering yet.
    #[default]
    Update,
    /// Rendering to the main window (user view). Uses `RenderingContext::user_view`.
    RenderToScreen,
    /// Rendering to offscreen targets (camera, mirror, portal, render-to-asset).
    /// [`process_render_tasks`](crate::session::Session::process_render_tasks) will eventually run in this phase.
    RenderToAsset,
}

thread_local! {
    /// Current rendering context for the active frame. Set during each phase.
    static CURRENT_CONTEXT: Cell<RenderingContext> = const { Cell::new(RenderingContext::user_view) };
}

/// Returns the current thread-local rendering context.
#[inline]
pub fn current_context() -> RenderingContext {
    CURRENT_CONTEXT.with(|c| c.get())
}

/// Sets the current thread-local rendering context.
#[inline]
pub fn set_context(ctx: RenderingContext) {
    CURRENT_CONTEXT.with(|c| c.set(ctx));
}

/// Runs a closure with the given context set. Restores the previous context afterward.
#[inline]
pub fn with_context<F, R>(ctx: RenderingContext, f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = current_context();
    set_context(ctx);
    let result = f();
    set_context(prev);
    result
}
