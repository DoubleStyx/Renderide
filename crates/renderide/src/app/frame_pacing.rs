//! Desktop redraw pacing from config FPS caps ([`crate::config::DisplaySettings`]).
//!
//! VR sessions are paced by OpenXR; the app skips this module when the host reports VR active.

use std::time::{Duration, Instant};

/// Wall-clock minimum spacing between redraws for a positive FPS cap.
///
/// Returns [`None`] when `cap == 0` (uncapped).
pub fn min_interval_for_fps_cap(cap: u32) -> Option<Duration> {
    if cap == 0 {
        None
    } else {
        Some(Duration::from_secs_f64(1.0 / f64::from(cap)))
    }
}

/// If [`Some`] `(deadline)` is returned, the event loop should set
/// `ControlFlow::WaitUntil(deadline)` and **not** call [`winit::window::Window::request_redraw`]
/// until the deadline.
///
/// The deadline is anchored to the **start** of the previous tick so the cap expresses a true
/// period between frame starts: a `cap` of `N` fps yields consecutive frames spaced at least
/// `1/N` seconds apart regardless of how long each [`crate::app::renderide_app::RenderideApp::tick_frame`]
/// runs. Anchoring to the frame end instead would stack tick duration on top of `1/cap` and
/// collapse the effective rate to `1 / (1/cap + tick_duration)`.
///
/// Returns [`None`] when a redraw may be requested immediately: uncapped (`cap == 0`), no prior
/// frame start time (cold start), or the minimum interval has already elapsed.
pub fn next_redraw_wait_until(
    last_frame_start: Option<Instant>,
    cap: u32,
    now: Instant,
) -> Option<Instant> {
    let min_interval = min_interval_for_fps_cap(cap)?;
    let last = last_frame_start?;
    let next = last.checked_add(min_interval)?;
    if now < next {
        Some(next)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncapped_never_waits() {
        let t0 = Instant::now();
        assert_eq!(next_redraw_wait_until(Some(t0), 0, t0), None);
        assert_eq!(
            next_redraw_wait_until(Some(t0), 0, t0 + Duration::from_secs(1)),
            None
        );
    }

    #[test]
    fn cold_start_never_waits() {
        let now = Instant::now();
        assert_eq!(next_redraw_wait_until(None, 60, now), None);
    }

    #[test]
    fn cap_60_waits_until_next_tick() {
        let t0 = Instant::now();
        let min_i = min_interval_for_fps_cap(60).expect("60 fps");
        let just_after = t0 + min_i / 4;
        assert_eq!(
            next_redraw_wait_until(Some(t0), 60, just_after),
            Some(t0 + min_i)
        );
    }

    #[test]
    fn cap_60_elapsed_allows_immediate_redraw() {
        let t0 = Instant::now();
        let min_i = min_interval_for_fps_cap(60).expect("60 fps");
        let after = t0 + min_i;
        assert_eq!(next_redraw_wait_until(Some(t0), 60, after), None);
        assert_eq!(
            next_redraw_wait_until(Some(t0), 60, after + Duration::from_micros(1)),
            None
        );
    }

    #[test]
    fn boundary_now_equals_deadline_allows_redraw() {
        let t0 = Instant::now();
        let min_i = min_interval_for_fps_cap(60).expect("60 fps");
        let deadline = t0 + min_i;
        assert_eq!(next_redraw_wait_until(Some(t0), 60, deadline), None);
    }
}
