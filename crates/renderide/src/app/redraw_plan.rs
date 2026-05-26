//! Redraw scheduling decisions for the windowed driver event loop.

use std::time::{Duration, Instant};

/// Redraw action for the next `about_to_wait` event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RedrawDecision {
    /// Park the event loop until this deadline.
    WaitUntil(Instant),
    /// Request a redraw immediately.
    RedrawNow,
    /// Do not request another redraw.
    Idle,
}

/// Fully resolved redraw scheduling plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RedrawPlan {
    /// Decision to apply to the winit event loop.
    pub(crate) decision: RedrawDecision,
    /// FPS cap active for diagnostics; `0` means uncapped or inactive.
    pub(crate) fps_cap: u32,
    /// Wait time plotted for diagnostics.
    pub(crate) wait_ms: f64,
}

/// Inputs used to compute a redraw scheduling decision.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RedrawInputs {
    /// Whether the app has a window that can receive redraw requests.
    pub(crate) has_window: bool,
    /// Whether the app has already requested event-loop exit.
    pub(crate) exit_requested: bool,
    /// Whether VR pacing owns frame cadence.
    pub(crate) vr_active: bool,
    /// Whether winit reports that the renderer window currently has keyboard focus.
    pub(crate) window_has_keyboard_focus: bool,
    /// FPS cap used while the renderer window is the foreground input window; `0` means uncapped.
    pub(crate) foreground_fps_cap: u32,
    /// FPS cap used while the renderer window is not the foreground input window; `0` means uncapped.
    pub(crate) background_fps_cap: u32,
    /// Last frame-start anchor used to schedule capped redraws.
    pub(crate) last_frame_start: Option<Instant>,
    /// Current wall-clock instant.
    pub(crate) now: Instant,
}

/// Wall-clock minimum spacing between redraws for a positive FPS cap.
pub(crate) fn min_interval_for_fps_cap(cap: u32) -> Option<Duration> {
    if cap == 0 {
        None
    } else {
        Some(Duration::from_secs_f64(1.0 / f64::from(cap)))
    }
}

/// Returns the next redraw deadline for the configured desktop FPS cap.
pub(crate) fn next_redraw_wait_until(
    last_frame_start: Option<Instant>,
    cap: u32,
    now: Instant,
) -> Option<Instant> {
    let min_interval = min_interval_for_fps_cap(cap)?;
    let last = last_frame_start?;
    let next = last.checked_add(min_interval)?;
    (now < next).then_some(next)
}

/// Computes the event-loop redraw action from the app state without touching winit.
pub(crate) fn plan_redraw(inputs: RedrawInputs) -> RedrawPlan {
    if !inputs.has_window || inputs.exit_requested {
        return RedrawPlan {
            decision: RedrawDecision::Idle,
            fps_cap: 0,
            wait_ms: 0.0,
        };
    }

    if inputs.vr_active {
        return RedrawPlan {
            decision: RedrawDecision::RedrawNow,
            fps_cap: 0,
            wait_ms: 0.0,
        };
    }

    let cap = if inputs.window_has_keyboard_focus {
        inputs.foreground_fps_cap
    } else {
        inputs.background_fps_cap
    };
    if let Some(deadline) = next_redraw_wait_until(inputs.last_frame_start, cap, inputs.now) {
        return RedrawPlan {
            decision: RedrawDecision::WaitUntil(deadline),
            fps_cap: cap,
            wait_ms: deadline.saturating_duration_since(inputs.now).as_secs_f64() * 1000.0,
        };
    }

    RedrawPlan {
        decision: RedrawDecision::RedrawNow,
        fps_cap: cap,
        wait_ms: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{
        RedrawDecision, RedrawInputs, min_interval_for_fps_cap, next_redraw_wait_until, plan_redraw,
    };

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
    fn boundary_now_equals_deadline_allows_redraw() {
        let t0 = Instant::now();
        let min_i = min_interval_for_fps_cap(60).expect("60 fps");
        let deadline = t0 + min_i;
        assert_eq!(next_redraw_wait_until(Some(t0), 60, deadline), None);
    }

    #[test]
    fn redraw_plan_waits_for_foreground_cap() {
        let t0 = Instant::now();
        let now = t0 + Duration::from_millis(1);
        let plan = plan_redraw(RedrawInputs {
            has_window: true,
            exit_requested: false,
            vr_active: false,
            window_has_keyboard_focus: true,
            foreground_fps_cap: 60,
            background_fps_cap: 15,
            last_frame_start: Some(t0),
            now,
        });
        assert_eq!(plan.fps_cap, 60);
        assert!(matches!(plan.decision, RedrawDecision::WaitUntil(_)));
        assert!(plan.wait_ms > 0.0);
    }

    #[test]
    fn redraw_plan_uses_background_cap() {
        let t0 = Instant::now();
        let now = t0 + Duration::from_millis(1);
        let plan = plan_redraw(RedrawInputs {
            has_window: true,
            exit_requested: false,
            vr_active: false,
            window_has_keyboard_focus: false,
            foreground_fps_cap: 60,
            background_fps_cap: 15,
            last_frame_start: Some(t0),
            now,
        });
        assert_eq!(plan.fps_cap, 15);
        assert!(matches!(plan.decision, RedrawDecision::WaitUntil(_)));
    }

    #[test]
    fn redraw_plan_keeps_waiting_after_pre_deadline_wakeups() {
        let t0 = Instant::now();
        let min_i = min_interval_for_fps_cap(30).expect("30 fps");
        for now in [
            t0 + min_i / 8,
            t0 + min_i / 2,
            (t0 + min_i).checked_sub(Duration::from_millis(1)).unwrap(),
        ] {
            let plan = plan_redraw(RedrawInputs {
                has_window: true,
                exit_requested: false,
                vr_active: false,
                window_has_keyboard_focus: true,
                foreground_fps_cap: 30,
                background_fps_cap: 15,
                last_frame_start: Some(t0),
                now,
            });
            assert_eq!(plan.fps_cap, 30);
            assert!(matches!(plan.decision, RedrawDecision::WaitUntil(_)));
            assert!(plan.wait_ms > 0.0);
        }
    }

    #[test]
    fn redraw_plan_redraws_immediately_when_uncapped_or_vr() {
        let now = Instant::now();
        assert_eq!(
            plan_redraw(RedrawInputs {
                has_window: true,
                exit_requested: false,
                vr_active: false,
                window_has_keyboard_focus: true,
                foreground_fps_cap: 0,
                background_fps_cap: 15,
                last_frame_start: Some(now),
                now,
            })
            .decision,
            RedrawDecision::RedrawNow
        );
        assert_eq!(
            plan_redraw(RedrawInputs {
                has_window: true,
                exit_requested: false,
                vr_active: true,
                window_has_keyboard_focus: true,
                foreground_fps_cap: 60,
                background_fps_cap: 15,
                last_frame_start: Some(now),
                now,
            })
            .decision,
            RedrawDecision::RedrawNow
        );
    }

    #[test]
    fn redraw_plan_idles_without_window_or_after_exit() {
        let now = Instant::now();
        assert_eq!(
            plan_redraw(RedrawInputs {
                has_window: false,
                exit_requested: false,
                vr_active: false,
                window_has_keyboard_focus: true,
                foreground_fps_cap: 60,
                background_fps_cap: 15,
                last_frame_start: Some(now),
                now,
            })
            .decision,
            RedrawDecision::Idle
        );
        assert_eq!(
            plan_redraw(RedrawInputs {
                has_window: true,
                exit_requested: true,
                vr_active: true,
                window_has_keyboard_focus: true,
                foreground_fps_cap: 60,
                background_fps_cap: 15,
                last_frame_start: Some(now),
                now,
            })
            .decision,
            RedrawDecision::Idle
        );
    }
}
