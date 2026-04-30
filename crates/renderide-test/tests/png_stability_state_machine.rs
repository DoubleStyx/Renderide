//! Sanity scenarios for the PNG stability state machine, expressed against the public lib API.
//!
//! These mirror the inline unit tests in `host::scene_session::png_readback` but reach the state
//! machine through the crate-public path used by integration tests, confirming that the
//! visibility re-exports are wired correctly.

use std::time::{Duration, Instant, SystemTime};

use renderide_test::host::scene_session::png_readback::{
    PngObservation, PngStabilityState, PngStabilityVerdict, PngStabilityWaitTiming,
};

fn timing(scene_submitted_at: SystemTime, scene_submit_instant: Instant) -> PngStabilityWaitTiming {
    PngStabilityWaitTiming {
        scene_submitted_at,
        scene_submit_instant,
        overall_timeout: Duration::from_secs(10),
        interval: Duration::from_secs(1),
    }
}

fn fresh_state() -> (PngStabilityState, Instant, SystemTime) {
    let i0 = Instant::now();
    let t0 = SystemTime::now();
    let state = PngStabilityState::new(&timing(t0, i0));
    (state, i0, t0)
}

#[test]
fn pending_before_render_window_opens() {
    let (mut s, i0, t0) = fresh_state();
    let now = i0 + Duration::from_millis(500);
    let mtime = t0 + Duration::from_millis(100);
    assert_eq!(
        s.observe(now, PngObservation::Present { mtime, size: 1024 }),
        PngStabilityVerdict::Pending
    );
}

#[test]
fn stable_after_window_elapses() {
    let (mut s, i0, t0) = fresh_state();
    let mtime = t0 + Duration::from_millis(2200);
    assert_eq!(
        s.observe(
            i0 + Duration::from_millis(2300),
            PngObservation::Present { mtime, size: 1024 }
        ),
        PngStabilityVerdict::Pending
    );
    assert_eq!(
        s.observe(
            i0 + Duration::from_millis(2500),
            PngObservation::Present { mtime, size: 1024 }
        ),
        PngStabilityVerdict::Stable
    );
}

#[test]
fn missing_keeps_pending_and_does_not_advance_state() {
    let (mut s, i0, _t0) = fresh_state();
    let now = i0 + Duration::from_millis(2500);
    assert_eq!(
        s.observe(now, PngObservation::Missing),
        PngStabilityVerdict::Pending
    );
    assert!(s.last_seen_mtime().is_none());
}
