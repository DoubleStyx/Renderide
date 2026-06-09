//! Host [`crate::shared::OutputState`] cursor lock policy: grab/visibility transitions and
//! software cursor-position anchoring while relative mouse mode is active.

use glam::IVec2;
use glam::Vec2;
use winit::dpi::LogicalSize;
use winit::dpi::{LogicalPosition, Position};
use winit::error::RequestError;
use winit::window::{CursorGrabMode, ImeRequest, Window};

use super::super::accumulator::WindowInputAccumulator;
use super::ime::enable_ime_on_window;
use crate::shared::OutputState;

#[cfg(any(test, not(target_os = "macos")))]
const CONFINED_RECENTER_MARGIN_PX: f32 = 8.0;
#[cfg(any(test, not(target_os = "macos")))]
const CONFINED_RECENTER_DRIFT_PX: f32 = 96.0;

/// Tracks host [`OutputState`] cursor fields between frames so unchanged lock state avoids
/// redundant window-system calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct CursorOutputTracking {
    last_lock_cursor: bool,
    last_lock_position: Option<IVec2>,
    active_grab_mode: Option<CursorGrabMode>,
    active_grab_can_warp: bool,
}

trait CursorWindowOps {
    fn cursor_scale_factor(&self) -> f64;

    fn cursor_logical_size(&self) -> LogicalSize<f64>;

    fn set_cursor_grab_mode(&self, mode: CursorGrabMode) -> Result<(), RequestError>;

    fn set_cursor_visible_mode(&self, visible: bool);

    fn warp_cursor_logical(&self, p: Vec2) -> Result<(), RequestError>;
}

impl<T> CursorWindowOps for T
where
    T: Window + ?Sized,
{
    fn cursor_scale_factor(&self) -> f64 {
        self.scale_factor()
    }

    fn cursor_logical_size(&self) -> LogicalSize<f64> {
        self.surface_size().to_logical(self.scale_factor())
    }

    fn set_cursor_grab_mode(&self, mode: CursorGrabMode) -> Result<(), RequestError> {
        self.set_cursor_grab(mode)
    }

    fn set_cursor_visible_mode(&self, visible: bool) {
        self.set_cursor_visible(visible);
    }

    fn warp_cursor_logical(&self, p: Vec2) -> Result<(), RequestError> {
        let logical = LogicalPosition::new(f64::from(p.x), f64::from(p.y));
        self.set_cursor_position(Position::Logical(logical))
    }
}

/// Returns the cursor anchor requested by the host, or the logical window center for relative lock.
fn cursor_anchor_from_lock_or_size(
    lock_cursor_position: Option<IVec2>,
    logical_size: LogicalSize<f64>,
) -> Vec2 {
    lock_cursor_position.map_or_else(
        || {
            Vec2::new(
                (logical_size.width / 2.0) as f32,
                (logical_size.height / 2.0) as f32,
            )
        },
        |p| Vec2::new(p.x as f32, p.y as f32),
    )
}

fn lock_cursor_position_or_center(
    lock_cursor_position: Option<IVec2>,
    window: &(impl CursorWindowOps + ?Sized),
) -> Vec2 {
    cursor_anchor_from_lock_or_size(lock_cursor_position, window.cursor_logical_size())
}

/// Mirrors a cursor-transition anchor into host-facing input state.
fn sync_accumulator_to_cursor_anchor(
    window: &(impl CursorWindowOps + ?Sized),
    acc: &mut WindowInputAccumulator,
    anchor: Vec2,
) {
    let logical_size = window.cursor_logical_size();
    acc.window_resolution = (
        logical_size.width.round() as u32,
        logical_size.height.round() as u32,
    );
    acc.set_window_position_from_logical(anchor, window.cursor_scale_factor());
    if acc.window_focused {
        acc.mouse_active = true;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CursorGrabPreference {
    primary: CursorGrabMode,
    fallback: CursorGrabMode,
}

fn grab_preference(lock_cursor_position: Option<IVec2>) -> CursorGrabPreference {
    if lock_cursor_position.is_some() {
        CursorGrabPreference {
            primary: CursorGrabMode::Confined,
            fallback: CursorGrabMode::Locked,
        }
    } else {
        CursorGrabPreference {
            primary: CursorGrabMode::Locked,
            fallback: CursorGrabMode::Confined,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CursorGrabActivation {
    mode: CursorGrabMode,
    can_warp: bool,
}

fn apply_cursor_grab(
    window: &(impl CursorWindowOps + ?Sized),
    preference: CursorGrabPreference,
) -> Result<CursorGrabMode, RequestError> {
    if window.set_cursor_grab_mode(preference.primary).is_ok() {
        return Ok(preference.primary);
    }
    window.set_cursor_grab_mode(preference.fallback)?;
    Ok(preference.fallback)
}

fn apply_cursor_grab_with_anchor(
    window: &(impl CursorWindowOps + ?Sized),
    preference: CursorGrabPreference,
    target: Vec2,
) -> Result<CursorGrabActivation, RequestError> {
    let primary = apply_cursor_grab(window, preference)?;
    if window.warp_cursor_logical(target).is_ok() {
        return Ok(CursorGrabActivation {
            mode: primary,
            can_warp: true,
        });
    }

    if primary == CursorGrabMode::Confined
        && preference.primary == CursorGrabMode::Confined
        && preference.fallback == CursorGrabMode::Locked
        && window.set_cursor_grab_mode(CursorGrabMode::Locked).is_ok()
    {
        return Ok(CursorGrabActivation {
            mode: CursorGrabMode::Locked,
            can_warp: window.warp_cursor_logical(target).is_ok(),
        });
    }

    Ok(CursorGrabActivation {
        mode: primary,
        can_warp: false,
    })
}

fn lock_state_changed(state: &OutputState, track: &CursorOutputTracking) -> bool {
    state.lock_cursor != track.last_lock_cursor
        || state.lock_cursor_position != track.last_lock_position
}

#[cfg(any(test, not(target_os = "macos")))]
fn should_recenter_confined_cursor(observed: Vec2, target: Vec2, resolution: (u32, u32)) -> bool {
    if resolution.0 == 0 || resolution.1 == 0 {
        return false;
    }
    let max_x = resolution.0 as f32;
    let max_y = resolution.1 as f32;
    let near_edge = observed.x <= CONFINED_RECENTER_MARGIN_PX
        || observed.y <= CONFINED_RECENTER_MARGIN_PX
        || observed.x >= max_x - CONFINED_RECENTER_MARGIN_PX
        || observed.y >= max_y - CONFINED_RECENTER_MARGIN_PX;
    let large_drift = observed.distance_squared(target)
        >= CONFINED_RECENTER_DRIFT_PX * CONFINED_RECENTER_DRIFT_PX;
    near_edge || large_drift
}

/// Maintains the reported cursor position while the host requests cursor lock.
///
/// Call after [`apply_output_state_to_window`] when [`OutputState::lock_cursor`] is true so relative
/// look and IPC [`crate::shared::MouseState::window_position`] stay anchored without issuing OS
/// grab or warp calls every frame. When the platform fell back to `Confined`, the hidden OS cursor
/// is re-centered only after edge contact or large drift.
#[cfg(not(target_os = "macos"))]
pub fn apply_per_frame_cursor_lock_when_locked(
    window: &dyn Window,
    acc: &mut WindowInputAccumulator,
    lock_cursor_position: Option<IVec2>,
    track: &CursorOutputTracking,
) -> Result<(), RequestError> {
    let sf = window.scale_factor();
    acc.sync_window_resolution_logical(window);
    let target = lock_cursor_position_or_center(lock_cursor_position, window);
    let observed = acc.window_position;
    let recenter_confined = track.active_grab_mode == Some(CursorGrabMode::Confined)
        && track.active_grab_can_warp
        && should_recenter_confined_cursor(observed, target, acc.window_resolution);
    acc.set_window_position_from_logical(target, sf);
    if recenter_confined {
        window.warp_cursor_logical(target)?;
        acc.set_window_position_from_logical(target, sf);
    }
    Ok(())
}

/// Maintains the reported cursor position while the host requests cursor lock.
///
/// On macOS this deliberately avoids OS cursor warps because reapplying center warps every frame
/// breaks relative mouse input with winit. Grab and visibility for [`OutputState::lock_cursor`] are
/// still applied from [`apply_output_state_to_window`].
#[cfg(target_os = "macos")]
pub fn apply_per_frame_cursor_lock_when_locked(
    window: &dyn Window,
    acc: &mut WindowInputAccumulator,
    lock_cursor_position: Option<IVec2>,
    _track: &CursorOutputTracking,
) -> Result<(), winit::error::RequestError> {
    let target = lock_cursor_position_or_center(lock_cursor_position, window);
    acc.sync_window_resolution_logical(window);
    acc.set_window_position_from_logical(target, window.scale_factor());
    Ok(())
}

/// Applies host [`OutputState`] to the winit window (IME, grab transitions, and transition warps).
/// Use [`apply_per_frame_cursor_lock_when_locked`] each frame while locked for software anchoring.
pub fn apply_output_state_to_window(
    window: &dyn Window,
    acc: &mut WindowInputAccumulator,
    state: &OutputState,
    track: &mut CursorOutputTracking,
) -> Result<(), RequestError> {
    if state.keyboard_input_active {
        if window.ime_capabilities().is_none() {
            enable_ime_on_window(window);
        }
    } else if window.ime_capabilities().is_some() {
        let _ = window.request_ime_update(ImeRequest::Disable);
    }

    apply_cursor_output_state(window, acc, state, track)
}

fn apply_cursor_output_state(
    window: &(impl CursorWindowOps + ?Sized),
    acc: &mut WindowInputAccumulator,
    state: &OutputState,
    track: &mut CursorOutputTracking,
) -> Result<(), RequestError> {
    if !lock_state_changed(state, track) {
        return Ok(());
    }

    if state.lock_cursor {
        let target = lock_cursor_position_or_center(state.lock_cursor_position, window);
        let activation = apply_cursor_grab_with_anchor(
            window,
            grab_preference(state.lock_cursor_position),
            target,
        )?;
        window.set_cursor_visible_mode(false);
        sync_accumulator_to_cursor_anchor(window, acc, target);
        track.last_lock_cursor = state.lock_cursor;
        track.last_lock_position = state.lock_cursor_position;
        track.active_grab_mode = Some(activation.mode);
        track.active_grab_can_warp = activation.can_warp;
        return Ok(());
    }

    let release_anchor = lock_cursor_position_or_center(track.last_lock_position, window);
    if track.active_grab_can_warp || track.active_grab_mode == Some(CursorGrabMode::Locked) {
        let _ = window.warp_cursor_logical(release_anchor);
    }
    window.set_cursor_grab_mode(CursorGrabMode::None)?;
    sync_accumulator_to_cursor_anchor(window, acc, release_anchor);
    window.set_cursor_visible_mode(false);
    track.last_lock_cursor = state.lock_cursor;
    track.last_lock_position = state.lock_cursor_position;
    track.active_grab_mode = None;
    track.active_grab_can_warp = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use glam::{IVec2, Vec2};
    use winit::dpi::LogicalSize;
    use winit::error::{NotSupportedError, RequestError};
    use winit::window::CursorGrabMode;

    use super::{
        CONFINED_RECENTER_DRIFT_PX, CONFINED_RECENTER_MARGIN_PX, CursorGrabActivation,
        CursorGrabPreference, CursorOutputTracking, CursorWindowOps, apply_cursor_grab_with_anchor,
        apply_cursor_output_state, cursor_anchor_from_lock_or_size, grab_preference,
        lock_state_changed, should_recenter_confined_cursor,
    };
    use crate::frontend::input::WindowInputAccumulator;
    use crate::shared::OutputState;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum FakeCursorEventKind {
        Grab,
        Visible,
        Warp,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct FakeCursorEvent {
        kind: FakeCursorEventKind,
        mode: CursorGrabMode,
        position: Vec2,
        visible: bool,
    }

    impl FakeCursorEvent {
        fn grab(mode: CursorGrabMode) -> Self {
            Self {
                kind: FakeCursorEventKind::Grab,
                mode,
                position: Vec2::ZERO,
                visible: false,
            }
        }

        fn visible(visible: bool) -> Self {
            Self {
                kind: FakeCursorEventKind::Visible,
                mode: CursorGrabMode::None,
                position: Vec2::ZERO,
                visible,
            }
        }

        fn warp(mode: CursorGrabMode, position: Vec2) -> Self {
            Self {
                kind: FakeCursorEventKind::Warp,
                mode,
                position,
                visible: false,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum FakeWarpRule {
        Always,
        OnlyLocked,
        Never,
    }

    struct FakeCursorWindow {
        state: RefCell<FakeCursorWindowState>,
    }

    struct FakeCursorWindowState {
        scale_factor: f64,
        logical_size: LogicalSize<f64>,
        grab_mode: CursorGrabMode,
        rejected_grabs: Vec<CursorGrabMode>,
        warp_rule: FakeWarpRule,
        events: Vec<FakeCursorEvent>,
    }

    impl FakeCursorWindow {
        fn new(warp_rule: FakeWarpRule) -> Self {
            Self {
                state: RefCell::new(FakeCursorWindowState {
                    scale_factor: 1.0,
                    logical_size: LogicalSize::new(800.0, 600.0),
                    grab_mode: CursorGrabMode::None,
                    rejected_grabs: Vec::new(),
                    warp_rule,
                    events: Vec::new(),
                }),
            }
        }

        fn with_grab_mode(self, grab_mode: CursorGrabMode) -> Self {
            self.state.borrow_mut().grab_mode = grab_mode;
            self
        }

        fn with_rejected_grab(self, grab_mode: CursorGrabMode) -> Self {
            self.state.borrow_mut().rejected_grabs.push(grab_mode);
            self
        }

        fn events(&self) -> Vec<FakeCursorEvent> {
            self.state.borrow().events.clone()
        }
    }

    impl CursorWindowOps for FakeCursorWindow {
        fn cursor_scale_factor(&self) -> f64 {
            self.state.borrow().scale_factor
        }

        fn cursor_logical_size(&self) -> LogicalSize<f64> {
            self.state.borrow().logical_size
        }

        fn set_cursor_grab_mode(&self, mode: CursorGrabMode) -> Result<(), RequestError> {
            let mut state = self.state.borrow_mut();
            state.events.push(FakeCursorEvent::grab(mode));
            if state.rejected_grabs.contains(&mode) {
                return Err(NotSupportedError::new("test grab rejection").into());
            }
            state.grab_mode = mode;
            Ok(())
        }

        fn set_cursor_visible_mode(&self, visible: bool) {
            self.state
                .borrow_mut()
                .events
                .push(FakeCursorEvent::visible(visible));
        }

        fn warp_cursor_logical(&self, p: Vec2) -> Result<(), RequestError> {
            let mut state = self.state.borrow_mut();
            let mode = state.grab_mode;
            state.events.push(FakeCursorEvent::warp(mode, p));
            let supported = match state.warp_rule {
                FakeWarpRule::Always => true,
                FakeWarpRule::OnlyLocked => mode == CursorGrabMode::Locked,
                FakeWarpRule::Never => false,
            };
            if supported {
                Ok(())
            } else {
                Err(NotSupportedError::new("test warp rejection").into())
            }
        }
    }

    #[test]
    fn grab_preference_uses_locked_for_relative_mode() {
        assert_eq!(
            grab_preference(None),
            CursorGrabPreference {
                primary: CursorGrabMode::Locked,
                fallback: CursorGrabMode::Confined,
            }
        );
    }

    #[test]
    fn grab_preference_uses_confined_for_explicit_position() {
        assert_eq!(
            grab_preference(Some(IVec2::new(12, 34))),
            CursorGrabPreference {
                primary: CursorGrabMode::Confined,
                fallback: CursorGrabMode::Locked,
            }
        );
    }

    #[test]
    fn cursor_anchor_uses_center_for_relative_lock() {
        assert_eq!(
            cursor_anchor_from_lock_or_size(None, LogicalSize::new(800.0, 600.0)),
            Vec2::new(400.0, 300.0)
        );
    }

    #[test]
    fn cursor_anchor_uses_explicit_lock_position() {
        assert_eq!(
            cursor_anchor_from_lock_or_size(
                Some(IVec2::new(120, 240)),
                LogicalSize::new(800.0, 600.0),
            ),
            Vec2::new(120.0, 240.0)
        );
    }

    #[test]
    fn cursor_anchor_keeps_zero_size_center() {
        assert_eq!(
            cursor_anchor_from_lock_or_size(None, LogicalSize::new(0.0, 0.0)),
            Vec2::ZERO
        );
    }

    #[test]
    fn unchanged_lock_state_skips_window_policy() {
        let track = CursorOutputTracking {
            last_lock_cursor: true,
            last_lock_position: Some(IVec2::new(100, 200)),
            active_grab_mode: Some(CursorGrabMode::Locked),
            active_grab_can_warp: true,
        };
        let state = OutputState {
            lock_cursor: true,
            lock_cursor_position: Some(IVec2::new(100, 200)),
            ..Default::default()
        };

        assert!(!lock_state_changed(&state, &track));
    }

    #[test]
    fn lock_position_change_reapplies_window_policy() {
        let track = CursorOutputTracking {
            last_lock_cursor: true,
            last_lock_position: Some(IVec2::new(100, 200)),
            active_grab_mode: Some(CursorGrabMode::Locked),
            active_grab_can_warp: true,
        };
        let state = OutputState {
            lock_cursor: true,
            lock_cursor_position: Some(IVec2::new(120, 200)),
            ..Default::default()
        };

        assert!(lock_state_changed(&state, &track));
    }

    #[test]
    fn confined_activation_retries_locked_when_confined_cannot_warp() {
        let window = FakeCursorWindow::new(FakeWarpRule::OnlyLocked);
        let target = Vec2::new(120.0, 240.0);

        let activation = apply_cursor_grab_with_anchor(
            &window,
            grab_preference(Some(IVec2::new(120, 240))),
            target,
        )
        .expect("confined grab falls back to locked");

        assert_eq!(
            activation,
            CursorGrabActivation {
                mode: CursorGrabMode::Locked,
                can_warp: true,
            }
        );
        assert_eq!(
            window.events(),
            vec![
                FakeCursorEvent::grab(CursorGrabMode::Confined),
                FakeCursorEvent::warp(CursorGrabMode::Confined, target),
                FakeCursorEvent::grab(CursorGrabMode::Locked),
                FakeCursorEvent::warp(CursorGrabMode::Locked, target),
            ]
        );
    }

    #[test]
    fn relative_activation_falls_back_to_confined_when_locked_grab_is_unsupported() {
        let window =
            FakeCursorWindow::new(FakeWarpRule::Always).with_rejected_grab(CursorGrabMode::Locked);
        let target = Vec2::new(400.0, 300.0);

        let activation = apply_cursor_grab_with_anchor(&window, grab_preference(None), target)
            .expect("confined fallback remains supported");

        assert_eq!(
            activation,
            CursorGrabActivation {
                mode: CursorGrabMode::Confined,
                can_warp: true,
            }
        );
        assert_eq!(
            window.events(),
            vec![
                FakeCursorEvent::grab(CursorGrabMode::Locked),
                FakeCursorEvent::grab(CursorGrabMode::Confined),
                FakeCursorEvent::warp(CursorGrabMode::Confined, target),
            ]
        );
    }

    #[test]
    fn unlock_warps_previous_explicit_anchor_before_releasing_grab() {
        let window =
            FakeCursorWindow::new(FakeWarpRule::Always).with_grab_mode(CursorGrabMode::Locked);
        let mut acc = WindowInputAccumulator::default();
        let mut track = CursorOutputTracking {
            last_lock_cursor: true,
            last_lock_position: Some(IVec2::new(120, 240)),
            active_grab_mode: Some(CursorGrabMode::Locked),
            active_grab_can_warp: true,
        };
        let state = OutputState {
            lock_cursor: false,
            ..Default::default()
        };

        apply_cursor_output_state(&window, &mut acc, &state, &mut track)
            .expect("unlock applies cleanly");

        assert_eq!(
            window.events(),
            vec![
                FakeCursorEvent::warp(CursorGrabMode::Locked, Vec2::new(120.0, 240.0)),
                FakeCursorEvent::grab(CursorGrabMode::None),
                FakeCursorEvent::visible(false),
            ]
        );
        assert_eq!(acc.window_position, Vec2::new(120.0, 240.0));
        assert_eq!(track.active_grab_mode, None);
        assert!(!track.active_grab_can_warp);
    }

    #[test]
    fn unlock_skips_os_warp_when_active_grab_cannot_warp() {
        let window =
            FakeCursorWindow::new(FakeWarpRule::Never).with_grab_mode(CursorGrabMode::Confined);
        let mut acc = WindowInputAccumulator::default();
        let mut track = CursorOutputTracking {
            last_lock_cursor: true,
            last_lock_position: None,
            active_grab_mode: Some(CursorGrabMode::Confined),
            active_grab_can_warp: false,
        };
        let state = OutputState {
            lock_cursor: false,
            ..Default::default()
        };

        apply_cursor_output_state(&window, &mut acc, &state, &mut track)
            .expect("unlock applies cleanly");

        assert_eq!(
            window.events(),
            vec![
                FakeCursorEvent::grab(CursorGrabMode::None),
                FakeCursorEvent::visible(false),
            ]
        );
        assert_eq!(acc.window_position, Vec2::new(400.0, 300.0));
        assert_eq!(track.active_grab_mode, None);
        assert!(!track.active_grab_can_warp);
    }

    #[test]
    fn confined_recenter_ignores_stable_cursor_near_target() {
        assert!(!should_recenter_confined_cursor(
            Vec2::new(400.0, 300.0),
            Vec2::new(402.0, 301.0),
            (800, 600),
        ));
    }

    #[test]
    fn confined_recenter_triggers_near_edge() {
        assert!(should_recenter_confined_cursor(
            Vec2::new(CONFINED_RECENTER_MARGIN_PX - 1.0, 300.0),
            Vec2::new(400.0, 300.0),
            (800, 600),
        ));
    }

    #[test]
    fn confined_recenter_triggers_on_large_drift() {
        assert!(should_recenter_confined_cursor(
            Vec2::new(400.0 + CONFINED_RECENTER_DRIFT_PX, 300.0),
            Vec2::new(400.0, 300.0),
            (800, 600),
        ));
    }

    #[test]
    fn confined_recenter_ignores_unknown_resolution() {
        assert!(!should_recenter_confined_cursor(
            Vec2::ZERO,
            Vec2::new(400.0, 300.0),
            (0, 600),
        ));
    }
}
