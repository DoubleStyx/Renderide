//! Integration test: `RendererFrontend::new(None)` standalone construction path is reachable by
//! an embedder and reports the invariants documented on the type.

use renderide::frontend::{InitState, RendererFrontend};

/// A fresh standalone frontend reports `Finalized` init state, no pending frame, and
/// `last_frame_index == -1`, matching `RendererFrontend::new(None)` contract.
#[test]
fn new_none_yields_finalized_standalone_state() {
    let frontend = RendererFrontend::new(None);
    assert_eq!(frontend.init_state(), InitState::Finalized);
    assert!(
        frontend.last_frame_data_processed(),
        "standalone frontend starts with last_frame_data_processed == true"
    );
    assert_eq!(frontend.last_frame_index(), -1);
    assert!(!frontend.shutdown_requested());
    assert!(!frontend.fatal_error());
    assert!(frontend.pending_init().is_none());
}

/// Standalone frontends are not connected to IPC, so `begin_frame_allowed` must return `false`
/// regardless of the other lock-step state. This guards the contract that lock-step only runs when
/// IPC is wired, documented on `begin_frame_allowed`.
#[test]
fn standalone_frontend_never_sends_begin_frame() {
    let frontend = RendererFrontend::new(None);
    assert!(
        !frontend.should_send_begin_frame(),
        "standalone frontend has no IPC and must not emit begin-frame"
    );
}

/// `note_frame_submit_processed` echoes the supplied frame index and keeps the processed flag set
/// so the next lock-step check would see the applied frame.
#[test]
fn note_frame_submit_processed_updates_last_index() {
    let mut frontend = RendererFrontend::new(None);
    frontend.note_frame_submit_processed(7);
    assert_eq!(frontend.last_frame_index(), 7);
    assert!(frontend.last_frame_data_processed());

    frontend.note_frame_submit_processed(42);
    assert_eq!(frontend.last_frame_index(), 42);
}

/// `set_fatal_error` and `set_shutdown_requested` flip the corresponding accessors. These are the
/// only mutators an embedder needs to drive a cooperative-exit flow without an IPC connection.
#[test]
fn setters_update_fatal_and_shutdown_flags() {
    let mut frontend = RendererFrontend::new(None);
    assert!(!frontend.fatal_error());
    frontend.set_fatal_error(true);
    assert!(frontend.fatal_error());
    frontend.set_fatal_error(false);
    assert!(!frontend.fatal_error());

    assert!(!frontend.shutdown_requested());
    frontend.set_shutdown_requested(true);
    assert!(frontend.shutdown_requested());
}
