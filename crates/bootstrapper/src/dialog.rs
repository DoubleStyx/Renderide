//! Interactive desktop/VR selection dialog used by `main.rs` before spawning the Host.
//!
//! This module deliberately lives in the bin target (not in the library) so the bootstrapper
//! lib's unit-test executable never references `rfd`. On Windows, `rfd`'s `common-controls-v6`
//! feature emits a static import of `TaskDialogIndirect` from `comctl32.dll`, which Windows only
//! resolves when the executable carries a Common Controls v6 side-by-side manifest. `build.rs`
//! embeds that manifest into the bootstrapper binary via `embed-manifest`, but `embed-manifest`
//! cannot reach the lib unit-test exe — so when this code lived in the library, the lib unit-test
//! exe failed to load with `STATUS_ENTRYPOINT_NOT_FOUND` (0xc0000139) on Windows CI. Keeping the
//! `rfd` reference in the bin keeps the lib (and its test exe) free of that import.
//!
//! ## Hang protection on Linux
//!
//! `rfd::MessageDialog::show()` uses a GTK3 or XDG portal backend on Linux and can block
//! indefinitely when the backend cannot surface a window (headless shell, missing GTK runtime,
//! broken portal). [`prompt_desktop_or_vr`] emits one log line before and after the blocking
//! `show()` call, and a watchdog thread aborts the process with an actionable error (pointing at
//! [`bootstrapper::vr_prompt::ENV_SKIP_VR_DIALOG`]) if the dialog does not return within
//! [`DIALOG_WATCHDOG_TIMEOUT`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use bootstrapper::vr_prompt::ENV_SKIP_VR_DIALOG;

/// Maximum time [`prompt_desktop_or_vr`] waits for `rfd::MessageDialog::show()` to return
/// before the watchdog thread aborts the process with an actionable log line.
const DIALOG_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(60);

/// Custom-button label for the VR choice; also returned verbatim by `rfd` as the
/// `MessageDialogResult::Custom(label)` payload, so the same string doubles as the match key.
const VR_BUTTON_LABEL: &str = "VR";
/// Custom-button label for the Desktop choice; also returned verbatim by `rfd` as the
/// `MessageDialogResult::Custom(label)` payload.
const DESKTOP_BUTTON_LABEL: &str = "Desktop";
/// Custom-button label for the Cancel choice; also returned verbatim by `rfd` as the
/// `MessageDialogResult::Custom(label)` payload.
const CANCEL_BUTTON_LABEL: &str = "Cancel";

/// Shows the desktop vs VR selection dialog and returns the choice: `Some(true)` for VR,
/// `Some(false)` for Desktop, [`None`] for Cancel/dismiss (callers treat the latter as a
/// request to abort the launch).
///
/// Requires the global logger to be initialized before invocation so that the before/after
/// log lines and the watchdog abort message reach disk. Installs a short-lived watchdog
/// thread that aborts the process via [`std::process::exit`] with a pointer to
/// [`ENV_SKIP_VR_DIALOG`] if `rfd::MessageDialog::show()` has not returned after
/// [`DIALOG_WATCHDOG_TIMEOUT`].
pub(crate) fn prompt_desktop_or_vr() -> Option<bool> {
    let completed = Arc::new(AtomicBool::new(false));
    spawn_dialog_watchdog(Arc::clone(&completed));

    logger::info!("Showing desktop/VR selection dialog via rfd backend.");
    let res = rfd::MessageDialog::new()
        .set_title("Renderide")
        .set_description("Launch Resonite in VR or desktop mode?")
        .set_buttons(rfd::MessageButtons::YesNoCancelCustom(
            VR_BUTTON_LABEL.into(),
            DESKTOP_BUTTON_LABEL.into(),
            CANCEL_BUTTON_LABEL.into(),
        ))
        .show();
    completed.store(true, Ordering::SeqCst);

    match res {
        // Native backends that honor custom labels return them verbatim.
        rfd::MessageDialogResult::Custom(label) if label == VR_BUTTON_LABEL => {
            logger::info!("Desktop/VR dialog returned: VR.");
            Some(true)
        }
        rfd::MessageDialogResult::Custom(label) if label == DESKTOP_BUTTON_LABEL => {
            logger::info!("Desktop/VR dialog returned: Desktop.");
            Some(false)
        }
        other => {
            logger::info!("Desktop/VR dialog cancelled or dismissed: {other:?}.");
            None
        }
    }
}

/// Spawns a detached watchdog thread that logs an error and exits the process if `completed`
/// is still `false` after [`DIALOG_WATCHDOG_TIMEOUT`].
///
/// The dialog thread flips `completed` to `true` once `rfd`'s `show()` returns; the watchdog
/// checks the flag after sleeping and quietly exits its own `thread::spawn` closure if the
/// dialog finished in time.
fn spawn_dialog_watchdog(completed: Arc<AtomicBool>) {
    let spawn_result = thread::Builder::new()
        .name("rfd-dialog-watchdog".into())
        .spawn(move || {
            thread::sleep(DIALOG_WATCHDOG_TIMEOUT);
            if completed.load(Ordering::SeqCst) {
                return;
            }
            logger::error!(
                "Desktop/VR dialog did not return within {secs}s. \
                 The rfd GTK/XDG portal backend appears to be hung. \
                 Set {ENV_SKIP_VR_DIALOG}=1 (or pass -Screen / -Device SteamVR) to bypass the dialog.",
                secs = DIALOG_WATCHDOG_TIMEOUT.as_secs(),
            );
            logger::flush();
            std::process::exit(1);
        });
    if let Err(e) = spawn_result {
        // If the OS cannot spawn a watchdog thread, the dialog can still hang silently — log and
        // continue rather than aborting; the dialog's own behavior is unchanged.
        logger::warn!(
            "Could not spawn rfd dialog watchdog thread: {e}. Dialog timeout is disabled."
        );
    }
}
