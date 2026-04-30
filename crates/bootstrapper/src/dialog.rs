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
/// log lines reach disk.
pub(crate) fn prompt_desktop_or_vr() -> Option<bool> {
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
