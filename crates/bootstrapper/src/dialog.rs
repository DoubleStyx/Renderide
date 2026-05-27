//! Interactive desktop/VR selection dialog used by `main.rs` before spawning the Host.
//!
//! This module deliberately lives in the bin target (not in the library) so the bootstrapper
//! lib's unit-test executable never references `rfd`. On Windows, `rfd`'s `common-controls-v6`
//! feature emits a static import of `TaskDialogIndirect` from `comctl32.dll`, which Windows only
//! resolves when the executable carries a Common Controls v6 side-by-side manifest. `build.rs`
//! embeds that manifest into the bootstrapper binary via `embed-manifest`, but `embed-manifest`
//! cannot reach the lib unit-test exe -- so when this code lived in the library, the lib unit-test
//! exe failed to load with `STATUS_ENTRYPOINT_NOT_FOUND` (0xc0000139) on Windows CI. Keeping the
//! `rfd` reference in the bin keeps the lib (and its test exe) free of that import.

use bootstrapper::updater::{UpdateNotice, UpdateNoticeLevel, UpdatePrompt, UpdatePromptChoice};

/// Custom-button label for the VR choice; also returned verbatim by `rfd` as the
/// `MessageDialogResult::Custom(label)` payload, so the same string doubles as the match key.
const VR_BUTTON_LABEL: &str = "VR";
/// Custom-button label for the Desktop choice; also returned verbatim by `rfd` as the
/// `MessageDialogResult::Custom(label)` payload.
const DESKTOP_BUTTON_LABEL: &str = "Desktop";
/// Custom-button label for the Cancel choice; also returned verbatim by `rfd` as the
/// `MessageDialogResult::Custom(label)` payload.
const CANCEL_BUTTON_LABEL: &str = "Cancel";
/// Custom-button label that starts the release update.
const UPDATE_BUTTON_LABEL: &str = "Update";
/// Custom-button label that skips the update for the current launch only.
const SKIP_ONCE_BUTTON_LABEL: &str = "Skip Once";
/// Custom-button label that persists a skip for the offered release tag.
const SKIP_RELEASE_BUTTON_LABEL: &str = "Skip This Release";

/// Shows the desktop vs VR selection dialog and returns the choice: `Some(true)` for VR,
/// `Some(false)` for Desktop, [`None`] for Cancel/dismiss (callers treat the latter as a
/// request to abort the launch).
///
/// Requires the global logger to be initialized before invocation so that the before/after
/// log lines reach disk.
pub fn prompt_desktop_or_vr() -> Option<bool> {
    logger::info!("Showing desktop/VR selection dialog via rfd backend.");
    let res = rfd::MessageDialog::new()
        .set_title("Renderide")
        .set_description("Launch Resonite in VR or desktop mode?")
        // Keep Desktop first so native default-button handling is safe if a pending keypress
        // confirms the dialog as soon as it appears.
        .set_buttons(rfd::MessageButtons::YesNoCancelCustom(
            DESKTOP_BUTTON_LABEL.into(),
            VR_BUTTON_LABEL.into(),
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

/// Shows the release update prompt and returns the selected update action.
pub fn prompt_release_update(prompt: &UpdatePrompt) -> UpdatePromptChoice {
    logger::info!(
        "Showing update dialog for release {} asset {}.",
        prompt.latest_tag,
        prompt.asset_name
    );
    let description = format!(
        "A new Renderide CI release is available.\n\nCurrent: {} ({})\nLatest: {} ({})\n\nUpdating will replace the launcher, renderer, and bundled runtime assets, then exit so you can restart into the new build.",
        prompt.current_tag,
        short_commit(&prompt.current_commit),
        prompt.latest_tag,
        short_commit(&prompt.latest_commit),
    );
    let result = rfd::MessageDialog::new()
        .set_title("Renderide Update")
        .set_description(description)
        .set_buttons(rfd::MessageButtons::YesNoCancelCustom(
            UPDATE_BUTTON_LABEL.into(),
            SKIP_ONCE_BUTTON_LABEL.into(),
            SKIP_RELEASE_BUTTON_LABEL.into(),
        ))
        .show();

    match result {
        rfd::MessageDialogResult::Custom(label) if label == UPDATE_BUTTON_LABEL => {
            logger::info!("Update dialog returned: update.");
            UpdatePromptChoice::Update
        }
        rfd::MessageDialogResult::Custom(label) if label == SKIP_RELEASE_BUTTON_LABEL => {
            logger::info!("Update dialog returned: skip this release.");
            UpdatePromptChoice::SkipRelease
        }
        other => {
            logger::info!("Update dialog returned skip-once or dismissed: {other:?}.");
            UpdatePromptChoice::SkipOnce
        }
    }
}

/// Shows an updater notification dialog.
pub fn show_update_notice(notice: UpdateNotice) {
    logger::info!("Showing update notice: {}", notice.title);
    let level = match notice.level {
        UpdateNoticeLevel::Info => rfd::MessageLevel::Info,
        UpdateNoticeLevel::Warning => rfd::MessageLevel::Warning,
        UpdateNoticeLevel::Error => rfd::MessageLevel::Error,
    };
    let _ = rfd::MessageDialog::new()
        .set_title(notice.title)
        .set_description(notice.message)
        .set_level(level)
        .show();
}

/// Returns a short display prefix for a full commit SHA.
fn short_commit(commit: &str) -> &str {
    commit.get(..8).unwrap_or(commit)
}
