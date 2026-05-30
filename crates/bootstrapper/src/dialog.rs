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
/// Custom-button label that shows the release changelog.
const VIEW_CHANGELOG_BUTTON_LABEL: &str = "View Changelog";
/// Custom-button label that skips the offered release until the next launch.
const SKIP_ONCE_BUTTON_LABEL: &str = "Not Now";
/// Custom-button label that persists a skip for the offered release tag.
const SKIP_RELEASE_BUTTON_LABEL: &str = "Skip This Release";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UpdateDialogAction {
    Update,
    ViewChangelog,
    SkipOnce,
    SkipRelease,
}

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
    loop {
        match show_release_update_prompt(prompt) {
            UpdateDialogAction::Update => {
                logger::info!("Update dialog returned: update.");
                return UpdatePromptChoice::Update;
            }
            UpdateDialogAction::ViewChangelog => {
                logger::info!("Update dialog returned: view changelog.");
                show_release_changelog(prompt);
            }
            UpdateDialogAction::SkipRelease => {
                logger::info!("Update dialog returned: skip this release.");
                return UpdatePromptChoice::SkipRelease;
            }
            UpdateDialogAction::SkipOnce => {
                logger::info!("Update dialog returned skip-once or dismissed.");
                return UpdatePromptChoice::SkipOnce;
            }
        }
    }
}

fn show_release_update_prompt(prompt: &UpdatePrompt) -> UpdateDialogAction {
    let description = release_update_description(prompt);
    #[cfg(target_os = "linux")]
    {
        if let Some(action) = show_release_update_prompt_with_zenity(&description) {
            return action;
        }
    }
    let result = rfd::MessageDialog::new()
        .set_title("Renderide Update")
        .set_description(description)
        .set_buttons(rfd::MessageButtons::YesNoCancelCustom(
            UPDATE_BUTTON_LABEL.into(),
            VIEW_CHANGELOG_BUTTON_LABEL.into(),
            SKIP_RELEASE_BUTTON_LABEL.into(),
        ))
        .show();
    action_from_dialog_result(result)
}

fn release_update_description(prompt: &UpdatePrompt) -> String {
    format!(
        "A new Renderide CI release is available.\n\nCurrent: {} ({})\nLatest: {} ({})\n\nUpdating will replace the launcher, renderer, and bundled runtime assets, then exit so you can restart into the new build.",
        prompt.current_tag,
        short_commit(&prompt.current_commit),
        prompt.latest_tag,
        short_commit(&prompt.latest_commit),
    )
}

fn action_from_dialog_result(result: rfd::MessageDialogResult) -> UpdateDialogAction {
    match result {
        rfd::MessageDialogResult::Custom(label) if label == UPDATE_BUTTON_LABEL => {
            UpdateDialogAction::Update
        }
        rfd::MessageDialogResult::Custom(label) if label == VIEW_CHANGELOG_BUTTON_LABEL => {
            UpdateDialogAction::ViewChangelog
        }
        rfd::MessageDialogResult::Custom(label) if label == SKIP_ONCE_BUTTON_LABEL => {
            UpdateDialogAction::SkipOnce
        }
        rfd::MessageDialogResult::Custom(label) if label == SKIP_RELEASE_BUTTON_LABEL => {
            UpdateDialogAction::SkipRelease
        }
        other => {
            logger::debug!("Update dialog dismissed or returned unhandled result: {other:?}.");
            UpdateDialogAction::SkipOnce
        }
    }
}

#[cfg(target_os = "linux")]
fn show_release_update_prompt_with_zenity(description: &str) -> Option<UpdateDialogAction> {
    let output = match std::process::Command::new("zenity")
        .arg("--no-markup")
        .arg("--question")
        .arg("--title")
        .arg("Renderide Update")
        .arg("--text")
        .arg(description)
        .arg("--ok-label")
        .arg(UPDATE_BUTTON_LABEL)
        .arg("--cancel-label")
        .arg(SKIP_ONCE_BUTTON_LABEL)
        .arg("--extra-button")
        .arg(VIEW_CHANGELOG_BUTTON_LABEL)
        .arg("--extra-button")
        .arg(SKIP_RELEASE_BUTTON_LABEL)
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            logger::warn!("Could not show update dialog via zenity: {e}");
            return None;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let label = stdout.trim();
    match label {
        VIEW_CHANGELOG_BUTTON_LABEL => Some(UpdateDialogAction::ViewChangelog),
        SKIP_ONCE_BUTTON_LABEL => Some(UpdateDialogAction::SkipOnce),
        SKIP_RELEASE_BUTTON_LABEL => Some(UpdateDialogAction::SkipRelease),
        _ if output.status.success() => Some(UpdateDialogAction::Update),
        _ => Some(UpdateDialogAction::SkipOnce),
    }
}

fn show_release_changelog(prompt: &UpdatePrompt) {
    let changelog = changelog_text_for_dialog(&prompt.changelog);
    let _ = rfd::MessageDialog::new()
        .set_title(format!("Renderide Changelog {}", prompt.latest_tag))
        .set_description(changelog)
        .set_level(rfd::MessageLevel::Info)
        .show();
}

fn changelog_text_for_dialog(changelog: &str) -> String {
    let mut output = String::with_capacity(changelog.len());
    let mut remaining = changelog;

    while let Some(start) = remaining.find('[') {
        output.push_str(&remaining[..start]);
        let candidate = &remaining[start..];
        let Some(label_end) = candidate.find(']') else {
            output.push_str(candidate);
            return output;
        };
        let after_label = &candidate[label_end + 1..];
        let Some(after_open) = after_label.strip_prefix('(') else {
            output.push('[');
            remaining = &candidate[1..];
            continue;
        };
        let Some(url_end) = after_open.find(')') else {
            output.push_str(candidate);
            return output;
        };

        output.push_str(&candidate[1..label_end]);
        remaining = &after_open[url_end + 1..];
    }

    output.push_str(remaining);
    output
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_dialog_result_maps_custom_buttons_and_dismissal() {
        assert_eq!(
            action_from_dialog_result(rfd::MessageDialogResult::Custom(
                UPDATE_BUTTON_LABEL.to_owned()
            )),
            UpdateDialogAction::Update
        );
        assert_eq!(
            action_from_dialog_result(rfd::MessageDialogResult::Custom(
                VIEW_CHANGELOG_BUTTON_LABEL.to_owned()
            )),
            UpdateDialogAction::ViewChangelog
        );
        assert_eq!(
            action_from_dialog_result(rfd::MessageDialogResult::Custom(
                SKIP_ONCE_BUTTON_LABEL.to_owned()
            )),
            UpdateDialogAction::SkipOnce
        );
        assert_eq!(
            action_from_dialog_result(rfd::MessageDialogResult::Custom(
                SKIP_RELEASE_BUTTON_LABEL.to_owned()
            )),
            UpdateDialogAction::SkipRelease
        );
        assert_eq!(
            action_from_dialog_result(rfd::MessageDialogResult::Cancel),
            UpdateDialogAction::SkipOnce
        );
    }

    #[test]
    fn changelog_dialog_text_renders_commit_links_as_text() {
        let changelog = "- [22222222](https://github.com/DoubleStyx/Renderide/commit/2222222222222222222222222222222222222222) Add updater changelog by Renderer Developer";

        assert_eq!(
            changelog_text_for_dialog(changelog),
            "- 22222222 Add updater changelog by Renderer Developer"
        );
    }

    #[test]
    fn changelog_dialog_text_renders_range_label_links_as_text() {
        let changelog = "Changes since [nightly-2026-05-29-1111111](https://github.com/DoubleStyx/Renderide/releases/tag/nightly-2026-05-29-1111111) ([11111111](https://github.com/DoubleStyx/Renderide/commit/1111111111111111111111111111111111111111)).";

        assert_eq!(
            changelog_text_for_dialog(changelog),
            "Changes since nightly-2026-05-29-1111111 (11111111)."
        );
    }

    #[test]
    fn changelog_dialog_text_keeps_fallback_text_unchanged() {
        let changelog = "No changelog is available for this release.";

        assert_eq!(changelog_text_for_dialog(changelog), changelog);
    }

    #[test]
    fn changelog_dialog_text_keeps_malformed_links_unchanged() {
        let changelog = "Keep [unfinished link text and [label] without a target.";

        assert_eq!(changelog_text_for_dialog(changelog), changelog);
    }
}
