//! Startup update and rollback orchestration.

use super::{
    ENV_SKIP_UPDATE_CHECK, ReleaseBuildMetadata, StartupUpdateOutcome, UpdateCandidate,
    UpdateNotice, UpdateNoticeLevel, UpdatePrompt, UpdatePromptChoice, github, install, state,
};

/// Runs the startup update check and optional install flow.
pub(super) fn run_startup_update_check<P, N>(prompt_update: P, notify: N) -> StartupUpdateOutcome
where
    P: FnOnce(&UpdatePrompt) -> UpdatePromptChoice,
    N: Fn(UpdateNotice),
{
    let Some(metadata) = ReleaseBuildMetadata::current() else {
        logger::debug!("Update check skipped: launcher was not built as a CI release.");
        return StartupUpdateOutcome::Continue;
    };
    if !update_check_allowed_by_environment() {
        return StartupUpdateOutcome::Continue;
    }

    let candidate = match github::fetch_latest_candidate(&metadata) {
        Ok(Some(candidate)) => candidate,
        Ok(None) => return StartupUpdateOutcome::Continue,
        Err(e) => {
            logger::warn!("Update check failed: {e}");
            return StartupUpdateOutcome::Continue;
        }
    };

    if state::skipped_release_tag().is_some_and(|tag| tag == candidate.tag) {
        logger::info!(
            "Update check skipped release {} by user preference.",
            candidate.tag
        );
        return StartupUpdateOutcome::Continue;
    }

    let update_prompt = prompt_from_candidate(&metadata, &candidate);
    match prompt_update(&update_prompt) {
        UpdatePromptChoice::SkipOnce => StartupUpdateOutcome::Continue,
        UpdatePromptChoice::SkipRelease => {
            if let Err(e) = state::persist_skipped_release(&candidate.tag) {
                logger::warn!("Could not persist skipped update release: {e}");
            }
            StartupUpdateOutcome::Continue
        }
        UpdatePromptChoice::Update => install_prompted_candidate(&metadata, &candidate, notify),
    }
}

/// Restores the newest local update backup and exits so the restored launcher can be restarted.
pub(super) fn run_startup_rollback<N>(notify: N) -> StartupUpdateOutcome
where
    N: Fn(UpdateNotice),
{
    match install::restore_latest_backup() {
        Ok(()) => notify(UpdateNotice {
            title: "Renderide rollback complete".to_owned(),
            message: "The previous Renderide bundle was restored. Start Renderide again to run it."
                .to_owned(),
            level: UpdateNoticeLevel::Info,
        }),
        Err(e) => notify(UpdateNotice {
            title: "Renderide rollback failed".to_owned(),
            message: format!("Renderide could not restore the previous bundle.\n\n{e}"),
            level: UpdateNoticeLevel::Error,
        }),
    }
    StartupUpdateOutcome::Exit
}

/// Returns whether startup is allowed to contact GitHub and show update UI.
fn update_check_allowed_by_environment() -> bool {
    if std::env::var("CI").is_ok() {
        logger::debug!("Update check skipped: CI is set.");
        return false;
    }
    if std::env::var_os(ENV_SKIP_UPDATE_CHECK).is_some_and(|v| !v.is_empty()) {
        logger::info!("Update check skipped: {ENV_SKIP_UPDATE_CHECK} is set.");
        return false;
    }
    if !graphical_session_available() {
        logger::warn!(
            "Update check skipped: no graphical session is available for the update prompt."
        );
        return false;
    }
    true
}

/// Returns whether a graphical dialog can reasonably be shown on this platform.
fn graphical_session_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        let has_x11 = std::env::var_os("DISPLAY").is_some_and(|v| !v.is_empty());
        let has_wayland = std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty());
        has_x11 || has_wayland
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

/// Builds the dialog prompt payload for an update candidate.
fn prompt_from_candidate(
    metadata: &ReleaseBuildMetadata,
    candidate: &UpdateCandidate,
) -> UpdatePrompt {
    UpdatePrompt {
        current_tag: metadata.tag.clone(),
        current_commit: metadata.commit.clone(),
        latest_tag: candidate.tag.clone(),
        latest_commit: candidate.commit.clone(),
        asset_name: candidate.asset.name.clone(),
        changelog: candidate.changelog.clone(),
    }
}

/// Installs a candidate selected by the user and reports the outcome through a dialog notice.
fn install_prompted_candidate<N>(
    metadata: &ReleaseBuildMetadata,
    candidate: &UpdateCandidate,
    notify: N,
) -> StartupUpdateOutcome
where
    N: Fn(UpdateNotice),
{
    match install::install_candidate(metadata, candidate) {
        Ok(()) => {
            notify(UpdateNotice {
                title: "Renderide update installed".to_owned(),
                message: format!(
                    "Renderide was updated to {}.\n\nStart Renderide again to run the new build.",
                    candidate.tag
                ),
                level: UpdateNoticeLevel::Info,
            });
            StartupUpdateOutcome::Exit
        }
        Err(e) => {
            logger::error!("Renderide update failed: {e}");
            notify(UpdateNotice {
                title: "Renderide update failed".to_owned(),
                message: format!(
                    "Renderide could not install {}. The current bundle was kept or restored.\n\n{e}",
                    candidate.tag
                ),
                level: UpdateNoticeLevel::Error,
            });
            StartupUpdateOutcome::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use self_update::update::ReleaseAsset;

    use super::*;
    use crate::updater::{RELEASE_CHANNEL, github::asset_name_for};

    #[test]
    fn prompt_from_candidate_carries_changelog() {
        let metadata = ReleaseBuildMetadata {
            channel: RELEASE_CHANNEL.to_owned(),
            tag: "nightly-2026-05-26-1111111".to_owned(),
            commit: "1111111111111111111111111111111111111111".to_owned(),
            platform: "linux-x86_64".to_owned(),
        };
        let changelog =
            "- [22222222](https://example.invalid) Add release notes by Developer".to_owned();
        let candidate = UpdateCandidate {
            tag: "nightly-2026-05-27-2222222".to_owned(),
            commit: "2222222222222222222222222222222222222222".to_owned(),
            changelog: changelog.clone(),
            asset: ReleaseAsset {
                name: asset_name_for(&metadata.platform, "nightly-2026-05-27-2222222"),
                download_url: String::new(),
            },
        };

        let prompt = prompt_from_candidate(&metadata, &candidate);

        assert_eq!(prompt.changelog, changelog);
    }
}
