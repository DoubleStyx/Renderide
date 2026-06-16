//! CI-release bundle updater for the bootstrapper.
//!
//! The updater intentionally keys off compile-time release metadata emitted by the release
//! workflow. Source checkouts and manual release-mode builds do not carry that metadata, so they
//! never contact GitHub or show update UI.

mod bundle;
mod github;
mod install;
mod release_metadata;
mod startup;
mod state;
/// Public updater type definitions.
mod types;

pub use types::{
    ReleaseBuildMetadata, StartupUpdateOutcome, UpdateCandidate, UpdateError, UpdateNotice,
    UpdateNoticeLevel, UpdatePrompt, UpdatePromptChoice,
};

/// Environment flag that disables update checks for release builds.
pub const ENV_SKIP_UPDATE_CHECK: &str = "RENDERIDE_SKIP_UPDATE_CHECK";
/// Environment flag that restores the latest local update backup.
pub const ENV_ROLLBACK_UPDATE: &str = "RENDERIDE_ROLLBACK_UPDATE";

/// Release channel string embedded into official GitHub CI builds.
const RELEASE_CHANNEL: &str = "github-ci";
/// GitHub repository owner used for release checks.
const REPO_OWNER: &str = "DoubleStyx";
/// GitHub repository name used for release checks.
const REPO_NAME: &str = "Renderide";
/// Prefix used by CI release tags that are eligible for auto-update.
const NIGHTLY_PREFIX: &str = "nightly-";
/// Manifest filename included in every release zip root.
const MANIFEST_FILE: &str = "renderide-release.json";
/// Detached Ed25519 signature filename for [`MANIFEST_FILE`].
const MANIFEST_SIGNATURE_FILE: &str = "renderide-release.json.sig";
/// Install-local directory that stores updater staging and backups.
const UPDATE_DIR: &str = ".renderide-update";
/// Child directory under [`UPDATE_DIR`] containing rollback backups.
const BACKUPS_DIR: &str = "backups";
/// Child directory under [`UPDATE_DIR`] containing downloaded and extracted updates.
const DOWNLOADS_DIR: &str = "downloads";
/// Per-user updater state file storing skipped release tags.
const STATE_FILE: &str = "updater-state.txt";

impl ReleaseBuildMetadata {
    /// Returns embedded release metadata when this launcher is an official CI release.
    pub fn current() -> Option<Self> {
        release_metadata::current()
    }
}

/// Runs the startup update check and optional install flow.
pub fn run_startup_update_check<P, N>(prompt_update: P, notify: N) -> StartupUpdateOutcome
where
    P: FnOnce(&UpdatePrompt) -> UpdatePromptChoice,
    N: Fn(UpdateNotice),
{
    startup::run_startup_update_check(prompt_update, notify)
}

/// Returns `true` when rollback is requested through the environment.
pub fn rollback_requested_from_env() -> bool {
    std::env::var_os(ENV_ROLLBACK_UPDATE).is_some_and(|v| !v.is_empty() && v != "0")
}

/// Restores the newest local update backup and exits so the restored launcher can be restarted.
pub fn run_startup_rollback<N>(notify: N) -> StartupUpdateOutcome
where
    N: Fn(UpdateNotice),
{
    startup::run_startup_rollback(notify)
}

/// Returns the compile-time platform token used by release asset names.
pub fn current_platform() -> Option<&'static str> {
    release_metadata::current_platform()
}

/// Converts a displayable self_update error into the updater error enum.
fn to_self_update_error(error: impl std::fmt::Display) -> UpdateError {
    UpdateError::SelfUpdate(error.to_string())
}
