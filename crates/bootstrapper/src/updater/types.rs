//! Public updater types shared by the launcher binary and updater internals.

use std::io;

use self_update::update::ReleaseAsset;

/// Release metadata embedded only by the GitHub release workflow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseBuildMetadata {
    /// Update channel. Only `github-ci` is update-enabled.
    pub channel: String,
    /// GitHub release tag that produced this launcher.
    pub tag: String,
    /// Full 40-character commit SHA used by the release workflow.
    pub commit: String,
    /// Release platform token, matching zip asset names.
    pub platform: String,
}

/// A GitHub release asset selected for this platform.
#[derive(Clone, Debug)]
pub struct UpdateCandidate {
    /// Release tag to install.
    pub tag: String,
    /// Release commit SHA parsed from the release body.
    pub commit: String,
    /// Markdown changelog parsed from the release body.
    pub changelog: String,
    /// Downloadable GitHub release asset.
    pub asset: ReleaseAsset,
}

/// Information shown in the update prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpdatePrompt {
    /// Currently installed release tag.
    pub current_tag: String,
    /// Currently installed release commit.
    pub current_commit: String,
    /// New release tag.
    pub latest_tag: String,
    /// New release commit.
    pub latest_commit: String,
    /// GitHub release asset name.
    pub asset_name: String,
    /// Markdown changelog for the offered release.
    pub changelog: String,
}

/// User choice from the update prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdatePromptChoice {
    /// Download and install the release.
    Update,
    /// Do not install now, but ask again next launch.
    SkipOnce,
    /// Persistently skip this release tag.
    SkipRelease,
}

/// Severity of an updater notification dialog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateNoticeLevel {
    /// Informational message.
    Info,
    /// Recoverable failure or anomaly.
    Warning,
    /// Update or rollback failed.
    Error,
}

/// Updater notification payload for the bin-only dialog module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpdateNotice {
    /// Dialog title.
    pub title: String,
    /// Dialog body text.
    pub message: String,
    /// Dialog severity.
    pub level: UpdateNoticeLevel,
}

/// Startup action requested by the updater.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartupUpdateOutcome {
    /// Continue with normal Host launch.
    Continue,
    /// Exit the launcher before starting Host.
    Exit,
}

/// Failures that can occur while checking, installing, or rolling back releases.
#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    /// Filesystem operation failed.
    #[error("{context}: {source}")]
    Io {
        /// Operation context.
        context: String,
        /// Original I/O error.
        #[source]
        source: io::Error,
    },
    /// `self_update` reported a backend, download, extraction, or replacement failure.
    #[error("self-update operation failed: {0}")]
    SelfUpdate(String),
    /// Zip archive was malformed or unsafe.
    #[error("invalid update archive: {0}")]
    InvalidArchive(String),
    /// Extracted release manifest was missing or invalid.
    #[error("invalid release manifest: {0}")]
    InvalidManifest(String),
    /// Installed or extracted bundle did not match the expected release shape.
    #[error("invalid update bundle: {0}")]
    InvalidBundle(String),
    /// No rollback backup was available.
    #[error("no update backup found under {0}")]
    NoBackup(String),
}
