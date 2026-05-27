//! CI-release bundle updater for the bootstrapper.
//!
//! The updater intentionally keys off compile-time release metadata emitted by the release
//! workflow. Source checkouts and manual release-mode builds do not carry that metadata, so they
//! never contact GitHub or show update UI.

use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use self_update::backends::github::ReleaseList;
use self_update::update::{Release, ReleaseAsset};

/// Unit tests for release filtering, manifests, and skip state.
#[cfg(test)]
mod tests;
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
        let platform = current_platform()?;
        let channel = option_env!("RENDERIDE_RELEASE_CHANNEL")?.trim();
        let tag = option_env!("RENDERIDE_RELEASE_TAG")?.trim();
        let commit = option_env!("RENDERIDE_RELEASE_COMMIT")?.trim();
        let embedded_platform = option_env!("RENDERIDE_RELEASE_PLATFORM")?.trim();
        if channel != RELEASE_CHANNEL
            || !tag.starts_with(NIGHTLY_PREFIX)
            || !is_full_sha(commit)
            || embedded_platform != platform
        {
            return None;
        }
        Some(Self {
            channel: channel.to_owned(),
            tag: tag.to_owned(),
            commit: commit.to_owned(),
            platform: embedded_platform.to_owned(),
        })
    }
}

/// Runs the startup update check and optional install flow.
pub fn run_startup_update_check<P, N>(prompt_update: P, notify: N) -> StartupUpdateOutcome
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

    let candidate = match fetch_latest_candidate(&metadata) {
        Ok(Some(candidate)) => candidate,
        Ok(None) => return StartupUpdateOutcome::Continue,
        Err(e) => {
            logger::warn!("Update check failed: {e}");
            return StartupUpdateOutcome::Continue;
        }
    };

    if skipped_release_tag().is_some_and(|tag| tag == candidate.tag) {
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
            if let Err(e) = persist_skipped_release(&candidate.tag) {
                logger::warn!("Could not persist skipped update release: {e}");
            }
            StartupUpdateOutcome::Continue
        }
        UpdatePromptChoice::Update => install_prompted_candidate(&metadata, &candidate, notify),
    }
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
    match restore_latest_backup() {
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

/// Returns the compile-time platform token used by release asset names.
pub fn current_platform() -> Option<&'static str> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        Some("linux-x86_64")
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        Some("windows-x86_64")
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        Some("macos-x86_64")
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        Some("macos-aarch64")
    }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    )))]
    {
        None
    }
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

/// Fetches GitHub releases and selects the newest eligible update candidate.
fn fetch_latest_candidate(
    metadata: &ReleaseBuildMetadata,
) -> Result<Option<UpdateCandidate>, UpdateError> {
    let releases = ReleaseList::configure()
        .repo_owner(REPO_OWNER)
        .repo_name(REPO_NAME)
        .build()
        .map_err(to_self_update_error)?
        .fetch()
        .map_err(to_self_update_error)?;
    Ok(select_update_candidate(&releases, metadata))
}

/// Selects the first eligible release asset that is not the running release tag.
fn select_update_candidate(
    releases: &[Release],
    metadata: &ReleaseBuildMetadata,
) -> Option<UpdateCandidate> {
    releases
        .iter()
        .filter(|release| release.version.starts_with(NIGHTLY_PREFIX))
        .filter(|release| release.version != metadata.tag)
        .find_map(|release| candidate_from_release(release, metadata))
}

/// Converts a GitHub release into an update candidate when it has this platform's asset.
fn candidate_from_release(
    release: &Release,
    metadata: &ReleaseBuildMetadata,
) -> Option<UpdateCandidate> {
    let commit = release_commit(release.body.as_deref())?;
    let asset_name = asset_name_for(&metadata.platform, &release.version);
    let asset = release
        .assets
        .iter()
        .find(|asset| asset.name == asset_name)
        .cloned()?;
    Some(UpdateCandidate {
        tag: release.version.clone(),
        commit,
        asset,
    })
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
    match install_candidate(metadata, candidate) {
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

/// Downloads, validates, extracts, and installs a selected release candidate.
fn install_candidate(
    metadata: &ReleaseBuildMetadata,
    candidate: &UpdateCandidate,
) -> Result<(), UpdateError> {
    let install_dir = current_install_dir()?;
    let stage_dir = unique_update_subdir(&install_dir, DOWNLOADS_DIR, &candidate.tag)?;
    let archive_path = download_asset(&candidate.asset, &stage_dir)?;
    validate_zip_paths(&archive_path)?;
    let bundle_root = extract_asset(&archive_path, &stage_dir, &candidate.asset.name)?;
    validate_bundle_root(&bundle_root, metadata, candidate)?;
    install_extracted_bundle(&install_dir, &bundle_root, metadata)
}

/// Resolves the directory containing the currently running launcher.
fn current_install_dir() -> Result<PathBuf, UpdateError> {
    let exe = std::env::current_exe().map_err(|source| UpdateError::Io {
        context: "resolve current executable".to_owned(),
        source,
    })?;
    exe.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| UpdateError::InvalidBundle("current executable has no parent".to_owned()))
}

/// Downloads the selected GitHub release asset into a staging directory.
fn download_asset(asset: &ReleaseAsset, stage_dir: &Path) -> Result<PathBuf, UpdateError> {
    fs::create_dir_all(stage_dir).map_err(|source| UpdateError::Io {
        context: format!("create update staging directory {}", stage_dir.display()),
        source,
    })?;
    let archive_path = stage_dir.join(&asset.name);
    let mut archive = fs::File::create(&archive_path).map_err(|source| UpdateError::Io {
        context: format!("create update archive {}", archive_path.display()),
        source,
    })?;
    let mut download = self_update::Download::from_url(&asset.download_url);
    download
        .set_header(
            http::header::ACCEPT,
            http::HeaderValue::from_static("application/octet-stream"),
        )
        .show_progress(false)
        .download_to(&mut archive)
        .map_err(to_self_update_error)?;
    Ok(archive_path)
}

/// Validates every zip entry path before extraction.
fn validate_zip_paths(archive_path: &Path) -> Result<(), UpdateError> {
    let archive = fs::File::open(archive_path).map_err(|source| UpdateError::Io {
        context: format!("open update archive {}", archive_path.display()),
        source,
    })?;
    let mut zip = zip::ZipArchive::new(archive)
        .map_err(|e| UpdateError::InvalidArchive(format!("{}: {e}", archive_path.display())))?;
    for index in 0..zip.len() {
        let entry = zip
            .by_index(index)
            .map_err(|e| UpdateError::InvalidArchive(e.to_string()))?;
        validate_zip_entry_name(entry.name())?;
    }
    Ok(())
}

/// Rejects absolute, parent-relative, or platform-ambiguous archive paths.
fn validate_zip_entry_name(name: &str) -> Result<(), UpdateError> {
    let normalized = name.strip_suffix('/').unwrap_or(name);
    if normalized.is_empty()
        || normalized.starts_with('/')
        || normalized.contains('\\')
        || normalized.contains(':')
        || normalized.split('/').any(str::is_empty)
    {
        return Err(UpdateError::InvalidArchive(format!(
            "unsafe archive path `{name}`"
        )));
    }
    for component in Path::new(normalized).components() {
        match component {
            Component::Normal(part) if !part.is_empty() => {}
            _ => {
                return Err(UpdateError::InvalidArchive(format!(
                    "unsafe archive path `{name}`"
                )));
            }
        }
    }
    Ok(())
}

/// Extracts the release archive and returns the expected bundle root directory.
fn extract_asset(
    archive_path: &Path,
    stage_dir: &Path,
    asset_name: &str,
) -> Result<PathBuf, UpdateError> {
    let extract_dir = stage_dir.join("extracted");
    fs::create_dir_all(&extract_dir).map_err(|source| UpdateError::Io {
        context: format!("create extraction directory {}", extract_dir.display()),
        source,
    })?;
    self_update::Extract::from_source(archive_path)
        .archive(self_update::ArchiveKind::Zip)
        .extract_into(&extract_dir)
        .map_err(to_self_update_error)?;
    let bundle_root = extract_dir.join(asset_stem(asset_name)?);
    if !bundle_root.is_dir() {
        return Err(UpdateError::InvalidBundle(format!(
            "expected extracted bundle root {}",
            bundle_root.display()
        )));
    }
    Ok(bundle_root)
}

/// Returns the expected top-level directory stem for a release zip asset.
fn asset_stem(asset_name: &str) -> Result<String, UpdateError> {
    Path::new(asset_name)
        .file_stem()
        .and_then(OsStr::to_str)
        .filter(|stem| !stem.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| UpdateError::InvalidBundle(format!("invalid asset name `{asset_name}`")))
}

/// Validates the extracted release manifest and required bundle entries.
fn validate_bundle_root(
    bundle_root: &Path,
    metadata: &ReleaseBuildMetadata,
    candidate: &UpdateCandidate,
) -> Result<(), UpdateError> {
    let manifest_path = bundle_root.join(MANIFEST_FILE);
    let manifest = fs::read_to_string(&manifest_path).map_err(|source| UpdateError::Io {
        context: format!("read release manifest {}", manifest_path.display()),
        source,
    })?;
    validate_manifest(&manifest, metadata, candidate)?;
    for entry in required_bundle_entries(&metadata.platform) {
        validate_required_bundle_entry(bundle_root, entry)?;
    }
    Ok(())
}

/// Verifies that an extracted required entry exists with the expected type.
fn validate_required_bundle_entry(
    bundle_root: &Path,
    entry: BundleEntry,
) -> Result<(), UpdateError> {
    let path = bundle_root.join(entry.relative);
    let metadata = fs::metadata(&path).map_err(|source| UpdateError::Io {
        context: format!("inspect bundle entry {}", path.display()),
        source,
    })?;
    match entry.kind {
        EntryKind::File if metadata.is_file() => Ok(()),
        EntryKind::Directory if metadata.is_dir() => Ok(()),
        _ => Err(UpdateError::InvalidBundle(format!(
            "{} has the wrong entry type",
            path.display()
        ))),
    }
}

/// Validates release manifest metadata against the selected release and running platform.
fn validate_manifest(
    text: &str,
    metadata: &ReleaseBuildMetadata,
    candidate: &UpdateCandidate,
) -> Result<(), UpdateError> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| UpdateError::InvalidManifest(e.to_string()))?;
    require_json_u64(&value, "schema", 1)?;
    require_json_str(&value, "channel", RELEASE_CHANNEL)?;
    require_json_str(&value, "tag", &candidate.tag)?;
    require_json_str(&value, "commit", &candidate.commit)?;
    require_json_str(&value, "platform", &metadata.platform)?;
    require_manifest_files(&value, &metadata.platform)
}

/// Requires a JSON integer field to match an expected value.
fn require_json_u64(
    value: &serde_json::Value,
    key: &str,
    expected: u64,
) -> Result<(), UpdateError> {
    match value.get(key).and_then(serde_json::Value::as_u64) {
        Some(actual) if actual == expected => Ok(()),
        _ => Err(UpdateError::InvalidManifest(format!(
            "`{key}` must be {expected}"
        ))),
    }
}

/// Requires a JSON string field to match an expected value.
fn require_json_str(
    value: &serde_json::Value,
    key: &str,
    expected: &str,
) -> Result<(), UpdateError> {
    match value.get(key).and_then(serde_json::Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(UpdateError::InvalidManifest(format!(
            "`{key}` was `{actual}`, expected `{expected}`"
        ))),
        None => Err(UpdateError::InvalidManifest(format!("`{key}` missing"))),
    }
}

/// Requires the manifest to list every platform-required bundle entry.
fn require_manifest_files(value: &serde_json::Value, platform: &str) -> Result<(), UpdateError> {
    let files = value
        .get("required_files")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| UpdateError::InvalidManifest("`required_files` missing".to_owned()))?;
    for entry in required_bundle_entries(platform) {
        let found = files
            .iter()
            .any(|file| file.as_str().is_some_and(|file| file == entry.relative));
        if !found {
            return Err(UpdateError::InvalidManifest(format!(
                "`required_files` missing `{}`",
                entry.relative
            )));
        }
    }
    Ok(())
}

/// Installs a validated bundle into the current install directory with rollback backup.
fn install_extracted_bundle(
    install_dir: &Path,
    bundle_root: &Path,
    metadata: &ReleaseBuildMetadata,
) -> Result<(), UpdateError> {
    let entries = required_bundle_entries(&metadata.platform);
    let backup_dir = unique_update_subdir(install_dir, BACKUPS_DIR, &metadata.tag)?;
    backup_existing_entries(install_dir, &backup_dir, &entries)?;
    write_backup_metadata(&backup_dir, metadata);
    if let Err(e) = replace_non_launcher_entries(install_dir, bundle_root, &entries) {
        let _ = restore_non_launcher_entries(install_dir, &backup_dir, &entries);
        return Err(e);
    }
    let launcher = entries
        .iter()
        .find(|entry| entry.launcher)
        .ok_or_else(|| UpdateError::InvalidBundle("launcher entry missing".to_owned()))?;
    let new_launcher = bundle_root.join(launcher.relative);
    if let Err(e) = self_update::self_replace::self_replace(&new_launcher) {
        let _ = restore_non_launcher_entries(install_dir, &backup_dir, &entries);
        return Err(to_self_update_error(e));
    }
    Ok(())
}

/// Copies current install entries to a rollback backup directory.
fn backup_existing_entries(
    install_dir: &Path,
    backup_dir: &Path,
    entries: &[BundleEntry],
) -> Result<(), UpdateError> {
    fs::create_dir_all(backup_dir).map_err(|source| UpdateError::Io {
        context: format!("create update backup {}", backup_dir.display()),
        source,
    })?;
    for entry in entries {
        let source = install_dir.join(entry.relative);
        if !source.exists() {
            return Err(UpdateError::InvalidBundle(format!(
                "required installed path missing: {}",
                source.display()
            )));
        }
        copy_path(&source, &backup_dir.join(entry.relative))?;
    }
    Ok(())
}

/// Writes best-effort metadata describing a rollback backup.
fn write_backup_metadata(backup_dir: &Path, metadata: &ReleaseBuildMetadata) {
    let text = format!(
        "tag={}\ncommit={}\nplatform={}\n",
        metadata.tag, metadata.commit, metadata.platform
    );
    let path = backup_dir.join("backup-release.txt");
    if let Err(e) = fs::write(&path, text) {
        logger::warn!(
            "Could not write update backup metadata {}: {e}",
            path.display()
        );
    }
}

/// Replaces all required entries except the currently running launcher executable.
fn replace_non_launcher_entries(
    install_dir: &Path,
    bundle_root: &Path,
    entries: &[BundleEntry],
) -> Result<(), UpdateError> {
    for entry in entries.iter().filter(|entry| !entry.launcher) {
        replace_entry_from_bundle(bundle_root, install_dir, *entry)?;
    }
    Ok(())
}

/// Restores all required entries except the launcher from a backup directory.
fn restore_non_launcher_entries(
    install_dir: &Path,
    backup_dir: &Path,
    entries: &[BundleEntry],
) -> Result<(), UpdateError> {
    for entry in entries.iter().filter(|entry| !entry.launcher) {
        restore_entry_from_backup(backup_dir, install_dir, *entry)?;
    }
    Ok(())
}

/// Replaces one installed entry from the extracted bundle.
fn replace_entry_from_bundle(
    bundle_root: &Path,
    install_dir: &Path,
    entry: BundleEntry,
) -> Result<(), UpdateError> {
    let source = bundle_root.join(entry.relative);
    let destination = install_dir.join(entry.relative);
    remove_destination(&destination)?;
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|source| UpdateError::Io {
            context: format!("create destination directory {}", parent.display()),
            source,
        })?;
    }
    match fs::rename(&source, &destination) {
        Ok(()) => Ok(()),
        Err(_) => {
            copy_path(&source, &destination)?;
            remove_source(&source).map_err(|source_error| UpdateError::Io {
                context: format!("remove staged entry {}", source.display()),
                source: source_error,
            })
        }
    }
}

/// Restores one installed entry from the rollback backup.
fn restore_entry_from_backup(
    backup_dir: &Path,
    install_dir: &Path,
    entry: BundleEntry,
) -> Result<(), UpdateError> {
    let source = backup_dir.join(entry.relative);
    let destination = install_dir.join(entry.relative);
    if !source.exists() {
        return Err(UpdateError::InvalidBundle(format!(
            "backup entry missing: {}",
            source.display()
        )));
    }
    remove_destination(&destination)?;
    copy_path(&source, &destination)
}

/// Recursively copies a file or directory.
fn copy_path(source: &Path, destination: &Path) -> Result<(), UpdateError> {
    let metadata = fs::metadata(source).map_err(|source_error| UpdateError::Io {
        context: format!("inspect {}", source.display()),
        source: source_error,
    })?;
    if metadata.is_dir() {
        copy_dir_all(source, destination)
    } else {
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|source_error| UpdateError::Io {
                context: format!("create directory {}", parent.display()),
                source: source_error,
            })?;
        }
        fs::copy(source, destination)
            .map(|_| ())
            .map_err(|source_error| UpdateError::Io {
                context: format!("copy {} to {}", source.display(), destination.display()),
                source: source_error,
            })
    }
}

/// Recursively copies a directory tree.
fn copy_dir_all(source: &Path, destination: &Path) -> Result<(), UpdateError> {
    fs::create_dir_all(destination).map_err(|source_error| UpdateError::Io {
        context: format!("create directory {}", destination.display()),
        source: source_error,
    })?;
    for entry in fs::read_dir(source).map_err(|source_error| UpdateError::Io {
        context: format!("read directory {}", source.display()),
        source: source_error,
    })? {
        let entry = entry.map_err(|source_error| UpdateError::Io {
            context: format!("read directory entry {}", source.display()),
            source: source_error,
        })?;
        let child_source = entry.path();
        let child_destination = destination.join(entry.file_name());
        copy_path(&child_source, &child_destination)?;
    }
    Ok(())
}

/// Removes an installed destination path when it exists.
fn remove_destination(path: &Path) -> Result<(), UpdateError> {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return Ok(());
    };
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
    .map_err(|source| UpdateError::Io {
        context: format!("remove {}", path.display()),
        source,
    })
}

/// Removes a staged source path after copy fallback succeeds.
fn remove_source(path: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

/// Restores the newest rollback backup and schedules launcher self-replacement.
fn restore_latest_backup() -> Result<(), UpdateError> {
    let install_dir = current_install_dir()?;
    let backup_dir = latest_backup_dir(&install_dir)?;
    let entries = required_bundle_entries(
        current_platform()
            .ok_or_else(|| UpdateError::InvalidBundle("unsupported platform".to_owned()))?,
    );
    restore_non_launcher_entries(&install_dir, &backup_dir, &entries)?;
    let launcher = entries
        .iter()
        .find(|entry| entry.launcher)
        .ok_or_else(|| UpdateError::InvalidBundle("launcher entry missing".to_owned()))?;
    let backup_launcher = backup_dir.join(launcher.relative);
    let temp_launcher =
        unique_update_subdir(&install_dir, DOWNLOADS_DIR, "rollback")?.join(launcher.relative);
    copy_path(&backup_launcher, &temp_launcher)?;
    self_update::self_replace::self_replace(&temp_launcher).map_err(to_self_update_error)
}

/// Returns the most recently modified rollback backup directory.
fn latest_backup_dir(install_dir: &Path) -> Result<PathBuf, UpdateError> {
    let backups = update_root(install_dir).join(BACKUPS_DIR);
    let mut latest: Option<(SystemTime, PathBuf)> = None;
    let read_dir =
        fs::read_dir(&backups).map_err(|source| UpdateError::NoBackup(source.to_string()))?;
    for entry in read_dir {
        let entry = entry.map_err(|source| UpdateError::Io {
            context: format!("read backup entry {}", backups.display()),
            source,
        })?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(UNIX_EPOCH);
        if latest
            .as_ref()
            .is_none_or(|(latest_time, _)| modified > *latest_time)
        {
            latest = Some((modified, path));
        }
    }
    latest
        .map(|(_, path)| path)
        .ok_or_else(|| UpdateError::NoBackup(backups.display().to_string()))
}

/// Returns the install-local updater working directory.
fn update_root(install_dir: &Path) -> PathBuf {
    install_dir.join(UPDATE_DIR)
}

/// Creates a unique updater subdirectory for one staged operation.
fn unique_update_subdir(
    install_dir: &Path,
    category: &str,
    tag: &str,
) -> Result<PathBuf, UpdateError> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis());
    let name = format!(
        "{}-{}-{timestamp}",
        sanitize_for_path(tag),
        std::process::id()
    );
    let path = update_root(install_dir).join(category).join(name);
    fs::create_dir_all(&path).map_err(|source| UpdateError::Io {
        context: format!("create update directory {}", path.display()),
        source,
    })?;
    Ok(path)
}

/// Reads the persisted release tag that the user chose to skip.
fn skipped_release_tag() -> Option<String> {
    let path = state_path()?;
    match fs::read_to_string(&path) {
        Ok(contents) => skipped_release_tag_from_contents(&contents),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => {
            logger::warn!("Could not read updater state {}: {e}", path.display());
            None
        }
    }
}

/// Persists that a release tag should not be prompted again.
fn persist_skipped_release(tag: &str) -> Result<(), UpdateError> {
    let Some(path) = state_path() else {
        logger::warn!("Could not resolve per-user updater state path.");
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| UpdateError::Io {
            context: format!("create updater state directory {}", parent.display()),
            source,
        })?;
    }
    fs::write(&path, format!("skip_release_tag={tag}\n")).map_err(|source| UpdateError::Io {
        context: format!("write updater state {}", path.display()),
        source,
    })
}

/// Resolves the per-user updater state file path.
fn state_path() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| {
        dirs.config_dir()
            .join("Renderide")
            .join("updater")
            .join(STATE_FILE)
    })
}

/// Parses the skipped release tag from state file contents.
fn skipped_release_tag_from_contents(contents: &str) -> Option<String> {
    contents.lines().find_map(|line| {
        line.strip_prefix("skip_release_tag=")
            .filter(|tag| !tag.trim().is_empty())
            .map(|tag| tag.trim().to_owned())
    })
}

/// Returns the required install entries for a release platform token.
fn required_bundle_entries(platform: &str) -> Vec<BundleEntry> {
    let mut entries = Vec::with_capacity(5);
    match platform {
        "windows-x86_64" => {
            entries.push(BundleEntry::file("renderide.exe", true));
            entries.push(BundleEntry::file("renderide-renderer.exe", false));
            entries.push(BundleEntry::directory("xr"));
            entries.push(BundleEntry::file("openxr_loader.dll", false));
        }
        "linux-x86_64" => {
            entries.push(BundleEntry::file("renderide", true));
            entries.push(BundleEntry::file("renderide-renderer", false));
            entries.push(BundleEntry::directory("xr"));
        }
        "macos-x86_64" | "macos-aarch64" => {
            entries.push(BundleEntry::file("renderide", true));
            entries.push(BundleEntry::file("renderide-renderer", false));
            entries.push(BundleEntry::directory("xr"));
            entries.push(BundleEntry::file("libopenxr_loader.dylib", false));
        }
        _ => {}
    }
    entries
}

/// A required file or directory in a release bundle.
#[derive(Clone, Copy)]
struct BundleEntry {
    /// Path relative to the bundle or install root.
    relative: &'static str,
    /// Required filesystem entry type.
    kind: EntryKind,
    /// Whether this entry is the running launcher executable.
    launcher: bool,
}

impl BundleEntry {
    /// Creates a required file entry.
    fn file(relative: &'static str, launcher: bool) -> Self {
        Self {
            relative,
            kind: EntryKind::File,
            launcher,
        }
    }

    /// Creates a required directory entry.
    fn directory(relative: &'static str) -> Self {
        Self {
            relative,
            kind: EntryKind::Directory,
            launcher: false,
        }
    }
}

/// Required filesystem entry kind.
#[derive(Clone, Copy)]
enum EntryKind {
    /// A regular file.
    File,
    /// A directory tree.
    Directory,
}

/// Builds the exact GitHub release asset name for a platform and tag.
fn asset_name_for(platform: &str, tag: &str) -> String {
    format!("renderide-{platform}-{tag}.zip")
}

/// Parses the commit SHA recorded in a GitHub release body.
fn release_commit(body: Option<&str>) -> Option<String> {
    body.and_then(|body| {
        body.lines().find_map(|line| {
            let commit = line.strip_prefix("Commit: ")?;
            is_full_sha(commit).then(|| commit.to_owned())
        })
    })
}

/// Returns whether a string is a complete hexadecimal Git commit SHA.
fn is_full_sha(value: &str) -> bool {
    value.len() == 40 && value.chars().all(|c| c.is_ascii_hexdigit())
}

/// Sanitizes user-controlled tag text for use in updater working directory names.
fn sanitize_for_path(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unknown".to_owned()
    } else {
        sanitized
    }
}

/// Converts a displayable self_update error into the updater error enum.
fn to_self_update_error(error: impl std::fmt::Display) -> UpdateError {
    UpdateError::SelfUpdate(error.to_string())
}
