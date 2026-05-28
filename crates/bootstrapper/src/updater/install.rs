//! Download, extraction, installation, and rollback for update bundles.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use self_update::update::ReleaseAsset;

use super::bundle::{BundleEntry, asset_stem, required_bundle_entries};
use super::{
    BACKUPS_DIR, DOWNLOADS_DIR, ReleaseBuildMetadata, UPDATE_DIR, UpdateCandidate, UpdateError,
    bundle, current_platform, to_self_update_error,
};

/// Downloads, validates, extracts, and installs a selected release candidate.
pub(super) fn install_candidate(
    metadata: &ReleaseBuildMetadata,
    candidate: &UpdateCandidate,
) -> Result<(), UpdateError> {
    let install_dir = current_install_dir()?;
    let stage_dir = unique_update_subdir(&install_dir, DOWNLOADS_DIR, &candidate.tag)?;
    let archive_path = download_asset(&candidate.asset, &stage_dir)?;
    bundle::validate_zip_paths(&archive_path)?;
    let bundle_root = extract_asset(&archive_path, &stage_dir, &candidate.asset.name)?;
    bundle::validate_bundle_root(&bundle_root, metadata, candidate)?;
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
pub(super) fn restore_latest_backup() -> Result<(), UpdateError> {
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
