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

/// Extracts the release archive and returns the validated bundle root directory.
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
    extract_zip_archive(archive_path, &extract_dir)?;
    extracted_bundle_root(&extract_dir, asset_name)
}

/// Extracts a zip archive into `extract_dir` after validating every entry path.
fn extract_zip_archive(archive_path: &Path, extract_dir: &Path) -> Result<(), UpdateError> {
    let archive = fs::File::open(archive_path).map_err(|source| UpdateError::Io {
        context: format!("open update archive {}", archive_path.display()),
        source,
    })?;
    let mut zip = zip::ZipArchive::new(archive)
        .map_err(|e| UpdateError::InvalidArchive(format!("{}: {e}", archive_path.display())))?;
    for index in 0..zip.len() {
        let mut entry = zip
            .by_index(index)
            .map_err(|e| UpdateError::InvalidArchive(e.to_string()))?;
        bundle::validate_zip_entry_name(entry.name())?;
        let output_path = extract_dir.join(entry.name());
        if entry.is_dir() {
            fs::create_dir_all(&output_path).map_err(|source| UpdateError::Io {
                context: format!("create extracted directory {}", output_path.display()),
                source,
            })?;
            continue;
        }
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).map_err(|source| UpdateError::Io {
                context: format!("create extracted directory {}", parent.display()),
                source,
            })?;
        }
        let mut output = fs::File::create(&output_path).map_err(|source| UpdateError::Io {
            context: format!("create extracted file {}", output_path.display()),
            source,
        })?;
        io::copy(&mut entry, &mut output)
            .map(|_| ())
            .map_err(|source| UpdateError::Io {
                context: format!("extract {} to {}", entry.name(), output_path.display()),
                source,
            })?;
        set_extracted_file_permissions(&output_path, entry.unix_mode())?;
    }
    Ok(())
}

/// Resolves either supported release archive root layout.
fn extracted_bundle_root(extract_dir: &Path, asset_name: &str) -> Result<PathBuf, UpdateError> {
    let flat_manifest = extract_dir.join(super::MANIFEST_FILE);
    if flat_manifest.is_file() {
        return Ok(extract_dir.to_path_buf());
    }

    let nested_root = extract_dir.join(asset_stem(asset_name)?);
    if nested_root.join(super::MANIFEST_FILE).is_file() {
        return Ok(nested_root);
    }

    Err(UpdateError::InvalidBundle(format!(
        "expected release manifest at {} or {}",
        flat_manifest.display(),
        nested_root.join(super::MANIFEST_FILE).display()
    )))
}

/// Applies executable permissions recorded in the zip metadata on Unix.
#[cfg(unix)]
fn set_extracted_file_permissions(path: &Path, unix_mode: Option<u32>) -> Result<(), UpdateError> {
    use std::os::unix::fs::PermissionsExt;

    let Some(mode) = unix_mode else {
        return Ok(());
    };
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(|source| UpdateError::Io {
        context: format!("set extracted file permissions {}", path.display()),
        source,
    })
}

/// Permissions are not represented portably on non-Unix platforms.
#[cfg(not(unix))]
fn set_extracted_file_permissions(
    _path: &Path,
    _unix_mode: Option<u32>,
) -> Result<(), UpdateError> {
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

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::{fs, process};

    use self_update::update::ReleaseAsset;
    use zip::write::SimpleFileOptions;

    use super::*;
    use crate::updater::{RELEASE_CHANNEL, github::asset_name_for};

    const PLATFORM: &str = "linux-x86_64";
    const CURRENT_TAG: &str = "nightly-2026-05-26-1111111";
    const UPDATE_TAG: &str = "nightly-2026-05-27-2222222";
    const CURRENT_COMMIT: &str = "1111111111111111111111111111111111111111";
    const UPDATE_COMMIT: &str = "2222222222222222222222222222222222222222";
    const ZIP_REGULAR_FILE_MODE: u32 = 0o100_644;
    const ZIP_EXECUTABLE_FILE_MODE: u32 = 0o100_755;

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> Self {
            let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "renderide_updater_{name}_{}_{}",
                process::id(),
                id
            ));
            fs::create_dir_all(&path).expect("create test temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    enum TestZipEntry<'a> {
        File {
            name: &'a str,
            contents: &'a [u8],
            unix_mode: Option<u32>,
        },
        Directory(&'a str),
    }

    fn metadata() -> ReleaseBuildMetadata {
        ReleaseBuildMetadata {
            channel: RELEASE_CHANNEL.to_owned(),
            tag: CURRENT_TAG.to_owned(),
            commit: CURRENT_COMMIT.to_owned(),
            platform: PLATFORM.to_owned(),
        }
    }

    fn candidate(asset_name: String) -> UpdateCandidate {
        UpdateCandidate {
            tag: UPDATE_TAG.to_owned(),
            commit: UPDATE_COMMIT.to_owned(),
            changelog: String::new(),
            asset: ReleaseAsset {
                name: asset_name,
                download_url: String::new(),
            },
        }
    }

    fn manifest() -> String {
        serde_json::json!({
            "schema": 1,
            "channel": RELEASE_CHANNEL,
            "tag": UPDATE_TAG,
            "commit": UPDATE_COMMIT,
            "platform": PLATFORM,
            "required_files": ["renderide", "renderide-renderer", "xr"]
        })
        .to_string()
    }

    fn write_test_zip(path: &Path, entries: &[TestZipEntry<'_>]) {
        let file = fs::File::create(path).expect("create test zip");
        let mut writer = zip::ZipWriter::new(file);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for entry in entries {
            match entry {
                TestZipEntry::File {
                    name,
                    contents,
                    unix_mode,
                } => {
                    let options = unix_mode.map_or(options, |mode| options.unix_permissions(mode));
                    writer.start_file(name, options).expect("start zip file");
                    writer.write_all(contents).expect("write zip file");
                }
                TestZipEntry::Directory(name) => {
                    writer.add_directory(*name, options).expect("add zip dir");
                }
            }
        }
        writer.finish().expect("finish test zip");
    }

    #[test]
    fn flat_release_archive_extracts_as_bundle_root() {
        let tmp = TempDir::new("flat");
        let asset_name = asset_name_for(PLATFORM, UPDATE_TAG);
        let archive_path = tmp.path().join(&asset_name);
        let manifest = manifest();
        write_test_zip(
            &archive_path,
            &[
                TestZipEntry::Directory("xr/"),
                TestZipEntry::File {
                    name: "renderide-release.json",
                    contents: manifest.as_bytes(),
                    unix_mode: Some(ZIP_REGULAR_FILE_MODE),
                },
                TestZipEntry::File {
                    name: "renderide",
                    contents: b"launcher",
                    unix_mode: Some(ZIP_EXECUTABLE_FILE_MODE),
                },
                TestZipEntry::File {
                    name: "renderide-renderer",
                    contents: b"renderer",
                    unix_mode: Some(ZIP_EXECUTABLE_FILE_MODE),
                },
                TestZipEntry::File {
                    name: "xr/actions.json",
                    contents: b"{}",
                    unix_mode: Some(ZIP_REGULAR_FILE_MODE),
                },
            ],
        );

        let stage_dir = tmp.path().join("stage");
        let root = extract_asset(&archive_path, &stage_dir, &asset_name).expect("extract flat zip");

        assert_eq!(root, stage_dir.join("extracted"));
        bundle::validate_bundle_root(&root, &metadata(), &candidate(asset_name))
            .expect("flat bundle root validates");
    }

    #[test]
    fn nested_release_archive_extracts_as_bundle_root() {
        let tmp = TempDir::new("nested");
        let asset_name = asset_name_for(PLATFORM, UPDATE_TAG);
        let archive_path = tmp.path().join(&asset_name);
        let root_name = asset_stem(&asset_name).expect("asset stem");
        let manifest = manifest();
        write_test_zip(
            &archive_path,
            &[
                TestZipEntry::Directory(&format!("{root_name}/xr/")),
                TestZipEntry::File {
                    name: &format!("{root_name}/renderide-release.json"),
                    contents: manifest.as_bytes(),
                    unix_mode: Some(ZIP_REGULAR_FILE_MODE),
                },
                TestZipEntry::File {
                    name: &format!("{root_name}/renderide"),
                    contents: b"launcher",
                    unix_mode: Some(ZIP_EXECUTABLE_FILE_MODE),
                },
                TestZipEntry::File {
                    name: &format!("{root_name}/renderide-renderer"),
                    contents: b"renderer",
                    unix_mode: Some(ZIP_EXECUTABLE_FILE_MODE),
                },
                TestZipEntry::File {
                    name: &format!("{root_name}/xr/actions.json"),
                    contents: b"{}",
                    unix_mode: Some(ZIP_REGULAR_FILE_MODE),
                },
            ],
        );

        let stage_dir = tmp.path().join("stage");
        let root =
            extract_asset(&archive_path, &stage_dir, &asset_name).expect("extract nested zip");

        assert_eq!(root, stage_dir.join("extracted").join(root_name));
        bundle::validate_bundle_root(&root, &metadata(), &candidate(asset_name))
            .expect("nested bundle root validates");
    }

    #[test]
    fn zip_directory_entries_extract_without_file_creation() {
        let tmp = TempDir::new("directory_entry");
        let archive_path = tmp.path().join("directory-entry.zip");
        write_test_zip(
            &archive_path,
            &[
                TestZipEntry::Directory("xr/"),
                TestZipEntry::File {
                    name: "xr/actions.json",
                    contents: b"{}",
                    unix_mode: Some(ZIP_REGULAR_FILE_MODE),
                },
            ],
        );
        let extract_dir = tmp.path().join("extracted");
        fs::create_dir_all(&extract_dir).expect("create extraction root");

        extract_zip_archive(&archive_path, &extract_dir).expect("extract directory zip entry");

        assert!(extract_dir.join("xr").is_dir());
        assert_eq!(
            fs::read_to_string(extract_dir.join("xr/actions.json")).expect("read extracted file"),
            "{}"
        );
    }

    #[test]
    fn zip_extraction_rejects_unsafe_paths() {
        let tmp = TempDir::new("unsafe");
        let archive_path = tmp.path().join("unsafe.zip");
        write_test_zip(
            &archive_path,
            &[TestZipEntry::File {
                name: "../escape",
                contents: b"nope",
                unix_mode: None,
            }],
        );
        let extract_dir = tmp.path().join("extracted");
        fs::create_dir_all(&extract_dir).expect("create extraction root");

        assert!(extract_zip_archive(&archive_path, &extract_dir).is_err());
        assert!(!tmp.path().join("escape").exists());
    }

    #[cfg(unix)]
    #[test]
    fn zip_extraction_preserves_unix_executable_bits() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = TempDir::new("permissions");
        let archive_path = tmp.path().join("permissions.zip");
        write_test_zip(
            &archive_path,
            &[TestZipEntry::File {
                name: "renderide",
                contents: b"launcher",
                unix_mode: Some(ZIP_EXECUTABLE_FILE_MODE),
            }],
        );
        let extract_dir = tmp.path().join("extracted");
        fs::create_dir_all(&extract_dir).expect("create extraction root");

        extract_zip_archive(&archive_path, &extract_dir).expect("extract permission zip");

        let mode = fs::metadata(extract_dir.join("renderide"))
            .expect("inspect extracted launcher")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o755);
    }
}
