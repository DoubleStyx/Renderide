//! Release archive and extracted bundle validation.

use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path};

use super::{MANIFEST_FILE, RELEASE_CHANNEL, ReleaseBuildMetadata, UpdateCandidate, UpdateError};

/// Validates every zip entry path before extraction.
pub(super) fn validate_zip_paths(archive_path: &Path) -> Result<(), UpdateError> {
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
pub(super) fn validate_zip_entry_name(name: &str) -> Result<(), UpdateError> {
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

/// Returns the expected top-level directory stem for a release zip asset.
pub(super) fn asset_stem(asset_name: &str) -> Result<String, UpdateError> {
    Path::new(asset_name)
        .file_stem()
        .and_then(OsStr::to_str)
        .filter(|stem| !stem.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| UpdateError::InvalidBundle(format!("invalid asset name `{asset_name}`")))
}

/// Validates the extracted release manifest and required bundle entries.
pub(super) fn validate_bundle_root(
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
pub(super) fn validate_manifest(
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

/// Returns the required install entries for a release platform token.
pub(super) fn required_bundle_entries(platform: &str) -> Vec<BundleEntry> {
    let mut entries = Vec::with_capacity(5);
    match platform {
        "windows-x86_64" | "windows-aarch64" => {
            entries.push(BundleEntry::file("renderide.exe", true));
            entries.push(BundleEntry::file("renderide-renderer.exe", false));
            entries.push(BundleEntry::directory("xr"));
            entries.push(BundleEntry::file("openxr_loader.dll", false));
        }
        "linux-x86_64" | "linux-aarch64" => {
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
pub(super) struct BundleEntry {
    /// Path relative to the bundle or install root.
    pub(super) relative: &'static str,
    /// Required filesystem entry type.
    pub(super) kind: EntryKind,
    /// Whether this entry is the running launcher executable.
    pub(super) launcher: bool,
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
pub(super) enum EntryKind {
    /// A regular file.
    File,
    /// A directory tree.
    Directory,
}

#[cfg(test)]
mod tests {
    use self_update::update::ReleaseAsset;

    use super::*;
    use crate::updater::github::asset_name_for;

    fn metadata() -> ReleaseBuildMetadata {
        ReleaseBuildMetadata {
            channel: RELEASE_CHANNEL.to_owned(),
            tag: "nightly-2026-05-26-1111111".to_owned(),
            commit: "1111111111111111111111111111111111111111".to_owned(),
            platform: "linux-x86_64".to_owned(),
        }
    }

    #[test]
    fn unsafe_zip_paths_are_rejected() {
        for name in ["/abs", "../up", "root/../up", "root\\file", "C:/file"] {
            assert!(
                validate_zip_entry_name(name).is_err(),
                "{name} should be rejected"
            );
        }
        assert!(validate_zip_entry_name("renderide-linux/foo").is_ok());
        assert!(validate_zip_entry_name("renderide-linux/xr/").is_ok());
    }

    #[test]
    fn manifest_validation_requires_expected_fields() {
        let metadata = metadata();
        let candidate = UpdateCandidate {
            tag: "nightly-2026-05-27-2222222".to_owned(),
            commit: "2222222222222222222222222222222222222222".to_owned(),
            changelog: String::new(),
            asset: ReleaseAsset {
                name: asset_name_for(&metadata.platform, "nightly-2026-05-27-2222222"),
                download_url: String::new(),
            },
        };
        let text = serde_json::json!({
            "schema": 1,
            "channel": RELEASE_CHANNEL,
            "tag": candidate.tag,
            "commit": candidate.commit,
            "platform": metadata.platform,
            "required_files": ["renderide", "renderide-renderer", "xr"]
        })
        .to_string();

        assert!(validate_manifest(&text, &metadata, &candidate).is_ok());
    }

    #[test]
    fn required_bundle_entries_include_launcher_for_known_platforms() {
        for platform in [
            "windows-x86_64",
            "windows-aarch64",
            "linux-x86_64",
            "linux-aarch64",
            "macos-x86_64",
            "macos-aarch64",
        ] {
            let entries = required_bundle_entries(platform);
            assert!(
                entries.iter().any(|entry| entry.launcher),
                "{platform} should include launcher"
            );
            assert!(
                entries
                    .iter()
                    .any(|entry| entry.relative == "renderide-renderer"
                        || entry.relative == "renderide-renderer.exe"),
                "{platform} should include renderer binary"
            );
        }
    }
}
