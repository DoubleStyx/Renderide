//! Release archive and extracted bundle validation.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path};

use base64::Engine;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};

use super::{
    MANIFEST_FILE, MANIFEST_SIGNATURE_FILE, RELEASE_CHANNEL, ReleaseBuildMetadata, UpdateCandidate,
    UpdateError,
};

#[cfg(not(test))]
const RELEASE_PUBLIC_KEY_ENV: &str = "RENDERIDE_RELEASE_PUBLIC_KEY_HEX";

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
    let manifest_bytes = fs::read(&manifest_path).map_err(|source| UpdateError::Io {
        context: format!("read release manifest {}", manifest_path.display()),
        source,
    })?;
    verify_release_manifest_signature(bundle_root, &manifest_bytes)?;
    let manifest = String::from_utf8(manifest_bytes)
        .map_err(|e| UpdateError::InvalidManifest(e.to_string()))?;
    validate_manifest(&manifest, metadata, candidate)?;
    for entry in required_bundle_entries(&metadata.platform) {
        validate_required_bundle_entry(bundle_root, entry)?;
    }
    validate_required_file_digests(bundle_root, &manifest, &metadata.platform)?;
    Ok(())
}

fn verify_release_manifest_signature(
    bundle_root: &Path,
    manifest: &[u8],
) -> Result<(), UpdateError> {
    let signature_path = bundle_root.join(MANIFEST_SIGNATURE_FILE);
    let signature_text = fs::read_to_string(&signature_path).map_err(|source| UpdateError::Io {
        context: format!(
            "read release manifest signature {}",
            signature_path.display()
        ),
        source,
    })?;
    verify_manifest_signature_bytes(manifest, signature_text.trim().as_bytes())
}

fn verify_manifest_signature_bytes(
    manifest: &[u8],
    signature_b64: &[u8],
) -> Result<(), UpdateError> {
    let signature_bytes = base64::engine::general_purpose::STANDARD
        .decode(signature_b64)
        .map_err(|e| UpdateError::InvalidSignature(e.to_string()))?;
    let signature = Signature::from_slice(&signature_bytes)
        .map_err(|e| UpdateError::InvalidSignature(e.to_string()))?;
    let public_key = release_manifest_public_key()?;
    public_key
        .verify(manifest, &signature)
        .map_err(|e| UpdateError::InvalidSignature(e.to_string()))
}

fn release_manifest_public_key() -> Result<VerifyingKey, UpdateError> {
    #[cfg(test)]
    {
        VerifyingKey::from_bytes(&test_release_public_key_bytes())
            .map_err(|e| UpdateError::InvalidSignature(e.to_string()))
    }
    #[cfg(not(test))]
    {
        let Some(hex) = option_env!("RENDERIDE_RELEASE_PUBLIC_KEY_HEX") else {
            return Err(UpdateError::InvalidSignature(format!(
                "{RELEASE_PUBLIC_KEY_ENV} was not set at compile time"
            )));
        };
        let bytes = decode_hex_32(hex)?;
        VerifyingKey::from_bytes(&bytes).map_err(|e| UpdateError::InvalidSignature(e.to_string()))
    }
}

#[cfg(test)]
const TEST_RELEASE_PRIVATE_KEY_BYTES: [u8; 32] = [7; 32];

#[cfg(test)]
fn test_release_public_key_bytes() -> [u8; 32] {
    [
        0xea, 0x4a, 0x6c, 0x63, 0xe2, 0x9c, 0x52, 0x0a, 0xbe, 0xf5, 0x50, 0x7b, 0x13, 0x2e, 0xc5,
        0xf9, 0x95, 0x47, 0x76, 0xae, 0xbe, 0xbe, 0x7b, 0x92, 0x42, 0x1e, 0xea, 0x69, 0x14, 0x46,
        0xd2, 0x2c,
    ]
}

#[cfg(test)]
pub(super) fn signed_manifest_for_test(manifest: &str) -> String {
    use base64::Engine;
    use ed25519_dalek::{Signer, SigningKey};

    let signing_key = SigningKey::from_bytes(&TEST_RELEASE_PRIVATE_KEY_BYTES);
    let signature = signing_key.sign(manifest.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(signature.to_bytes())
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

fn validate_required_file_digests(
    bundle_root: &Path,
    text: &str,
    platform: &str,
) -> Result<(), UpdateError> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| UpdateError::InvalidManifest(e.to_string()))?;
    let digests = value
        .get("sha256")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| UpdateError::InvalidManifest("`sha256` missing".to_owned()))?;
    let required_files = collect_required_regular_files(bundle_root, platform)?;
    let mut seen = BTreeSet::new();

    for (relative, expected) in digests {
        validate_digest_entry_name(relative)?;
        let Some(expected) = expected.as_str() else {
            return Err(UpdateError::InvalidManifest(format!(
                "`sha256.{relative}` must be a hex string"
            )));
        };
        if !required_files.contains(relative.as_str()) {
            return Err(UpdateError::InvalidManifest(format!(
                "`sha256` contains non-installed file `{relative}`"
            )));
        }
        let expected = decode_hex_32(expected)?;
        let actual = sha256_file(&bundle_root.join(relative))?;
        if actual != expected {
            return Err(UpdateError::InvalidBundle(format!(
                "{relative} failed sha256 verification"
            )));
        }
        seen.insert(relative.to_owned());
    }

    for relative in required_files {
        if !seen.contains(&relative) {
            return Err(UpdateError::InvalidManifest(format!(
                "`sha256` missing `{relative}`"
            )));
        }
    }
    Ok(())
}

fn collect_required_regular_files(
    bundle_root: &Path,
    platform: &str,
) -> Result<BTreeSet<String>, UpdateError> {
    let mut files = BTreeSet::new();
    for entry in required_bundle_entries(platform) {
        collect_required_regular_files_for_entry(bundle_root, entry, &mut files)?;
    }
    Ok(files)
}

fn collect_required_regular_files_for_entry(
    bundle_root: &Path,
    entry: BundleEntry,
    files: &mut BTreeSet<String>,
) -> Result<(), UpdateError> {
    let path = bundle_root.join(entry.relative);
    match entry.kind {
        EntryKind::File => {
            files.insert(entry.relative.to_owned());
            Ok(())
        }
        EntryKind::Directory => collect_regular_files_recursive(bundle_root, &path, files),
    }
}

fn collect_regular_files_recursive(
    bundle_root: &Path,
    dir: &Path,
    files: &mut BTreeSet<String>,
) -> Result<(), UpdateError> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .map_err(|source| UpdateError::Io {
            context: format!("read bundle directory {}", dir.display()),
            source,
        })?
        .collect::<Result<_, _>>()
        .map_err(|source| UpdateError::Io {
            context: format!("read bundle directory entry {}", dir.display()),
            source,
        })?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        let metadata = fs::metadata(&path).map_err(|source| UpdateError::Io {
            context: format!("inspect bundle entry {}", path.display()),
            source,
        })?;
        if metadata.is_dir() {
            collect_regular_files_recursive(bundle_root, &path, files)?;
        } else if metadata.is_file() {
            files.insert(bundle_relative_file_name(bundle_root, &path)?);
        }
    }
    Ok(())
}

fn bundle_relative_file_name(bundle_root: &Path, path: &Path) -> Result<String, UpdateError> {
    let relative = path.strip_prefix(bundle_root).map_err(|source| {
        UpdateError::InvalidBundle(format!(
            "{} is outside bundle root {}: {source}",
            path.display(),
            bundle_root.display()
        ))
    })?;
    path_to_manifest_name(relative)
}

fn path_to_manifest_name(path: &Path) -> Result<String, UpdateError> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let Some(part) = part.to_str() else {
                    return Err(UpdateError::InvalidBundle(format!(
                        "bundle path {} is not valid UTF-8",
                        path.display()
                    )));
                };
                parts.push(part);
            }
            _ => {
                return Err(UpdateError::InvalidBundle(format!(
                    "bundle path {} is not relative",
                    path.display()
                )));
            }
        }
    }
    if parts.is_empty() {
        return Err(UpdateError::InvalidBundle(
            "empty bundle file path".to_owned(),
        ));
    }
    Ok(parts.join("/"))
}

fn validate_digest_entry_name(name: &str) -> Result<(), UpdateError> {
    if name.ends_with('/') {
        return Err(UpdateError::InvalidManifest(format!(
            "unsafe sha256 path `{name}`"
        )));
    }
    let normalized = name;
    if normalized.is_empty()
        || normalized.starts_with('/')
        || normalized.contains('\\')
        || normalized.contains(':')
        || normalized.split('/').any(str::is_empty)
    {
        return Err(UpdateError::InvalidManifest(format!(
            "unsafe sha256 path `{name}`"
        )));
    }
    for component in Path::new(normalized).components() {
        match component {
            Component::Normal(part) if !part.is_empty() => {}
            _ => {
                return Err(UpdateError::InvalidManifest(format!(
                    "unsafe sha256 path `{name}`"
                )));
            }
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<[u8; 32], UpdateError> {
    let bytes = fs::read(path).map_err(|source| UpdateError::Io {
        context: format!("read bundle file {}", path.display()),
        source,
    })?;
    Ok(Sha256::digest(bytes).into())
}

fn decode_hex_32(hex: &str) -> Result<[u8; 32], UpdateError> {
    let hex = hex.trim();
    if hex.len() != 64 {
        return Err(UpdateError::InvalidManifest(format!(
            "expected 64 hex chars, got {}",
            hex.len()
        )));
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&hex[offset..offset + 2], 16)
            .map_err(|e| UpdateError::InvalidManifest(e.to_string()))?;
    }
    Ok(out)
}

/// Returns the required install entries for a release platform token.
pub(super) fn required_bundle_entries(platform: &str) -> Vec<BundleEntry> {
    let mut entries = Vec::with_capacity(6);
    match platform {
        "windows-x86_64" | "windows-aarch64" => {
            entries.push(BundleEntry::file("renderide.exe", true));
            entries.push(BundleEntry::file("renderide-renderer.exe", false));
            entries.push(BundleEntry::directory("xr"));
            entries.push(BundleEntry::directory("shaders"));
            entries.push(BundleEntry::file("openxr_loader.dll", false));
        }
        "linux-x86_64" | "linux-aarch64" => {
            entries.push(BundleEntry::file("renderide", true));
            entries.push(BundleEntry::file("renderide-renderer", false));
            entries.push(BundleEntry::directory("xr"));
            entries.push(BundleEntry::directory("shaders"));
        }
        "macos-x86_64" | "macos-aarch64" => {
            entries.push(BundleEntry::file("renderide", true));
            entries.push(BundleEntry::file("renderide-renderer", false));
            entries.push(BundleEntry::directory("xr"));
            entries.push(BundleEntry::directory("shaders"));
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
    use std::fs;

    use self_update::update::ReleaseAsset;
    use tempfile::tempdir;

    use super::*;
    use crate::updater::github::asset_name_for;

    const LAUNCHER_SHA256: &str =
        "ec9a6e9fe278eb1a471fbab6f40367d8548078b651d9c71581c57c2a6ca379e0";
    const RENDERER_SHA256: &str =
        "6bd52b204f5b4cffb267597f37d0fa62bae229341394dfec0e5d42439d8b722c";
    const EMPTY_JSON_SHA256: &str =
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a";

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
            "required_files": ["renderide", "renderide-renderer", "xr", "shaders"],
            "sha256": {
                "renderide": LAUNCHER_SHA256,
                "renderide-renderer": RENDERER_SHA256
            }
        })
        .to_string();

        assert!(validate_manifest(&text, &metadata, &candidate).is_ok());
    }

    #[test]
    fn manifest_signature_validation_rejects_tampering() {
        let manifest = "{}";
        let signature = signed_manifest_for_test(manifest);

        assert!(verify_manifest_signature_bytes(manifest.as_bytes(), signature.as_bytes()).is_ok());
        assert!(
            verify_manifest_signature_bytes(b"{\"tampered\":true}", signature.as_bytes()).is_err()
        );
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
            assert!(
                entries.iter().any(|entry| entry.relative == "shaders"),
                "{platform} should include runtime shader package"
            );
        }
    }

    fn write_linux_bundle(root: &Path, xr_actions: &[u8]) {
        fs::write(root.join("renderide"), b"launcher").expect("write launcher");
        fs::write(root.join("renderide-renderer"), b"renderer").expect("write renderer");
        fs::create_dir_all(root.join("xr")).expect("create xr");
        fs::write(root.join("xr/actions.toml"), xr_actions).expect("write xr actions");
        fs::create_dir_all(root.join("shaders")).expect("create shaders");
        fs::write(root.join("shaders/shader_manifest.toml"), b"{}").expect("write shader manifest");
    }

    fn linux_manifest_with_sha256(sha256: serde_json::Value) -> String {
        serde_json::json!({
            "schema": 1,
            "channel": RELEASE_CHANNEL,
            "tag": "nightly-2026-05-27-2222222",
            "commit": "2222222222222222222222222222222222222222",
            "platform": "linux-x86_64",
            "required_files": ["renderide", "renderide-renderer", "xr", "shaders"],
            "sha256": sha256
        })
        .to_string()
    }

    fn valid_linux_sha256() -> serde_json::Value {
        serde_json::json!({
            "renderide": LAUNCHER_SHA256,
            "renderide-renderer": RENDERER_SHA256,
            "xr/actions.toml": EMPTY_JSON_SHA256,
            "shaders/shader_manifest.toml": EMPTY_JSON_SHA256
        })
    }

    #[test]
    fn required_file_digests_cover_directory_contents() {
        let tmp = tempdir().expect("tempdir");
        write_linux_bundle(tmp.path(), b"{}");
        let manifest = linux_manifest_with_sha256(valid_linux_sha256());

        validate_required_file_digests(tmp.path(), &manifest, "linux-x86_64")
            .expect("directory file digest should validate");
    }

    #[test]
    fn required_file_digests_reject_directory_tampering() {
        let tmp = tempdir().expect("tempdir");
        write_linux_bundle(tmp.path(), b"changed");
        let manifest = linux_manifest_with_sha256(valid_linux_sha256());

        assert!(validate_required_file_digests(tmp.path(), &manifest, "linux-x86_64").is_err());
    }

    #[test]
    fn required_file_digests_reject_unsigned_directory_file() {
        let tmp = tempdir().expect("tempdir");
        write_linux_bundle(tmp.path(), b"{}");
        fs::create_dir_all(tmp.path().join("xr/bindings")).expect("create bindings");
        fs::write(tmp.path().join("xr/bindings/profile.toml"), b"unsigned")
            .expect("write unsigned file");
        let manifest = linux_manifest_with_sha256(valid_linux_sha256());

        assert!(validate_required_file_digests(tmp.path(), &manifest, "linux-x86_64").is_err());
    }

    #[test]
    fn required_file_digests_reject_extra_digest_entries() {
        let tmp = tempdir().expect("tempdir");
        write_linux_bundle(tmp.path(), b"{}");
        let manifest = linux_manifest_with_sha256(serde_json::json!({
            "renderide": LAUNCHER_SHA256,
            "renderide-renderer": RENDERER_SHA256,
            "xr/actions.toml": EMPTY_JSON_SHA256,
            "shaders/shader_manifest.toml": EMPTY_JSON_SHA256,
            "renderide-release.json": EMPTY_JSON_SHA256
        }));

        assert!(validate_required_file_digests(tmp.path(), &manifest, "linux-x86_64").is_err());
    }

    #[test]
    fn required_file_digests_reject_unsafe_digest_paths() {
        let tmp = tempdir().expect("tempdir");
        write_linux_bundle(tmp.path(), b"{}");
        let manifest = linux_manifest_with_sha256(serde_json::json!({
            "renderide": LAUNCHER_SHA256,
            "renderide-renderer": RENDERER_SHA256,
            "xr/actions.toml": EMPTY_JSON_SHA256,
            "shaders/shader_manifest.toml": EMPTY_JSON_SHA256,
            "xr/../escape": EMPTY_JSON_SHA256
        }));

        assert!(validate_required_file_digests(tmp.path(), &manifest, "linux-x86_64").is_err());
    }
}
