//! Resolve shader asset names from on-disk **Unity AssetBundle** files using `unity-asset`.
//!
//! [`crate::shared::ShaderUpload::file`] is typically an **extensionless path** (or any path) whose bytes
//! parse as UnityFS / AssetBundle data--not a Unity `.asset` YAML file. The resolved name is taken **only**
//! from [`unity_asset::environment::Environment::bundle_container_entries`]: `AssetBundle.m_Container`
//! asset paths matched to embedded Shader objects, then stemmed (e.g. `.../ui_unlit.shader` -> `ui_unlit`).
//!
//! Serialized shader objects are **not** read for internal labels, typetree names, or other object names.

use std::fmt::Display;
use std::path::Path;

use unity_asset::AssetBundle;
use unity_asset::class_ids::SHADER;
use unity_asset::environment::BinarySource;
use unity_asset::environment::Environment;
use unity_asset::load_bundle_from_memory;

/// Maximum file size to read when probing a bundle.
const MAX_READ_BYTES: usize = 32 * 1024 * 1024;

/// Maximum regular files examined under a directory hint (dev / loose layouts).
const MAX_DIR_FILES: usize = 256;

/// Maximum characters from parse errors included in logs.
const MAX_ERR_LOG_CHARS: usize = 240;

/// Hex prefix length for short probe lines.
const PROBE_HEX_SHORT: usize = 8;

/// Shader asset filename or stem from a filesystem path: **AssetBundle `m_Container` stem** only.
pub(crate) fn try_resolve_shader_asset_name_from_path(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let name = if meta.is_file() {
        try_from_file(path)
    } else if meta.is_dir() {
        try_from_directory(path)
    } else {
        None
    };
    if let Some(parsed) = &name {
        logger::info!(
            "shader_unity_asset: resolved {:?} from path {}",
            parsed,
            path.display()
        );
    }
    name
}

fn try_from_file(path: &Path) -> Option<String> {
    try_from_file_inner(path, true).0
}

/// When `log_failure` is `false` (directory scan), probe data is returned without per-file [`logger::warn!`].
fn try_from_file_inner(
    path: &Path,
    log_failure: bool,
) -> (Option<String>, Option<FileBinaryProbe>) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            logger::warn!(
                "shader_unity_asset: cannot read {:?} for binary probe: {}",
                path.display(),
                e
            );
            return (None, None);
        }
    };

    let mut probe = FileBinaryProbe::new(&bytes);
    if bytes.is_empty() {
        if log_failure {
            probe.warn_short(path, "empty file");
        }
        return (None, Some(probe));
    }
    if bytes.len() > MAX_READ_BYTES {
        if log_failure {
            probe.warn_short(path, "file too large");
        }
        return (None, Some(probe));
    }

    let mut env = Environment::new();
    let _ = env.load_file(path);
    let source = BinarySource::path(path);

    let mut memory_bundle: Option<AssetBundle> = None;
    if env.bundles().get(&source).is_none() {
        match load_bundle_from_memory(bytes) {
            Ok(b) => memory_bundle = Some(b),
            Err(e) => {
                probe.bundle_err = Some(truncate_display(&e, MAX_ERR_LOG_CHARS));
                logger::debug!(
                    "shader_unity_asset: {:?} not an AssetBundle: {}",
                    path.display(),
                    probe.bundle_err.as_deref().unwrap_or("")
                );
            }
        }
    }
    let bundle_ref: Option<&AssetBundle> = env.bundles().get(&source).or(memory_bundle.as_ref());

    if let Some(bundle) = bundle_ref {
        probe.bundle_parse_ok = true;
        probe.bundle_assets = bundle.assets.len();
        log_bundle_parse_debug(path, bundle);
        if let Some(name) = shader_name_from_bundle_container(path, bundle) {
            return (Some(name), None);
        }
        if log_failure {
            probe.warn_short(path, "AssetBundle: no shader name from m_Container");
            probe.log_debug_detail();
        }
        return (None, Some(probe));
    }

    if log_failure {
        probe.warn_short(path, "not an AssetBundle");
        probe.log_debug_detail();
    }
    (None, Some(probe))
}

fn log_bundle_parse_debug(path: &Path, bundle: &AssetBundle) {
    logger::debug!(
        "shader_unity_asset: parsed AssetBundle {:?}: {} SerializedFile(s)",
        path.display(),
        bundle.assets.len()
    );
}

fn log_container_resolution(path_id: i64, name: &str, container_asset_path: &str) {
    logger::debug!(
        "shader_unity_asset: Shader path_id={} source=m_Container asset_path={:?} name={:?}",
        path_id,
        container_asset_path,
        name
    );
}

/// Per-file binary probe state for structured failure logs.
struct FileBinaryProbe {
    bytes_len: usize,
    prefix_hex: String,
    prefix_ascii: String,
    bundle_parse_ok: bool,
    bundle_assets: usize,
    bundle_err: Option<String>,
}

impl FileBinaryProbe {
    fn new(bytes: &[u8]) -> Self {
        Self {
            bytes_len: bytes.len(),
            prefix_hex: format_hex_prefix(bytes, 24),
            prefix_ascii: ascii_prefix_hint(bytes, 40),
            bundle_parse_ok: false,
            bundle_assets: 0,
            bundle_err: None,
        }
    }

    /// One short [`logger::warn!`] line; full fields via [`Self::log_debug_detail`].
    fn warn_short(&self, path: &Path, reason: &str) {
        logger::warn!(
            "shader_unity_asset: {:?} -- {} | bytes={} hex8={} | bundle_ok={} | err {:?}",
            path.display(),
            reason,
            self.bytes_len,
            short_hex_prefix(&self.prefix_hex, PROBE_HEX_SHORT),
            self.bundle_parse_ok,
            self.bundle_err.as_deref().unwrap_or("")
        );
    }

    fn log_debug_detail(&self) {
        logger::debug!(
            "shader_unity_asset: probe detail bytes={} prefix_hex={} prefix_ascii={:?} bundle_ok={} bundle_assets={} bundle_err={:?}",
            self.bytes_len,
            self.prefix_hex,
            self.prefix_ascii,
            self.bundle_parse_ok,
            self.bundle_assets,
            self.bundle_err
        );
    }
}

fn short_hex_prefix(space_separated_hex: &str, max_bytes: usize) -> String {
    space_separated_hex
        .split_whitespace()
        .take(max_bytes)
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_hex_prefix(bytes: &[u8], max: usize) -> String {
    bytes
        .iter()
        .take(max)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn ascii_prefix_hint(bytes: &[u8], max: usize) -> String {
    let take = bytes.iter().copied().take(max).collect::<Vec<u8>>();
    if take.is_empty() {
        return String::new();
    }
    if take
        .iter()
        .all(|b| b.is_ascii_graphic() || matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
    {
        String::from_utf8_lossy(&take).chars().take(40).collect()
    } else {
        String::new()
    }
}

fn truncate_display(err: impl Display, max: usize) -> String {
    let s = err.to_string();
    if s.len() <= max {
        return s;
    }
    format!("{}...", &s[..max.saturating_sub(1)])
}

fn try_from_directory(dir: &Path) -> Option<String> {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(d) => d,
        Err(e) => {
            logger::warn!(
                "shader_unity_asset: cannot read directory {:?}: {}",
                dir.display(),
                e
            );
            return None;
        }
    };

    let mut paths: Vec<std::path::PathBuf> = read_dir
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();

    let files_total = paths.len();
    if files_total == 0 {
        logger::warn!(
            "shader_unity_asset: directory {:?} contains no regular files (only subdirs or empty); cannot probe Unity binaries here",
            dir.display()
        );
        return None;
    }
    paths.sort_unstable();
    // Prefer typical Unity extensions when scanning a loose directory (extensionless bundles sort last).
    paths.sort_by_key(|p| {
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match ext.as_str() {
            "asset" | "unity" | "shader" => 0,
            _ => 1,
        }
    });

    let mut examined = 0usize;
    let mut bundle_parse_hits = 0usize;
    let mut first_probe: Option<FileBinaryProbe> = None;

    for (idx, p) in paths.into_iter().enumerate() {
        if idx >= MAX_DIR_FILES {
            break;
        }
        examined += 1;
        logger::debug!(
            "shader_unity_asset: directory {:?} examining [{}/{}] {:?}",
            dir.display(),
            examined,
            files_total.min(MAX_DIR_FILES),
            p.display()
        );
        let (name, probe) = try_from_file_inner(&p, false);
        if let Some(name) = name {
            return Some(name);
        }
        if let Some(probe) = probe {
            if probe.bundle_parse_ok {
                bundle_parse_hits += 1;
            }
            if first_probe.is_none() {
                first_probe = Some(probe);
            }
        }
    }

    logger::warn!(
        "shader_unity_asset: directory {:?} -- no shader name (files_total={} examined={} cap={} bundle_hits={})",
        dir.display(),
        files_total,
        examined,
        MAX_DIR_FILES,
        bundle_parse_hits
    );
    if let Some(ref fp) = first_probe {
        logger::debug!("shader_unity_asset: first failed file probe sample");
        fp.log_debug_detail();
    }

    None
}

/// Shader stem from [`Environment::bundle_container_entries`] by matching Shader `path_id` to
/// `AssetBundle.m_Container`.
fn shader_name_from_bundle_container(bundle_path: &Path, bundle: &AssetBundle) -> Option<String> {
    let mut env = Environment::new();
    let _ = env.load_file(bundle_path);
    let source = BinarySource::path(bundle_path);
    if env.bundles().get(&source).is_none() {
        logger::debug!(
            "shader_unity_asset: Environment has no bundle for {:?} (m_Container unavailable)",
            bundle_path.display()
        );
        return None;
    }
    let entries = env.bundle_container_entries(bundle_path).ok()?;
    if entries.is_empty() {
        logger::debug!(
            "shader_unity_asset: no m_Container entries for {:?}",
            bundle_path.display()
        );
        return None;
    }

    let shader_path_ids: Vec<i64> = bundle
        .assets
        .iter()
        .flat_map(|sf| {
            sf.object_handles()
                .filter(|h| h.class_id() == SHADER)
                .map(|h| h.path_id())
        })
        .collect();

    for pid in shader_path_ids {
        if let Some(entry) = entries.iter().find(|e| e.path_id == pid)
            && let Some(name) = shader_asset_name_from_container_asset_path(&entry.asset_path)
        {
            log_container_resolution(pid, &name, &entry.asset_path);
            return Some(name);
        }
    }
    None
}

/// Derives a shader asset name from a Unity `m_Container` asset path (e.g. `.../ui_unlit.shader` -> `ui_unlit`).
fn shader_asset_name_from_container_asset_path(asset_path: &str) -> Option<String> {
    let p = asset_path.replace('\\', "/");
    let seg = p.rsplit('/').next()?.trim();
    if seg.is_empty() {
        return None;
    }
    let base = seg
        .strip_suffix(".shader")
        .unwrap_or(seg)
        .rsplit('/')
        .next()
        .unwrap_or(seg)
        .trim();
    if base.is_empty() {
        return None;
    }
    let lower = base.to_ascii_lowercase();
    if lower.starts_with("cab-") {
        return None;
    }
    Some(base.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_asset_path_strips_shader_suffix() {
        assert_eq!(
            shader_asset_name_from_container_asset_path("assets/foo/my_shader.shader").as_deref(),
            Some("my_shader")
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("archive:/CAB-deadbeef").as_deref(),
            None
        );
    }

    #[test]
    fn container_asset_path_handles_backslashes_whitespace_and_plain_stems() {
        assert_eq!(
            shader_asset_name_from_container_asset_path("Assets\\Shaders\\UI Text Unlit.shader")
                .as_deref(),
            Some("UI Text Unlit")
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("  assets/foo/ToonLit.shader  ").as_deref(),
            Some("ToonLit")
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("assets/foo/AlreadyStem").as_deref(),
            Some("AlreadyStem")
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("").as_deref(),
            None
        );
        assert_eq!(
            shader_asset_name_from_container_asset_path("assets/foo/   ").as_deref(),
            None
        );
    }

    #[test]
    fn prefix_formatters_are_stable_for_empty_short_and_truncated_inputs() {
        assert_eq!(format_hex_prefix(&[], 8), "");
        assert_eq!(format_hex_prefix(&[0, 1, 0xab, 0xff], 8), "00 01 ab ff");
        assert_eq!(format_hex_prefix(&[0, 1, 2, 3], 2), "00 01");
        assert_eq!(short_hex_prefix("00 01 02 03", 2), "00 01");
        assert_eq!(short_hex_prefix("00 01", 8), "00 01");
    }

    #[test]
    fn ascii_prefix_hint_only_returns_printable_prefixes() {
        assert_eq!(ascii_prefix_hint(b"", 8), "");
        assert_eq!(ascii_prefix_hint(b"UnityFS\nBundle", 32), "UnityFS\nBundle");
        assert_eq!(ascii_prefix_hint(&[0xff, b'A', b'B'], 32), "");
        assert_eq!(ascii_prefix_hint(b"abcdef", 3), "abc");
    }

    #[test]
    fn truncate_display_preserves_short_errors_and_truncates_long_errors() {
        assert_eq!(truncate_display("short", 16), "short");
        let truncated = truncate_display("abcdefghijklmnopqrstuvwxyz", 8);
        assert_eq!(truncated, "abcdefg...");
    }

    #[test]
    fn file_binary_probe_records_prefixes_without_parsing() {
        let probe = FileBinaryProbe::new(b"UnityFS\0binary");
        assert_eq!(probe.bytes_len, 14);
        assert!(probe.prefix_hex.starts_with("55 6e 69 74 79 46 53 00"));
        assert_eq!(probe.prefix_ascii, "");
        assert!(!probe.bundle_parse_ok);
        assert_eq!(probe.bundle_assets, 0);
        assert_eq!(probe.bundle_err, None);
    }

    #[test]
    fn path_hint_rejects_missing_paths_and_empty_directories() {
        let temp = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            try_resolve_shader_asset_name_from_path(&temp.path().join("missing")).as_deref(),
            None
        );
        assert_eq!(
            try_resolve_shader_asset_name_from_path(temp.path()).as_deref(),
            None
        );
    }
}
