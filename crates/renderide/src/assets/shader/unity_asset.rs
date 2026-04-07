//! Resolve Unity shader logical names from filesystem paths using the `unity-asset` stack.
//!
//! [`crate::shared::ShaderUpload::file`] may point at cache bundles (UnityFS), YAML assets, or raw
//! serialized files. For **AssetBundles** loaded from disk, the usual cache case is a name from
//! [`unity_asset::environment::Environment::bundle_container_entries`] (`AssetBundle.m_Container`
//! path stem, e.g. `.../ui_unlit.shader` → `ui_unlit`). If that fails, TypeTree `peek_name` / full
//! [`unity_asset_binary::object::ObjectHandle::read`] / ShaderLab substring fallbacks apply.

use std::fmt::Display;
use std::path::Path;

use unity_asset::class_ids::SHADER;
use unity_asset::environment::BinarySource;
use unity_asset::environment::Environment;
use unity_asset::load_bundle_from_memory;
use unity_asset::AssetBundle;
use unity_asset::UnityClass;
use unity_asset::UnityDocument;
use unity_asset::UnityValue;
use unity_asset::YamlDocument;
use unity_asset_binary::asset::SerializedFile;
use unity_asset_binary::asset::SerializedFileParser;
use unity_asset_binary::object::UnityObject;

use super::logical_name::parse_shader_lab_quoted_name;

/// Maximum file size to read for parsing (bundle / serialized file).
const MAX_READ_BYTES: usize = 32 * 1024 * 1024;

/// Maximum regular files examined under a directory hint.
const MAX_DIR_FILES: usize = 256;

/// Unity YAML class tag for [`Shader`](https://docs.unity3d.com/ScriptReference/Shader.html).
const SHADER_CLASS: &str = "Shader";

/// Maximum characters from parse errors included in logs.
const MAX_ERR_LOG_CHARS: usize = 240;

/// Hex prefix length for short probe lines.
const PROBE_HEX_SHORT: usize = 8;

/// How a shader logical name was obtained (for [`log_resolution_debug`] only).
#[derive(Clone, Copy)]
enum ResolutionSource {
    MNamePeek,
    UnityObjectTypetree,
    ShaderLabBytes,
    Container,
}

/// Attempts to extract a ShaderLab logical name when `file_field` is a path to a file or directory
/// containing Unity-serialized shader data.
///
/// Logs [`logger::info!`] on success (`resolved … from path`). Failures use one concise [`logger::warn!`]
/// per file or directory; detailed probe output is [`logger::debug!`].
pub(crate) fn try_resolve_shader_name_from_path_hint(path: &Path) -> Option<String> {
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
    let mut yaml_loaded = false;
    if let Ok(doc) = YamlDocument::load_yaml(path, false) {
        yaml_loaded = true;
        if let Some(name) = shader_name_from_yaml(&doc) {
            return (Some(name), None);
        }
        logger::debug!(
            "shader_unity_asset: {:?} loaded as YAML but no Shader name extracted",
            path.display()
        );
    } else {
        logger::debug!(
            "shader_unity_asset: {:?} not Unity YAML or YAML load failed",
            path.display()
        );
    }

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

    let mut probe = FileBinaryProbe::new(yaml_loaded, &bytes);
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
        match load_bundle_from_memory(bytes.clone()) {
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
        for (i, asset) in bundle.assets.iter().enumerate() {
            log_serialized_file_debug(path, asset, Some(i));
        }
        if let Some(name) = shader_name_from_bundle(path, bundle) {
            return (Some(name), None);
        }
        if log_failure {
            probe.warn_short(path, "AssetBundle: no shader name");
            probe.log_debug_detail();
        }
        return (None, Some(probe));
    }

    match SerializedFileParser::from_bytes(bytes) {
        Ok(sf) => {
            probe.serialized_parse_ok = true;
            probe.serialized_object_count = sf.object_count();
            probe.serialized_shader_count = shader_object_count(&sf);
            probe.serialized_unity_version = Some(sf.unity_version.clone());
            probe.serialized_format_version = Some(sf.header.version);
            probe.serialized_enable_type_tree = Some(sf.enable_type_tree);
            log_serialized_file_debug(path, &sf, None);
            if let Some(name) = shader_name_from_serialized_file(&sf) {
                return (Some(name), None);
            }
            if log_failure {
                probe.warn_short(path, "SerializedFile: no shader name");
                probe.log_debug_detail();
            }
            (None, Some(probe))
        }
        Err(e) => {
            probe.serialized_err = Some(truncate_display(&e, MAX_ERR_LOG_CHARS));
            logger::debug!(
                "shader_unity_asset: {:?} not a SerializedFile: {}",
                path.display(),
                probe.serialized_err.as_deref().unwrap_or("")
            );
            if log_failure {
                probe.warn_short(path, "not AssetBundle or SerializedFile");
                probe.log_debug_detail();
            }
            (None, Some(probe))
        }
    }
}

fn log_bundle_parse_debug(path: &Path, bundle: &AssetBundle) {
    logger::debug!(
        "shader_unity_asset: parsed AssetBundle {:?}: {} SerializedFile(s)",
        path.display(),
        bundle.assets.len()
    );
}

fn log_serialized_file_debug(path: &Path, sf: &SerializedFile, bundle_index: Option<usize>) {
    let n = shader_object_count(sf);
    let label = match bundle_index {
        Some(i) => format!("bundle[{i}]"),
        None => "standalone".to_string(),
    };
    logger::debug!(
        "shader_unity_asset: SerializedFile {:?} ({}): objects={} unity={} fmt={} type_tree={} shaders={}",
        path.display(),
        label,
        sf.object_count(),
        sf.unity_version,
        sf.header.version,
        sf.enable_type_tree,
        n
    );
}

fn log_resolution_debug(
    path_id: i64,
    class_id: i32,
    source: ResolutionSource,
    name: &str,
    container_asset_path: Option<&str>,
) {
    let src = match source {
        ResolutionSource::MNamePeek => "m_Name_peek",
        ResolutionSource::UnityObjectTypetree => "typetree",
        ResolutionSource::ShaderLabBytes => "ShaderLab_bytes",
        ResolutionSource::Container => "m_Container",
    };
    match (source, container_asset_path) {
        (ResolutionSource::Container, Some(ap)) => {
            logger::debug!(
                "shader_unity_asset: Shader path_id={} class_id={} source={} asset_path={:?} name={:?}",
                path_id,
                class_id,
                src,
                ap,
                name
            );
        }
        _ => {
            logger::debug!(
                "shader_unity_asset: Shader path_id={} class_id={} source={} name={:?}",
                path_id,
                class_id,
                src,
                name
            );
        }
    }
}

/// Per-file binary probe state for structured failure logs.
struct FileBinaryProbe {
    yaml_loaded: bool,
    bytes_len: usize,
    prefix_hex: String,
    prefix_ascii: String,
    bundle_parse_ok: bool,
    bundle_assets: usize,
    bundle_err: Option<String>,
    serialized_parse_ok: bool,
    serialized_object_count: usize,
    serialized_shader_count: usize,
    serialized_unity_version: Option<String>,
    serialized_format_version: Option<u32>,
    serialized_enable_type_tree: Option<bool>,
    serialized_err: Option<String>,
}

impl FileBinaryProbe {
    fn new(yaml_loaded: bool, bytes: &[u8]) -> Self {
        Self {
            yaml_loaded,
            bytes_len: bytes.len(),
            prefix_hex: format_hex_prefix(bytes, 24),
            prefix_ascii: ascii_prefix_hint(bytes, 40),
            bundle_parse_ok: false,
            bundle_assets: 0,
            bundle_err: None,
            serialized_parse_ok: false,
            serialized_object_count: 0,
            serialized_shader_count: 0,
            serialized_unity_version: None,
            serialized_format_version: None,
            serialized_enable_type_tree: None,
            serialized_err: None,
        }
    }

    /// One short [`logger::warn!`] line; full fields via [`Self::log_debug_detail`].
    fn warn_short(&self, path: &Path, reason: &str) {
        logger::warn!(
            "shader_unity_asset: {:?} — {} | yaml={} bytes={} hex8={} | bundle_ok={} ser_ok={} | err {:?} / {:?}",
            path.display(),
            reason,
            self.yaml_loaded,
            self.bytes_len,
            short_hex_prefix(&self.prefix_hex, PROBE_HEX_SHORT),
            self.bundle_parse_ok,
            self.serialized_parse_ok,
            self.bundle_err.as_deref().unwrap_or(""),
            self.serialized_err.as_deref().unwrap_or("")
        );
    }

    fn log_debug_detail(&self) {
        logger::debug!(
            "shader_unity_asset: probe detail yaml_loaded={} bytes={} prefix_hex={} prefix_ascii={:?} bundle_ok={} bundle_assets={} bundle_err={:?} serialized_ok={} objects={} shader_objects={} unity_ver={:?} format_ver={:?} type_tree={:?} serialized_err={:?}",
            self.yaml_loaded,
            self.bytes_len,
            self.prefix_hex,
            self.prefix_ascii,
            self.bundle_parse_ok,
            self.bundle_assets,
            self.bundle_err,
            self.serialized_parse_ok,
            self.serialized_object_count,
            self.serialized_shader_count,
            self.serialized_unity_version,
            self.serialized_format_version,
            self.serialized_enable_type_tree,
            self.serialized_err
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
    format!("{}…", &s[..max.saturating_sub(1)])
}

fn shader_object_count(sf: &SerializedFile) -> usize {
    sf.object_handles()
        .filter(|h| h.class_id() == SHADER)
        .count()
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
        .filter_map(|e| e.ok())
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
    let mut serialized_parse_hits = 0usize;
    let mut shader_objects_sum = 0usize;
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
            if probe.serialized_parse_ok {
                serialized_parse_hits += 1;
                shader_objects_sum += probe.serialized_shader_count;
            }
            if first_probe.is_none() {
                first_probe = Some(probe);
            }
        }
    }

    logger::warn!(
        "shader_unity_asset: directory {:?} — no shader name (files_total={} examined={} cap={} bundle_hits={} ser_hits={} shader_objs_sum={})",
        dir.display(),
        files_total,
        examined,
        MAX_DIR_FILES,
        bundle_parse_hits,
        serialized_parse_hits,
        shader_objects_sum
    );
    if let Some(ref fp) = first_probe {
        logger::debug!("shader_unity_asset: first failed file probe sample");
        fp.log_debug_detail();
    }

    None
}

fn shader_name_from_yaml(doc: &YamlDocument) -> Option<String> {
    for c in doc.filter_by_class(SHADER_CLASS) {
        if let Some(name) = shader_name_from_unity_class(c) {
            return Some(name);
        }
    }
    None
}

fn shader_name_from_unity_class(c: &UnityClass) -> Option<String> {
    for key in ["m_ParsedForm", "m_Script"] {
        if let Some(v) = c.get(key) {
            let text = unity_value_searchable_text(v);
            if let Some(n) = find_shader_lab_name_in_text(&text) {
                return Some(n);
            }
        }
    }
    c.name()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(std::string::ToString::to_string)
}

fn unity_value_searchable_text(v: &UnityValue) -> String {
    match v {
        UnityValue::Null => String::new(),
        UnityValue::Bool(b) => b.to_string(),
        UnityValue::Integer(i) => i.to_string(),
        UnityValue::Float(f) => f.to_string(),
        UnityValue::String(s) => s.clone(),
        UnityValue::Array(a) => a
            .iter()
            .map(unity_value_searchable_text)
            .collect::<Vec<_>>()
            .join(" "),
        UnityValue::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
        UnityValue::Object(map) => map
            .values()
            .map(unity_value_searchable_text)
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn find_shader_lab_name_in_text(s: &str) -> Option<String> {
    let idx = s.find("Shader \"")?;
    let tail = s.get(idx..)?;
    let window: String = tail.chars().take(4096).collect();
    parse_shader_lab_quoted_name(&window)
}

fn find_shader_lab_name_in_bytes(data: &[u8]) -> Option<String> {
    let s = String::from_utf8_lossy(data);
    find_shader_lab_name_in_text(&s)
}

/// Shader logical name from a fully parsed [`UnityObject`]: `m_Name` / `name`, then ShaderLab in
/// `m_ParsedForm` / `m_Script` (same strategy as [`shader_name_from_unity_class`] for YAML).
fn shader_name_from_loaded_unity_object(obj: &UnityObject) -> Option<String> {
    let from_named = obj
        .name()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            obj.get("name")
                .and_then(UnityValue::as_str)
                .map(std::string::ToString::to_string)
        })
        .filter(|s| !s.trim().is_empty());
    if let Some(n) = from_named {
        return Some(n);
    }
    for key in ["m_ParsedForm", "m_Script"] {
        if let Some(v) = obj.get(key) {
            let text = unity_value_searchable_text(v);
            if let Some(n) = find_shader_lab_name_in_text(&text) {
                return Some(n);
            }
        }
    }
    None
}

/// Best-effort name from [`Environment::bundle_container_entries`] by matching Shader `path_id` to
/// `AssetBundle.m_Container`.
fn shader_name_from_bundle_container_fallback(
    bundle_path: &Path,
    bundle: &AssetBundle,
) -> Option<String> {
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
        if let Some(entry) = entries.iter().find(|e| e.path_id == pid) {
            if let Some(name) = shader_logical_name_from_container_asset_path(&entry.asset_path) {
                log_resolution_debug(
                    pid,
                    SHADER,
                    ResolutionSource::Container,
                    &name,
                    Some(&entry.asset_path),
                );
                return Some(name);
            }
        }
    }
    None
}

/// Derives a shader stem from a Unity `m_Container` asset path (e.g. `.../ui_unlit.shader` → `ui_unlit`).
fn shader_logical_name_from_container_asset_path(asset_path: &str) -> Option<String> {
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

/// [`ObjectHandle::peek_name`], full [`ObjectHandle::read`], then ShaderLab bytes scan.
fn shader_name_from_serialized_file(sf: &SerializedFile) -> Option<String> {
    for handle in sf.object_handles() {
        if handle.class_id() != SHADER {
            continue;
        }
        let pid = handle.path_id();
        let cid = handle.class_id();
        match handle.peek_name() {
            Ok(Some(name)) if !name.trim().is_empty() => {
                log_resolution_debug(pid, cid, ResolutionSource::MNamePeek, &name, None);
                return Some(name);
            }
            Ok(Some(_)) => {}
            Ok(None) => {
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} peek_name None; typetree read",
                    pid
                );
            }
            Err(e) => {
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} peek_name err {}; typetree read",
                    pid,
                    e
                );
            }
        }

        match handle.read() {
            Ok(obj) => {
                if let Some(name) = shader_name_from_loaded_unity_object(&obj) {
                    log_resolution_debug(
                        pid,
                        cid,
                        ResolutionSource::UnityObjectTypetree,
                        &name,
                        None,
                    );
                    return Some(name);
                }
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} typetree ok; keys_sample={:?}",
                    pid,
                    obj.property_names().iter().take(24).collect::<Vec<_>>()
                );
            }
            Err(e) => {
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} ObjectHandle::read failed: {}",
                    pid,
                    e
                );
            }
        }

        let bytes = match handle.raw_data() {
            Ok(b) => b,
            Err(e) => {
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} raw_data failed: {}",
                    pid,
                    e
                );
                continue;
            }
        };
        if let Some(name) = find_shader_lab_name_in_bytes(bytes) {
            log_resolution_debug(pid, cid, ResolutionSource::ShaderLabBytes, &name, None);
            return Some(name);
        }
    }
    None
}

/// Prefer `m_Container` path stems (typical for UnityFS cache), then TypeTree / ShaderLab per asset.
fn shader_name_from_bundle(bundle_path: &Path, bundle: &AssetBundle) -> Option<String> {
    if let Some(name) = shader_name_from_bundle_container_fallback(bundle_path, bundle) {
        return Some(name);
    }
    for asset in &bundle.assets {
        if let Some(name) = shader_name_from_serialized_file(asset) {
            return Some(name);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn container_asset_path_strips_shader_suffix() {
        assert_eq!(
            shader_logical_name_from_container_asset_path("assets/foo/ui_unlit.shader").as_deref(),
            Some("ui_unlit")
        );
        assert_eq!(
            shader_logical_name_from_container_asset_path("archive:/CAB-deadbeef").as_deref(),
            None
        );
    }

    #[test]
    fn yaml_shader_extracts_quoted_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("TestShader.asset");
        let mut f = std::fs::File::create(&path).expect("create");
        let yaml = r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!48 &4800000
Shader:
  m_ObjectHideFlags: 0
  m_Name:
  m_ParsedForm: "Shader \"Custom/TestFromYaml\"
{
}
"
"#;
        f.write_all(yaml.as_bytes()).expect("write");

        let name = try_from_file(&path).expect("resolve");
        assert_eq!(name, "Custom/TestFromYaml");
    }
}
