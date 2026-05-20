//! Resolve shader asset names from on-disk **Unity AssetBundle** files using `unity-asset`.
//!
//! [`crate::shared::ShaderUpload::file`] is typically an **extensionless path** (or any path) whose bytes
//! parse as UnityFS / AssetBundle data--not a Unity `.asset` YAML file. Route selection still prefers
//! [`unity_asset::environment::Environment::bundle_container_entries`]: `AssetBundle.m_Container`
//! asset paths matched to embedded Shader objects, then lowercased and stemmed
//! (e.g. `.../UI_Unlit.shader` -> `ui_unlit`).
//!
//! Serialized shader objects are also read for the top-level ShaderLab name so Froox variant
//! suffixes (`{shader_name}_{variant_bits:08X}`) can be stripped and carried as metadata.

use std::path::Path;

use unity_asset::AssetBundle;
use unity_asset::SerializedFile;
use unity_asset::UnityValue;
use unity_asset::class_ids::SHADER;
use unity_asset::environment::BinarySource;
use unity_asset::environment::Environment;
use unity_asset::load_bundle_from_memory;

mod container;
mod directory;
mod probe;

use container::{container_name_for_path_id, shader_container_names_from_bundle};
use directory::try_from_directory;
use probe::{FileBinaryProbe, truncate_display};

#[cfg(test)]
use container::shader_asset_name_from_container_asset_path;
#[cfg(test)]
use probe::{ascii_prefix_hint, format_hex_prefix, short_hex_prefix};

/// Maximum file size to read when probing a bundle.
const MAX_READ_BYTES: usize = 32 * 1024 * 1024;

/// Maximum regular files examined under a directory hint (dev / loose layouts).
const MAX_DIR_FILES: usize = 256;

/// Maximum characters from parse errors included in logs.
const MAX_ERR_LOG_CHARS: usize = 240;

/// Shader asset route metadata resolved from an uploaded Unity shader AssetBundle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedUnityShaderAsset {
    /// Lowercase shader asset filename stem used for route selection.
    pub shader_asset_name: String,
    /// Froox shader variant bitmask parsed from the internal Shader name suffix, when present.
    pub shader_variant_bits: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InternalShaderName {
    full_name: String,
    shader_asset_name: String,
    shader_variant_bits: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShaderObjectCandidate {
    path_id: i64,
    class_id: i32,
    container_name: Option<String>,
    internal_name: Option<InternalShaderName>,
    internal_source: Option<InternalShaderNameSource>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InternalShaderNameSource {
    TypeTreeShaderLab,
    RawShaderLab,
    ParsedFormName,
}

impl InternalShaderNameSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::TypeTreeShaderLab => "typetree_shaderlab",
            Self::RawShaderLab => "raw_shaderlab",
            Self::ParsedFormName => "typetree_m_ParsedForm",
        }
    }
}

/// Shader asset filename or stem plus optional Froox variant bitmask from a filesystem path.
pub(crate) fn try_resolve_shader_asset_name_from_path(
    path: &Path,
) -> Option<ResolvedUnityShaderAsset> {
    let meta = std::fs::metadata(path).ok()?;
    let resolved = if meta.is_file() {
        try_from_file(path)
    } else if meta.is_dir() {
        try_from_directory(path)
    } else {
        None
    };
    if let Some(parsed) = &resolved {
        logger::info!(
            "shader_unity_asset: resolved shader_asset_name={:?} shader_variant_bits={} from path {}",
            parsed.shader_asset_name,
            super::shader_variant_bits_log(parsed.shader_variant_bits),
            path.display()
        );
    }
    resolved
}

fn try_from_file(path: &Path) -> Option<ResolvedUnityShaderAsset> {
    try_from_file_inner(path, true).0
}

/// When `log_failure` is `false` (directory scan), probe data is returned without per-file [`logger::warn!`].
fn try_from_file_inner(
    path: &Path,
    log_failure: bool,
) -> (Option<ResolvedUnityShaderAsset>, Option<FileBinaryProbe>) {
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
        if let Some(resolved) = shader_resolution_from_bundle(path, bundle) {
            return (Some(resolved), None);
        }
        if log_failure {
            probe.warn_short(path, "AssetBundle: no shader name");
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

fn log_internal_name_resolution(
    path_id: i64,
    class_id: i32,
    source: InternalShaderNameSource,
    name: &InternalShaderName,
) {
    logger::info!(
        "shader_unity_asset: Shader path_id={} class_id={} source={} full_name={:?} stem={:?} variant_bits={}",
        path_id,
        class_id,
        source.as_str(),
        name.full_name,
        name.shader_asset_name,
        super::shader_variant_bits_log(name.shader_variant_bits)
    );
}

fn shader_resolution_from_bundle(
    bundle_path: &Path,
    bundle: &AssetBundle,
) -> Option<ResolvedUnityShaderAsset> {
    let container_names = shader_container_names_from_bundle(bundle_path, bundle);
    let candidates = shader_candidates_from_bundle(bundle, &container_names);
    shader_resolution_from_candidates(&candidates)
}

fn shader_resolution_from_candidates(
    candidates: &[ShaderObjectCandidate],
) -> Option<ResolvedUnityShaderAsset> {
    let shader_asset_name = shader_asset_name_from_candidates(candidates)?;
    let variant_candidate = shader_variant_candidate(candidates);
    log_shader_candidate_selection(&shader_asset_name, variant_candidate, candidates);
    Some(ResolvedUnityShaderAsset {
        shader_asset_name,
        shader_variant_bits: variant_candidate
            .and_then(|candidate| candidate.internal_name.as_ref())
            .and_then(|name| name.shader_variant_bits),
    })
}

fn shader_asset_name_from_candidates(candidates: &[ShaderObjectCandidate]) -> Option<String> {
    candidates
        .iter()
        .find_map(|candidate| candidate.container_name.clone())
}

fn shader_variant_candidate(
    candidates: &[ShaderObjectCandidate],
) -> Option<&ShaderObjectCandidate> {
    candidates.iter().find(|candidate| {
        candidate
            .internal_name
            .as_ref()
            .is_some_and(non_fallback_variant_internal_name)
    })
}

fn non_fallback_variant_internal_name(name: &InternalShaderName) -> bool {
    name.shader_variant_bits.is_some() && !is_fallback_internal_shader_name(&name.full_name)
}

fn is_fallback_internal_shader_name(full_name: &str) -> bool {
    full_name
        .trim_start()
        .to_ascii_lowercase()
        .starts_with("legacy shaders/")
}

fn shader_candidate_skip_reason(candidate: &ShaderObjectCandidate) -> &'static str {
    let Some(name) = &candidate.internal_name else {
        return "no internal name";
    };
    if name.shader_variant_bits.is_none() {
        return "no variant bits";
    }
    if is_fallback_internal_shader_name(&name.full_name) {
        return "fallback internal name";
    }
    "variant not selected"
}

fn log_shader_candidate_selection(
    shader_asset_name: &str,
    selected_variant: Option<&ShaderObjectCandidate>,
    candidates: &[ShaderObjectCandidate],
) {
    if let Some(candidate) = selected_variant
        && let (Some(source), Some(name)) = (candidate.internal_source, &candidate.internal_name)
    {
        log_internal_name_resolution(candidate.path_id, candidate.class_id, source, name);
    }

    for candidate in candidates {
        if selected_variant.is_some_and(|selected| selected.path_id == candidate.path_id) {
            continue;
        }
        let (full_name, variant_bits, source) =
            candidate
                .internal_name
                .as_ref()
                .map_or((None, None, None), |name| {
                    (
                        Some(name.full_name.as_str()),
                        name.shader_variant_bits,
                        candidate.internal_source,
                    )
                });
        logger::debug!(
            "shader_unity_asset: skipped Shader path_id={} class_id={} reason={} route={:?} container_name={:?} source={} full_name={:?} variant_bits={}",
            candidate.path_id,
            candidate.class_id,
            shader_candidate_skip_reason(candidate),
            shader_asset_name,
            candidate.container_name,
            source.map_or("none", InternalShaderNameSource::as_str),
            full_name,
            super::shader_variant_bits_log(variant_bits)
        );
    }
}

fn shader_candidates_from_bundle(
    bundle: &AssetBundle,
    container_names: &[(i64, String)],
) -> Vec<ShaderObjectCandidate> {
    let mut candidates = Vec::new();
    for asset in &bundle.assets {
        shader_candidates_from_serialized_file(asset, container_names, &mut candidates);
    }
    candidates
}

fn shader_candidates_from_serialized_file(
    sf: &SerializedFile,
    container_names: &[(i64, String)],
    candidates: &mut Vec<ShaderObjectCandidate>,
) {
    for handle in sf.object_handles() {
        if handle.class_id() != SHADER {
            continue;
        }
        let path_id = handle.path_id();
        let class_id = handle.class_id();
        let mut parsed_form_name = None;
        let mut keys_sample = Vec::new();
        let mut internal_name = None;
        let mut internal_source = None;

        match handle.read() {
            Ok(obj) => {
                if let Some(parsed) = shader_lab_internal_name_from_loaded_unity_object(&obj) {
                    internal_name = Some(parsed);
                    internal_source = Some(InternalShaderNameSource::TypeTreeShaderLab);
                }
                parsed_form_name = shader_internal_name_from_loaded_unity_object(&obj);
                keys_sample = obj
                    .property_names()
                    .iter()
                    .take(24)
                    .map(|key| (*key).clone())
                    .collect::<Vec<_>>();
            }
            Err(e) => {
                logger::debug!(
                    "shader_unity_asset: Shader path_id={} ObjectHandle::read failed: {}",
                    path_id,
                    e
                );
            }
        }

        if internal_name.is_none() {
            match handle.raw_data() {
                Ok(raw) => {
                    if let Some(parsed) = shader_lab_internal_name_from_bytes(raw) {
                        internal_name = Some(parsed);
                        internal_source = Some(InternalShaderNameSource::RawShaderLab);
                    }
                }
                Err(e) => {
                    logger::debug!(
                        "shader_unity_asset: Shader path_id={} ObjectHandle::raw_data failed: {}",
                        path_id,
                        e
                    );
                }
            }
        }

        if internal_name.is_none()
            && let Some(parsed) = parsed_form_name
        {
            internal_name = Some(parsed);
            internal_source = Some(InternalShaderNameSource::ParsedFormName);
        }

        if internal_name.is_none() {
            logger::debug!(
                "shader_unity_asset: Shader path_id={} typetree ok; no ShaderLab declaration or m_ParsedForm.m_Name; keys_sample={:?}",
                path_id,
                keys_sample
            );
        }

        candidates.push(ShaderObjectCandidate {
            path_id,
            class_id,
            container_name: container_name_for_path_id(container_names, path_id),
            internal_name,
            internal_source,
        });
    }
}

fn shader_lab_internal_name_from_loaded_unity_object(
    obj: &unity_asset_binary::object::UnityObject,
) -> Option<InternalShaderName> {
    for key in [
        "m_ParsedForm",
        "m_Script",
        "m_SerializedShader",
        "m_SubProgramBlob",
    ] {
        if let Some(parsed) = obj
            .get(key)
            .and_then(shader_lab_internal_name_from_unity_value)
        {
            return Some(parsed);
        }
    }

    obj.property_names().into_iter().find_map(|key| {
        obj.get(key)
            .and_then(shader_lab_internal_name_from_unity_value)
    })
}

fn shader_internal_name_from_loaded_unity_object(
    obj: &unity_asset_binary::object::UnityObject,
) -> Option<InternalShaderName> {
    obj.get("m_ParsedForm")
        .and_then(parsed_form_internal_shader_name)
}

fn parsed_form_internal_shader_name(value: &UnityValue) -> Option<InternalShaderName> {
    let UnityValue::Object(fields) = value else {
        return None;
    };
    ["m_Name", "name"].into_iter().find_map(|key| {
        fields
            .get(key)
            .and_then(UnityValue::as_str)
            .filter(|name| !name.trim().is_empty())
            .and_then(parse_internal_shader_name)
    })
}

fn shader_lab_internal_name_from_unity_value(value: &UnityValue) -> Option<InternalShaderName> {
    match value {
        UnityValue::String(text) => parse_shader_lab_internal_name(text),
        UnityValue::Bytes(bytes) => shader_lab_internal_name_from_bytes(bytes),
        UnityValue::Array(values) => values
            .iter()
            .find_map(shader_lab_internal_name_from_unity_value),
        UnityValue::Object(fields) => fields
            .values()
            .find_map(shader_lab_internal_name_from_unity_value),
        UnityValue::Null | UnityValue::Bool(_) | UnityValue::Integer(_) | UnityValue::Float(_) => {
            None
        }
    }
}

fn shader_lab_internal_name_from_bytes(bytes: &[u8]) -> Option<InternalShaderName> {
    if bytes.is_empty() {
        return None;
    }
    let text = String::from_utf8_lossy(bytes);
    parse_shader_lab_internal_name(text.as_ref())
}

fn parse_shader_lab_internal_name(text: &str) -> Option<InternalShaderName> {
    shader_lab_declared_name(text).and_then(|name| parse_internal_shader_name(&name))
}

fn shader_lab_declared_name(text: &str) -> Option<String> {
    const SHADER_KEYWORD: &[u8] = b"Shader";

    let bytes = text.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'/' if bytes.get(index + 1) == Some(&b'/') => {
                index = skip_shader_lab_line_comment(bytes, index);
                continue;
            }
            b'/' if bytes.get(index + 1) == Some(&b'*') => {
                index = skip_shader_lab_block_comment(bytes, index);
                continue;
            }
            b'"' => {
                index = skip_shader_lab_string_literal(bytes, index);
                continue;
            }
            b'S' if bytes[index..].starts_with(SHADER_KEYWORD) => {
                let after_keyword = index + SHADER_KEYWORD.len();
                let previous_is_boundary = index == 0
                    || !bytes
                        .get(index - 1)
                        .is_some_and(|byte| shader_lab_identifier_byte(*byte));
                let next_is_whitespace = bytes
                    .get(after_keyword)
                    .is_some_and(u8::is_ascii_whitespace);
                if previous_is_boundary && next_is_whitespace {
                    let quote_index = skip_shader_lab_whitespace(bytes, after_keyword);
                    if bytes.get(quote_index) == Some(&b'"')
                        && let Some(name) = shader_lab_quoted_string(text, quote_index)
                        && !name.trim().is_empty()
                    {
                        return Some(name);
                    }
                }
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn shader_lab_identifier_byte(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

fn skip_shader_lab_whitespace(bytes: &[u8], mut index: usize) -> usize {
    while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
        index += 1;
    }
    index
}

fn skip_shader_lab_line_comment(bytes: &[u8], mut index: usize) -> usize {
    index += 2;
    while index < bytes.len() && bytes[index] != b'\n' {
        index += 1;
    }
    index
}

fn skip_shader_lab_block_comment(bytes: &[u8], mut index: usize) -> usize {
    index += 2;
    while index + 1 < bytes.len() {
        if bytes[index] == b'*' && bytes[index + 1] == b'/' {
            return index + 2;
        }
        index += 1;
    }
    bytes.len()
}

fn skip_shader_lab_string_literal(bytes: &[u8], mut index: usize) -> usize {
    index += 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index = (index + 2).min(bytes.len()),
            b'"' => return index + 1,
            _ => index += 1,
        }
    }
    index
}

fn shader_lab_quoted_string(text: &str, quote_index: usize) -> Option<String> {
    let content_start = quote_index + 1;
    let mut escaped = false;
    for (offset, ch) in text[content_start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => {
                let content_end = content_start + offset;
                return Some(unescape_shader_lab_quoted_name(
                    &text[content_start..content_end],
                ));
            }
            _ => {}
        }
    }
    None
}

fn unescape_shader_lab_quoted_name(name: &str) -> String {
    let mut unescaped = String::with_capacity(name.len());
    let mut chars = name.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            unescaped.push(ch);
            continue;
        }
        match chars.next() {
            Some(next @ ('"' | '\\')) => unescaped.push(next),
            Some(next) => {
                unescaped.push(ch);
                unescaped.push(next);
            }
            None => unescaped.push(ch),
        }
    }
    unescaped
}

fn parse_internal_shader_name(name: &str) -> Option<InternalShaderName> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (stem, shader_variant_bits) =
        split_variant_suffix(trimmed).map_or((trimmed, None), |(stem, bits)| (stem, Some(bits)));
    let shader_asset_name = shader_asset_stem_from_internal_name(stem)?;
    Some(InternalShaderName {
        full_name: trimmed.to_string(),
        shader_asset_name,
        shader_variant_bits,
    })
}

fn split_variant_suffix(name: &str) -> Option<(&str, u32)> {
    let (stem, suffix) = name.rsplit_once('_')?;
    if stem.trim().is_empty() || suffix.len() != 8 || !suffix.chars().all(|c| c.is_ascii_hexdigit())
    {
        return None;
    }
    u32::from_str_radix(suffix, 16)
        .ok()
        .map(|bits| (stem, bits))
}

fn shader_asset_stem_from_internal_name(name: &str) -> Option<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return None;
    }
    let base = trimmed
        .rsplit('/')
        .next()
        .and_then(|segment| segment.rsplit('\\').next())
        .unwrap_or(trimmed)
        .trim();
    if base.is_empty() {
        return None;
    }
    Some(base.to_string())
}

#[cfg(test)]
mod tests;
