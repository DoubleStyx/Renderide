//! Unity ShaderLab logical names (`Shader "…"`), WGSL `unity-shader-name` banners, and plain logical stems
//! for shader uploads (e.g. from a host mod that sets [`ShaderUpload::file`](ShaderUpload::file) to `UI_Unlit`).
//!
//! FrooxEngine sends a sequential [`crate::shared::ShaderUpload::asset_id`] and a `file` path. For an optional
//! host-appended logical name after the stock payload, see [`crate::shared::shader_upload_extras::unpack_appended_shader_logical_name`]
//! and [`resolve_logical_shader_name_from_upload_with_host_hint`].
//!
//! When `file` is a filesystem path to an on-disk **Unity AssetBundle** (often extensionless) or a directory
//! scanned for bundles, [`super::unity_asset`] resolves the shader stem from **`m_Container` only** (not ShaderLab inside serialized shaders).

use crate::shared::ShaderUpload;

/// Maximum length for a single logical stem or first-token label from [`try_resolve_plain_shader_label`].
const PLAIN_SHADER_LABEL_MAX_LEN: usize = 256;

/// Returns the first token of `resolved` trimmed, or empty. Routing matches host names to embedded WGSL via
/// [`crate::materials::embedded_shader_stem`]; material source stems under `shaders/source/materials/*.wgsl` align with
/// normalized Unity shader **asset** names (see crate `build.rs`).
pub fn canonical_shader_lab_logical_name(resolved: &str) -> String {
    let t = resolved
        .split_whitespace()
        .next()
        .unwrap_or(resolved)
        .trim();
    if t.is_empty() {
        return String::new();
    }
    t.to_string()
}

/// Parses the quoted name from a ShaderLab `Shader "Name"` opening line or small file prelude.
pub fn parse_shader_lab_quoted_name(source: &str) -> Option<String> {
    let s = source.trim_start_matches('\u{feff}').trim_start();
    let rest = s.strip_prefix("Shader")?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    let name = rest[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Reads `// unity-shader-name: …` from the first lines of WGSL (or embedded text).
pub fn parse_wgsl_unity_shader_name_banner(source: &str) -> Option<String> {
    for line in source.lines().take(64) {
        let t = line.trim();
        let Some(after_comment) = t.strip_prefix("//") else {
            continue;
        };
        let after_slash = after_comment.trim();
        let Some(name_part) = after_slash.strip_prefix("unity-shader-name:") else {
            continue;
        };
        let name_part = name_part.trim();
        if !name_part.is_empty() {
            return Some(name_part.to_string());
        }
    }
    None
}

fn looks_like_shader_lab_inline(s: &str) -> bool {
    let t = s.trim_start_matches('\u{feff}').trim_start();
    t.starts_with("Shader \"") || t.starts_with("Shader\"")
}

fn looks_like_wgsl_with_banner(s: &str) -> bool {
    s.lines()
        .take(48)
        .any(|line| line.trim().starts_with("// unity-shader-name:"))
}

/// Returns a logical Unity-style shader name when `file_field` is a short host-provided stem (e.g. `UI_Unlit`)
/// or `stem kw1` (first token only), not ShaderLab source or a filesystem path.
///
/// Multi-line payloads with non-empty lines after the first are rejected so full ShaderLab/WGSL resolution can run.
pub fn try_resolve_plain_shader_label(file_field: &str) -> Option<String> {
    let trimmed = file_field.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut lines = trimmed.lines();
    let first_line = lines.next()?;
    if lines.any(|l| !l.trim().is_empty()) {
        return None;
    }
    let mut token = first_line.split_whitespace().next()?;
    if let Some(rest) = token.strip_prefix("renderide:") {
        token = rest.trim();
    }
    if token.is_empty() {
        return None;
    }
    if token == "." || token == ".." {
        return None;
    }
    if token.len() > PLAIN_SHADER_LABEL_MAX_LEN {
        return None;
    }
    if token.starts_with("\\\\") || token.starts_with('\\') {
        return None;
    }
    if token.contains(":\\") {
        return None;
    }
    if token.starts_with('/') {
        return None;
    }
    if !token
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '/' || c == '-' || c == '.')
    {
        return None;
    }
    Some(token.to_string())
}

const UPLOAD_SOURCE_READ_CAP_BYTES: usize = 262_144;

/// Resolves the logical Unity shader name from [`ShaderUpload::file`](ShaderUpload::file): path, inline text, or disk read.
///
/// For **material routing** and embedded stem matching, prefer [`resolve_shader_routing_name_from_upload`],
/// which favors raw AssetBundle / container stems from filesystem paths before ShaderLab first-token
/// canonicalization.
pub fn resolve_logical_shader_name_from_upload(data: &ShaderUpload) -> Option<String> {
    resolve_logical_shader_name_from_upload_with_host_hint(data, None)
}

/// Shader identifier for **routing**, embedded `{key}_default` stem probes, and registry display when
/// no separate label is needed.
///
/// Resolution order:
/// 1. Non-empty `host_hint` (trimmed).
/// 2. Raw name from [`super::unity_asset::try_resolve_shader_name_from_path_hint_raw`] when `file` is an
///    existing filesystem path to an AssetBundle file or directory (container path stem from `m_Container` only).
/// 3. Otherwise same sources as [`resolve_logical_shader_name_from_upload_inner`], then
///    [`canonical_shader_lab_logical_name`] on the result.
pub fn resolve_shader_routing_name_from_upload(
    data: &ShaderUpload,
    host_hint: Option<&str>,
) -> Option<String> {
    if let Some(h) = host_hint {
        let t = h.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    let file_field = data.file.as_deref()?;
    let path = std::path::Path::new(file_field);
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.is_file() || meta.is_dir() {
            if let Some(name) = super::unity_asset::try_resolve_shader_name_from_path_hint_raw(path)
            {
                return Some(name);
            }
        }
    }
    let raw = resolve_logical_shader_name_from_upload_inner(data, None)?;
    Some(canonical_shader_lab_logical_name(&raw))
}

/// Like [`resolve_logical_shader_name_from_upload`], but uses `host_hint` first when set (e.g. from
/// [`crate::shared::shader_upload_extras::unpack_appended_shader_logical_name`]).
///
/// The returned name is passed through [`canonical_shader_lab_logical_name`] (first token only) before routing.
pub fn resolve_logical_shader_name_from_upload_with_host_hint(
    data: &ShaderUpload,
    host_hint: Option<&str>,
) -> Option<String> {
    let raw = resolve_logical_shader_name_from_upload_inner(data, host_hint)?;
    Some(canonical_shader_lab_logical_name(&raw))
}

fn resolve_logical_shader_name_from_upload_inner(
    data: &ShaderUpload,
    host_hint: Option<&str>,
) -> Option<String> {
    if let Some(h) = host_hint {
        let t = h.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    let file_field = data.file.as_deref()?;
    if looks_like_shader_lab_inline(file_field) {
        return parse_shader_lab_quoted_name(file_field)
            .or_else(|| parse_wgsl_unity_shader_name_banner(file_field));
    }
    if looks_like_wgsl_with_banner(file_field) {
        return parse_wgsl_unity_shader_name_banner(file_field);
    }
    if let Some(stem) = try_resolve_plain_shader_label(file_field) {
        return Some(stem);
    }
    if file_field.len() < UPLOAD_SOURCE_READ_CAP_BYTES {
        if let Some(from_lab) = parse_shader_lab_quoted_name(file_field) {
            return Some(from_lab);
        }
        if let Some(from_wgsl) = parse_wgsl_unity_shader_name_banner(file_field) {
            return Some(from_wgsl);
        }
    }
    let path = std::path::Path::new(file_field);
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.is_file() || meta.is_dir() {
            if let Some(name) = super::unity_asset::try_resolve_shader_name_from_path_hint(path) {
                return Some(name);
            }
        }
    }
    if std::path::Path::new(file_field).is_file() {
        match std::fs::read_to_string(file_field) {
            Ok(contents) if contents.len() <= UPLOAD_SOURCE_READ_CAP_BYTES => {
                return parse_shader_lab_quoted_name(&contents)
                    .or_else(|| parse_wgsl_unity_shader_name_banner(&contents));
            }
            Ok(_) | Err(_) => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_resolve_plain_rejects_absolute_path_like() {
        assert_eq!(
            try_resolve_plain_shader_label("/etc/shaders/x.shader"),
            None
        );
        assert_eq!(try_resolve_plain_shader_label("C:\\a\\b.shader"), None);
    }

    #[test]
    fn try_resolve_plain_accepts_stem_and_first_token() {
        assert_eq!(
            try_resolve_plain_shader_label("UI_Unlit"),
            Some("UI_Unlit".to_string())
        );
        assert_eq!(
            try_resolve_plain_shader_label("UI_Unlit kw1 kw2"),
            Some("UI_Unlit".to_string())
        );
        assert_eq!(
            try_resolve_plain_shader_label("  Foo-Bar/baz.v2  "),
            Some("Foo-Bar/baz.v2".to_string())
        );
    }

    #[test]
    fn try_resolve_plain_strips_renderide_prefix() {
        assert_eq!(
            try_resolve_plain_shader_label("renderide:MyShader"),
            Some("MyShader".to_string())
        );
    }

    #[test]
    fn try_resolve_plain_rejects_multi_line_and_bad_tokens() {
        assert_eq!(try_resolve_plain_shader_label(""), None);
        assert_eq!(try_resolve_plain_shader_label("."), None);
        assert_eq!(try_resolve_plain_shader_label(".."), None);
        assert_eq!(try_resolve_plain_shader_label("Shader \"X\"\n{\n}\n"), None);
        assert_eq!(try_resolve_plain_shader_label("bad!char"), None);
        let too_long = "a".repeat(PLAIN_SHADER_LABEL_MAX_LEN + 1);
        assert_eq!(try_resolve_plain_shader_label(&too_long), None);
    }

    #[test]
    fn parse_shader_lab_quoted_name_extracts_name() {
        assert_eq!(
            parse_shader_lab_quoted_name("Shader \"Custom/UI_Unlit\""),
            Some("Custom/UI_Unlit".to_string())
        );
        assert_eq!(
            parse_shader_lab_quoted_name("\u{feff}Shader \"Boxed\" { }"),
            Some("Boxed".to_string())
        );
    }

    #[test]
    fn parse_shader_lab_quoted_name_rejects_malformed() {
        assert_eq!(parse_shader_lab_quoted_name("Shader {}"), None);
        assert_eq!(parse_shader_lab_quoted_name("Shader \"\""), None);
        assert_eq!(parse_shader_lab_quoted_name("not shader source"), None);
    }

    #[test]
    fn parse_wgsl_banner_extracts_and_rejects() {
        let src = "// unity-shader-name: UI/TextUnlit\n@vertex fn vs() {}";
        assert_eq!(
            parse_wgsl_unity_shader_name_banner(src),
            Some("UI/TextUnlit".to_string())
        );
        assert_eq!(
            parse_wgsl_unity_shader_name_banner("// unrelated\n@vertex fn vs() {}"),
            None
        );
        assert_eq!(
            parse_wgsl_unity_shader_name_banner("// unity-shader-name:\n"),
            None
        );
    }

    #[test]
    fn canonical_takes_first_token_and_handles_empty() {
        assert_eq!(
            canonical_shader_lab_logical_name("Custom/UI_Unlit kw1 kw2"),
            "Custom/UI_Unlit"
        );
        assert_eq!(canonical_shader_lab_logical_name("   "), "");
        assert_eq!(canonical_shader_lab_logical_name("Solo"), "Solo");
    }
}
