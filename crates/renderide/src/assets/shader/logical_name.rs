//! Unity ShaderLab logical names (`Shader "…"`), WGSL `unity-shader-name` banners, and plain logical stems
//! for shader uploads (e.g. from a host mod that sets [`ShaderUpload::file`](ShaderUpload::file) to `UI_Unlit`).
//!
//! FrooxEngine sends a sequential [`crate::shared::ShaderUpload::asset_id`] and a `file` path. For an optional
//! host-appended logical name after the stock payload, see [`crate::shared::unpack_appended_shader_logical_name`]
//! and [`resolve_logical_shader_name_from_upload_with_host_hint`].
//!
//! When `file` is a filesystem path to a Unity YAML asset, AssetBundle, serialized file, or a directory
//! containing such files, [`super::shader_unity_asset`] tries to extract the ShaderLab name via the `unity-asset` crate.

use crate::shared::ShaderUpload;

/// Maximum length for a single logical stem or first-token label from [`try_resolve_plain_shader_label`].
const PLAIN_SHADER_LABEL_MAX_LEN: usize = 256;

/// Canonical name from Resonite `UI_Unlit.shader`: line `Shader "UI/Unlit"`.
pub const CANONICAL_UNITY_UI_UNLIT: &str = "UI/Unlit";

/// Canonical name: `UI_TextUnlit.shader` → `Shader "UI/Text/Unlit"`.
pub const CANONICAL_UNITY_UI_TEXT_UNLIT: &str = "UI/Text/Unlit";

/// Canonical name from Resonite `Common/Unlit.shader`: line `Shader "Unlit"`.
///
/// Distinct from [`CANONICAL_UNITY_UI_UNLIT`] (`UI/Unlit`); the world-mesh unlit WGSL path keys on this.
pub const CANONICAL_UNITY_WORLD_UNLIT: &str = "Unlit";

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
pub fn resolve_logical_shader_name_from_upload(data: &ShaderUpload) -> Option<String> {
    resolve_logical_shader_name_from_upload_with_host_hint(data, None)
}

/// Like [`resolve_logical_shader_name_from_upload`], but uses `host_hint` first when set (e.g. from
/// [`crate::shared::unpack_appended_shader_logical_name`]).
pub fn resolve_logical_shader_name_from_upload_with_host_hint(
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
    if let Ok(meta) = std::fs::metadata(path)
        && (meta.is_file() || meta.is_dir())
        && let Some(name) = super::shader_unity_asset::try_resolve_shader_name_from_path_hint(path)
    {
        return Some(name);
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
    fn parse_ui_unlit_shader_line() {
        let src = "Shader \"UI/Unlit\"\n{\n";
        assert_eq!(
            parse_shader_lab_quoted_name(src).as_deref(),
            Some(CANONICAL_UNITY_UI_UNLIT)
        );
    }

    #[test]
    fn parse_ui_text_unlit_shader_line() {
        let src = "Shader \"UI/Text/Unlit\"\r\n{";
        assert_eq!(
            parse_shader_lab_quoted_name(src).as_deref(),
            Some(CANONICAL_UNITY_UI_TEXT_UNLIT)
        );
    }

    #[test]
    fn parse_wgsl_banner() {
        let wgsl = "// unity-shader-name: UI/Unlit\nfn vs() {}\n";
        assert_eq!(
            parse_wgsl_unity_shader_name_banner(wgsl).as_deref(),
            Some(CANONICAL_UNITY_UI_UNLIT)
        );
    }

    #[test]
    fn resolve_from_host_hint_without_file() {
        let u = ShaderUpload::default();
        assert_eq!(
            resolve_logical_shader_name_from_upload_with_host_hint(&u, Some("UI/Text/Unlit"))
                .as_deref(),
            Some(CANONICAL_UNITY_UI_TEXT_UNLIT)
        );
    }

    #[test]
    fn resolve_from_inline_shader_lab_in_file_field() {
        let u = ShaderUpload {
            file: Some("Shader \"UI/Unlit\"\n{\n".to_string()),
            ..Default::default()
        };
        assert_eq!(
            resolve_logical_shader_name_from_upload(&u).as_deref(),
            Some(CANONICAL_UNITY_UI_UNLIT)
        );
    }

    #[test]
    fn resolve_from_plain_stem_ui_unlit() {
        let u = ShaderUpload {
            file: Some("UI_Unlit".to_string()),
            ..Default::default()
        };
        assert_eq!(
            resolve_logical_shader_name_from_upload(&u).as_deref(),
            Some("UI_Unlit")
        );
    }

    #[test]
    fn resolve_from_plain_stem_ui_text_unlit() {
        let u = ShaderUpload {
            file: Some("UI_TextUnlit".to_string()),
            ..Default::default()
        };
        assert_eq!(
            resolve_logical_shader_name_from_upload(&u).as_deref(),
            Some("UI_TextUnlit")
        );
    }

    #[test]
    fn resolve_plain_label_first_token_only() {
        let u = ShaderUpload {
            file: Some("UI_Unlit ALPHACLIP".to_string()),
            ..Default::default()
        };
        assert_eq!(
            resolve_logical_shader_name_from_upload(&u).as_deref(),
            Some("UI_Unlit")
        );
    }

    #[test]
    fn try_resolve_plain_rejects_absolute_path_like() {
        assert_eq!(
            try_resolve_plain_shader_label("/etc/shaders/x.shader"),
            None
        );
        assert_eq!(try_resolve_plain_shader_label("C:\\a\\b.shader"), None);
    }

    #[test]
    fn try_resolve_plain_strips_renderide_prefix() {
        assert_eq!(
            try_resolve_plain_shader_label("renderide:UI_Unlit").as_deref(),
            Some("UI_Unlit")
        );
    }

    #[test]
    fn parse_world_unlit_shader_line() {
        let src = "Shader \"Unlit\"\n{\n";
        assert_eq!(
            parse_shader_lab_quoted_name(src).as_deref(),
            Some(CANONICAL_UNITY_WORLD_UNLIT)
        );
    }
}
