//! Unity ShaderLab logical names (`Shader "…"`) and WGSL `unity-shader-name` banners for shader uploads.
//!
//! FrooxEngine sends a sequential [`crate::shared::ShaderUpload::asset_id`] and a `file` path. For an optional
//! host-appended logical name after the stock payload, see [`crate::shared::unpack_appended_shader_logical_name`]
//! and [`resolve_logical_shader_name_from_upload_with_host_hint`].

use crate::shared::ShaderUpload;

/// Canonical name from Resonite `UI_Unlit.shader`: line `Shader "UI/Unlit"`.
pub const CANONICAL_UNITY_UI_UNLIT: &str = "UI/Unlit";

/// Canonical name: `UI_TextUnlit.shader` → `Shader "UI/Text/Unlit"`.
pub const CANONICAL_UNITY_UI_TEXT_UNLIT: &str = "UI/Text/Unlit";

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
    if file_field.len() < UPLOAD_SOURCE_READ_CAP_BYTES {
        if let Some(from_lab) = parse_shader_lab_quoted_name(file_field) {
            return Some(from_lab);
        }
        if let Some(from_wgsl) = parse_wgsl_unity_shader_name_banner(file_field) {
            return Some(from_wgsl);
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
}
