//! Small helpers shared across asset ingestion (shader name normalization, etc.).

/// Normalizes a Unity `Shader "…"` label or path / AssetBundle filename for stable dictionary lookup.
///
/// Rule: lowercase every character, map `/` → `_` (so nested ShaderLab paths become single
/// underscore-separated stems), and map space → `-` (so `XSToon2.0 Outlined` becomes
/// `xstoon2.0-outlined`, distinct from the underscore-spelled `XSToon2.0_Outlined` →
/// `xstoon2.0_outlined`). The WGSL stem filenames under `shaders/source/materials/` are named
/// directly against this rule applied to the Unity `.shader` filename.
///
/// Shared by shader routing and embedded `{key}_default` stem resolution so lookups stay
/// consistent without import cycles between `assets::shader::route` and materials.
pub fn normalize_unity_shader_lookup_key(name: &str) -> String {
    name.trim()
        .chars()
        .map(|c| match c {
            '/' => '_',
            ' ' => '-',
            c => c.to_ascii_lowercase(),
        })
        .collect()
}

#[cfg(test)]
mod normalize_unity_shader_lookup_key_tests {
    use super::normalize_unity_shader_lookup_key;

    #[test]
    fn folds_slashes_to_underscores_and_spaces_to_dashes() {
        assert_eq!(
            normalize_unity_shader_lookup_key("Custom/UI/TextUnlit"),
            "custom_ui_textunlit"
        );
        assert_eq!(
            normalize_unity_shader_lookup_key("XSToon2.0 Outlined"),
            "xstoon2.0-outlined"
        );
        assert_eq!(
            normalize_unity_shader_lookup_key("XSToon2.0_Outlined"),
            "xstoon2.0_outlined"
        );
    }

    #[test]
    fn empty_input_yields_empty() {
        assert_eq!(normalize_unity_shader_lookup_key(""), "");
        assert_eq!(normalize_unity_shader_lookup_key("   "), "");
    }

    #[test]
    fn preserves_inner_ascii_other_chars() {
        assert_eq!(normalize_unity_shader_lookup_key("Foo-Bar_1"), "foo-bar_1");
    }
}
