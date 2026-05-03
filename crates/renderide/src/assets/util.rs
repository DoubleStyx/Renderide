//! Small helpers shared across asset ingestion (shader name normalization, etc.).

/// Normalizes a Unity shader asset filename or `m_Container` asset path for stable dictionary lookup.
///
/// Rule: lowercase every character, map `/` to `_`, and map space to `-` (so `XSToon2.0 Outlined` becomes
/// `xstoon2.0-outlined`, distinct from the underscore-spelled `XSToon2.0_Outlined` to
/// `xstoon2.0_outlined`). The WGSL stem filenames under `shaders/materials/` are named
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

    #[test]
    fn handles_mixed_slashes_and_spaces_in_one_key() {
        assert_eq!(
            normalize_unity_shader_lookup_key("Custom UI/Text Unlit"),
            "custom-ui_text-unlit"
        );
    }

    #[test]
    fn preserves_inner_tab_characters() {
        assert_eq!(
            normalize_unity_shader_lookup_key("Foo\tBar Baz"),
            "foo\tbar-baz"
        );
    }

    #[test]
    fn preserves_non_ascii_characters_during_ascii_lowercase() {
        assert_eq!(
            normalize_unity_shader_lookup_key("Ümlaut/Foo"),
            "Ümlaut_foo"
        );
    }

    #[test]
    fn trims_outer_whitespace_and_maps_inner_spaces() {
        assert_eq!(
            normalize_unity_shader_lookup_key("  Foo Bar/Baz Qux  "),
            "foo-bar_baz-qux"
        );
    }
}
