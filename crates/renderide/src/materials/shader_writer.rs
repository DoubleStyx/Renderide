//! Identifier-name normalization for the WGSL writer / naga-oil composer.
//!
//! Composed material WGSL feeds through the WGSL writer, which escapes identifiers that end in
//! digits by appending a trailing `_`, and through naga-oil, which mangles cross-module names by
//! appending an `X_naga_oil_mod_<base32>` suffix. The renderer needs the original host property
//! name to look up properties in [`crate::materials::host_data::MaterialPropertyStore`], so these
//! helpers reverse those two transformations.

/// Strip naga-oil's cross-module name mangling suffix (`X_naga_oil_mod_<base32>`) from `name`.
///
/// If `name` does not contain the suffix it is returned unchanged.
pub(crate) fn strip_naga_oil_mod_suffix(name: &str) -> &str {
    name.split_once("X_naga_oil_mod_")
        .map_or(name, |(base, _)| base)
}

/// Strip the WGSL writer's trailing-`_` digit-escape from `name`.
///
/// The writer appends `_` after a digit to keep the identifier from clashing with reserved
/// suffixes; e.g. host property `_Tex0` becomes WGSL field `_Tex0_`. Returns the original
/// `name` unchanged when no digit-escape is present.
pub(crate) fn strip_writer_digit_escape(name: &str) -> &str {
    let Some(stripped) = name.strip_suffix('_') else {
        return name;
    };
    if stripped
        .chars()
        .next_back()
        .is_some_and(|c| c.is_ascii_digit())
    {
        stripped
    } else {
        name
    }
}

/// Reverse both naga-oil mangling and the writer digit-escape to recover the host property name.
pub(crate) fn unescape_property_name(name: &str) -> &str {
    strip_writer_digit_escape(strip_naga_oil_mod_suffix(name))
}

#[cfg(test)]
mod tests {
    use super::{strip_naga_oil_mod_suffix, strip_writer_digit_escape, unescape_property_name};

    #[test]
    fn writer_digit_escape_drops_trailing_underscore_after_digit() {
        assert_eq!(strip_writer_digit_escape("_Tex0_"), "_Tex0");
        assert_eq!(strip_writer_digit_escape("_Tint0_"), "_Tint0");
    }

    #[test]
    fn writer_digit_escape_preserves_non_digit_tail() {
        assert_eq!(strip_writer_digit_escape("_Color_"), "_Color_");
        assert_eq!(strip_writer_digit_escape("_Color"), "_Color");
        assert_eq!(strip_writer_digit_escape("_MainTex_ST"), "_MainTex_ST");
    }

    #[test]
    fn naga_oil_suffix_is_stripped() {
        assert_eq!(
            strip_naga_oil_mod_suffix(
                "_MainTexX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ4GSZLYMU5DU5DPN5XDEX"
            ),
            "_MainTex"
        );
    }

    #[test]
    fn unescape_property_name_combines_both_passes() {
        assert_eq!(
            unescape_property_name("_Tint0_X_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ4GSZLYMU5DU5DPN5XDEX"),
            "_Tint0"
        );
        assert_eq!(unescape_property_name("_MainTex_ST"), "_MainTex_ST");
    }
}
