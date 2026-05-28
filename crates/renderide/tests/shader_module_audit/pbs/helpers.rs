//! Shared helpers for PBS shader source audits.

use super::super::*;

/// Returns the declared pass names or pass types from source pass directives.
pub(super) fn pass_directives(src: &str) -> Vec<&str> {
    src.lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("//#pass ")?;
            let pass_type = rest
                .split_whitespace()
                .find_map(|token| token.strip_prefix("type="))?;
            Some(
                rest.split_whitespace()
                    .find_map(|token| token.strip_prefix("name="))
                    .unwrap_or(pass_type),
            )
        })
        .collect()
}

/// Asserts one expected shader variant bit constant for a material root.
pub(super) fn assert_keyword_bit(src: &str, file_name: &str, constant_name: &str, bit_index: u32) {
    let needle = format!("const {constant_name}: u32 = 1u << {bit_index}u;");
    assert!(src.contains(&needle), "{file_name} must define `{needle}`");
}

/// Asserts all expected shader variant bit constants for a material root.
pub(super) fn assert_keyword_bits(file_name: &str, expected: &[(&str, u32)]) -> io::Result<()> {
    let src = material_source(file_name)?;
    for (constant_name, bit_index) in expected.iter().copied() {
        assert_keyword_bit(&src, file_name, constant_name, bit_index);
    }
    Ok(())
}

/// Returns true when a source path belongs to the shader families that should use the modern PBS BRDF.
pub(super) fn modern_brdf_family_label(label: &str) -> bool {
    label.starts_with("shaders/modules/pbs/")
        || label.starts_with("shaders/modules/xiexe/")
        || label.starts_with("shaders/modules/fur/")
        || label.starts_with("shaders/materials/pbs")
        || label.starts_with("shaders/materials/paintpbs")
        || label.starts_with("shaders/materials/xstoon")
        || label.starts_with("shaders/materials/furfx")
}
