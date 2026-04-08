//! Small helpers shared across asset ingestion (shader name normalization, etc.).

/// Normalizes a Unity `Shader "…"` label or path for stable dictionary lookup (whitespace, `/` → `_`, lowercased).
///
/// Shared by shader routing and [`crate::materials::stem_manifest::StemResolver`] so manifest lookups stay
/// consistent without import cycles between `assets::shader::route` and materials.
pub fn normalize_unity_shader_lookup_key(name: &str) -> String {
    let token = name.split_whitespace().next().unwrap_or(name).trim();
    token
        .chars()
        .map(|c| {
            if c.is_whitespace() || c == '/' {
                '_'
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect()
}

/// Normalizes a shader token for comparison: keeps only ASCII alphanumeric characters and folds to lowercase.
///
/// Used when mapping Unity shader names and path hints to compact comparison keys.
pub fn compact_alnum_lower(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}
