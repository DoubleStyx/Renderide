//! Small helpers shared across asset ingestion (shader name normalization, etc.).

/// Normalizes a shader token for comparison: keeps only ASCII alphanumeric characters and folds to lowercase.
///
/// Used when mapping Unity shader names and path hints to compact comparison keys.
pub fn compact_alnum_lower(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}
