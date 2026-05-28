//! Source-alias wrapper directive parsing.

use super::super::error::BuildError;

/// Parses an optional `//#source_alias <stem>` directive from a thin shader wrapper.
pub(in super::super) fn parse_source_alias(
    source: &str,
    file: &str,
) -> Result<Option<String>, BuildError> {
    let mut alias = None;
    for (line_idx, line) in source.lines().enumerate() {
        let line_no = line_idx + 1;
        let Some(rest) = line.trim_start().strip_prefix("//#source_alias") else {
            continue;
        };
        if alias.is_some() {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: duplicate `//#source_alias` directive"
            )));
        }
        let mut tokens = rest.split_whitespace();
        let Some(stem) = tokens.next() else {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#source_alias` requires a source file stem"
            )));
        };
        if tokens.next().is_some() {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#source_alias` accepts exactly one source file stem"
            )));
        }
        if stem.contains('/')
            || stem.contains('\\')
            || std::path::Path::new(stem)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("wgsl"))
        {
            return Err(BuildError::Message(format!(
                "{file}:{line_no}: `//#source_alias` must be a sibling WGSL file stem, got `{stem}`"
            )));
        }
        alias = Some(stem.to_string());
    }
    Ok(alias)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Source-alias wrappers carry exactly one sibling WGSL stem.
    #[test]
    fn source_alias_parses_sibling_stem() -> Result<(), BuildError> {
        let source = "//! wrapper\n//#source_alias blur\n";

        assert_eq!(
            parse_source_alias(source, "blur_perobject.wgsl")?.as_deref(),
            Some("blur")
        );
        Ok(())
    }

    /// Source-alias wrappers reject paths so build output stays deterministic and local.
    #[test]
    fn source_alias_rejects_paths() {
        let err = parse_source_alias("//#source_alias ../blur\n", "bad.wgsl")
            .expect_err("path aliases must be rejected");

        assert!(err.to_string().contains("sibling WGSL file stem"));
    }
}
