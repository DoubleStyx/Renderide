//! Build identity embedded into the renderer binary.

use std::sync::LazyLock;

const UNAVAILABLE: &str = "<unavailable>";

static RENDERER_IDENTIFIER: LazyLock<String> =
    LazyLock::new(|| renderer_identifier_for(env!("CARGO_PKG_VERSION"), renderer_commit_sha8()));

/// Returns the human-readable renderer identity reported to the host and logs.
pub(crate) fn renderer_identifier() -> &'static str {
    RENDERER_IDENTIFIER.as_str()
}

/// Returns the embedded 8-character renderer commit SHA when available.
pub(crate) fn renderer_commit_sha8() -> Option<&'static str> {
    normalized_commit_sha8(env!("RENDERIDE_GIT_COMMIT"))
}

/// Returns the source used to embed [`renderer_commit_sha8`].
pub(crate) fn renderer_commit_source() -> &'static str {
    if renderer_commit_sha8().is_none() {
        return "unavailable";
    }

    let source = option_env!("RENDERIDE_GIT_COMMIT_SOURCE").unwrap_or("unavailable");
    if source.trim().is_empty() {
        "unavailable"
    } else {
        source
    }
}

/// Returns a printable commit label for structured startup logs.
pub(crate) fn renderer_commit_sha8_label() -> &'static str {
    renderer_commit_sha8().unwrap_or(UNAVAILABLE)
}

fn renderer_identifier_for(version: &str, commit_sha8: Option<&str>) -> String {
    match commit_sha8 {
        Some(commit) => format!("Renderide {version}-{commit}"),
        None => format!("Renderide {version}"),
    }
}

fn normalized_commit_sha8(commit_sha8: &'static str) -> Option<&'static str> {
    (commit_sha8.len() == 8 && commit_sha8.chars().all(|c| c.is_ascii_hexdigit()))
        .then_some(commit_sha8)
}

#[cfg(test)]
mod tests {
    use super::{normalized_commit_sha8, renderer_identifier_for};

    #[test]
    fn renderer_identifier_includes_commit_when_present() {
        assert_eq!(
            renderer_identifier_for("0.1.1", Some("03b605ad")),
            "Renderide 0.1.1-03b605ad"
        );
    }

    #[test]
    fn renderer_identifier_omits_commit_when_unavailable() {
        assert_eq!(renderer_identifier_for("0.1.1", None), "Renderide 0.1.1");
    }

    #[test]
    fn normalized_commit_sha8_requires_eight_hex_chars() {
        assert_eq!(normalized_commit_sha8("03b605ad"), Some("03b605ad"));
        assert_eq!(normalized_commit_sha8("03B605AD"), Some("03B605AD"));
        assert_eq!(normalized_commit_sha8("03b605a"), None);
        assert_eq!(normalized_commit_sha8("03b605adx"), None);
        assert_eq!(normalized_commit_sha8(""), None);
    }
}
