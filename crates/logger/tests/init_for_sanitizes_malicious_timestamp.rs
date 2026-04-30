//! Integration: [`logger::init_for`] sanitizes hostile timestamps before opening the log file.
//!
//! The sanitization is a defense-in-depth guard against future callers — or attacker-influenced
//! input — passing path-like strings through the public API. The opened file must remain inside
//! the resolved component directory and have a single safe `.log` extension.

/// Verifies that a malicious timestamp containing path traversal segments cannot escape the
/// renderer component directory under an env-overridden logs root.
#[test]
fn init_for_sanitizes_malicious_timestamp_into_component_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    // SAFETY: env mutation in test; serialized via cargo test integration-binary isolation.
    unsafe {
        std::env::set_var("RENDERIDE_LOGS_ROOT", dir.path().as_os_str());
    }

    let log_path = logger::init_for(
        logger::LogComponent::Renderer,
        "../../etc/passwd",
        logger::LogLevel::Info,
        false,
    )
    .expect("init_for must succeed; sanitization is silent");

    let component_dir = dir.path().join("renderer");
    assert_eq!(
        log_path.parent().expect("parent dir"),
        component_dir.as_path(),
        "expected log file directly under {component_dir:?}, got {log_path:?}"
    );

    let stem = log_path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("file stem");
    assert!(
        !stem.contains(".."),
        "file stem must not contain `..`: {stem:?}"
    );
    assert!(
        !stem.contains('/') && !stem.contains('\\'),
        "file stem must not contain path separators: {stem:?}"
    );

    let extension = log_path
        .extension()
        .and_then(|s| s.to_str())
        .expect("file extension");
    assert_eq!(extension, "log", "unexpected file extension: {extension:?}");

    assert!(log_path.exists(), "log file must exist at {log_path:?}");

    let escaped = dir.path().join("etc").join("passwd");
    assert!(
        !escaped.exists(),
        "sanitization must prevent writes outside component dir: {escaped:?}"
    );

    // SAFETY: env mutation in test; serialized via cargo test integration-binary isolation.
    unsafe {
        std::env::remove_var("RENDERIDE_LOGS_ROOT");
    }
}
