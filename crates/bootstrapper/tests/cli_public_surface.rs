//! Integration: [`bootstrapper::cli`] parsing as a downstream crate would call it (separate test binary).

use logger::LogLevel;

/// Strips bootstrapper-only flags and preserves Host forwarding semantics.
#[test]
fn parse_host_args_tokens_forwards_host_argv_and_log_level() {
    let args = vec!["--log-level".into(), "info".into(), "-Invisible".into()];
    let (host, level) = bootstrapper::cli::parse_host_args_tokens(&args);
    assert_eq!(host, vec!["-Invisible".to_string()]);
    assert_eq!(level, Some(LogLevel::Info));
}

/// Empty argv yields empty host list and no log level override.
#[test]
fn parse_host_args_tokens_empty_slice() {
    let (host, level) = bootstrapper::cli::parse_host_args_tokens(&[]);
    assert!(host.is_empty());
    assert!(level.is_none());
}
