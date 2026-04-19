//! Integration: log macros after [`logger::init_for`] for [`logger::LogComponent::Bootstrapper`].

use std::time::{SystemTime, UNIX_EPOCH};

/// Exercises every public log macro and post-init max-level filtering.
#[test]
fn macros_under_init_for_bootstrapper() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var("RENDERIDE_LOGS_ROOT", dir.path().as_os_str());

    let ts = format!(
        "macros_bootstrap_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let log_path = logger::init_for(
        logger::LogComponent::Bootstrapper,
        &ts,
        logger::LogLevel::Trace,
        false,
    )
    .expect("init_for");

    assert_eq!(
        log_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str()),
        Some("bootstrapper")
    );

    logger::error!("macro_err_marker");
    logger::warn!("macro_warn_marker");
    logger::info!("macro_info_marker");
    logger::debug!("macro_debug_marker");
    logger::trace!("macro_trace_marker");
    logger::flush();

    let contents = std::fs::read_to_string(&log_path).expect("read log");
    for (level_token, needle) in [
        ("ERROR", "macro_err_marker"),
        ("WARN", "macro_warn_marker"),
        ("INFO", "macro_info_marker"),
        ("DEBUG", "macro_debug_marker"),
        ("TRACE", "macro_trace_marker"),
    ] {
        assert!(
            contents.contains(level_token) && contents.contains(needle),
            "missing {level_token} line with {needle}: {contents:?}"
        );
    }

    logger::set_max_level(logger::LogLevel::Error);
    logger::info!("macro_hidden_after_filter");
    logger::flush();

    let after = std::fs::read_to_string(&log_path).expect("read log");
    assert!(
        !after.contains("macro_hidden_after_filter"),
        "info line should be filtered: {after:?}"
    );
}
