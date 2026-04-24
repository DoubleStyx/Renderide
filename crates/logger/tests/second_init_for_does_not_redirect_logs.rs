//! Integration: a second [`logger::init_for`] does not replace the global logger; logs keep using the first file.

use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn second_init_for_keeps_logging_to_first_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var("RENDERIDE_LOGS_ROOT", dir.path().as_os_str());

    let ts_first = format!(
        "dup_first_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let path_first = logger::init_for(
        logger::LogComponent::Bootstrapper,
        &ts_first,
        logger::LogLevel::Trace,
        false,
    )
    .expect("first init_for");

    logger::info!("first_path_marker");
    logger::flush();

    let ts_second = format!(
        "dup_second_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let path_second = logger::init_for(
        logger::LogComponent::Renderer,
        &ts_second,
        logger::LogLevel::Trace,
        false,
    )
    .expect("second init_for returns Ok");

    assert_ne!(path_first, path_second);

    logger::info!("second_call_marker");
    logger::flush();

    let first_contents = std::fs::read_to_string(&path_first).expect("read first");
    assert!(
        first_contents.contains("first_path_marker"),
        "first log missing: {first_contents:?}"
    );
    assert!(
        first_contents.contains("second_call_marker"),
        "log after second init_for should still go to first file: {first_contents:?}"
    );

    std::env::remove_var("RENDERIDE_LOGS_ROOT");
}
