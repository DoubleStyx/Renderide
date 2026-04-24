//! Integration: [`logger::init_for`] with `append: false` truncates an existing log file.

use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn init_for_append_false_truncates_existing_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var("RENDERIDE_LOGS_ROOT", dir.path().as_os_str());

    let ts = format!(
        "append_false_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let path = logger::log_file_path(logger::LogComponent::Bootstrapper, &ts);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    const SENTINEL: &str = "stale_sentinel_should_vanish\n";
    std::fs::write(&path, SENTINEL).expect("seed file");

    let log_path = logger::init_for(
        logger::LogComponent::Bootstrapper,
        &ts,
        logger::LogLevel::Trace,
        false,
    )
    .expect("init_for");
    assert_eq!(log_path, path);

    logger::info!("fresh_after_truncate_marker");
    logger::flush();

    let contents = std::fs::read_to_string(&log_path).expect("read log");
    assert!(
        !contents.contains("stale_sentinel_should_vanish"),
        "truncated file must not retain sentinel: {contents:?}"
    );
    assert!(
        contents.contains("fresh_after_truncate_marker"),
        "expected new log line: {contents:?}"
    );

    std::env::remove_var("RENDERIDE_LOGS_ROOT");
}
