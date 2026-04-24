//! Integration: [`logger::init_for`] with `append: true` keeps bytes already on disk.

use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn init_for_append_true_keeps_prior_file_content() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var("RENDERIDE_LOGS_ROOT", dir.path().as_os_str());

    let ts = format!(
        "append_true_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let path = logger::log_file_path(logger::LogComponent::Host, &ts);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    const SENTINEL: &str = "prior_sentinel_line\n";
    std::fs::write(&path, SENTINEL).expect("seed file");

    let log_path = logger::init_for(
        logger::LogComponent::Host,
        &ts,
        logger::LogLevel::Trace,
        true,
    )
    .expect("init_for");
    assert_eq!(log_path, path);

    logger::info!("after_append_marker");
    logger::flush();

    let contents = std::fs::read_to_string(&log_path).expect("read log");
    assert!(
        contents.contains(SENTINEL.trim_end()),
        "sentinel should remain: {contents:?}"
    );
    assert!(
        contents.contains("after_append_marker"),
        "new line should appear: {contents:?}"
    );

    std::env::remove_var("RENDERIDE_LOGS_ROOT");
}
