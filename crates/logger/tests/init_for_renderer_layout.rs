//! Integration: [`logger::init_for`] layout under a temporary [`RENDERIDE_LOGS_ROOT`].

use std::time::{SystemTime, UNIX_EPOCH};

/// Ensures renderer logs land under `<temp>/renderer/<ts>.log` with expected line formatting.
#[test]
fn init_for_renderer_under_temp_logs_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var("RENDERIDE_LOGS_ROOT", dir.path().as_os_str());

    let ts = format!(
        "init_for_layout_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let log_path = logger::init_for(
        logger::LogComponent::Renderer,
        &ts,
        logger::LogLevel::Debug,
        false,
    )
    .expect("init_for");

    assert!(
        log_path.starts_with(dir.path()),
        "expected {:?} under {:?}",
        log_path,
        dir.path()
    );
    assert!(
        log_path.ends_with(format!("{ts}.log")),
        "expected {:?} to end with {ts}.log",
        log_path
    );
    assert_eq!(
        log_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str()),
        Some("renderer")
    );
    assert!(log_path.parent().expect("parent").is_dir());
    assert!(logger::is_initialized());

    logger::info!("init_for_marker");
    logger::flush();

    let contents = std::fs::read_to_string(&log_path).expect("read log");
    assert!(
        contents.contains("init_for_marker"),
        "expected marker in file: {contents:?}"
    );
    assert!(
        contents.contains(" INFO "),
        "expected INFO level token: {contents:?}"
    );
    assert!(
        contents.contains('[') && contents.contains(']'),
        "expected bracketed timestamp prefix: {contents:?}"
    );
}
