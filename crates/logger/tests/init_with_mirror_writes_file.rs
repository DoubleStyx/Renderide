//! Integration: [`logger::init_with_mirror`] still writes to the log file when stderr mirroring is on.

use std::time::SystemTime;

#[test]
fn init_with_mirror_true_writes_log_file() {
    let path = std::env::temp_dir().join(format!(
        "logger_mirror_smoke_{}.log",
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_file(&path);

    logger::init_with_mirror(&path, logger::LogLevel::Info, false, true).expect("init_with_mirror");
    assert!(logger::is_initialized());

    logger::info!("mirror_file_marker");
    logger::flush();

    let contents = std::fs::read_to_string(&path).expect("read log");
    assert!(
        contents.contains("mirror_file_marker"),
        "expected marker in file: {contents:?}"
    );
    assert!(
        contents.contains(" INFO "),
        "expected INFO token: {contents:?}"
    );

    let _ = std::fs::remove_file(&path);
}
