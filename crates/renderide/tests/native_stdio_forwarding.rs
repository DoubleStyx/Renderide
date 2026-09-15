//! Integration test: native stdout is forwarded into the log file under a temp `RENDERIDE_LOGS_ROOT`.
//!
//! This binary runs in its own process so the global logger is not initialized by other unit tests.

use std::io::Write;
use std::time::{Duration, Instant, SystemTime};

#[test]
fn stdout_redirected_to_log_under_temp_logs_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    // SAFETY: env mutation in test; cargo test runs each integration test in its own process.
    unsafe {
        std::env::set_var("RENDERIDE_LOGS_ROOT", dir.path().as_os_str());
        std::env::set_var("RENDERIDE_LOG_TEE_TERMINAL", "0");
    }

    let ts = format!(
        "stdio_fwd_{}",
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let log_path = logger::init_for(
        logger::LogComponent::Renderer,
        &ts,
        logger::LogLevel::Debug,
        false,
    )
    .expect("logger init");

    renderide::ensure_native_stdio_forwarded_to_logger();

    const MARKER: &str = "RENDERIDE_STDIO_REDIRECT_TEST_MARKER";
    writeln!(std::io::stdout(), "{MARKER}").expect("write stdout");

    // Forwarding is asynchronous. Flush after each read attempt so the pipe reader's log write is
    // made visible regardless of whether it ran before or after the initial stdout write returned.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        logger::flush();
        let contents = std::fs::read_to_string(&log_path).expect("read log");
        if contents.contains(MARKER) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "expected log file to contain forwarded stdout line; got len {}",
            contents.len()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}
