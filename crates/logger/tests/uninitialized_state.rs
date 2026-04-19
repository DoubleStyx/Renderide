//! Integration: global logger state in a process that never calls [`logger::init`].

/// Verifies filtering, flushing, and logging are inert before [`logger::init`].
#[test]
fn uninitialized_logger_is_inert() {
    let decoy =
        std::env::temp_dir().join(format!("logger_uninit_decoy_{}.log", std::process::id()));
    let _ = std::fs::remove_file(&decoy);

    assert!(!logger::is_initialized());
    assert!(!logger::enabled(logger::LogLevel::Error));
    assert!(!logger::try_log(
        logger::LogLevel::Error,
        format_args!("try_should_not_run")
    ));
    logger::flush();
    logger::set_max_level(logger::LogLevel::Trace);
    logger::log(
        logger::LogLevel::Error,
        format_args!("log_should_not_create_files"),
    );

    assert!(
        !decoy.exists(),
        "logger must not create unrelated paths when uninitialized"
    );
}
