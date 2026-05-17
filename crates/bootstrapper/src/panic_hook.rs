//! Panic hook that records panics to the bootstrapper log file before chaining the previous hook.

use std::io::Write;
use std::path::PathBuf;

/// Installs a process-wide panic hook that appends the full report to the bootstrapper log file,
/// invokes the hook that was active when this function ran, then prints the log-directory footer.
///
/// Called from `main.rs` immediately after [`logger::init_for`] so that any panic occurring
/// before [`crate::run`] (for example inside the `rfd` desktop/VR dialog path) still reaches
/// the bootstrapper log file.
pub fn install(log_path: PathBuf) {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let logs_root = logger::logs_root();
        let footer = logger::log_directory_footer(&logs_root);
        let mut report = logger::panic_report(info);
        report.push_str(&footer);
        logger::append_panic_report_to_file(&log_path, &report);
        default_hook(info);
        let _ = std::io::stderr().write_all(footer.as_bytes());
        let _ = std::io::stderr().flush();
    }));
}
