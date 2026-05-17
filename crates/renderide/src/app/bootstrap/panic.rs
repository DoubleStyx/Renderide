//! Panic-hook installation for the renderer process.

use std::path::Path;

use crate::diagnostics::crash_context;

/// Installs a panic hook that appends a fully formatted crash snapshot to the open log file and
/// echoes the report to the preserved native stderr handle.
pub(super) fn install_panic_hook(log_path: &Path) {
    let log_path_hook = log_path.to_path_buf();
    std::panic::set_hook(Box::new(move |info| {
        let mut report = logger::panic_report(info);
        report.push('\n');
        report.push_str(&crash_context::format_snapshot());
        logger::append_log_directory_footer(&mut report, logger::logs_root());
        logger::append_panic_report_to_file(&log_path_hook, &report);
        crate::native_stdio::try_write_preserved_stderr(report.as_bytes());
    }));
}
