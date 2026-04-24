//! Panic hook that records panics to the bootstrapper log file before chaining the previous hook.

use std::path::PathBuf;

/// Installs a process-wide panic hook that appends via [`logger::log_panic`], then invokes the
/// hook that was active when this function ran.
///
/// Called from `main.rs` immediately after [`logger::init_for`] so that any panic occurring
/// before [`crate::run`] (for example inside the `rfd` desktop/VR dialog path) still reaches
/// the bootstrapper log file.
pub fn install(log_path: PathBuf) {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        logger::log_panic(&log_path, info);
        default_hook(info);
    }));
}
