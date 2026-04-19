//! Bootstrapper binary entry point.

#![warn(missing_docs)]
#![cfg_attr(windows, windows_subsystem = "windows")]

/// Parses CLI args, optionally prompts for desktop vs VR, then runs [`bootstrapper::run`].
fn main() {
    let opts = bootstrapper::cli::prepare_run_inputs();
    if let Err(e) = bootstrapper::run(opts) {
        logger::error!("{e}");
        std::process::exit(1);
    }
}
