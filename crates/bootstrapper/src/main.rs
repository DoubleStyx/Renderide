//! Bootstrapper binary entry point.

#![warn(missing_docs)]
#![cfg_attr(windows, windows_subsystem = "windows")]

/// Parses CLI args, optionally prompts for desktop vs VR, then runs [`bootstrapper::run`].
fn main() {
    let (mut host_args, log_level) = bootstrapper::config::parse_args();
    if bootstrapper::vr_prompt::should_prompt_vr_dialog(&host_args) {
        if let Some(vr) = bootstrapper::vr_prompt::prompt_desktop_or_vr() {
            host_args = bootstrapper::vr_prompt::apply_host_vr_choice(host_args, vr);
        }
    }
    let timestamp = logger::log_filename_timestamp();

    if let Err(e) = bootstrapper::run(bootstrapper::BootstrapOptions {
        host_args,
        log_level,
        log_timestamp: timestamp,
    }) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
