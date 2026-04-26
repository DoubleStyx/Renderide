//! Bootstrapper binary entry point.

#![warn(missing_docs)]
#![cfg_attr(windows, windows_subsystem = "windows")]

mod dialog;

/// Parses CLI args, initializes the bootstrapper log file and panic hook, optionally prompts
/// for desktop vs VR, then runs [`bootstrapper::run`].
///
/// Logging is initialized **before** the desktop/VR dialog so that an `rfd` backend hang on
/// headless or display-less Linux systems leaves an actionable line in
/// `logs/bootstrapper/*.log` instead of producing a silent "nothing happens" failure.
///
/// On Linux, [`bootstrapper::vr_prompt::sanitize_linux_display_env`] runs after the logger
/// is initialized and before the dialog so that an empty `WAYLAND_DISPLAY` (the folk
/// "force X11" idiom) does not poison the in-process GTK3 backend `rfd` uses for the dialog.
///
/// The interactive dialog itself lives in the bin-only [`dialog`] module so the bootstrapper
/// library never references `rfd` (see `dialog`'s module docs for why).
///
/// Exits with status `0` without spawning the Host when the user cancels the
/// desktop vs VR dialog.
fn main() {
    let (host_args, log_level) = bootstrapper::cli::parse_args();
    let log_timestamp = logger::log_filename_timestamp();
    let max_level = log_level.unwrap_or(logger::LogLevel::Trace);

    let log_path = match logger::init_for(
        logger::LogComponent::Bootstrapper,
        &log_timestamp,
        max_level,
        false,
    ) {
        Ok(path) => path,
        Err(e) => {
            // The logger is the sink we would normally route through; stderr is the only channel
            // left when its own initialization fails.
            #[expect(clippy::print_stderr, reason = "logger failed to initialize")]
            {
                eprintln!("bootstrapper: failed to initialize logging: {e}");
            }
            std::process::exit(1);
        }
    };
    bootstrapper::panic_hook::install(log_path);

    bootstrapper::vr_prompt::sanitize_linux_display_env();

    let Some(host_args) =
        bootstrapper::cli::resolve_vr_choice(host_args, dialog::prompt_desktop_or_vr)
    else {
        logger::info!("Desktop/VR dialog cancelled; exiting without spawning Host.");
        return;
    };

    let opts = bootstrapper::BootstrapOptions {
        host_args,
        log_level,
        log_timestamp,
    };
    if let Err(e) = bootstrapper::run(opts) {
        logger::error!("{e}");
        std::process::exit(1);
    }
}
