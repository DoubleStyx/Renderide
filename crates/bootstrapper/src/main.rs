//! Bootstrapper binary - starts Renderite.Host and the Renderide renderer.
//! Communicates with the Resonite host via IPC queues.

#![cfg_attr(windows, windows_subsystem = "windows")]

mod config;
mod host_spawner;
mod paths;
mod process_lifetime;
mod queue_commands;
mod resoboot;
mod wine_helpers;

use logger::LogLevel;

const BOOTSTRAPPER_LOG: &str = "logs/Bootstrapper.log";

fn main() {
    let _ = std::fs::create_dir_all("logs");
    let (host_args, log_level) = config::parse_args();
    if let Err(e) = logger::init(
        BOOTSTRAPPER_LOG,
        log_level.unwrap_or(LogLevel::Trace),
        false,
    ) {
        eprintln!(
            "Failed to initialize logging to {}: {}",
            BOOTSTRAPPER_LOG, e
        );
        std::process::exit(1);
    }

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        logger::log_panic(BOOTSTRAPPER_LOG, info);
        default_hook(info);
    }));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        resoboot::run(&host_args, log_level);
    }));

    if let Err(ex) = result {
        logger::error!("Exception in bootstrapper:\n{:?}", ex);
        logger::flush();
    }
}
