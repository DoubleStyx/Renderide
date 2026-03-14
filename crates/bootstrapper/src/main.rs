//! Bootstrapper binary - starts Renderite.Host and the Renderide renderer.
//! Communicates with the Resonite host via IPC queues.

mod config;
mod host_spawner;
mod logger;
mod orphan;
mod paths;
mod queue_commands;
mod resoboot;
mod wine_helpers;

use std::fs::OpenOptions;
use std::io::Write;

const BOOTSTRAPPER_LOG: &str = "logs/Bootstrapper.log";

fn main() {
    let _ = std::fs::create_dir_all("logs");
    let mut logger = logger::Logger::new(BOOTSTRAPPER_LOG);

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(BOOTSTRAPPER_LOG)
        {
            let _ = writeln!(f, "PANIC: {}", info);
            let _ = writeln!(f, "Backtrace:\n{:?}", std::backtrace::Backtrace::capture());
            let _ = f.flush();
        }
        default_hook(info);
    }));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        resoboot::run(&mut logger);
    }));

    if let Err(ex) = result {
        logger.log(&format!("Exception in bootstrapper:\n{:?}", ex));
    }

    logger.flush();
}
