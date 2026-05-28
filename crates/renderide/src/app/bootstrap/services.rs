//! Process services shared by the windowed and headless app drivers.

use crate::config::WatchdogSettings;
use crate::diagnostics::{Heartbeat, Watchdog};

pub(crate) use super::signals::ExternalShutdownCoordinator;

/// Long-lived services installed for the renderer process.
pub(crate) struct AppServices {
    /// OS-driven cooperative shutdown observer.
    pub(crate) external_shutdown: Option<ExternalShutdownCoordinator>,
    /// Watchdog guard; dropping joins the watchdog thread.
    pub(crate) watchdog: Option<Watchdog>,
    /// Main-thread heartbeat registered with the watchdog.
    pub(crate) main_heartbeat: Option<Heartbeat>,
}

/// Installs shutdown handling, watchdog, profiling main-thread state, and Rayon worker services.
pub(crate) fn install_app_services(watchdog_settings: WatchdogSettings) -> AppServices {
    let external_shutdown = super::signals::install_external_shutdown();
    let watchdog = Watchdog::install(watchdog_settings);
    let main_heartbeat = watchdog.as_ref().map(|w| w.register_current_thread("main"));

    crate::profiling::register_main_thread();
    init_rayon_pool();

    AppServices {
        external_shutdown,
        watchdog,
        main_heartbeat,
    }
}

/// Builds the global Rayon pool with renderer thread names and profiling registration.
fn init_rayon_pool() {
    match rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("rayon-worker-{i}"))
        .start_handler(crate::profiling::rayon_thread_start_handler())
        .build_global()
    {
        Ok(()) => warm_rayon_pool(),
        Err(e) => {
            logger::warn!("Rayon global pool already initialized or build_global failed: {e}");
        }
    }
}

/// Runs one startup task on every Rayon worker so first-frame jobs do not pay thread wake-up cost.
fn warm_rayon_pool() {
    profiling::scope!("startup::rayon_pool_warmup");
    rayon::broadcast(|_| {
        profiling::scope!("startup::rayon_pool_warmup::worker");
    });
}
