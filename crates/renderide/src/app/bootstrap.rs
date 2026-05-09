//! Process bootstrap before choosing the windowed or headless app driver.

mod config;
mod logging;
mod runtime;
pub(crate) mod services;

use std::cell::RefCell;
use std::rc::Rc;

use winit::event_loop::EventLoop;

use crate::ipc::get_headless_params;
use crate::run_error::RunError;
use crate::{app::exit::ExitState, connection::try_claim_renderer_singleton};

use self::services::AppServices;
use super::driver::AppDriver;
use super::exit::RunExit;
use super::headless::run_headless;

pub(crate) use config::{GpuStartupConfig, effective_renderer_log_level};
pub(crate) use services::ExternalShutdownCoordinator;

/// Runs the renderer process until the selected app driver exits normally.
pub fn run() -> Result<RunExit, RunError> {
    try_claim_renderer_singleton().map_err(RunError::connection)?;

    let logging = logging::init_logging()?;
    let app_config = config::load_app_config(logging.log_level_cli);
    let mut runtime = runtime::build_runtime(&app_config.load)?;
    let services = services::install_app_services(app_config.load.settings.watchdog);

    if let Some(headless_params) = get_headless_params() {
        let AppServices {
            external_shutdown,
            watchdog,
            main_heartbeat,
        } = services;
        let result = run_headless(
            &mut runtime,
            headless_params,
            external_shutdown,
            app_config.gpu,
        );
        drop(main_heartbeat);
        drop(watchdog);
        return result;
    }

    let event_loop = EventLoop::new().map_err(|e| {
        logger::error!("EventLoop::new failed: {e}");
        RunError::event_loop_create(e)
    })?;

    let exit_state = Rc::new(RefCell::new(ExitState::default()));

    let AppServices {
        external_shutdown,
        watchdog,
        main_heartbeat,
    } = services;
    let app = AppDriver::new(
        runtime,
        app_config.gpu,
        logging.log_level_cli,
        external_shutdown,
        main_heartbeat,
        exit_state.clone(),
    );

    let _ = event_loop.run_app(app);
    drop(watchdog);
    Ok(exit_state.borrow().run_exit())
}
