//! Process bootstrap before choosing the windowed or headless app driver.

mod config;
mod logging;
mod panic;
mod runtime;
pub(crate) mod services;
mod signals;

use std::cell::RefCell;
use std::rc::Rc;

use winit::event_loop::EventLoop;
#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;

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

    let event_loop = create_window_event_loop(app_config.gpu.graphics_api)?;
    let display_handle = event_loop.owned_display_handle();

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
        display_handle,
        exit_state.clone(),
    );

    let _ = event_loop.run_app(app);
    drop(watchdog);
    Ok(exit_state.borrow().run_exit())
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LinuxWindowBackendPreference {
    Default,
    ForceX11ForOpenGl,
}

#[cfg(target_os = "linux")]
fn linux_window_backend_preference(
    graphics_api: crate::config::GraphicsApiSetting,
    display_env: Option<&str>,
) -> LinuxWindowBackendPreference {
    if graphics_api == crate::config::GraphicsApiSetting::Gl
        && display_env.is_some_and(|value| !value.is_empty())
    {
        LinuxWindowBackendPreference::ForceX11ForOpenGl
    } else {
        LinuxWindowBackendPreference::Default
    }
}

#[cfg(target_os = "linux")]
fn create_window_event_loop(
    graphics_api: crate::config::GraphicsApiSetting,
) -> Result<EventLoop, RunError> {
    let display_env = std::env::var("DISPLAY").ok();
    let preference = linux_window_backend_preference(graphics_api, display_env.as_deref());
    let mut builder = EventLoop::builder();
    if preference == LinuxWindowBackendPreference::ForceX11ForOpenGl {
        builder.with_x11();
    }

    let event_loop = builder.build().map_err(map_event_loop_create_error)?;
    log_linux_window_backend(&event_loop, graphics_api, preference);
    Ok(event_loop)
}

#[cfg(not(target_os = "linux"))]
fn create_window_event_loop(
    graphics_api: crate::config::GraphicsApiSetting,
) -> Result<EventLoop, RunError> {
    let event_loop = EventLoop::new().map_err(map_event_loop_create_error)?;
    logger::info!(
        "Window backend selected: backend=platform-default graphics_api={} display_handle=event-loop-owned",
        graphics_api.as_persist_str()
    );
    Ok(event_loop)
}

fn map_event_loop_create_error(error: winit::error::EventLoopError) -> RunError {
    logger::error!("EventLoop creation failed: {error}");
    RunError::event_loop_create(error)
}

#[cfg(target_os = "linux")]
fn log_linux_window_backend(
    event_loop: &EventLoop,
    graphics_api: crate::config::GraphicsApiSetting,
    preference: LinuxWindowBackendPreference,
) {
    use winit::platform::wayland::EventLoopExtWayland;
    use winit::platform::x11::EventLoopExtX11;

    let backend = if event_loop.is_x11() {
        "x11"
    } else if event_loop.is_wayland() {
        "wayland"
    } else {
        "unknown"
    };
    logger::info!(
        "Window backend selected: backend={} graphics_api={} x11_for_opengl={} display_handle=event-loop-owned",
        backend,
        graphics_api.as_persist_str(),
        preference == LinuxWindowBackendPreference::ForceX11ForOpenGl,
    );
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::config::GraphicsApiSetting;

    #[test]
    fn opengl_with_display_env_forces_x11() {
        assert_eq!(
            linux_window_backend_preference(GraphicsApiSetting::Gl, Some(":0")),
            LinuxWindowBackendPreference::ForceX11ForOpenGl
        );
    }

    #[test]
    fn opengl_without_display_env_keeps_default_backend() {
        assert_eq!(
            linux_window_backend_preference(GraphicsApiSetting::Gl, None),
            LinuxWindowBackendPreference::Default
        );
        assert_eq!(
            linux_window_backend_preference(GraphicsApiSetting::Gl, Some("")),
            LinuxWindowBackendPreference::Default
        );
    }

    #[test]
    fn non_opengl_api_keeps_default_backend() {
        assert_eq!(
            linux_window_backend_preference(GraphicsApiSetting::Auto, Some(":0")),
            LinuxWindowBackendPreference::Default
        );
        assert_eq!(
            linux_window_backend_preference(GraphicsApiSetting::Vulkan, Some(":0")),
            LinuxWindowBackendPreference::Default
        );
    }
}
