//! Runtime construction and IPC init wait for app bootstrap.

use std::thread;
use std::time::{Duration, Instant};

use crate::config::{ConfigLoadResult, settings_handle_from};
use crate::connection::get_connection_parameters;
use crate::crash_context::{self, InitState as CrashInitState, RenderMode, TickPhase};
use crate::frontend::InitState;
use crate::run_error::RunError;
use crate::runtime::RendererRuntime;

/// Max time to wait for host init data after IPC connect.
const IPC_INIT_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Builds the renderer runtime and completes IPC init when queue parameters are present.
pub(crate) fn build_runtime(config: &ConfigLoadResult) -> Result<RendererRuntime, RunError> {
    crash_context::set_tick_phase(TickPhase::RuntimeInit);
    let params = get_connection_parameters();
    if let Some(params) = params.as_ref() {
        logger::info!(
            "IPC parameters detected: queue_base={} capacity={} bytes",
            params.queue_name,
            params.queue_capacity
        );
    }
    let mut runtime = RendererRuntime::new(
        params.clone(),
        settings_handle_from(config),
        config.save_path.clone(),
    );
    runtime.set_suppress_renderer_config_disk_writes(config.suppress_config_disk_writes);

    if let Err(e) = runtime.connect_ipc()
        && params.is_some()
    {
        logger::error!("IPC connect failed: {e}");
        return Err(RunError::connection(e));
    }

    if params.is_some() && runtime.is_ipc_connected() {
        logger::info!("IPC connected (Primary/Background)");
        crash_context::set_render_mode(RenderMode::IpcDesktop);
        wait_for_renderer_init_data(&mut runtime)?;
    } else if params.is_some() {
        logger::warn!("IPC params present but connection state unexpected");
    } else {
        logger::info!("Standalone mode (no -QueueName/-QueueCapacity; desktop GPU, no host init)");
        crash_context::set_render_mode(RenderMode::Standalone);
    }

    Ok(runtime)
}

fn wait_for_renderer_init_data(runtime: &mut RendererRuntime) -> Result<(), RunError> {
    crash_context::set_init_state(CrashInitState::WaitingForInitData);
    let start = Instant::now();
    let deadline = Instant::now() + IPC_INIT_WAIT_TIMEOUT;
    while runtime.init_state() == InitState::Uninitialized {
        if Instant::now() > deadline {
            logger::error!(
                "Timed out waiting for RendererInitData from host after {:.3}s\n{}",
                start.elapsed().as_secs_f64(),
                crash_context::format_snapshot()
            );
            return Err(RunError::renderer_init_data_timeout());
        }
        runtime.poll_ipc();
        if runtime.fatal_error() {
            logger::error!("Fatal IPC error while waiting for RendererInitData");
            return Err(RunError::renderer_init_data_fatal_ipc());
        }
        thread::sleep(Duration::from_millis(1));
    }
    logger::info!(
        "RendererInitData received after {:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
