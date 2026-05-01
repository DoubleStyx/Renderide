//! IPC command routing by [`crate::frontend::InitState`]: init handshake vs running dispatch.

use crate::assets::texture::supported_host_formats_for_init;
use crate::config::RendererSettings;
use crate::frontend::InitState;
use crate::ipc::DualQueueIpc;
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, RendererCommand, RendererInitResult};
use crate::xr::output_device::head_output_device_wants_openxr;

use super::command_kind::{RendererCommandLifecycle, classify_renderer_command};

/// `Renderide` plus the `renderide` crate version (`env!("CARGO_PKG_VERSION")` at compile time).
const RENDERER_IDENTIFIER: &str = concat!("Renderide ", env!("CARGO_PKG_VERSION"));

/// Pure init-routing decision for one command in the current init phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InitDispatchDecision {
    /// Ignore the command.
    Ignore,
    /// Apply host init data and enter `InitReceived`.
    ApplyInitData,
    /// Mark init as finalized.
    Finalize,
    /// Route to normal running-command dispatch.
    DispatchRunning,
    /// Defer until init is finalized.
    DeferUntilFinalized,
    /// Treat the command as a fatal ordering error.
    FatalExpectedInitData,
}

/// Computes the init-routing action without touching runtime state.
pub(crate) fn init_dispatch_decision(
    init_state: InitState,
    lifecycle: RendererCommandLifecycle,
) -> InitDispatchDecision {
    match init_state {
        InitState::Uninitialized => match lifecycle {
            RendererCommandLifecycle::KeepAlive => InitDispatchDecision::Ignore,
            RendererCommandLifecycle::InitData => InitDispatchDecision::ApplyInitData,
            _ => InitDispatchDecision::FatalExpectedInitData,
        },
        InitState::InitReceived => match lifecycle {
            RendererCommandLifecycle::KeepAlive
            | RendererCommandLifecycle::InitProgressUpdate
            | RendererCommandLifecycle::EngineReady => InitDispatchDecision::Ignore,
            RendererCommandLifecycle::InitFinalize => InitDispatchDecision::Finalize,
            _ => InitDispatchDecision::DeferUntilFinalized,
        },
        InitState::Finalized => InitDispatchDecision::DispatchRunning,
    }
}

/// Sends [`RendererInitResult`] to the host after [`crate::shared::RendererInitData`] is applied.
///
/// Returns `false` if the primary queue rejected the message (caller should treat as fatal / retry init).
///
/// `gpu_max_texture_dim_2d` should be [`None`] until a [`wgpu::Device`] exists; the host only
/// accepts **one** init result (see FrooxEngine `RenderSystem.HandleCommand`), so this is sent once
/// from [`crate::runtime::RendererRuntime::on_init_data`] with [`None`] before GPU init. The
/// [`RendererSettings::reported_max_texture_dimension_for_host`] fallback ([`crate::gpu::REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE`]
/// when GPU limits are unknown) matches typical GPUs; non-zero config caps are still clamped.
pub(crate) fn send_renderer_init_result(
    ipc: &mut DualQueueIpc,
    output_device: HeadOutputDevice,
    settings: &RendererSettings,
    gpu_max_texture_dim_2d: Option<u32>,
) -> bool {
    let stereo = if head_output_device_wants_openxr(output_device) {
        "OpenXR(multiview)"
    } else {
        "None"
    };
    let max_texture_size = settings.reported_max_texture_dimension_for_host(gpu_max_texture_dim_2d);
    let result = RendererInitResult {
        actual_output_device: output_device,
        renderer_identifier: Some(RENDERER_IDENTIFIER.to_string()),
        main_window_handle_ptr: 0,
        stereo_rendering_mode: Some(stereo.to_string()),
        max_texture_size,
        is_gpu_texture_pot_byte_aligned: true,
        supported_texture_formats: supported_host_formats_for_init(),
    };
    ipc.send_primary(RendererCommand::RendererInitResult(result))
}

/// Dispatches a single command according to the current init phase.
pub(crate) fn dispatch_ipc_command(runtime: &mut RendererRuntime, cmd: RendererCommand) {
    let decision = init_dispatch_decision(
        runtime.frontend.init_state(),
        classify_renderer_command(&cmd).lifecycle(),
    );
    match decision {
        InitDispatchDecision::Ignore => {}
        InitDispatchDecision::ApplyInitData => {
            if let RendererCommand::RendererInitData(d) = cmd {
                runtime.on_init_data(d);
            }
        }
        InitDispatchDecision::Finalize => {
            logger::info!("IPC init finalized; renderer entering running command dispatch");
            runtime.frontend.set_init_state(InitState::Finalized);
        }
        InitDispatchDecision::DispatchRunning => runtime.handle_running_command(cmd),
        InitDispatchDecision::DeferUntilFinalized => {
            logger::trace!("IPC: deferring command until init finalized (skeleton)");
        }
        InitDispatchDecision::FatalExpectedInitData => {
            logger::error!("IPC: expected RendererInitData first");
            runtime.frontend.set_fatal_error(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InitDispatchDecision, init_dispatch_decision};
    use crate::frontend::InitState;
    use crate::frontend::dispatch::command_kind::RendererCommandLifecycle;

    #[test]
    fn uninitialized_accepts_keepalive_and_init_only() {
        assert_eq!(
            init_dispatch_decision(
                InitState::Uninitialized,
                RendererCommandLifecycle::KeepAlive
            ),
            InitDispatchDecision::Ignore
        );
        assert_eq!(
            init_dispatch_decision(InitState::Uninitialized, RendererCommandLifecycle::InitData),
            InitDispatchDecision::ApplyInitData
        );
        assert_eq!(
            init_dispatch_decision(InitState::Uninitialized, RendererCommandLifecycle::Running),
            InitDispatchDecision::FatalExpectedInitData
        );
    }

    #[test]
    fn init_received_ignores_lifecycle_noise_finalizes_and_defers_running() {
        assert_eq!(
            init_dispatch_decision(
                InitState::InitReceived,
                RendererCommandLifecycle::EngineReady
            ),
            InitDispatchDecision::Ignore
        );
        assert_eq!(
            init_dispatch_decision(
                InitState::InitReceived,
                RendererCommandLifecycle::InitFinalize
            ),
            InitDispatchDecision::Finalize
        );
        assert_eq!(
            init_dispatch_decision(
                InitState::InitReceived,
                RendererCommandLifecycle::FrameSubmit
            ),
            InitDispatchDecision::DeferUntilFinalized
        );
    }

    #[test]
    fn finalized_dispatches_everything_to_running_router() {
        assert_eq!(
            init_dispatch_decision(InitState::Finalized, RendererCommandLifecycle::KeepAlive),
            InitDispatchDecision::DispatchRunning
        );
    }
}
