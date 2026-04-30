//! IPC command routing by [`crate::frontend::InitState`]: init handshake vs running dispatch.

use crate::assets::texture::supported_host_formats_for_init;
use crate::config::RendererSettings;
use crate::frontend::InitState;
use crate::ipc::DualQueueIpc;
use crate::output_device::head_output_device_wants_openxr;
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, RendererCommand, RendererInitResult};

/// `Renderide` plus the `renderide` crate version (`env!("CARGO_PKG_VERSION")` at compile time).
const RENDERER_IDENTIFIER: &str = concat!("Renderide ", env!("CARGO_PKG_VERSION"));

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
    match runtime.frontend.init_state() {
        InitState::Uninitialized => match cmd {
            RendererCommand::KeepAlive(_) => {}
            RendererCommand::RendererInitData(d) => runtime.on_init_data(d),
            _ => {
                logger::error!("IPC: expected RendererInitData first");
                runtime.frontend.set_fatal_error(true);
            }
        },
        InitState::InitReceived => match cmd {
            RendererCommand::KeepAlive(_) => {}
            RendererCommand::RendererInitFinalizeData(_) => {
                runtime.frontend.set_init_state(InitState::Finalized);
            }
            RendererCommand::RendererInitProgressUpdate(_) => {}
            RendererCommand::RendererEngineReady(_) => {}
            _ => {
                logger::trace!("IPC: deferring command until init finalized (skeleton)");
            }
        },
        InitState::Finalized => runtime.handle_running_command(cmd),
    }
}
