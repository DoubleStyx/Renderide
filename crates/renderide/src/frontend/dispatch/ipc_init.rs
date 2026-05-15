//! Pure IPC command routing by [`crate::frontend::InitState`].

use std::{borrow::Cow, sync::LazyLock};

use crate::frontend::InitState;
use crate::shared::{
    HeadOutputDevice, RendererCommand, RendererInitData, RendererInitResult, TextureFormat,
};

use super::command_dispatch::RunningCommandEffect;
use super::command_kind::{RendererCommandLifecycle, classify_renderer_command};
use super::commands::handle_running_command;
use super::renderer_command_kind::renderer_command_variant_tag;

/// `Renderide <version>` or `Renderide <version>-<8-char-commit>`.
///
/// The commit suffix is supplied by `build.rs` via the `RENDERIDE_GIT_COMMIT`
/// rustc env var; an empty value means git was unavailable at build time and
/// the suffix is omitted.
static RENDERER_IDENTIFIER: LazyLock<String> = LazyLock::new(|| {
    let version = env!("CARGO_PKG_VERSION");
    let commit = env!("RENDERIDE_GIT_COMMIT");
    if commit.is_empty() {
        format!("Renderide {version}")
    } else {
        format!("Renderide {version}-{commit}")
    }
});

/// Renderer capabilities reported during the init handshake.
///
/// Runtime builds this from GPU, asset, and output-device policy so pure frontend routing does not
/// depend on backend or XR modules.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RendererInitCapabilities {
    /// Human-readable stereo mode reported to the host.
    pub stereo_rendering_mode: Cow<'static, str>,
    /// Maximum host texture dimension accepted by the renderer.
    pub max_texture_size: i32,
    /// Host texture formats accepted by the renderer.
    pub supported_texture_formats: Vec<TextureFormat>,
}

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

/// Runtime-application effect for an init-routed IPC command.
#[derive(Debug)]
pub(crate) enum IpcDispatchEffect {
    /// Ignore the command.
    Ignore,
    /// Apply host init data and enter `InitReceived`.
    ApplyInitData(RendererInitData),
    /// Mark init as finalized.
    Finalize,
    /// Apply a decoded post-init running command.
    DispatchRunning(RunningCommandEffect),
    /// Defer the command until init is finalized.
    DeferUntilFinalized(Box<RendererCommand>),
    /// Mark init as fatally invalid because init data was expected first.
    FatalExpectedInitData {
        /// Actual command tag observed while waiting for init data.
        actual_tag: &'static str,
    },
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

/// Builds [`RendererInitResult`] after [`crate::shared::RendererInitData`] is applied.
///
/// `gpu_max_texture_dim_2d` should be [`None`] until a [`wgpu::Device`] exists; the host only
/// accepts one init result, so startup normally reports the renderer policy max before GPU init.
pub(crate) fn build_renderer_init_result(
    output_device: HeadOutputDevice,
    capabilities: RendererInitCapabilities,
) -> RendererInitResult {
    RendererInitResult {
        actual_output_device: output_device,
        renderer_identifier: Some(Cow::Borrowed(RENDERER_IDENTIFIER.as_str())),
        main_window_handle_ptr: 0,
        stereo_rendering_mode: Some(capabilities.stereo_rendering_mode),
        max_texture_size: capabilities.max_texture_size,
        is_gpu_texture_pot_byte_aligned: true,
        supported_texture_formats: capabilities.supported_texture_formats,
    }
}

/// Decodes a single command according to the current init phase.
pub(crate) fn dispatch_ipc_command(
    init_state: InitState,
    cmd: RendererCommand,
) -> IpcDispatchEffect {
    let decision = init_dispatch_decision(init_state, classify_renderer_command(&cmd).lifecycle());
    match decision {
        InitDispatchDecision::Ignore => IpcDispatchEffect::Ignore,
        InitDispatchDecision::ApplyInitData => match cmd {
            RendererCommand::RendererInitData(d) => IpcDispatchEffect::ApplyInitData(d),
            _ => IpcDispatchEffect::FatalExpectedInitData {
                actual_tag: renderer_command_variant_tag(&cmd),
            },
        },
        InitDispatchDecision::Finalize => IpcDispatchEffect::Finalize,
        InitDispatchDecision::DispatchRunning => {
            IpcDispatchEffect::DispatchRunning(handle_running_command(cmd))
        }
        InitDispatchDecision::DeferUntilFinalized => {
            IpcDispatchEffect::DeferUntilFinalized(Box::new(cmd))
        }
        InitDispatchDecision::FatalExpectedInitData => IpcDispatchEffect::FatalExpectedInitData {
            actual_tag: renderer_command_variant_tag(&cmd),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{
        InitDispatchDecision, IpcDispatchEffect, RendererInitCapabilities,
        build_renderer_init_result, dispatch_ipc_command, init_dispatch_decision,
    };
    use crate::frontend::InitState;
    use crate::frontend::dispatch::command_dispatch::RunningCommandEffect;
    use crate::frontend::dispatch::command_kind::RendererCommandLifecycle;
    use crate::shared::{
        FrameSubmitData, KeepAlive, QualityConfig, RendererCommand, RendererEngineReady,
        RendererInitData, RendererInitFinalizeData, RendererInitProgressUpdate, TextureFormat,
    };

    #[test]
    fn init_result_reports_supported_formats_and_renderer_identity() {
        let result = build_renderer_init_result(
            Default::default(),
            RendererInitCapabilities {
                stereo_rendering_mode: "None".into(),
                max_texture_size: 4096,
                supported_texture_formats: vec![TextureFormat::RGBA32],
            },
        );

        assert!(matches!(
            result.renderer_identifier,
            Some(Cow::Borrowed(identifier)) if identifier.starts_with("Renderide ")
        ));
        assert!(!result.supported_texture_formats.is_empty());
        assert_eq!(result.max_texture_size, 4096);
    }

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

    #[test]
    fn dispatch_decodes_init_data_effect() {
        assert!(matches!(
            dispatch_ipc_command(
                InitState::Uninitialized,
                RendererCommand::RendererInitData(RendererInitData::default())
            ),
            IpcDispatchEffect::ApplyInitData(_)
        ));
    }

    #[test]
    fn fatal_init_order_error_reports_actual_tag() {
        assert!(matches!(
            dispatch_ipc_command(
                InitState::Uninitialized,
                RendererCommand::QualityConfig(QualityConfig::default())
            ),
            IpcDispatchEffect::FatalExpectedInitData {
                actual_tag: "QualityConfig"
            }
        ));
    }

    #[test]
    fn dispatch_decodes_finalize_effect() {
        assert!(matches!(
            dispatch_ipc_command(
                InitState::InitReceived,
                RendererCommand::RendererInitFinalizeData(RendererInitFinalizeData::default())
            ),
            IpcDispatchEffect::Finalize
        ));
    }

    #[test]
    fn dispatch_defers_running_command_during_init_received() {
        assert!(matches!(
            dispatch_ipc_command(
                InitState::InitReceived,
                RendererCommand::QualityConfig(QualityConfig::default())
            ),
            IpcDispatchEffect::DeferUntilFinalized(cmd)
                if matches!(*cmd, RendererCommand::QualityConfig(_))
        ));
    }

    #[test]
    fn dispatch_ignores_init_received_noise() {
        for cmd in [
            RendererCommand::KeepAlive(KeepAlive::default()),
            RendererCommand::RendererInitProgressUpdate(RendererInitProgressUpdate::default()),
            RendererCommand::RendererEngineReady(RendererEngineReady::default()),
        ] {
            assert!(matches!(
                dispatch_ipc_command(InitState::InitReceived, cmd),
                IpcDispatchEffect::Ignore
            ));
        }
    }

    #[test]
    fn dispatch_decodes_finalized_frame_submit_to_running_effect() {
        match dispatch_ipc_command(
            InitState::Finalized,
            RendererCommand::FrameSubmitData(FrameSubmitData {
                frame_index: 9,
                ..Default::default()
            }),
        ) {
            IpcDispatchEffect::DispatchRunning(RunningCommandEffect::FrameSubmit(data)) => {
                assert_eq!(data.frame_index, 9);
            }
            other => panic!("unexpected effect: {other:?}"),
        }
    }
}
