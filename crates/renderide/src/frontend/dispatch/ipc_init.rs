//! Pure IPC command routing by [`crate::frontend::InitState`].

use std::borrow::Cow;

use crate::build_info::renderer_identifier;
use crate::frontend::InitState;
use crate::shared::{
    HeadOutputDevice, RendererCommand, RendererInitData, RendererInitResult, TextureFormat,
};

use super::command_dispatch::RunningCommandEffect;
use super::command_kind::{RendererCommandLifecycle, classify_renderer_command};
use super::commands::handle_running_command;
use super::renderer_command_kind::renderer_command_variant_tag;

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
            RendererCommandLifecycle::KeepAlive | RendererCommandLifecycle::InitProgressUpdate => {
                InitDispatchDecision::Ignore
            }
            RendererCommandLifecycle::InitFinalize => InitDispatchDecision::Finalize,
            _ => InitDispatchDecision::DeferUntilFinalized,
        },
        InitState::Finalized => InitDispatchDecision::DispatchRunning,
    }
}

/// Returns whether a command may be processed after init data and before init finalization.
pub(crate) fn can_dispatch_before_init_finalize(cmd: &RendererCommand) -> bool {
    matches!(
        cmd,
        RendererCommand::FreeSharedMemoryView(_)
            | RendererCommand::SetWindowIcon(_)
            | RendererCommand::MeshUploadData(_)
            | RendererCommand::MeshUnload(_)
            | RendererCommand::ShaderUpload(_)
            | RendererCommand::ShaderUnload(_)
            | RendererCommand::MaterialPropertyIdRequest(_)
            | RendererCommand::MaterialsUpdateBatch(_)
            | RendererCommand::UnloadMaterial(_)
            | RendererCommand::UnloadMaterialPropertyBlock(_)
            | RendererCommand::SetTexture2DFormat(_)
            | RendererCommand::SetTexture2DProperties(_)
            | RendererCommand::SetTexture2DData(_)
            | RendererCommand::UnloadTexture2D(_)
            | RendererCommand::SetDesktopTextureProperties(_)
            | RendererCommand::UnloadDesktopTexture(_)
            | RendererCommand::SetTexture3DFormat(_)
            | RendererCommand::SetTexture3DProperties(_)
            | RendererCommand::SetTexture3DData(_)
            | RendererCommand::UnloadTexture3D(_)
            | RendererCommand::SetCubemapFormat(_)
            | RendererCommand::SetCubemapProperties(_)
            | RendererCommand::SetCubemapData(_)
            | RendererCommand::UnloadCubemap(_)
            | RendererCommand::SetRenderTextureFormat(_)
            | RendererCommand::UnloadRenderTexture(_)
            | RendererCommand::VideoTextureLoad(_)
            | RendererCommand::VideoTextureUpdate(_)
            | RendererCommand::VideoTextureProperties(_)
            | RendererCommand::VideoTextureStartAudioTrack(_)
            | RendererCommand::UnloadVideoTexture(_)
            | RendererCommand::PointRenderBufferUpload(_)
            | RendererCommand::PointRenderBufferUnload(_)
            | RendererCommand::TrailRenderBufferUpload(_)
            | RendererCommand::TrailRenderBufferUnload(_)
            | RendererCommand::GaussianSplatUploadRaw(_)
            | RendererCommand::GaussianSplatUploadEncoded(_)
            | RendererCommand::UnloadGaussianSplat(_)
            | RendererCommand::LightsBufferRendererSubmission(_)
    )
}

/// Computes the init-routing action for a concrete command.
pub(crate) fn init_dispatch_decision_for_command(
    init_state: InitState,
    cmd: &RendererCommand,
) -> InitDispatchDecision {
    if init_state == InitState::InitReceived && can_dispatch_before_init_finalize(cmd) {
        return InitDispatchDecision::DispatchRunning;
    }
    init_dispatch_decision(init_state, classify_renderer_command(cmd).lifecycle())
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
        renderer_identifier: Some(Cow::Borrowed(renderer_identifier())),
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
    let decision = init_dispatch_decision_for_command(init_state, &cmd);
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
        build_renderer_init_result, can_dispatch_before_init_finalize, dispatch_ipc_command,
        init_dispatch_decision, init_dispatch_decision_for_command,
    };
    use crate::frontend::InitState;
    use crate::frontend::dispatch::command_dispatch::RunningCommandEffect;
    use crate::frontend::dispatch::command_kind::RendererCommandLifecycle;
    use crate::shared::*;

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
    fn init_received_ignores_progress_finalizes_and_defers_lifecycle_commands() {
        assert_eq!(
            init_dispatch_decision(
                InitState::InitReceived,
                RendererCommandLifecycle::EngineReady
            ),
            InitDispatchDecision::DeferUntilFinalized
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

    fn pre_finalize_asset_commands() -> Vec<RendererCommand> {
        vec![
            RendererCommand::FreeSharedMemoryView(FreeSharedMemoryView::default()),
            RendererCommand::SetWindowIcon(SetWindowIcon::default()),
            RendererCommand::MeshUploadData(MeshUploadData::default()),
            RendererCommand::MeshUnload(MeshUnload::default()),
            RendererCommand::ShaderUpload(ShaderUpload::default()),
            RendererCommand::ShaderUnload(ShaderUnload::default()),
            RendererCommand::MaterialPropertyIdRequest(MaterialPropertyIdRequest::default()),
            RendererCommand::MaterialsUpdateBatch(MaterialsUpdateBatch::default()),
            RendererCommand::UnloadMaterial(UnloadMaterial::default()),
            RendererCommand::UnloadMaterialPropertyBlock(UnloadMaterialPropertyBlock::default()),
            RendererCommand::SetTexture2DFormat(SetTexture2DFormat::default()),
            RendererCommand::SetTexture2DProperties(SetTexture2DProperties::default()),
            RendererCommand::SetTexture2DData(SetTexture2DData::default()),
            RendererCommand::UnloadTexture2D(UnloadTexture2D::default()),
            RendererCommand::SetDesktopTextureProperties(SetDesktopTextureProperties::default()),
            RendererCommand::UnloadDesktopTexture(UnloadDesktopTexture::default()),
            RendererCommand::SetTexture3DFormat(SetTexture3DFormat::default()),
            RendererCommand::SetTexture3DProperties(SetTexture3DProperties::default()),
            RendererCommand::SetTexture3DData(SetTexture3DData::default()),
            RendererCommand::UnloadTexture3D(UnloadTexture3D::default()),
            RendererCommand::SetCubemapFormat(SetCubemapFormat::default()),
            RendererCommand::SetCubemapProperties(SetCubemapProperties::default()),
            RendererCommand::SetCubemapData(SetCubemapData::default()),
            RendererCommand::UnloadCubemap(UnloadCubemap::default()),
            RendererCommand::SetRenderTextureFormat(SetRenderTextureFormat::default()),
            RendererCommand::UnloadRenderTexture(UnloadRenderTexture::default()),
            RendererCommand::VideoTextureLoad(VideoTextureLoad::default()),
            RendererCommand::VideoTextureUpdate(VideoTextureUpdate::default()),
            RendererCommand::VideoTextureProperties(VideoTextureProperties::default()),
            RendererCommand::VideoTextureStartAudioTrack(VideoTextureStartAudioTrack::default()),
            RendererCommand::UnloadVideoTexture(UnloadVideoTexture::default()),
            RendererCommand::PointRenderBufferUpload(PointRenderBufferUpload::default()),
            RendererCommand::PointRenderBufferUnload(PointRenderBufferUnload::default()),
            RendererCommand::TrailRenderBufferUpload(TrailRenderBufferUpload::default()),
            RendererCommand::TrailRenderBufferUnload(TrailRenderBufferUnload::default()),
            RendererCommand::GaussianSplatUploadRaw(GaussianSplatUploadRaw::default()),
            RendererCommand::GaussianSplatUploadEncoded(GaussianSplatUploadEncoded::default()),
            RendererCommand::UnloadGaussianSplat(UnloadGaussianSplat::default()),
            RendererCommand::LightsBufferRendererSubmission(
                LightsBufferRendererSubmission::default(),
            ),
        ]
    }

    #[test]
    fn init_received_dispatches_startup_asset_commands() {
        for cmd in pre_finalize_asset_commands() {
            assert!(
                can_dispatch_before_init_finalize(&cmd),
                "command should be allowed before init finalization: {cmd:?}"
            );
            assert_eq!(
                init_dispatch_decision_for_command(InitState::InitReceived, &cmd),
                InitDispatchDecision::DispatchRunning
            );
            assert!(
                matches!(
                    dispatch_ipc_command(InitState::InitReceived, cmd),
                    IpcDispatchEffect::DispatchRunning(_)
                ),
                "asset command should dispatch before init finalization"
            );
        }
    }

    #[test]
    fn init_received_defers_non_asset_commands() {
        for cmd in [
            RendererCommand::RendererEngineReady(RendererEngineReady::default()),
            RendererCommand::FrameSubmitData(FrameSubmitData::default()),
            RendererCommand::DesktopConfig(DesktopConfig::default()),
            RendererCommand::QualityConfig(QualityConfig::default()),
            RendererCommand::RenderDecouplingConfig(RenderDecouplingConfig::default()),
        ] {
            assert!(!can_dispatch_before_init_finalize(&cmd));
            assert_eq!(
                init_dispatch_decision_for_command(InitState::InitReceived, &cmd),
                InitDispatchDecision::DeferUntilFinalized
            );
        }
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
        ] {
            assert!(matches!(
                dispatch_ipc_command(InitState::InitReceived, cmd),
                IpcDispatchEffect::Ignore
            ));
        }
    }

    #[test]
    fn dispatch_defers_engine_ready_during_init_received() {
        assert!(matches!(
            dispatch_ipc_command(
                InitState::InitReceived,
                RendererCommand::RendererEngineReady(RendererEngineReady::default())
            ),
            IpcDispatchEffect::DeferUntilFinalized(cmd)
                if matches!(*cmd, RendererCommand::RendererEngineReady(_))
        ));
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
