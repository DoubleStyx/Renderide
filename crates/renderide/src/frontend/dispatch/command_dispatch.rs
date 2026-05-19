//! Pure post-handshake [`RendererCommand`] decoding.
//!
//! This module maps host IPC commands to domain effects. Runtime code owns applying those effects
//! to frontend, scene, backend, settings, and IPC state.

use crate::shared::{
    DesktopConfig, DesktopTexturePropertiesUpdate, FrameStartData, FrameSubmitData,
    GaussianSplatConfig, GaussianSplatUploadEncoded, GaussianSplatUploadRaw,
    LightsBufferRendererSubmission, MaterialPropertyIdRequest, MaterialsUpdateBatch, MeshUnload,
    MeshUploadData, PointRenderBufferUnload, PointRenderBufferUpload, QualityConfig,
    RenderDecouplingConfig, RendererCommand, SetCubemapData, SetCubemapFormat,
    SetCubemapProperties, SetDesktopTextureProperties, SetRenderTextureFormat, SetTexture2DData,
    SetTexture2DFormat, SetTexture2DProperties, SetTexture3DData, SetTexture3DFormat,
    SetTexture3DProperties, ShaderUnload, ShaderUpload, TrailRenderBufferUnload,
    TrailRenderBufferUpload, UnloadCubemap, UnloadDesktopTexture, UnloadGaussianSplat,
    UnloadRenderTexture, UnloadTexture2D, UnloadTexture3D, UnloadVideoTexture, VideoTextureLoad,
    VideoTextureProperties, VideoTextureStartAudioTrack, VideoTextureUpdate,
};

use super::renderer_command_kind::renderer_command_variant_tag;

/// Decoded post-handshake command effect for runtime-owned application.
#[derive(Debug)]
pub(crate) enum RunningCommandEffect {
    /// Host keep-alive command with no runtime state change.
    KeepAlive,
    /// Host requested orderly renderer shutdown.
    RequestShutdown,
    /// Host frame submit payload to apply to lock-step, scene, camera, and diagnostics state.
    FrameSubmit(FrameSubmitData),
    /// Mesh upload payload requiring shared memory and optional IPC acknowledgement.
    MeshUpload(MeshUploadData),
    /// Mesh unload payload for the backend mesh pool.
    MeshUnload(MeshUnload),
    /// Texture 2D format declaration.
    SetTexture2DFormat(SetTexture2DFormat),
    /// Texture 2D property update.
    SetTexture2DProperties(SetTexture2DProperties),
    /// Texture 2D data upload.
    SetTexture2DData(SetTexture2DData),
    /// Texture 2D unload command.
    UnloadTexture2D(UnloadTexture2D),
    /// Texture 3D format declaration.
    SetTexture3DFormat(SetTexture3DFormat),
    /// Texture 3D property update.
    SetTexture3DProperties(SetTexture3DProperties),
    /// Texture 3D data upload.
    SetTexture3DData(SetTexture3DData),
    /// Texture 3D unload command.
    UnloadTexture3D(UnloadTexture3D),
    /// Cubemap format declaration.
    SetCubemapFormat(SetCubemapFormat),
    /// Cubemap property update.
    SetCubemapProperties(SetCubemapProperties),
    /// Cubemap data upload.
    SetCubemapData(SetCubemapData),
    /// Cubemap unload command.
    UnloadCubemap(UnloadCubemap),
    /// Render texture format declaration.
    SetRenderTextureFormat(SetRenderTextureFormat),
    /// Render texture unload command.
    UnloadRenderTexture(UnloadRenderTexture),
    /// Desktop texture display properties.
    SetDesktopTextureProperties(SetDesktopTextureProperties),
    /// Desktop texture size/properties update.
    DesktopTexturePropertiesUpdate(DesktopTexturePropertiesUpdate),
    /// Desktop texture unload command.
    UnloadDesktopTexture(UnloadDesktopTexture),
    /// Point render buffer upload command.
    PointRenderBufferUpload(PointRenderBufferUpload),
    /// Point render buffer unload command.
    PointRenderBufferUnload(PointRenderBufferUnload),
    /// Trail render buffer upload command.
    TrailRenderBufferUpload(TrailRenderBufferUpload),
    /// Trail render buffer unload command.
    TrailRenderBufferUnload(TrailRenderBufferUnload),
    /// Gaussian splat renderer config.
    GaussianSplatConfig(GaussianSplatConfig),
    /// Raw Gaussian splat upload command.
    GaussianSplatUploadRaw(GaussianSplatUploadRaw),
    /// Encoded Gaussian splat upload command.
    GaussianSplatUploadEncoded(GaussianSplatUploadEncoded),
    /// Gaussian splat unload command.
    UnloadGaussianSplat(UnloadGaussianSplat),
    /// Renderer-sent point render buffer consumed ACK received from host, ignored.
    PointRenderBufferConsumed,
    /// Renderer-sent trail render buffer consumed ACK received from host, ignored.
    TrailRenderBufferConsumed,
    /// Renderer-sent Gaussian splat result received from host, ignored.
    GaussianSplatResult,
    /// Video texture load command.
    VideoTextureLoad(VideoTextureLoad),
    /// Video texture frame/update command.
    VideoTextureUpdate(VideoTextureUpdate),
    /// Video texture property update.
    VideoTextureProperties(VideoTextureProperties),
    /// Video texture audio-track start command.
    VideoTextureStartAudioTrack(VideoTextureStartAudioTrack),
    /// Video texture unload command.
    UnloadVideoTexture(UnloadVideoTexture),
    /// Host released a shared-memory view lease.
    FreeSharedMemoryView { buffer_id: i32 },
    /// Host request for material property IDs.
    MaterialPropertyIdRequest(MaterialPropertyIdRequest),
    /// Host material/property-block update batch.
    MaterialsUpdateBatch(MaterialsUpdateBatch),
    /// Material unload by asset ID.
    UnloadMaterial { asset_id: i32 },
    /// Material property-block unload by asset ID.
    UnloadMaterialPropertyBlock { asset_id: i32 },
    /// Shader upload command whose route resolution may run asynchronously.
    ShaderUpload(ShaderUpload),
    /// Shader unload command.
    ShaderUnload(ShaderUnload),
    /// Host-sent frame-start payload, currently used only for tracing.
    FrameStartData(Box<FrameStartData>),
    /// Host lights-buffer payload for scene light cache update.
    LightsBufferRendererSubmission(LightsBufferRendererSubmission),
    /// Host-sent lights consumed ACK, currently ignored.
    LightsBufferRendererConsumed,
    /// Host-sent render-texture result, currently ignored.
    RenderTextureResult,
    /// Host engine-ready lifecycle ACK received after init, currently ignored.
    RendererEngineReady,
    /// Host desktop display/framerate config.
    DesktopConfig(DesktopConfig),
    /// Host render decoupling config.
    RenderDecouplingConfig(RenderDecouplingConfig),
    /// Host quality settings used for realtime raster shadows and future render-quality gates.
    QualityConfig(QualityConfig),
    /// Command variant with no running-state handler yet.
    Unhandled { tag: &'static str },
}

/// Routes a post-handshake [`RendererCommand`] to a runtime-application effect.
pub(crate) fn dispatch_running_command(cmd: RendererCommand) -> RunningCommandEffect {
    match cmd {
        RendererCommand::KeepAlive(_) => RunningCommandEffect::KeepAlive,
        RendererCommand::RendererShutdown(_) | RendererCommand::RendererShutdownRequest(_) => {
            RunningCommandEffect::RequestShutdown
        }
        RendererCommand::FrameSubmitData(data) => RunningCommandEffect::FrameSubmit(data),
        RendererCommand::MeshUploadData(d) => RunningCommandEffect::MeshUpload(d),
        RendererCommand::MeshUnload(u) => RunningCommandEffect::MeshUnload(u),
        RendererCommand::SetTexture2DFormat(f) => RunningCommandEffect::SetTexture2DFormat(f),
        RendererCommand::SetTexture2DProperties(p) => {
            RunningCommandEffect::SetTexture2DProperties(p)
        }
        RendererCommand::SetTexture2DData(d) => RunningCommandEffect::SetTexture2DData(d),
        RendererCommand::UnloadTexture2D(u) => RunningCommandEffect::UnloadTexture2D(u),
        RendererCommand::SetTexture3DFormat(f) => RunningCommandEffect::SetTexture3DFormat(f),
        RendererCommand::SetTexture3DProperties(p) => {
            RunningCommandEffect::SetTexture3DProperties(p)
        }
        RendererCommand::SetTexture3DData(d) => RunningCommandEffect::SetTexture3DData(d),
        RendererCommand::UnloadTexture3D(u) => RunningCommandEffect::UnloadTexture3D(u),
        RendererCommand::SetCubemapFormat(f) => RunningCommandEffect::SetCubemapFormat(f),
        RendererCommand::SetCubemapProperties(p) => RunningCommandEffect::SetCubemapProperties(p),
        RendererCommand::SetCubemapData(d) => RunningCommandEffect::SetCubemapData(d),
        RendererCommand::UnloadCubemap(u) => RunningCommandEffect::UnloadCubemap(u),
        RendererCommand::SetRenderTextureFormat(f) => {
            RunningCommandEffect::SetRenderTextureFormat(f)
        }
        RendererCommand::UnloadRenderTexture(u) => RunningCommandEffect::UnloadRenderTexture(u),
        RendererCommand::VideoTextureLoad(l) => RunningCommandEffect::VideoTextureLoad(l),
        RendererCommand::VideoTextureUpdate(u) => RunningCommandEffect::VideoTextureUpdate(u),
        RendererCommand::VideoTextureProperties(p) => {
            RunningCommandEffect::VideoTextureProperties(p)
        }
        RendererCommand::VideoTextureStartAudioTrack(s) => {
            RunningCommandEffect::VideoTextureStartAudioTrack(s)
        }
        RendererCommand::UnloadVideoTexture(u) => RunningCommandEffect::UnloadVideoTexture(u),
        RendererCommand::FreeSharedMemoryView(f) => RunningCommandEffect::FreeSharedMemoryView {
            buffer_id: f.buffer_id,
        },
        RendererCommand::MaterialPropertyIdRequest(req) => {
            RunningCommandEffect::MaterialPropertyIdRequest(req)
        }
        RendererCommand::MaterialsUpdateBatch(batch) => {
            RunningCommandEffect::MaterialsUpdateBatch(batch)
        }
        RendererCommand::UnloadMaterial(u) => RunningCommandEffect::UnloadMaterial {
            asset_id: u.asset_id,
        },
        RendererCommand::UnloadMaterialPropertyBlock(u) => {
            RunningCommandEffect::UnloadMaterialPropertyBlock {
                asset_id: u.asset_id,
            }
        }
        RendererCommand::ShaderUpload(u) => RunningCommandEffect::ShaderUpload(u),
        RendererCommand::ShaderUnload(u) => RunningCommandEffect::ShaderUnload(u),
        RendererCommand::FrameStartData(fs) => RunningCommandEffect::FrameStartData(Box::new(fs)),
        RendererCommand::LightsBufferRendererSubmission(sub) => {
            RunningCommandEffect::LightsBufferRendererSubmission(sub)
        }
        RendererCommand::LightsBufferRendererConsumed(_) => {
            RunningCommandEffect::LightsBufferRendererConsumed
        }
        RendererCommand::RenderTextureResult(_) => RunningCommandEffect::RenderTextureResult,
        RendererCommand::RendererEngineReady(_) => RunningCommandEffect::RendererEngineReady,
        RendererCommand::DesktopConfig(cfg) => RunningCommandEffect::DesktopConfig(cfg),
        RendererCommand::RenderDecouplingConfig(cfg) => {
            RunningCommandEffect::RenderDecouplingConfig(cfg)
        }
        RendererCommand::QualityConfig(cfg) => RunningCommandEffect::QualityConfig(cfg),
        cmd => dispatch_auxiliary_asset_command(cmd),
    }
}

fn dispatch_auxiliary_asset_command(cmd: RendererCommand) -> RunningCommandEffect {
    match cmd {
        RendererCommand::SetDesktopTextureProperties(p) => {
            RunningCommandEffect::SetDesktopTextureProperties(p)
        }
        RendererCommand::DesktopTexturePropertiesUpdate(u) => {
            RunningCommandEffect::DesktopTexturePropertiesUpdate(u)
        }
        RendererCommand::UnloadDesktopTexture(u) => RunningCommandEffect::UnloadDesktopTexture(u),
        RendererCommand::PointRenderBufferUpload(u) => {
            RunningCommandEffect::PointRenderBufferUpload(u)
        }
        RendererCommand::PointRenderBufferUnload(u) => {
            RunningCommandEffect::PointRenderBufferUnload(u)
        }
        RendererCommand::TrailRenderBufferUpload(u) => {
            RunningCommandEffect::TrailRenderBufferUpload(u)
        }
        RendererCommand::TrailRenderBufferUnload(u) => {
            RunningCommandEffect::TrailRenderBufferUnload(u)
        }
        RendererCommand::GaussianSplatConfig(c) => RunningCommandEffect::GaussianSplatConfig(c),
        RendererCommand::GaussianSplatUploadRaw(u) => {
            RunningCommandEffect::GaussianSplatUploadRaw(u)
        }
        RendererCommand::GaussianSplatUploadEncoded(u) => {
            RunningCommandEffect::GaussianSplatUploadEncoded(u)
        }
        RendererCommand::UnloadGaussianSplat(u) => RunningCommandEffect::UnloadGaussianSplat(u),
        RendererCommand::PointRenderBufferConsumed(_) => {
            RunningCommandEffect::PointRenderBufferConsumed
        }
        RendererCommand::TrailRenderBufferConsumed(_) => {
            RunningCommandEffect::TrailRenderBufferConsumed
        }
        RendererCommand::GaussianSplatResult(_) => RunningCommandEffect::GaussianSplatResult,
        ref cmd => {
            let tag = renderer_command_variant_tag(cmd);
            RunningCommandEffect::Unhandled { tag }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::frontend::dispatch::command_dispatch::{
        RunningCommandEffect, dispatch_running_command,
    };
    use crate::shared::{
        DesktopConfig, FrameSubmitData, GaussianSplatUploadRaw, MaterialPropertyIdRequest,
        PointRenderBufferUpload, PostProcessingConfig, QualityConfig, RendererCommand,
        RendererShutdown, SetDesktopTextureProperties, SetTexture2DFormat, TrailRenderBufferUpload,
    };

    #[test]
    fn decodes_shutdown_to_session_effect() {
        assert!(matches!(
            dispatch_running_command(RendererCommand::RendererShutdown(
                RendererShutdown::default()
            )),
            RunningCommandEffect::RequestShutdown
        ));
    }

    #[test]
    fn decodes_frame_submit_to_scene_effect() {
        let effect = dispatch_running_command(RendererCommand::FrameSubmitData(FrameSubmitData {
            frame_index: 17,
            ..Default::default()
        }));

        match effect {
            RunningCommandEffect::FrameSubmit(data) => assert_eq!(data.frame_index, 17),
            other => panic!("unexpected effect: {other:?}"),
        }
    }

    #[test]
    fn decodes_backend_asset_commands() {
        assert!(matches!(
            dispatch_running_command(RendererCommand::SetTexture2DFormat(
                SetTexture2DFormat::default()
            )),
            RunningCommandEffect::SetTexture2DFormat(_)
        ));
    }

    #[test]
    fn decodes_material_property_request() {
        assert!(matches!(
            dispatch_running_command(RendererCommand::MaterialPropertyIdRequest(
                MaterialPropertyIdRequest::default()
            )),
            RunningCommandEffect::MaterialPropertyIdRequest(_)
        ));
    }

    #[test]
    fn decodes_desktop_config_to_settings_effect() {
        assert!(matches!(
            dispatch_running_command(RendererCommand::DesktopConfig(DesktopConfig::default())),
            RunningCommandEffect::DesktopConfig(_)
        ));
    }

    #[test]
    fn decodes_unity_asset_family_commands_to_handlers() {
        assert!(matches!(
            dispatch_running_command(RendererCommand::SetDesktopTextureProperties(
                SetDesktopTextureProperties::default()
            )),
            RunningCommandEffect::SetDesktopTextureProperties(_)
        ));
        assert!(matches!(
            dispatch_running_command(RendererCommand::PointRenderBufferUpload(
                PointRenderBufferUpload::default()
            )),
            RunningCommandEffect::PointRenderBufferUpload(_)
        ));
        assert!(matches!(
            dispatch_running_command(RendererCommand::TrailRenderBufferUpload(
                TrailRenderBufferUpload::default()
            )),
            RunningCommandEffect::TrailRenderBufferUpload(_)
        ));
        assert!(matches!(
            dispatch_running_command(RendererCommand::GaussianSplatUploadRaw(
                GaussianSplatUploadRaw::default()
            )),
            RunningCommandEffect::GaussianSplatUploadRaw(_)
        ));
    }

    #[test]
    fn decodes_unhandled_commands_with_stable_tag() {
        match dispatch_running_command(RendererCommand::PostProcessingConfig(
            PostProcessingConfig::default(),
        )) {
            RunningCommandEffect::Unhandled { tag } => assert_eq!(tag, "PostProcessingConfig"),
            other => panic!("unexpected effect: {other:?}"),
        }
    }

    #[test]
    fn decodes_quality_config_to_backend_effect() {
        assert!(matches!(
            dispatch_running_command(RendererCommand::QualityConfig(QualityConfig::default())),
            RunningCommandEffect::QualityConfig(_)
        ));
    }
}
