//! Exhaustive command classification for init routing and diagnostics.

use crate::shared::RendererCommand;

/// Command lifecycle class used by pure init-routing decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RendererCommandLifecycle {
    /// Host keep-alive command.
    KeepAlive,
    /// Initial host renderer init payload.
    InitData,
    /// Init progress notification.
    InitProgressUpdate,
    /// Host engine-ready lifecycle notification.
    EngineReady,
    /// Host init-finalize payload.
    InitFinalize,
    /// Frame submit payload.
    FrameSubmit,
    /// Any normal post-init command.
    Running,
}

/// Stable metadata for a [`RendererCommand`] variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RendererCommandInfo {
    tag: &'static str,
    lifecycle: RendererCommandLifecycle,
}

impl RendererCommandInfo {
    /// Stable diagnostic tag for the command variant.
    pub(crate) fn tag(self) -> &'static str {
        self.tag
    }

    /// Lifecycle classification used by init routing.
    pub(crate) fn lifecycle(self) -> RendererCommandLifecycle {
        self.lifecycle
    }
}

/// Returns stable metadata for the given renderer command.
pub(crate) fn classify_renderer_command(cmd: &RendererCommand) -> RendererCommandInfo {
    use RendererCommandLifecycle::{
        EngineReady, FrameSubmit, InitData, InitFinalize, InitProgressUpdate, KeepAlive, Running,
    };

    let (tag, lifecycle) = match cmd {
        RendererCommand::RendererInitData(_) => ("RendererInitData", InitData),
        RendererCommand::RendererInitResult(_) => ("RendererInitResult", Running),
        RendererCommand::RendererInitProgressUpdate(_) => {
            ("RendererInitProgressUpdate", InitProgressUpdate)
        }
        RendererCommand::RendererInitFinalizeData(_) => ("RendererInitFinalizeData", InitFinalize),
        RendererCommand::RendererEngineReady(_) => ("RendererEngineReady", EngineReady),
        RendererCommand::RendererShutdownRequest(_) => ("RendererShutdownRequest", Running),
        RendererCommand::RendererShutdown(_) => ("RendererShutdown", Running),
        RendererCommand::KeepAlive(_) => ("KeepAlive", KeepAlive),
        RendererCommand::RendererParentWindow(_) => ("RendererParentWindow", Running),
        RendererCommand::FreeSharedMemoryView(_) => ("FreeSharedMemoryView", Running),
        RendererCommand::SetWindowIcon(_) => ("SetWindowIcon", Running),
        RendererCommand::SetWindowIconResult(_) => ("SetWindowIconResult", Running),
        RendererCommand::SetTaskbarProgress(_) => ("SetTaskbarProgress", Running),
        RendererCommand::FrameStartData(_) => ("FrameStartData", Running),
        RendererCommand::FrameSubmitData(_) => ("FrameSubmitData", FrameSubmit),
        RendererCommand::PostProcessingConfig(_) => ("PostProcessingConfig", Running),
        RendererCommand::QualityConfig(_) => ("QualityConfig", Running),
        RendererCommand::ResolutionConfig(_) => ("ResolutionConfig", Running),
        RendererCommand::DesktopConfig(_) => ("DesktopConfig", Running),
        RendererCommand::GaussianSplatConfig(_) => ("GaussianSplatConfig", Running),
        RendererCommand::RenderDecouplingConfig(_) => ("RenderDecouplingConfig", Running),
        RendererCommand::MeshUploadData(_) => ("MeshUploadData", Running),
        RendererCommand::MeshUnload(_) => ("MeshUnload", Running),
        RendererCommand::MeshUploadResult(_) => ("MeshUploadResult", Running),
        RendererCommand::ShaderUpload(_) => ("ShaderUpload", Running),
        RendererCommand::ShaderUnload(_) => ("ShaderUnload", Running),
        RendererCommand::ShaderUploadResult(_) => ("ShaderUploadResult", Running),
        RendererCommand::MaterialPropertyIdRequest(_) => ("MaterialPropertyIdRequest", Running),
        RendererCommand::MaterialPropertyIdResult(_) => ("MaterialPropertyIdResult", Running),
        RendererCommand::MaterialsUpdateBatch(_) => ("MaterialsUpdateBatch", Running),
        RendererCommand::MaterialsUpdateBatchResult(_) => ("MaterialsUpdateBatchResult", Running),
        RendererCommand::UnloadMaterial(_) => ("UnloadMaterial", Running),
        RendererCommand::UnloadMaterialPropertyBlock(_) => ("UnloadMaterialPropertyBlock", Running),
        RendererCommand::SetTexture2DFormat(_) => ("SetTexture2DFormat", Running),
        RendererCommand::SetTexture2DProperties(_) => ("SetTexture2DProperties", Running),
        RendererCommand::SetTexture2DData(_) => ("SetTexture2DData", Running),
        RendererCommand::SetTexture2DResult(_) => ("SetTexture2DResult", Running),
        RendererCommand::UnloadTexture2D(_) => ("UnloadTexture2D", Running),
        RendererCommand::SetTexture3DFormat(_) => ("SetTexture3DFormat", Running),
        RendererCommand::SetTexture3DProperties(_) => ("SetTexture3DProperties", Running),
        RendererCommand::SetTexture3DData(_) => ("SetTexture3DData", Running),
        RendererCommand::SetTexture3DResult(_) => ("SetTexture3DResult", Running),
        RendererCommand::UnloadTexture3D(_) => ("UnloadTexture3D", Running),
        RendererCommand::SetCubemapFormat(_) => ("SetCubemapFormat", Running),
        RendererCommand::SetCubemapProperties(_) => ("SetCubemapProperties", Running),
        RendererCommand::SetCubemapData(_) => ("SetCubemapData", Running),
        RendererCommand::SetCubemapResult(_) => ("SetCubemapResult", Running),
        RendererCommand::UnloadCubemap(_) => ("UnloadCubemap", Running),
        RendererCommand::SetRenderTextureFormat(_) => ("SetRenderTextureFormat", Running),
        RendererCommand::RenderTextureResult(_) => ("RenderTextureResult", Running),
        RendererCommand::UnloadRenderTexture(_) => ("UnloadRenderTexture", Running),
        RendererCommand::SetDesktopTextureProperties(_) => ("SetDesktopTextureProperties", Running),
        RendererCommand::DesktopTexturePropertiesUpdate(_) => {
            ("DesktopTexturePropertiesUpdate", Running)
        }
        RendererCommand::UnloadDesktopTexture(_) => ("UnloadDesktopTexture", Running),
        RendererCommand::PointRenderBufferUpload(_) => ("PointRenderBufferUpload", Running),
        RendererCommand::PointRenderBufferConsumed(_) => ("PointRenderBufferConsumed", Running),
        RendererCommand::PointRenderBufferUnload(_) => ("PointRenderBufferUnload", Running),
        RendererCommand::TrailRenderBufferUpload(_) => ("TrailRenderBufferUpload", Running),
        RendererCommand::TrailRenderBufferConsumed(_) => ("TrailRenderBufferConsumed", Running),
        RendererCommand::TrailRenderBufferUnload(_) => ("TrailRenderBufferUnload", Running),
        RendererCommand::GaussianSplatUploadRaw(_) => ("GaussianSplatUploadRaw", Running),
        RendererCommand::GaussianSplatUploadEncoded(_) => ("GaussianSplatUploadEncoded", Running),
        RendererCommand::GaussianSplatResult(_) => ("GaussianSplatResult", Running),
        RendererCommand::UnloadGaussianSplat(_) => ("UnloadGaussianSplat", Running),
        RendererCommand::LightsBufferRendererSubmission(_) => {
            ("LightsBufferRendererSubmission", Running)
        }
        RendererCommand::LightsBufferRendererConsumed(_) => {
            ("LightsBufferRendererConsumed", Running)
        }
        RendererCommand::ReflectionProbeRenderResult(_) => ("ReflectionProbeRenderResult", Running),
        RendererCommand::VideoTextureLoad(_) => ("VideoTextureLoad", Running),
        RendererCommand::VideoTextureUpdate(_) => ("VideoTextureUpdate", Running),
        RendererCommand::VideoTextureReady(_) => ("VideoTextureReady", Running),
        RendererCommand::VideoTextureChanged(_) => ("VideoTextureChanged", Running),
        RendererCommand::VideoTextureProperties(_) => ("VideoTextureProperties", Running),
        RendererCommand::VideoTextureStartAudioTrack(_) => ("VideoTextureStartAudioTrack", Running),
        RendererCommand::UnloadVideoTexture(_) => ("UnloadVideoTexture", Running),
    };

    RendererCommandInfo { tag, lifecycle }
}

/// Returns a stable tag for logging and unhandled-command counters.
pub(crate) fn renderer_command_variant_tag(cmd: &RendererCommand) -> &'static str {
    classify_renderer_command(cmd).tag()
}

#[cfg(test)]
mod tests {
    use super::{
        RendererCommandLifecycle, classify_renderer_command, renderer_command_variant_tag,
    };
    use crate::shared::*;

    fn assert_tag(cmd: RendererCommand, expected: &'static str) {
        assert_eq!(renderer_command_variant_tag(&cmd), expected);
    }

    #[test]
    fn lifecycle_classifies_init_and_frame_commands() {
        assert_eq!(
            classify_renderer_command(&RendererCommand::RendererInitData(
                RendererInitData::default()
            ))
            .lifecycle(),
            RendererCommandLifecycle::InitData
        );
        assert_eq!(
            classify_renderer_command(&RendererCommand::RendererInitProgressUpdate(
                RendererInitProgressUpdate::default()
            ))
            .lifecycle(),
            RendererCommandLifecycle::InitProgressUpdate
        );
        assert_eq!(
            classify_renderer_command(&RendererCommand::RendererEngineReady(
                RendererEngineReady::default()
            ))
            .lifecycle(),
            RendererCommandLifecycle::EngineReady
        );
        assert_eq!(
            classify_renderer_command(&RendererCommand::RendererInitFinalizeData(
                RendererInitFinalizeData::default()
            ))
            .lifecycle(),
            RendererCommandLifecycle::InitFinalize
        );
        assert_eq!(
            classify_renderer_command(
                &RendererCommand::FrameSubmitData(FrameSubmitData::default())
            )
            .lifecycle(),
            RendererCommandLifecycle::FrameSubmit
        );
    }

    #[test]
    fn lifecycle_classifies_keepalive_and_running_commands() {
        assert_eq!(
            classify_renderer_command(&RendererCommand::KeepAlive(KeepAlive::default()))
                .lifecycle(),
            RendererCommandLifecycle::KeepAlive
        );
        assert_eq!(
            classify_renderer_command(&RendererCommand::QualityConfig(QualityConfig::default()))
                .lifecycle(),
            RendererCommandLifecycle::Running
        );
    }

    #[test]
    fn lifecycle_window_and_frame_command_tags_are_stable() {
        assert_tag(
            RendererCommand::RendererInitData(RendererInitData::default()),
            "RendererInitData",
        );
        assert_tag(
            RendererCommand::RendererInitResult(RendererInitResult::default()),
            "RendererInitResult",
        );
        assert_tag(
            RendererCommand::RendererInitProgressUpdate(RendererInitProgressUpdate::default()),
            "RendererInitProgressUpdate",
        );
        assert_tag(
            RendererCommand::RendererInitFinalizeData(RendererInitFinalizeData::default()),
            "RendererInitFinalizeData",
        );
        assert_tag(
            RendererCommand::RendererEngineReady(RendererEngineReady::default()),
            "RendererEngineReady",
        );
        assert_tag(
            RendererCommand::RendererShutdownRequest(RendererShutdownRequest::default()),
            "RendererShutdownRequest",
        );
        assert_tag(
            RendererCommand::RendererShutdown(RendererShutdown::default()),
            "RendererShutdown",
        );
        assert_tag(
            RendererCommand::KeepAlive(KeepAlive::default()),
            "KeepAlive",
        );
        assert_tag(
            RendererCommand::RendererParentWindow(RendererParentWindow::default()),
            "RendererParentWindow",
        );
        assert_tag(
            RendererCommand::FreeSharedMemoryView(FreeSharedMemoryView::default()),
            "FreeSharedMemoryView",
        );
        assert_tag(
            RendererCommand::SetWindowIcon(SetWindowIcon::default()),
            "SetWindowIcon",
        );
        assert_tag(
            RendererCommand::SetWindowIconResult(SetWindowIconResult::default()),
            "SetWindowIconResult",
        );
        assert_tag(
            RendererCommand::SetTaskbarProgress(SetTaskbarProgress::default()),
            "SetTaskbarProgress",
        );
        assert_tag(
            RendererCommand::FrameStartData(FrameStartData::default()),
            "FrameStartData",
        );
        assert_tag(
            RendererCommand::FrameSubmitData(FrameSubmitData::default()),
            "FrameSubmitData",
        );
    }

    #[test]
    fn config_mesh_shader_and_material_command_tags_are_stable() {
        assert_tag(
            RendererCommand::PostProcessingConfig(PostProcessingConfig::default()),
            "PostProcessingConfig",
        );
        assert_tag(
            RendererCommand::QualityConfig(QualityConfig::default()),
            "QualityConfig",
        );
        assert_tag(
            RendererCommand::ResolutionConfig(ResolutionConfig::default()),
            "ResolutionConfig",
        );
        assert_tag(
            RendererCommand::DesktopConfig(DesktopConfig::default()),
            "DesktopConfig",
        );
        assert_tag(
            RendererCommand::GaussianSplatConfig(GaussianSplatConfig::default()),
            "GaussianSplatConfig",
        );
        assert_tag(
            RendererCommand::RenderDecouplingConfig(RenderDecouplingConfig::default()),
            "RenderDecouplingConfig",
        );
        assert_tag(
            RendererCommand::MeshUploadData(MeshUploadData::default()),
            "MeshUploadData",
        );
        assert_tag(
            RendererCommand::MeshUnload(MeshUnload::default()),
            "MeshUnload",
        );
        assert_tag(
            RendererCommand::MeshUploadResult(MeshUploadResult::default()),
            "MeshUploadResult",
        );
        assert_tag(
            RendererCommand::ShaderUpload(ShaderUpload::default()),
            "ShaderUpload",
        );
        assert_tag(
            RendererCommand::ShaderUnload(ShaderUnload::default()),
            "ShaderUnload",
        );
        assert_tag(
            RendererCommand::ShaderUploadResult(ShaderUploadResult::default()),
            "ShaderUploadResult",
        );
        assert_tag(
            RendererCommand::MaterialPropertyIdRequest(MaterialPropertyIdRequest::default()),
            "MaterialPropertyIdRequest",
        );
        assert_tag(
            RendererCommand::MaterialPropertyIdResult(MaterialPropertyIdResult::default()),
            "MaterialPropertyIdResult",
        );
        assert_tag(
            RendererCommand::MaterialsUpdateBatch(MaterialsUpdateBatch::default()),
            "MaterialsUpdateBatch",
        );
        assert_tag(
            RendererCommand::MaterialsUpdateBatchResult(MaterialsUpdateBatchResult::default()),
            "MaterialsUpdateBatchResult",
        );
        assert_tag(
            RendererCommand::UnloadMaterial(UnloadMaterial::default()),
            "UnloadMaterial",
        );
        assert_tag(
            RendererCommand::UnloadMaterialPropertyBlock(UnloadMaterialPropertyBlock::default()),
            "UnloadMaterialPropertyBlock",
        );
    }

    #[test]
    fn texture_command_tags_are_stable() {
        assert_tag(
            RendererCommand::SetTexture2DFormat(SetTexture2DFormat::default()),
            "SetTexture2DFormat",
        );
        assert_tag(
            RendererCommand::SetTexture2DProperties(SetTexture2DProperties::default()),
            "SetTexture2DProperties",
        );
        assert_tag(
            RendererCommand::SetTexture2DData(SetTexture2DData::default()),
            "SetTexture2DData",
        );
        assert_tag(
            RendererCommand::SetTexture2DResult(SetTexture2DResult::default()),
            "SetTexture2DResult",
        );
        assert_tag(
            RendererCommand::UnloadTexture2D(UnloadTexture2D::default()),
            "UnloadTexture2D",
        );
        assert_tag(
            RendererCommand::SetTexture3DFormat(SetTexture3DFormat::default()),
            "SetTexture3DFormat",
        );
        assert_tag(
            RendererCommand::SetTexture3DProperties(SetTexture3DProperties::default()),
            "SetTexture3DProperties",
        );
        assert_tag(
            RendererCommand::SetTexture3DData(SetTexture3DData::default()),
            "SetTexture3DData",
        );
        assert_tag(
            RendererCommand::SetTexture3DResult(SetTexture3DResult::default()),
            "SetTexture3DResult",
        );
        assert_tag(
            RendererCommand::UnloadTexture3D(UnloadTexture3D::default()),
            "UnloadTexture3D",
        );
        assert_tag(
            RendererCommand::SetCubemapFormat(SetCubemapFormat::default()),
            "SetCubemapFormat",
        );
        assert_tag(
            RendererCommand::SetCubemapProperties(SetCubemapProperties::default()),
            "SetCubemapProperties",
        );
        assert_tag(
            RendererCommand::SetCubemapData(SetCubemapData::default()),
            "SetCubemapData",
        );
        assert_tag(
            RendererCommand::SetCubemapResult(SetCubemapResult::default()),
            "SetCubemapResult",
        );
        assert_tag(
            RendererCommand::UnloadCubemap(UnloadCubemap::default()),
            "UnloadCubemap",
        );
        assert_tag(
            RendererCommand::SetRenderTextureFormat(SetRenderTextureFormat::default()),
            "SetRenderTextureFormat",
        );
        assert_tag(
            RendererCommand::RenderTextureResult(RenderTextureResult::default()),
            "RenderTextureResult",
        );
        assert_tag(
            RendererCommand::UnloadRenderTexture(UnloadRenderTexture::default()),
            "UnloadRenderTexture",
        );
        assert_tag(
            RendererCommand::SetDesktopTextureProperties(SetDesktopTextureProperties::default()),
            "SetDesktopTextureProperties",
        );
        assert_tag(
            RendererCommand::DesktopTexturePropertiesUpdate(
                DesktopTexturePropertiesUpdate::default(),
            ),
            "DesktopTexturePropertiesUpdate",
        );
        assert_tag(
            RendererCommand::UnloadDesktopTexture(UnloadDesktopTexture::default()),
            "UnloadDesktopTexture",
        );
    }

    #[test]
    fn render_buffer_splat_light_probe_and_video_command_tags_are_stable() {
        assert_tag(
            RendererCommand::PointRenderBufferUpload(PointRenderBufferUpload::default()),
            "PointRenderBufferUpload",
        );
        assert_tag(
            RendererCommand::PointRenderBufferConsumed(PointRenderBufferConsumed::default()),
            "PointRenderBufferConsumed",
        );
        assert_tag(
            RendererCommand::PointRenderBufferUnload(PointRenderBufferUnload::default()),
            "PointRenderBufferUnload",
        );
        assert_tag(
            RendererCommand::TrailRenderBufferUpload(TrailRenderBufferUpload::default()),
            "TrailRenderBufferUpload",
        );
        assert_tag(
            RendererCommand::TrailRenderBufferConsumed(TrailRenderBufferConsumed::default()),
            "TrailRenderBufferConsumed",
        );
        assert_tag(
            RendererCommand::TrailRenderBufferUnload(TrailRenderBufferUnload::default()),
            "TrailRenderBufferUnload",
        );
        assert_tag(
            RendererCommand::GaussianSplatUploadRaw(GaussianSplatUploadRaw::default()),
            "GaussianSplatUploadRaw",
        );
        assert_tag(
            RendererCommand::GaussianSplatUploadEncoded(GaussianSplatUploadEncoded::default()),
            "GaussianSplatUploadEncoded",
        );
        assert_tag(
            RendererCommand::GaussianSplatResult(GaussianSplatResult::default()),
            "GaussianSplatResult",
        );
        assert_tag(
            RendererCommand::UnloadGaussianSplat(UnloadGaussianSplat::default()),
            "UnloadGaussianSplat",
        );
        assert_tag(
            RendererCommand::LightsBufferRendererSubmission(
                LightsBufferRendererSubmission::default(),
            ),
            "LightsBufferRendererSubmission",
        );
        assert_tag(
            RendererCommand::LightsBufferRendererConsumed(LightsBufferRendererConsumed::default()),
            "LightsBufferRendererConsumed",
        );
        assert_tag(
            RendererCommand::ReflectionProbeRenderResult(ReflectionProbeRenderResult::default()),
            "ReflectionProbeRenderResult",
        );
        assert_tag(
            RendererCommand::VideoTextureLoad(VideoTextureLoad::default()),
            "VideoTextureLoad",
        );
        assert_tag(
            RendererCommand::VideoTextureUpdate(VideoTextureUpdate::default()),
            "VideoTextureUpdate",
        );
        assert_tag(
            RendererCommand::VideoTextureReady(VideoTextureReady::default()),
            "VideoTextureReady",
        );
        assert_tag(
            RendererCommand::VideoTextureChanged(VideoTextureChanged::default()),
            "VideoTextureChanged",
        );
        assert_tag(
            RendererCommand::VideoTextureProperties(VideoTextureProperties::default()),
            "VideoTextureProperties",
        );
        assert_tag(
            RendererCommand::VideoTextureStartAudioTrack(VideoTextureStartAudioTrack::default()),
            "VideoTextureStartAudioTrack",
        );
        assert_tag(
            RendererCommand::UnloadVideoTexture(UnloadVideoTexture::default()),
            "UnloadVideoTexture",
        );
    }
}
