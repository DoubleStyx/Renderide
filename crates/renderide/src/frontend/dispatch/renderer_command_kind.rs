//! Stable string tags for [`crate::shared::RendererCommand`] variants (diagnostics, counters).
//!
//! Kept exhaustive so adding a new IPC variant forces an update here and in the host generator.

use crate::shared::RendererCommand;

/// Returns a stable tag for logging and unhandled-command counters when the enum grows.
pub(crate) fn renderer_command_variant_tag(cmd: &RendererCommand) -> &'static str {
    match cmd {
        RendererCommand::RendererInitData(_) => "RendererInitData",
        RendererCommand::RendererInitResult(_) => "RendererInitResult",
        RendererCommand::RendererInitProgressUpdate(_) => "RendererInitProgressUpdate",
        RendererCommand::RendererInitFinalizeData(_) => "RendererInitFinalizeData",
        RendererCommand::RendererEngineReady(_) => "RendererEngineReady",
        RendererCommand::RendererShutdownRequest(_) => "RendererShutdownRequest",
        RendererCommand::RendererShutdown(_) => "RendererShutdown",
        RendererCommand::KeepAlive(_) => "KeepAlive",
        RendererCommand::RendererParentWindow(_) => "RendererParentWindow",
        RendererCommand::FreeSharedMemoryView(_) => "FreeSharedMemoryView",
        RendererCommand::SetWindowIcon(_) => "SetWindowIcon",
        RendererCommand::SetWindowIconResult(_) => "SetWindowIconResult",
        RendererCommand::SetTaskbarProgress(_) => "SetTaskbarProgress",
        RendererCommand::FrameStartData(_) => "FrameStartData",
        RendererCommand::FrameSubmitData(_) => "FrameSubmitData",
        RendererCommand::PostProcessingConfig(_) => "PostProcessingConfig",
        RendererCommand::QualityConfig(_) => "QualityConfig",
        RendererCommand::ResolutionConfig(_) => "ResolutionConfig",
        RendererCommand::DesktopConfig(_) => "DesktopConfig",
        RendererCommand::GaussianSplatConfig(_) => "GaussianSplatConfig",
        RendererCommand::RenderDecouplingConfig(_) => "RenderDecouplingConfig",
        RendererCommand::MeshUploadData(_) => "MeshUploadData",
        RendererCommand::MeshUnload(_) => "MeshUnload",
        RendererCommand::MeshUploadResult(_) => "MeshUploadResult",
        RendererCommand::ShaderUpload(_) => "ShaderUpload",
        RendererCommand::ShaderUnload(_) => "ShaderUnload",
        RendererCommand::ShaderUploadResult(_) => "ShaderUploadResult",
        RendererCommand::MaterialPropertyIdRequest(_) => "MaterialPropertyIdRequest",
        RendererCommand::MaterialPropertyIdResult(_) => "MaterialPropertyIdResult",
        RendererCommand::MaterialsUpdateBatch(_) => "MaterialsUpdateBatch",
        RendererCommand::MaterialsUpdateBatchResult(_) => "MaterialsUpdateBatchResult",
        RendererCommand::UnloadMaterial(_) => "UnloadMaterial",
        RendererCommand::UnloadMaterialPropertyBlock(_) => "UnloadMaterialPropertyBlock",
        RendererCommand::SetTexture2DFormat(_) => "SetTexture2DFormat",
        RendererCommand::SetTexture2DProperties(_) => "SetTexture2DProperties",
        RendererCommand::SetTexture2DData(_) => "SetTexture2DData",
        RendererCommand::SetTexture2DResult(_) => "SetTexture2DResult",
        RendererCommand::UnloadTexture2D(_) => "UnloadTexture2D",
        RendererCommand::SetTexture3DFormat(_) => "SetTexture3DFormat",
        RendererCommand::SetTexture3DProperties(_) => "SetTexture3DProperties",
        RendererCommand::SetTexture3DData(_) => "SetTexture3DData",
        RendererCommand::SetTexture3DResult(_) => "SetTexture3DResult",
        RendererCommand::UnloadTexture3D(_) => "UnloadTexture3D",
        RendererCommand::SetCubemapFormat(_) => "SetCubemapFormat",
        RendererCommand::SetCubemapProperties(_) => "SetCubemapProperties",
        RendererCommand::SetCubemapData(_) => "SetCubemapData",
        RendererCommand::SetCubemapResult(_) => "SetCubemapResult",
        RendererCommand::UnloadCubemap(_) => "UnloadCubemap",
        RendererCommand::SetRenderTextureFormat(_) => "SetRenderTextureFormat",
        RendererCommand::RenderTextureResult(_) => "RenderTextureResult",
        RendererCommand::UnloadRenderTexture(_) => "UnloadRenderTexture",
        RendererCommand::SetDesktopTextureProperties(_) => "SetDesktopTextureProperties",
        RendererCommand::DesktopTexturePropertiesUpdate(_) => "DesktopTexturePropertiesUpdate",
        RendererCommand::UnloadDesktopTexture(_) => "UnloadDesktopTexture",
        RendererCommand::PointRenderBufferUpload(_) => "PointRenderBufferUpload",
        RendererCommand::PointRenderBufferConsumed(_) => "PointRenderBufferConsumed",
        RendererCommand::PointRenderBufferUnload(_) => "PointRenderBufferUnload",
        RendererCommand::TrailRenderBufferUpload(_) => "TrailRenderBufferUpload",
        RendererCommand::TrailRenderBufferConsumed(_) => "TrailRenderBufferConsumed",
        RendererCommand::TrailRenderBufferUnload(_) => "TrailRenderBufferUnload",
        RendererCommand::GaussianSplatUploadRaw(_) => "GaussianSplatUploadRaw",
        RendererCommand::GaussianSplatUploadEncoded(_) => "GaussianSplatUploadEncoded",
        RendererCommand::GaussianSplatResult(_) => "GaussianSplatResult",
        RendererCommand::UnloadGaussianSplat(_) => "UnloadGaussianSplat",
        RendererCommand::LightsBufferRendererSubmission(_) => "LightsBufferRendererSubmission",
        RendererCommand::LightsBufferRendererConsumed(_) => "LightsBufferRendererConsumed",
        RendererCommand::ReflectionProbeRenderResult(_) => "ReflectionProbeRenderResult",
        RendererCommand::VideoTextureLoad(_) => "VideoTextureLoad",
        RendererCommand::VideoTextureUpdate(_) => "VideoTextureUpdate",
        RendererCommand::VideoTextureReady(_) => "VideoTextureReady",
        RendererCommand::VideoTextureChanged(_) => "VideoTextureChanged",
        RendererCommand::VideoTextureProperties(_) => "VideoTextureProperties",
        RendererCommand::VideoTextureStartAudioTrack(_) => "VideoTextureStartAudioTrack",
        RendererCommand::UnloadVideoTexture(_) => "UnloadVideoTexture",
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::*;

    use super::renderer_command_variant_tag;

    fn assert_tag(cmd: RendererCommand, expected: &'static str) {
        assert_eq!(renderer_command_variant_tag(&cmd), expected);
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
