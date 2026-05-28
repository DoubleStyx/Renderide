//! Texture2D upload helpers.

use std::path::Path;
use std::time::Duration;

use glam::IVec2;
use renderide_shared::SharedMemoryWriter;
use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::{
    ColorProfile, RendererCommand, SetTexture2DData, SetTexture2DFormat, SetTexture2DProperties,
    TextureFilterMode, TextureFormat, TextureUpdateResultType, TextureUploadHint, TextureWrapMode,
};

use crate::error::HarnessError;

use super::super::command_wait::wait_for_command;
use super::super::lockstep::LockstepDriver;
use super::shared::open_writer;

/// Per-call inputs for [`upload_texture2d_rgba8`].
pub(in crate::host) struct Texture2DUploadRequest<'a> {
    /// Shared-memory prefix matching `RendererInitData.shared_memory_prefix`.
    pub shared_memory_prefix: &'a str,
    /// Per-session backing directory passed to `SharedMemoryWriterConfig::dir_override`.
    pub backing_dir: &'a Path,
    /// Buffer id assigned to the texture-data SHM region.
    pub buffer_id: i32,
    /// Renderer-side asset id echoed back in `SetTexture2DResult`.
    pub asset_id: i32,
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
    /// `width * height * 4` RGBA8 bytes in row-major order.
    pub rgba_bytes: &'a [u8],
    /// sRGB vs Linear color profile.
    pub color_profile: ColorProfile,
    /// Deadline for receiving the data-upload portion of `SetTexture2DResult`.
    pub timeout: Duration,
}

/// Owns the live SHM writer backing a Texture2D upload.
pub(in crate::host) struct UploadedTexture {
    /// Live writer keeping the texture-data SHM region alive until the ack arrives.
    _writer: SharedMemoryWriter,
}

/// Uploads an RGBA8 texture and waits for the data-upload acknowledgement.
pub(in crate::host) fn upload_texture2d_rgba8(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    request: Texture2DUploadRequest<'_>,
) -> Result<UploadedTexture, HarnessError> {
    let expected_bytes = (request.width as usize) * (request.height as usize) * 4;
    if request.rgba_bytes.len() != expected_bytes {
        return Err(HarnessError::QueueOptions(format!(
            "RGBA8 texture {width}x{height} expects {expected_bytes} bytes, got {got}",
            width = request.width,
            height = request.height,
            got = request.rgba_bytes.len(),
        )));
    }

    let format = SetTexture2DFormat {
        asset_id: request.asset_id,
        width: request.width as i32,
        height: request.height as i32,
        mipmap_count: 1,
        format: TextureFormat::RGBA32,
        profile: request.color_profile,
    };
    if !queues.send_background(RendererCommand::SetTexture2DFormat(format)) {
        return Err(HarnessError::QueueOptions(
            "send_background(SetTexture2DFormat) returned false (queue full?)".to_string(),
        ));
    }

    let properties = SetTexture2DProperties {
        asset_id: request.asset_id,
        filter_mode: TextureFilterMode::Bilinear,
        aniso_level: 1,
        wrap_u: TextureWrapMode::Repeat,
        wrap_v: TextureWrapMode::Repeat,
        mipmap_bias: 0.0,
        apply_immediatelly: true,
        high_priority: true,
    };
    if !queues.send_background(RendererCommand::SetTexture2DProperties(properties)) {
        return Err(HarnessError::QueueOptions(
            "send_background(SetTexture2DProperties) returned false (queue full?)".to_string(),
        ));
    }

    let writer = open_writer(
        request.shared_memory_prefix,
        request.backing_dir,
        request.buffer_id,
        request.rgba_bytes,
        "texture2d_data",
    )?;
    let descriptor = writer.descriptor_for(0, request.rgba_bytes.len() as i32);
    let data = SetTexture2DData {
        asset_id: request.asset_id,
        data: descriptor,
        start_mip_level: 0,
        mip_map_sizes: vec![IVec2 {
            x: request.width as i32,
            y: request.height as i32,
        }],
        mip_starts: vec![0],
        flip_y: false,
        hint: TextureUploadHint::default(),
        high_priority: true,
    };
    if !queues.send_background(RendererCommand::SetTexture2DData(data)) {
        return Err(HarnessError::QueueOptions(
            "send_background(SetTexture2DData) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!(
        "AssetUpload: sent SetTexture2D{{Format,Properties,Data}}(asset_id={asset}, {w}x{h}, bytes={bytes})",
        asset = request.asset_id,
        w = request.width,
        h = request.height,
        bytes = request.rgba_bytes.len(),
    );

    wait_for_texture_data_upload_result(queues, lockstep, request.asset_id, request.timeout)?;
    logger::info!(
        "AssetUpload: received SetTexture2DResult(asset_id={asset}, DATA_UPLOAD)",
        asset = request.asset_id
    );

    Ok(UploadedTexture { _writer: writer })
}

/// Waits for the matching data-upload `SetTexture2DResult`.
fn wait_for_texture_data_upload_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    asset_id: i32,
    timeout: Duration,
) -> Result<(), HarnessError> {
    wait_for_command(
        queues,
        lockstep,
        timeout,
        |wait| HarnessError::AssetAckTimeout(wait, "SetTexture2DResult(DATA_UPLOAD) never arrived"),
        |msg| {
            if let RendererCommand::SetTexture2DResult(result) = msg
                && result.asset_id == asset_id
                && (result.r#type.0 & TextureUpdateResultType::DATA_UPLOAD) != 0
            {
                Some(())
            } else {
                None
            }
        },
    )
}
