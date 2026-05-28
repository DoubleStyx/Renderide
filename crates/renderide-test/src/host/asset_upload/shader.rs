//! Shader-upload helpers for test-only embedded WGSL stems.

use std::time::Duration;

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::{RendererCommand, ShaderUpload};
use renderide_shared::test_hooks::RENDERIDE_TEST_STEM_PREFIX;

use crate::error::HarnessError;

use super::super::command_wait::wait_for_command;
use super::super::lockstep::LockstepDriver;

/// Sends a `ShaderUpload` carrying the test-only stem sentinel and waits for its result.
pub(in crate::host) fn upload_shader(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    asset_id: i32,
    shader_name: &str,
    shader_variant_bits: Option<u32>,
    timeout: Duration,
) -> Result<(), HarnessError> {
    let sentinel = format_test_shader_upload_file(shader_name, shader_variant_bits);
    let upload = ShaderUpload {
        asset_id,
        file: Some(sentinel.clone()),
    };
    if !queues.send_background(RendererCommand::ShaderUpload(upload)) {
        return Err(HarnessError::QueueOptions(
            "send_background(ShaderUpload) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!("AssetUpload: sent ShaderUpload(asset_id={asset_id}, sentinel={sentinel:?})");
    wait_for_shader_upload_result(queues, lockstep, asset_id, timeout)?;
    logger::info!("AssetUpload: received ShaderUploadResult(asset_id={asset_id})");
    Ok(())
}

/// Formats the renderer test-stem sentinel for an uploaded shader name and optional variant mask.
fn format_test_shader_upload_file(shader_name: &str, shader_variant_bits: Option<u32>) -> String {
    match shader_variant_bits {
        Some(bits) => {
            let base = shader_name
                .strip_suffix(".shader")
                .or_else(|| shader_name.strip_suffix(".SHADER"))
                .or_else(|| shader_name.strip_suffix(".Shader"))
                .unwrap_or(shader_name);
            format!("{RENDERIDE_TEST_STEM_PREFIX}{base}_{bits:08X}.shader")
        }
        None => format!("{RENDERIDE_TEST_STEM_PREFIX}{shader_name}"),
    }
}

/// Waits for the matching `ShaderUploadResult`.
fn wait_for_shader_upload_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    asset_id: i32,
    timeout: Duration,
) -> Result<(), HarnessError> {
    wait_for_command(
        queues,
        lockstep,
        timeout,
        |wait| HarnessError::AssetAckTimeout(wait, "ShaderUploadResult never arrived"),
        |msg| match msg {
            RendererCommand::ShaderUploadResult(result) if result.asset_id == asset_id => Some(()),
            _ => None,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::format_test_shader_upload_file;
    use renderide_shared::test_hooks::RENDERIDE_TEST_STEM_PREFIX;

    #[test]
    fn shader_upload_file_without_variant_preserves_shader_name() {
        assert_eq!(
            format_test_shader_upload_file("Unlit.shader", None),
            format!("{RENDERIDE_TEST_STEM_PREFIX}Unlit.shader")
        );
    }

    #[test]
    fn shader_upload_file_with_variant_uses_unity_style_suffix() {
        assert_eq!(
            format_test_shader_upload_file("Unlit.shader", Some(0x0000_0200)),
            format!("{RENDERIDE_TEST_STEM_PREFIX}Unlit_00000200.shader")
        );
    }
}
