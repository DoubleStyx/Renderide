//! Shader command handlers: shader_upload.

use crate::shared::{RendererCommand, ShaderUploadResult};

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `shader_upload`. Stores shader in asset registry and sends result on success.
pub struct ShaderCommandHandler;

impl CommandHandler for ShaderCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::shader_upload(data) => {
                let asset_id = data.asset_id;
                let (success, existed_before) =
                    ctx.assets.asset_registry.handle_shader_upload(data.clone());
                if success {
                    ctx.receiver
                        .send_background(RendererCommand::shader_upload_result(
                            ShaderUploadResult {
                                asset_id,
                                instance_changed: !existed_before,
                            },
                        ));
                }
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
