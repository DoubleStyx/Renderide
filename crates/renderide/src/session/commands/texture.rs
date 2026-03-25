//! Texture command handlers: `SetTexture2D*` and `unload_texture_2d`.

use crate::shared::{RendererCommand, SetTexture2DResult, TextureUpdateResultType};

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles host `Texture2D` upload commands.
pub struct TextureCommandHandler;

impl CommandHandler for TextureCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::set_texture_2d_format(x) => {
                let (ok, existed_before) =
                    ctx.assets.asset_registry.set_texture_2d_format(x.clone());
                if ok {
                    ctx.receiver
                        .send_background(RendererCommand::set_texture_2d_result(
                            SetTexture2DResult {
                                asset_id: x.asset_id,
                                instance_changed: !existed_before,
                                r#type: TextureUpdateResultType(
                                    TextureUpdateResultType::FORMAT_SET,
                                ),
                            },
                        ));
                }
                CommandResult::Handled
            }
            RendererCommand::set_texture_2d_properties(x) => {
                ctx.receiver
                    .send_background(RendererCommand::set_texture_2d_result(SetTexture2DResult {
                        asset_id: x.asset_id,
                        instance_changed: false,
                        r#type: TextureUpdateResultType(TextureUpdateResultType::PROPERTIES_SET),
                    }));
                CommandResult::Handled
            }
            RendererCommand::set_texture_2d_data(data) => {
                if let Some(shm) = ctx.assets.shared_memory.as_mut() {
                    let (ok, first_upload) =
                        ctx.assets.asset_registry.set_texture_2d_data(shm, data);
                    if ok {
                        ctx.receiver
                            .send_background(RendererCommand::set_texture_2d_result(
                                SetTexture2DResult {
                                    asset_id: data.asset_id,
                                    instance_changed: first_upload,
                                    r#type: TextureUpdateResultType(
                                        TextureUpdateResultType::DATA_UPLOAD,
                                    ),
                                },
                            ));
                    }
                }
                CommandResult::Handled
            }
            RendererCommand::unload_texture_2d(u) => {
                let asset_id = u.asset_id;
                ctx.assets.asset_registry.unload_texture_2d(asset_id);
                ctx.frame.pending_texture_unloads.push(asset_id);
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
