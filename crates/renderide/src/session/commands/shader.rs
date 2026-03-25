//! Shader command handlers: `shader_upload`, `shader_unload`.

use crate::assets::{NativeUiShaderFamily, native_ui_family_from_shader_path_hint};
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
                    if let Some(ref hint) = data.file
                        && let Some(family) = native_ui_family_from_shader_path_hint(hint)
                    {
                        match family {
                            NativeUiShaderFamily::UiUnlit
                                if ctx.render_config.native_ui_unlit_shader_id < 0 =>
                            {
                                ctx.render_config.native_ui_unlit_shader_id = asset_id;
                                logger::info!(
                                    "native_ui: auto-registered UI_Unlit shader_id={} from upload hint",
                                    asset_id
                                );
                            }
                            NativeUiShaderFamily::UiTextUnlit
                                if ctx.render_config.native_ui_text_unlit_shader_id < 0 =>
                            {
                                ctx.render_config.native_ui_text_unlit_shader_id = asset_id;
                                logger::info!(
                                    "native_ui: auto-registered UI_TextUnlit shader_id={} from upload hint",
                                    asset_id
                                );
                            }
                            _ => {}
                        }
                    }
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
            RendererCommand::shader_unload(cmd) => {
                ctx.assets.asset_registry.handle_shader_unload(cmd.asset_id);
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
