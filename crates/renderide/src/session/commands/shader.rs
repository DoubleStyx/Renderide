//! Shader command handlers: `shader_upload`, `shader_unload`.
//!
//! Native UI shader ids default to auto-fill from resolved Unity shader names, then path hints, when
//! INI ids are `-1`. [`crate::config::RenderConfig::native_ui_force_shader_hint_registration`] overwrites
//! ids on every matching upload (stale positive INI ids otherwise block auto-reg).

use crate::assets::{
    NativeUiShaderFamily, native_ui_family_from_shader_path_hint,
    native_ui_family_from_unity_shader_name,
};
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
                    let unity_name = ctx
                        .assets
                        .asset_registry
                        .get_shader(asset_id)
                        .and_then(|s| s.unity_shader_name.clone());
                    let family = unity_name
                        .as_deref()
                        .and_then(native_ui_family_from_unity_shader_name)
                        .or_else(|| {
                            data.file
                                .as_deref()
                                .and_then(native_ui_family_from_shader_path_hint)
                        });
                    logger::info!(
                        "shader_upload: asset_id={} unity_shader_name={:?} wgsl_path_hint={:?} resolved_native_ui_family={:?}",
                        asset_id,
                        unity_name.as_deref(),
                        data.file.as_deref(),
                        family
                    );
                    if let Some(family) = family {
                        let force = ctx.render_config.native_ui_force_shader_hint_registration;
                        match family {
                            NativeUiShaderFamily::UiUnlit
                                if force || ctx.render_config.native_ui_unlit_shader_id < 0 =>
                            {
                                ctx.render_config.native_ui_unlit_shader_id = asset_id;
                                if force {
                                    logger::info!(
                                        "native_ui: force-registered UI_Unlit shader_id={} from upload hint",
                                        asset_id
                                    );
                                } else {
                                    logger::info!(
                                        "native_ui: auto-registered UI_Unlit shader_id={} from upload hint",
                                        asset_id
                                    );
                                }
                            }
                            NativeUiShaderFamily::UiTextUnlit
                                if force
                                    || ctx.render_config.native_ui_text_unlit_shader_id < 0 =>
                            {
                                ctx.render_config.native_ui_text_unlit_shader_id = asset_id;
                                if force {
                                    logger::info!(
                                        "native_ui: force-registered UI_TextUnlit shader_id={} from upload hint",
                                        asset_id
                                    );
                                } else {
                                    logger::info!(
                                        "native_ui: auto-registered UI_TextUnlit shader_id={} from upload hint",
                                        asset_id
                                    );
                                }
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
