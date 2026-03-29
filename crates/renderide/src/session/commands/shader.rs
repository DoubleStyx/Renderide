//! Shader command handlers: `shader_upload`, `shader_unload`.
//!
//! `shader_unload` removes the shader from the asset registry and queues the asset id for
//! [`crate::render::RenderLoop::evict_shader_pipelines`] on the next main-view tick (see
//! [`crate::session::Session::drain_pending_shader_unloads`]), so cached WGSL pipelines for that
//! shader are dropped.

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
                        .map(|s| {
                            (
                                s.unity_shader_name.clone(),
                                s.pipeline_family,
                                s.renderide_shader_rel_path,
                            )
                        });
                    let (unity_name, family, rel_path) = unity_name.unwrap_or((
                        None,
                        crate::assets::ShaderPipelineFamily::Unsupported,
                        None,
                    ));
                    logger::info!(
                        "shader_upload: asset_id={} unity_shader_name={:?} pipeline_family={:?} renderide_shader={:?} upload_file_label={:?}",
                        asset_id,
                        unity_name.as_deref(),
                        family,
                        rel_path,
                        data.file.as_deref()
                    );
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
                let id = cmd.asset_id;
                ctx.assets.asset_registry.handle_shader_unload(id);
                ctx.frame.pending_shader_unloads.push(id);
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
