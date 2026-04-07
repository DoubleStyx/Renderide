//! Mesh command handlers: mesh_upload_data, mesh_unload.

use crate::shared::{MeshUploadResult, RendererCommand};

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `mesh_upload_data` and `mesh_unload`. Sends mesh upload result on success.
pub struct MeshCommandHandler;

impl CommandHandler for MeshCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::mesh_upload_data(data) => {
                let asset_id = data.asset_id;
                let (success, existed_before) = match ctx.assets.shared_memory {
                    Some(shm) => ctx
                        .assets
                        .asset_registry
                        .handle_mesh_upload(shm, data.clone()),
                    None => (false, false),
                };
                if success {
                    ctx.receiver
                        .send_background(RendererCommand::mesh_upload_result(MeshUploadResult {
                            asset_id,
                            instance_changed: !existed_before,
                        }));
                }
                CommandResult::Handled
            }
            RendererCommand::mesh_unload(x) => {
                ctx.assets.asset_registry.handle_mesh_unload(x.asset_id);
                ctx.frame.pending_mesh_unloads.push(x.asset_id);
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
