//! Material command handlers: `material_property_id_request`, `materials_update_batch`,
//! `unload_material_property_block`, etc.
//!
//! `material_property_id_request` interns property names to integers and sends
//! [`MaterialPropertyIdResult`](crate::shared::MaterialPropertyIdResult); see
//! [`crate::assets::material_property_host`]. `materials_update_batch` parses buffers into
//! [`MaterialPropertyStore`](crate::assets::MaterialPropertyStore) for draw-time uniform and texture lookup.

use crate::assets::{
    apply_froox_material_property_name_to_native_ui_config,
    apply_froox_material_property_name_to_pbr_host_config, intern_host_material_property_id,
    material_update_batch::{ParseMaterialBatchOptions, parse_materials_update_batch_into_store},
};
use crate::shared::{MaterialPropertyIdResult, MaterialsUpdateBatchResult, RendererCommand};

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `materials_update_batch` and `unload_material_property_block`.
pub struct MaterialCommandHandler;

impl CommandHandler for MaterialCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::material_property_id_request(req) => {
                let mut property_ids = Vec::with_capacity(req.property_names.len());
                for name_opt in &req.property_names {
                    let name = name_opt.as_deref().unwrap_or("");
                    let id = intern_host_material_property_id(name);
                    apply_froox_material_property_name_to_native_ui_config(
                        ctx.render_config,
                        name,
                        id,
                    );
                    apply_froox_material_property_name_to_pbr_host_config(
                        ctx.render_config,
                        name,
                        id,
                    );
                    property_ids.push(id);
                }
                ctx.receiver
                    .send_background(RendererCommand::material_property_id_result(
                        MaterialPropertyIdResult {
                            request_id: req.request_id,
                            property_ids,
                        },
                    ));
                CommandResult::Handled
            }
            RendererCommand::materials_update_batch(batch) => {
                if let Some(shm) = ctx.assets.shared_memory.as_mut() {
                    let opts = ParseMaterialBatchOptions {
                        persist_extended_payloads: ctx
                            .render_config
                            .material_batch_persist_extended_payloads,
                        record_wire_metrics: ctx.render_config.material_batch_wire_metrics,
                    };
                    parse_materials_update_batch_into_store(
                        shm,
                        batch,
                        &mut ctx.assets.asset_registry.material_property_store,
                        &opts,
                    );
                    ctx.receiver
                        .send_background(RendererCommand::materials_update_batch_result(
                            MaterialsUpdateBatchResult {
                                update_batch_id: batch.update_batch_id,
                            },
                        ));
                }
                CommandResult::Handled
            }
            RendererCommand::unload_material_property_block(cmd) => {
                ctx.assets
                    .asset_registry
                    .material_property_store
                    .remove_block(cmd.asset_id);
                CommandResult::Handled
            }
            RendererCommand::unload_material(cmd) => {
                ctx.frame.pending_material_unloads.push(cmd.asset_id);
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
