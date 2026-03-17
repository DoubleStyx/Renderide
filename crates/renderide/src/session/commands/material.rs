//! Material command handlers: material_property_id_request, materials_update_batch,
//! unload_material_property_block, etc.
//!
//! Parses MaterialsUpdateBatch buffers and stores values in MaterialPropertyStore for
//! stencil lookup when building draw entries.

use std::mem::size_of;

use crate::assets::MaterialPropertyValue;
use crate::shared::{
    MaterialPropertyUpdate, MaterialPropertyUpdateType, MaterialsUpdateBatchResult,
    RendererCommand,
};

use super::{CommandContext, CommandHandler, CommandResult};

/// Size of MaterialPropertyUpdate in bytes (property_id: i32, update_type: u8, padding: 3).
const MATERIAL_PROPERTY_UPDATE_SIZE: usize = size_of::<MaterialPropertyUpdate>();

/// Handles `materials_update_batch` and `unload_material_property_block`.
pub struct MaterialCommandHandler;

impl CommandHandler for MaterialCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::materials_update_batch(batch) => {
                if let Some(shm) = ctx.shared_memory.as_mut() {
                    parse_and_store_materials_batch(shm, &batch, &mut ctx.asset_registry.material_property_store);
                    ctx.receiver.send_background(RendererCommand::materials_update_batch_result(
                        MaterialsUpdateBatchResult {
                            update_batch_id: batch.update_batch_id,
                        },
                    ));
                }
                CommandResult::Handled
            }
            RendererCommand::unload_material_property_block(cmd) => {
                ctx.asset_registry
                    .material_property_store
                    .remove_block(cmd.asset_id);
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}

/// Parses material_updates buffers and stores values in the property store.
///
/// Buffer layout: sequence of (MaterialPropertyUpdate, value) pairs. Value size depends on
/// update_type. Bounds checks prevent reading past buffer end.
fn parse_and_store_materials_batch(
    shm: &mut crate::ipc::shared_memory::SharedMemoryAccessor,
    batch: &crate::shared::MaterialsUpdateBatch,
    store: &mut crate::assets::MaterialPropertyStore,
) {
    if batch.material_updates.is_empty() {
        return;
    }

    for (block_index, desc) in batch.material_updates.iter().enumerate() {
        if desc.length <= 0 {
            continue;
        }
        let bytes = match shm.access_copy::<u8>(desc) {
            Some(b) => b,
            None => continue,
        };

        let mut offset = 0;
        let mut current_block_id = block_index as i32;

        while offset + MATERIAL_PROPERTY_UPDATE_SIZE <= bytes.len() {
            let update: MaterialPropertyUpdate =
                bytemuck::pod_read_unaligned(&bytes[offset..offset + MATERIAL_PROPERTY_UPDATE_SIZE]);
            offset += MATERIAL_PROPERTY_UPDATE_SIZE;

            let value_size = match update.update_type {
                MaterialPropertyUpdateType::select_target => 4,
                MaterialPropertyUpdateType::set_float => 4,
                MaterialPropertyUpdateType::set_float4 => 16,
                MaterialPropertyUpdateType::set_float4x4 => 64,
                MaterialPropertyUpdateType::update_batch_end => 0,
                _ => {
                    break;
                }
            };

            if value_size > 0 && offset + value_size > bytes.len() {
                break;
            }

            match update.update_type {
                MaterialPropertyUpdateType::select_target => {
                    if value_size == 4 {
                        current_block_id = i32::from_le_bytes(
                            bytes[offset..offset + 4]
                                .try_into()
                                .unwrap_or([0; 4]),
                        );
                    }
                }
                MaterialPropertyUpdateType::set_float => {
                    if value_size == 4 {
                        let val = f32::from_le_bytes(
                            bytes[offset..offset + 4]
                                .try_into()
                                .unwrap_or([0; 4]),
                        );
                        store.set(
                            current_block_id,
                            update.property_id,
                            MaterialPropertyValue::Float(val),
                        );
                    }
                }
                MaterialPropertyUpdateType::set_float4 => {
                    if value_size == 16 && offset + 16 <= bytes.len() {
                        let mut arr = [0.0f32; 4];
                        for (i, chunk) in bytes[offset..offset + 16].chunks_exact(4).enumerate() {
                            if i < 4 {
                                arr[i] = f32::from_le_bytes(chunk.try_into().unwrap_or([0; 4]));
                            }
                        }
                        store.set(
                            current_block_id,
                            update.property_id,
                            MaterialPropertyValue::Float4(arr),
                        );
                    }
                }
                MaterialPropertyUpdateType::set_float4x4 | MaterialPropertyUpdateType::update_batch_end => {}
                _ => {}
            }

            offset += value_size;
        }
    }
}
