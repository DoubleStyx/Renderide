//! Material and shader command effects on [`RendererRuntime`].

use crate::frontend::dispatch::command_dispatch::RunningCommandEffect;
use crate::shared::{MaterialPropertyIdRequest, MaterialPropertyIdResult, RendererCommand};

use super::super::super::RendererRuntime;
use super::super::shader_material;

impl RendererRuntime {
    pub(in crate::runtime) fn apply_material_shader_effect(
        &mut self,
        effect: RunningCommandEffect,
    ) {
        match effect {
            RunningCommandEffect::MaterialPropertyIdRequest(req) => {
                self.material_property_id_request(req);
            }
            RunningCommandEffect::MaterialsUpdateBatch(batch) => {
                shader_material::on_materials_update_batch(
                    &mut self.frontend,
                    &mut self.backend,
                    batch,
                );
            }
            RunningCommandEffect::UnloadMaterial { asset_id } => {
                self.backend.on_unload_material(asset_id);
            }
            RunningCommandEffect::UnloadMaterialPropertyBlock { asset_id } => {
                self.backend.on_unload_material_property_block(asset_id);
            }
            RunningCommandEffect::ShaderUpload(u) => {
                shader_material::on_shader_upload(
                    &mut self.ipc_state.pending_shader_resolutions,
                    u,
                );
            }
            RunningCommandEffect::ShaderUnload(u) => {
                shader_material::on_shader_unload(&mut self.backend, u);
            }
            _ => {}
        }
    }

    fn material_property_id_request(&mut self, req: MaterialPropertyIdRequest) {
        profiling::scope!("command::material_property_id_request");
        let property_ids: Vec<i32> = {
            let reg = self.backend.property_id_registry();
            req.property_names
                .iter()
                .map(|n| reg.intern_for_host_request(n.as_deref().unwrap_or("")))
                .collect()
        };
        if let Some(ipc) = self.frontend.ipc_mut() {
            let ack_queued = ipc.send_background_reliable(
                RendererCommand::MaterialPropertyIdResult(MaterialPropertyIdResult {
                    request_id: req.request_id,
                    property_ids,
                }),
            );
            if !ack_queued {
                logger::warn!(
                    "material property id request {}: failed to enqueue reliable result",
                    req.request_id
                );
            }
        }
    }
}
