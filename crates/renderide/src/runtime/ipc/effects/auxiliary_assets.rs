//! Auxiliary-asset command effects (desktop textures, point/trail buffers, gaussian splats) and
//! shared-memory view release on [`RendererRuntime`].

use crate::frontend::dispatch::command_dispatch::RunningCommandEffect;
use crate::shared::MeshUploadData;

use super::super::super::RendererRuntime;

impl RendererRuntime {
    pub(in crate::runtime) fn apply_auxiliary_asset_effect(
        &mut self,
        effect: RunningCommandEffect,
    ) {
        match effect {
            RunningCommandEffect::SetDesktopTextureProperties(p) => self
                .backend
                .on_set_desktop_texture_properties(p, self.frontend.ipc_mut()),
            RunningCommandEffect::DesktopTexturePropertiesUpdate(u) => {
                self.backend.on_desktop_texture_properties_update(u);
            }
            RunningCommandEffect::UnloadDesktopTexture(u) => {
                self.backend.on_unload_desktop_texture(u);
            }
            RunningCommandEffect::PointRenderBufferUpload(u) => self
                .backend
                .on_point_render_buffer_upload(u, self.frontend.ipc_mut()),
            RunningCommandEffect::PointRenderBufferUnload(u) => {
                self.backend.on_point_render_buffer_unload(u);
            }
            RunningCommandEffect::TrailRenderBufferUpload(u) => self
                .backend
                .on_trail_render_buffer_upload(u, self.frontend.ipc_mut()),
            RunningCommandEffect::TrailRenderBufferUnload(u) => {
                self.backend.on_trail_render_buffer_unload(u);
            }
            RunningCommandEffect::GaussianSplatConfig(c) => {
                self.backend.on_gaussian_splat_config(c);
            }
            RunningCommandEffect::GaussianSplatUploadRaw(u) => self
                .backend
                .on_gaussian_splat_upload_raw(u, self.frontend.ipc_mut()),
            RunningCommandEffect::GaussianSplatUploadEncoded(u) => self
                .backend
                .on_gaussian_splat_upload_encoded(u, self.frontend.ipc_mut()),
            RunningCommandEffect::UnloadGaussianSplat(u) => {
                self.backend.on_unload_gaussian_splat(u);
            }
            RunningCommandEffect::PointRenderBufferConsumed => {
                logger::trace!(
                    "runtime: point_render_buffer_consumed from host (ignored; renderer is source)"
                );
            }
            RunningCommandEffect::TrailRenderBufferConsumed => {
                logger::trace!(
                    "runtime: trail_render_buffer_consumed from host (ignored; renderer is source)"
                );
            }
            RunningCommandEffect::GaussianSplatResult => {
                logger::trace!(
                    "runtime: gaussian_splat_result from host (ignored; renderer is source)"
                );
            }
            _ => {}
        }
    }

    pub(in crate::runtime) fn process_mesh_upload(&mut self, d: MeshUploadData) {
        let (shm, ipc) = self.frontend.transport_pair_mut();
        self.backend.try_process_mesh_upload(d, shm, ipc);
    }

    pub(in crate::runtime) fn release_shared_memory_view(&mut self, buffer_id: i32) {
        if let Some(shm) = self.frontend.shared_memory_mut() {
            shm.release_view(buffer_id);
        }
    }
}
