//! Thin [`super::RendererRuntime`] accessors and forwards to the frontend, backend, and settings.

use std::path::PathBuf;

use crate::assets::AssetTransferQueue;
use crate::config::RendererSettingsHandle;
use crate::connection::InitError;
use crate::diagnostics::DebugHudInput;
use crate::frontend::InitState;
use crate::gpu::GpuContext;
use crate::shared::RendererInitData;

use super::RendererRuntime;

impl RendererRuntime {
    /// Shared settings store ([`crate::config::RendererSettings`]).
    pub fn settings(&self) -> &RendererSettingsHandle {
        &self.settings
    }

    /// Path written by the **Renderer config** ImGui window and [`crate::config::save_renderer_settings`].
    pub fn config_save_path(&self) -> &PathBuf {
        &self.config_save_path
    }

    /// Mesh deformation compute pipelines when GPU init succeeded.
    pub fn mesh_preprocess(&self) -> Option<&crate::backend::mesh_deform::MeshPreprocessPipelines> {
        self.backend.mesh_preprocess()
    }

    /// Opens Primary/Background queues when [`Self::new`] was given connection parameters.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        self.frontend.connect_ipc()
    }

    /// Whether IPC queues are open.
    pub fn is_ipc_connected(&self) -> bool {
        self.frontend.is_ipc_connected()
    }

    /// Host/renderer init handshake phase (see [`crate::frontend::RendererFrontend::init_state`]).
    pub fn init_state(&self) -> InitState {
        self.frontend.init_state()
    }

    /// After a successful [`FrameSubmitData`] application, host may expect another begin-frame.
    pub fn last_frame_data_processed(&self) -> bool {
        self.frontend.last_frame_data_processed()
    }

    /// Current lock-step frame index echoed to the host.
    pub fn last_frame_index(&self) -> i32 {
        self.frontend.last_frame_index()
    }

    /// Host requested an orderly renderer shutdown over IPC.
    pub fn shutdown_requested(&self) -> bool {
        self.frontend.shutdown_requested()
    }

    /// Unrecoverable IPC/init error; begin-frame is suppressed until reset.
    pub fn fatal_error(&self) -> bool {
        self.frontend.fatal_error()
    }

    /// Whether the host last reported VR mode as active (see [`crate::render_graph::HostCameraFrame::vr_active`]).
    pub fn vr_active(&self) -> bool {
        self.host_camera.vr_active
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &crate::resources::MeshPool {
        self.backend.mesh_pool()
    }

    /// Mutable mesh pool (eviction experiments).
    pub fn mesh_pool_mut(&mut self) -> &mut crate::resources::MeshPool {
        self.backend.mesh_pool_mut()
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &crate::resources::TexturePool {
        self.backend.texture_pool()
    }

    /// Mutable texture pool.
    pub fn texture_pool_mut(&mut self) -> &mut crate::resources::TexturePool {
        self.backend.texture_pool_mut()
    }

    /// Mesh/texture upload queues, pools, and IPC budgets ([`AssetTransferQueue`]).
    pub fn asset_transfers_mut(&mut self) -> &mut AssetTransferQueue {
        &mut self.backend.asset_transfers
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &crate::assets::material::MaterialPropertyStore {
        self.backend.material_property_store()
    }

    /// Mutable store for tests and tooling.
    pub fn material_property_store_mut(
        &mut self,
    ) -> &mut crate::assets::material::MaterialPropertyStore {
        self.backend.material_property_store_mut()
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &crate::assets::material::PropertyIdRegistry {
        self.backend.property_id_registry()
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.backend.material_registry()
    }

    /// Mutable registry (pipeline cache and shader routes).
    pub fn material_registry_mut(&mut self) -> Option<&mut crate::materials::MaterialRegistry> {
        self.backend.material_registry_mut()
    }

    /// Host [`RendererInitData`] after connect, before [`Self::take_pending_init`] consumes it.
    pub fn pending_init(&self) -> Option<&RendererInitData> {
        self.frontend.pending_init()
    }

    /// Applies pending init once a GPU/window stack exists (e.g. window title).
    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.frontend.take_pending_init()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    ///
    /// Logs at `error` when attach fails; the backend stays usable for CPU work but GPU draws are not configured.
    pub fn attach_gpu(&mut self, gpu: &GpuContext) {
        use std::sync::Arc;

        let device = gpu.device().clone();
        let queue = Arc::clone(gpu.queue());
        let shm = self.frontend.shared_memory_mut();
        if let Err(e) = self.backend.attach(
            crate::backend::RenderBackendAttachDesc {
                device,
                queue,
                gpu_limits: Arc::clone(gpu.limits()),
                surface_format: gpu.config_format(),
                renderer_settings: Arc::clone(&self.settings),
                config_save_path: self.config_save_path.clone(),
                suppress_renderer_config_disk_writes: self.suppress_renderer_config_disk_writes,
            },
            shm,
        ) {
            logger::error!("RenderBackend::attach failed: {e}");
        }
    }

    /// Per-frame pointer state and timing for the ImGui overlay ([`diagnostics::DebugHud`]).
    pub fn set_debug_hud_frame_data(&mut self, input: DebugHudInput, frame_time_ms: f64) {
        self.backend.set_debug_hud_frame_data(input, frame_time_ms);
    }

    /// Last ImGui `want_capture_mouse` after the previous successful HUD encode; used when filtering [`InputState`] for the host.
    pub fn debug_hud_last_want_capture_mouse(&self) -> bool {
        self.backend.debug_hud_last_want_capture_mouse()
    }

    /// Last ImGui `want_capture_keyboard` after the previous successful HUD encode; used when filtering [`InputState`] for the host.
    pub fn debug_hud_last_want_capture_keyboard(&self) -> bool {
        self.backend.debug_hud_last_want_capture_keyboard()
    }
}

#[cfg(test)]
impl RendererRuntime {
    /// Installs a shared-memory accessor for tests that apply [`crate::shared::FrameSubmitData`].
    pub(crate) fn test_set_shared_memory(&mut self, prefix: impl Into<String>) {
        use crate::ipc::SharedMemoryAccessor;
        self.frontend
            .set_shared_memory(SharedMemoryAccessor::new(prefix.into()));
    }
}
