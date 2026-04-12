//! Read-only snapshot of renderer state for the debug HUD “Renderer” tab (no ImGui types).

use crate::backend::RenderBackend;
use crate::frontend::InitState;
use crate::scene::SceneCoordinator;

/// Per-frame diagnostic snapshot built on the CPU before the render graph executes.
#[derive(Clone, Debug)]
pub struct RendererInfoSnapshot {
    /// Primary/Background queues open.
    pub ipc_connected: bool,
    /// Host init handshake phase.
    pub init_state: InitState,
    /// Lock-step index last sent toward the host.
    pub last_frame_index: i32,
    /// [`wgpu::AdapterInfo::name`].
    pub adapter_name: String,
    /// Selected API backend.
    pub adapter_backend: wgpu::Backend,
    /// Integrated vs discrete, etc.
    pub adapter_device_type: wgpu::DeviceType,
    pub adapter_driver: String,
    pub adapter_driver_info: String,
    pub surface_format: wgpu::TextureFormat,
    pub viewport_px: (u32, u32),
    pub present_mode: wgpu::PresentMode,
    /// Wall-clock time between redraw ticks (ms): same basis as HUD **total frame time**; FPS = `1000.0 / frame_time_ms`.
    pub frame_time_ms: f64,
    pub render_space_count: usize,
    pub mesh_renderable_count: usize,
    pub resident_mesh_count: usize,
    pub resident_texture_count: usize,
    pub material_property_slots: usize,
    pub property_block_slots: usize,
    pub material_shader_bindings: usize,
    pub frame_graph_pass_count: usize,
    /// Packed lights after [`RenderBackend::prepare_lights_from_scene`].
    pub gpu_light_count: usize,
}

impl RendererInfoSnapshot {
    /// Fills all fields from the scene, backend, and swapchain (call after light prep for `gpu_light_count`).
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        ipc_connected: bool,
        init_state: InitState,
        last_frame_index: i32,
        adapter_info: &wgpu::AdapterInfo,
        surface_format: wgpu::TextureFormat,
        viewport_px: (u32, u32),
        present_mode: wgpu::PresentMode,
        frame_time_ms: f64,
        scene: &SceneCoordinator,
        backend: &RenderBackend,
    ) -> Self {
        let store = backend.material_property_store();
        Self {
            ipc_connected,
            init_state,
            last_frame_index,
            adapter_name: adapter_info.name.clone(),
            adapter_backend: adapter_info.backend,
            adapter_device_type: adapter_info.device_type,
            adapter_driver: adapter_info.driver.clone(),
            adapter_driver_info: adapter_info.driver_info.clone(),
            surface_format,
            viewport_px,
            present_mode,
            frame_time_ms,
            render_space_count: scene.render_space_count(),
            mesh_renderable_count: scene.total_mesh_renderable_count(),
            resident_mesh_count: backend.mesh_pool().meshes().len(),
            resident_texture_count: backend.texture_pool().resident_texture_count(),
            material_property_slots: store.material_property_slot_count(),
            property_block_slots: store.property_block_slot_count(),
            material_shader_bindings: store.material_shader_binding_count(),
            frame_graph_pass_count: backend.frame_graph_pass_count(),
            gpu_light_count: backend.frame_resources.frame_lights().len(),
        }
    }
}
