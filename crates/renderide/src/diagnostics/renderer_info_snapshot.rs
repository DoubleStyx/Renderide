//! Read-only snapshot of renderer state for the debug HUD “Renderer” tab (no ImGui types).

use crate::backend::RenderBackend;
use crate::frontend::InitState;
use crate::gpu::{GpuContext, GpuLimits};
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
    /// Adapter driver name (when reported by wgpu).
    pub adapter_driver: String,
    /// Extra driver details string from the adapter.
    pub adapter_driver_info: String,
    /// Swapchain surface format in use.
    pub surface_format: wgpu::TextureFormat,
    /// Swapchain extent in physical pixels.
    pub viewport_px: (u32, u32),
    /// Swapchain present mode (fifo, mailbox, etc.).
    pub present_mode: wgpu::PresentMode,
    /// Wall-clock time between redraw ticks (ms): same basis as HUD **total frame time**; FPS = `1000.0 / frame_time_ms`.
    pub frame_time_ms: f64,
    /// Active render spaces in the scene coordinator.
    pub render_space_count: usize,
    /// Mesh renderable records across spaces.
    pub mesh_renderable_count: usize,
    /// Resident [`crate::resources::MeshPool`] entries.
    pub resident_mesh_count: usize,
    /// Resident entries in [`crate::resources::TexturePool`].
    pub resident_texture_count: usize,
    /// Host [`crate::resources::GpuRenderTexture`] entries in [`crate::resources::RenderTexturePool`].
    pub resident_render_texture_count: usize,
    /// Allocated material property uniform slots.
    pub material_property_slots: usize,
    /// Allocated material property block slots.
    pub property_block_slots: usize,
    /// Distinct shader binding sets registered for materials.
    pub material_shader_bindings: usize,
    /// Pass count in the compiled main render graph.
    pub frame_graph_pass_count: usize,
    /// Packed lights after [`RenderBackend::prepare_lights_from_scene`].
    pub gpu_light_count: usize,
    /// `max_texture_dimension_2d` from [`GpuLimits`].
    pub gpu_max_texture_dim_2d: u32,
    /// `max_buffer_size` from [`GpuLimits`].
    pub gpu_max_buffer_size: u64,
    /// `max_storage_buffer_binding_size` from [`GpuLimits`].
    pub gpu_max_storage_binding: u64,
    /// Whether the device exposes non-zero `first_instance` (merged mesh draws).
    pub gpu_supports_base_instance: bool,
    /// Whether stereo multiview shaders may be used.
    pub gpu_supports_multiview: bool,
    /// MSAA sample count from [`crate::config::RenderingSettings::msaa`] (before GPU clamp).
    pub msaa_requested_samples: u32,
    /// Effective MSAA for the swapchain forward path after clamping to [`Self::msaa_max_samples`].
    pub msaa_effective_samples: u32,
    /// Maximum MSAA sample count supported for the swapchain color + depth formats on this adapter.
    pub msaa_max_samples: u32,
}

/// Inputs for [`RendererInfoSnapshot::capture`] (IPC, adapter, swapchain, scene, and backend refs).
pub struct RendererInfoSnapshotCapture<'a> {
    /// Primary/Background IPC queues connected.
    pub ipc_connected: bool,
    /// Host/renderer init handshake state.
    pub init_state: InitState,
    /// Last lock-step frame index sent to the host.
    pub last_frame_index: i32,
    /// Selected adapter metadata.
    pub adapter_info: &'a wgpu::AdapterInfo,
    /// Device limits for HUD lines.
    pub gpu_limits: &'a GpuLimits,
    /// Swapchain surface format.
    pub surface_format: wgpu::TextureFormat,
    /// Swapchain extent in physical pixels.
    pub viewport_px: (u32, u32),
    /// Swapchain present mode.
    pub present_mode: wgpu::PresentMode,
    /// Wall-clock ms between redraw ticks (HUD frame time).
    pub frame_time_ms: f64,
    /// Scene coordinator for space/renderable counts.
    pub scene: &'a SceneCoordinator,
    /// Backend pools, graph, and lights.
    pub backend: &'a RenderBackend,
    /// GPU context (MSAA effective/max).
    pub gpu: &'a GpuContext,
    /// Requested MSAA sample count from settings (before clamp).
    pub msaa_requested_samples: u32,
}

impl RendererInfoSnapshot {
    /// Fills all fields from the scene, backend, and swapchain (call after light prep for `gpu_light_count`).
    pub fn capture(args: RendererInfoSnapshotCapture<'_>) -> Self {
        let store = args.backend.material_property_store();
        Self {
            ipc_connected: args.ipc_connected,
            init_state: args.init_state,
            last_frame_index: args.last_frame_index,
            adapter_name: args.adapter_info.name.clone(),
            adapter_backend: args.adapter_info.backend,
            adapter_device_type: args.adapter_info.device_type,
            adapter_driver: args.adapter_info.driver.clone(),
            adapter_driver_info: args.adapter_info.driver_info.clone(),
            surface_format: args.surface_format,
            viewport_px: args.viewport_px,
            present_mode: args.present_mode,
            frame_time_ms: args.frame_time_ms,
            render_space_count: args.scene.render_space_count(),
            mesh_renderable_count: args.scene.total_mesh_renderable_count(),
            resident_mesh_count: args.backend.mesh_pool().meshes().len(),
            resident_texture_count: args.backend.texture_pool().resident_texture_count(),
            resident_render_texture_count: args.backend.render_texture_pool().len(),
            material_property_slots: store.material_property_slot_count(),
            property_block_slots: store.property_block_slot_count(),
            material_shader_bindings: store.material_shader_binding_count(),
            frame_graph_pass_count: args.backend.frame_graph_pass_count(),
            gpu_light_count: args.backend.frame_resources.frame_lights().len(),
            gpu_max_texture_dim_2d: args.gpu_limits.max_texture_dimension_2d(),
            gpu_max_buffer_size: args.gpu_limits.max_buffer_size(),
            gpu_max_storage_binding: args.gpu_limits.max_storage_buffer_binding_size(),
            gpu_supports_base_instance: args.gpu_limits.supports_base_instance,
            gpu_supports_multiview: args.gpu_limits.supports_multiview,
            msaa_requested_samples: args.msaa_requested_samples,
            msaa_effective_samples: args.gpu.swapchain_msaa_effective(),
            msaa_max_samples: args.gpu.msaa_max_sample_count(),
        }
    }
}
