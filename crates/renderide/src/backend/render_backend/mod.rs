//! [`RenderBackend`] — thin coordinator for frame execution and IPC-facing GPU work.
//!
//! Core subsystems live in [`super::MaterialSystem`], [`crate::assets::AssetTransferQueue`],
//! [`super::FrameResourceManager`], and [`super::OcclusionSystem`]; this type wires attach,
//! the compiled render graph, mesh deform preprocess, and debug HUD.
//!
//! Graph execution lives in the `execute` submodule; IPC-facing asset handlers in `asset_ipc`.

mod asset_ipc;
mod execute;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::assets::asset_transfer_queue::{self as asset_uploads, AssetTransferQueue};
use crate::assets::material::MaterialPropertyStore;
use crate::backend::mesh_deform::{MeshDeformScratch, MeshPreprocessPipelines};
use crate::config::RendererSettingsHandle;
use crate::diagnostics::{DebugHudEncodeError, DebugHudInput, SceneTransformsSnapshot};
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::render_graph::{GraphCache, WorldMeshDrawStats};
use crate::resources::{CubemapPool, MeshPool, RenderTexturePool, Texture3dPool, TexturePool};

use super::debug_hud_bundle::DebugHudBundle;
use super::embedded::EmbeddedTexturePools;
use super::material_system::MaterialSystem;
use super::occlusion::OcclusionSystem;

pub use crate::assets::asset_transfer_queue::{
    MAX_ASSET_INTEGRATION_QUEUED, MAX_PENDING_MESH_UPLOADS, MAX_PENDING_TEXTURE_UPLOADS,
};

/// Device, queue, and settings passed to [`RenderBackend::attach`] (shared-memory flush is passed separately for borrow reasons).
pub struct RenderBackendAttachDesc {
    /// Logical device for uploads and graph encoding.
    pub device: Arc<wgpu::Device>,
    /// Queue used for submits and GPU writes.
    pub queue: Arc<Mutex<wgpu::Queue>>,
    /// Capabilities for buffer sizing and MSAA.
    pub gpu_limits: Arc<GpuLimits>,
    /// Swapchain / main surface format for HUD and pipelines.
    pub surface_format: wgpu::TextureFormat,
    /// Live renderer settings (HUD, VR budgets, etc.).
    pub renderer_settings: RendererSettingsHandle,
    /// Path for persisting HUD/config from the debug overlay.
    pub config_save_path: PathBuf,
}

/// Coordinates materials, asset uploads, per-frame GPU binds, occlusion, optional deform + ImGui HUD, and the render graph.
pub struct RenderBackend {
    /// Material property store, shader routes, pipeline registry, embedded `@group(1)` binds.
    pub(crate) materials: MaterialSystem,
    /// Mesh/texture upload queues, budgets, format tables, pools, and GPU device/queue for uploads.
    pub(crate) asset_transfers: AssetTransferQueue,
    /// Optional mesh skinning / blendshape compute pipelines (after [`Self::attach`]).
    mesh_preprocess: Option<MeshPreprocessPipelines>,
    /// Last compiled render graph keyed by surface extent, MSAA, multiview, and format; see [`crate::render_graph`].
    graph_cache: GraphCache,
    /// Scratch buffers for mesh deformation compute (after [`Self::attach`]).
    mesh_deform_scratch: Option<MeshDeformScratch>,
    /// Per-frame bind groups, light staging, and debug draw slab.
    pub(crate) frame_resources: super::FrameResourceManager,
    /// Dear ImGui overlay and capture state.
    debug_hud: DebugHudBundle,
    /// Hierarchical depth pyramid, CPU readback, and temporal cull state for occlusion culling.
    pub(crate) occlusion: OcclusionSystem,
    /// MSAA depth → R32F → [`wgpu::TextureFormat::Depth32Float`] resolve (after [`Self::attach`]).
    pub(crate) msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
}

impl Default for RenderBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderBackend {
    /// Empty pools and material store; no GPU until [`Self::attach`].
    pub fn new() -> Self {
        Self {
            materials: MaterialSystem::new(),
            asset_transfers: AssetTransferQueue::new(),
            mesh_preprocess: None,
            graph_cache: GraphCache::default(),
            mesh_deform_scratch: None,
            frame_resources: super::FrameResourceManager::new(),
            debug_hud: DebugHudBundle::new(),
            occlusion: OcclusionSystem::new(),
            msaa_depth_resolve: None,
        }
    }

    /// Count of host Texture2D asset ids that have received a [`crate::shared::SetTexture2DFormat`] (CPU-side table).
    pub fn texture_format_registration_count(&self) -> usize {
        self.asset_transfers.texture_formats.len()
    }

    /// Count of GPU-resident textures with `mip_levels_resident > 0` (at least mip0 uploaded).
    pub fn texture_mip0_ready_count(&self) -> usize {
        self.asset_transfers
            .texture_pool
            .textures()
            .values()
            .filter(|t| t.mip_levels_resident > 0)
            .count()
    }

    /// Mesh deformation compute pipelines when GPU init succeeded.
    pub fn mesh_preprocess(&self) -> Option<&MeshPreprocessPipelines> {
        self.mesh_preprocess.as_ref()
    }

    /// GPU limits snapshot after [`Self::attach`], if attach succeeded.
    pub fn gpu_limits(&self) -> Option<&Arc<GpuLimits>> {
        self.asset_transfers.gpu_limits.as_ref()
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &MeshPool {
        &self.asset_transfers.mesh_pool
    }

    /// Mutable mesh pool (eviction experiments).
    pub fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        &mut self.asset_transfers.mesh_pool
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &TexturePool {
        &self.asset_transfers.texture_pool
    }

    /// Resident Texture3D table.
    pub fn texture3d_pool(&self) -> &Texture3dPool {
        &self.asset_transfers.texture3d_pool
    }

    /// Resident cubemap table.
    pub fn cubemap_pool(&self) -> &CubemapPool {
        &self.asset_transfers.cubemap_pool
    }

    /// Host render texture targets (secondary cameras, material sampling).
    pub fn render_texture_pool(&self) -> &RenderTexturePool {
        &self.asset_transfers.render_texture_pool
    }

    /// Borrowed view of all texture pools used for embedded material `@group(1)` bind resolution.
    pub fn embedded_texture_pools(&self) -> EmbeddedTexturePools<'_> {
        EmbeddedTexturePools {
            texture: self.texture_pool(),
            texture3d: self.texture3d_pool(),
            cubemap: self.cubemap_pool(),
            render_texture: self.render_texture_pool(),
        }
    }

    /// Mutable texture pool.
    pub fn texture_pool_mut(&mut self) -> &mut TexturePool {
        &mut self.asset_transfers.texture_pool
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &MaterialPropertyStore {
        self.materials.material_property_store()
    }

    /// Mutable store for tests and tooling.
    pub fn material_property_store_mut(&mut self) -> &mut MaterialPropertyStore {
        self.materials.material_property_store_mut()
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &crate::assets::material::PropertyIdRegistry {
        self.materials.property_id_registry()
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.materials.material_registry()
    }

    /// Mutable registry (pipeline cache and shader routes).
    pub fn material_registry_mut(&mut self) -> Option<&mut crate::materials::MaterialRegistry> {
        self.materials.material_registry_mut()
    }

    /// Embedded material bind groups (world Unlit, etc.) after [`Self::attach`].
    pub fn embedded_material_bind(
        &self,
    ) -> Option<&super::embedded::EmbeddedMaterialBindResources> {
        self.materials.embedded_material_bind()
    }

    /// Number of scheduled passes in the cached frame graph, or `0` if none has been built yet.
    pub fn frame_graph_pass_count(&self) -> usize {
        self.graph_cache.pass_count()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    ///
    /// `shm` is used to flush pending mesh/texture payloads that require shared-memory reads; omit
    /// when none is available yet (uploads stay queued).
    pub fn attach(
        &mut self,
        desc: RenderBackendAttachDesc,
        shm: Option<&mut crate::ipc::SharedMemoryAccessor>,
    ) {
        let RenderBackendAttachDesc {
            device,
            queue,
            gpu_limits,
            surface_format,
            renderer_settings,
            config_save_path,
        } = desc;
        self.asset_transfers.gpu_device = Some(device.clone());
        self.asset_transfers.gpu_queue = Some(queue.clone());
        self.asset_transfers.gpu_limits = Some(Arc::clone(&gpu_limits));
        {
            let s = renderer_settings
                .read()
                .map(|g| g.clone())
                .unwrap_or_default();
            self.asset_transfers.render_texture_hdr_color = s.rendering.render_texture_hdr_color;
            self.asset_transfers.texture_vram_budget_bytes =
                u64::from(s.rendering.texture_vram_budget_mib).saturating_mul(1024 * 1024);
        }
        self.mesh_deform_scratch = Some(MeshDeformScratch::new(device.as_ref()));
        self.frame_resources.attach(device.as_ref(), gpu_limits);
        {
            let q = queue.lock().unwrap_or_else(|e| e.into_inner());
            self.debug_hud.attach(
                device.as_ref(),
                &q,
                surface_format,
                renderer_settings,
                config_save_path,
            );
        }
        match MeshPreprocessPipelines::new(device.as_ref()) {
            Ok(p) => self.mesh_preprocess = Some(p),
            Err(e) => {
                logger::warn!("mesh preprocess compute pipelines not created: {e}");
                self.mesh_preprocess = None;
            }
        }
        self.materials.attach_gpu(device.clone(), &queue);
        asset_uploads::attach_flush_pending_asset_uploads(&mut self.asset_transfers, &device, shm);

        self.msaa_depth_resolve = MsaaDepthResolveResources::try_new(device.as_ref()).map(Arc::new);
    }

    /// Updates whether main HUD diagnostics run (mirrors [`crate::config::DebugSettings::debug_hud_enabled`]).
    pub fn set_debug_hud_main_enabled(&mut self, enabled: bool) {
        self.debug_hud.set_main_enabled(enabled);
    }

    /// Whether main debug HUD is on (mesh-draw stats for [`crate::render_graph::passes::WorldMeshForwardPass`]).
    pub(crate) fn debug_hud_main_enabled(&self) -> bool {
        self.debug_hud.main_enabled()
    }

    /// Updates pointer state and frame delta for the ImGui overlay.
    pub fn set_debug_hud_frame_data(&mut self, input: DebugHudInput, frame_time_ms: f64) {
        self.debug_hud.set_frame_data(input, frame_time_ms);
    }

    /// Last inter-frame time in milliseconds supplied by the app for HUD FPS.
    pub(crate) fn debug_frame_time_ms(&self) -> f64 {
        self.debug_hud.frame_time_ms()
    }

    /// [`imgui::Io::want_capture_mouse`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_mouse(&self) -> bool {
        self.debug_hud.last_want_capture_mouse()
    }

    /// [`imgui::Io::want_capture_keyboard`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_keyboard(&self) -> bool {
        self.debug_hud.last_want_capture_keyboard()
    }

    /// Stores [`crate::diagnostics::RendererInfoSnapshot`] for the next HUD frame.
    pub(crate) fn set_debug_hud_snapshot(
        &mut self,
        snapshot: crate::diagnostics::RendererInfoSnapshot,
    ) {
        self.debug_hud.set_snapshot(snapshot);
    }

    pub(crate) fn set_debug_hud_frame_diagnostics(
        &mut self,
        snapshot: crate::diagnostics::FrameDiagnosticsSnapshot,
    ) {
        self.debug_hud.set_frame_diagnostics(snapshot);
    }

    pub(crate) fn set_debug_hud_frame_timing(
        &mut self,
        snapshot: crate::diagnostics::FrameTimingHudSnapshot,
    ) {
        self.debug_hud.set_frame_timing(snapshot);
    }

    /// Clears Stats / Shader routes payloads only (not frame timing or scene transforms).
    pub(crate) fn clear_debug_hud_stats_snapshots(&mut self) {
        self.debug_hud.clear_stats_snapshots();
    }

    /// Clears the **Scene transforms** HUD payload.
    pub(crate) fn clear_debug_hud_scene_transforms_snapshot(&mut self) {
        self.debug_hud.clear_scene_transforms_snapshot();
    }

    pub(crate) fn set_last_world_mesh_draw_stats(&mut self, stats: WorldMeshDrawStats) {
        self.debug_hud.set_last_world_mesh_draw_stats(stats);
    }

    pub(crate) fn last_world_mesh_draw_stats(&self) -> WorldMeshDrawStats {
        self.debug_hud.last_world_mesh_draw_stats()
    }

    /// Updates the **Scene transforms** Dear ImGui window payload for the next composite pass.
    pub(crate) fn set_debug_hud_scene_transforms_snapshot(
        &mut self,
        snapshot: SceneTransformsSnapshot,
    ) {
        self.debug_hud.set_scene_transforms_snapshot(snapshot);
    }

    /// Composites the debug HUD with `LoadOp::Load` onto the swapchain in `encoder`.
    pub(crate) fn encode_debug_hud_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
    ) -> Result<(), DebugHudEncodeError> {
        self.debug_hud
            .encode_overlay(device, queue, encoder, backbuffer, extent)
    }

    /// Scratch buffers for mesh deformation (`MeshDeformPass`).
    pub fn mesh_deform_scratch_mut(&mut self) -> Option<&mut MeshDeformScratch> {
        self.mesh_deform_scratch.as_mut()
    }

    /// Compute preprocess pipelines + deform scratch (`MeshDeformPass`) as one disjoint borrow.
    pub fn mesh_deform_pre_and_scratch(
        &mut self,
    ) -> Option<(
        &crate::backend::mesh_deform::MeshPreprocessPipelines,
        &mut MeshDeformScratch,
    )> {
        let pre = self.mesh_preprocess.as_ref()?;
        let scratch = self.mesh_deform_scratch.as_mut()?;
        Some((pre, scratch))
    }
}
