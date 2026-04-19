//! [`RenderBackend`] — thin coordinator for frame execution and IPC-facing GPU work.
//!
//! Core subsystems live in [`super::MaterialSystem`], [`crate::assets::AssetTransferQueue`],
//! [`super::FrameResourceManager`], and [`super::OcclusionSystem`]; this type wires attach,
//! the compiled render graph, mesh deform preprocess, and debug HUD.
//!
//! Graph execution lives in the `execute` submodule; IPC-facing asset handlers in `asset_ipc`.

mod asset_ipc;
mod execute;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;

use crate::assets::asset_transfer_queue::{self as asset_uploads, AssetTransferQueue};
use crate::assets::material::MaterialPropertyStore;
use crate::backend::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::config::RendererSettingsHandle;
use crate::diagnostics::{DebugHudEncodeError, DebugHudInput, SceneTransformsSnapshot};
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::render_graph::{TransientPool, WorldMeshDrawStateRow, WorldMeshDrawStats};
use crate::resources::{CubemapPool, MeshPool, RenderTexturePool, Texture3dPool, TexturePool};

use super::debug_hud_bundle::DebugHudBundle;
use super::embedded::{EmbeddedMaterialBindError, EmbeddedTexturePools};
use super::material_system::MaterialSystem;
use super::occlusion::OcclusionSystem;
use super::FrameGpuBindingsError;

pub use crate::assets::asset_transfer_queue::{
    MAX_ASSET_INTEGRATION_QUEUED, MAX_PENDING_MESH_UPLOADS, MAX_PENDING_TEXTURE_UPLOADS,
};

/// GPU attach failed for frame binds (`@group(0/1/2)`) or embedded materials (`@group(1)`).
#[derive(Debug, Error)]
pub enum RenderBackendAttachError {
    /// Frame / empty material / per-draw allocation failed atomically.
    #[error(transparent)]
    FrameGpuBindings(#[from] FrameGpuBindingsError),
    /// Embedded raster `@group(1)` bind resources could not be created.
    #[error(transparent)]
    EmbeddedMaterialBind(#[from] EmbeddedMaterialBindError),
}

/// Device, queue, and settings passed to [`RenderBackend::attach`] (shared-memory flush is passed separately for borrow reasons).
pub struct RenderBackendAttachDesc {
    /// Logical device for uploads and graph encoding.
    pub device: Arc<wgpu::Device>,
    /// Queue used for submits and GPU writes.
    pub queue: Arc<wgpu::Queue>,
    /// Capabilities for buffer sizing and MSAA.
    pub gpu_limits: Arc<GpuLimits>,
    /// Swapchain / main surface format for HUD and pipelines.
    pub surface_format: wgpu::TextureFormat,
    /// Live renderer settings (HUD, VR budgets, etc.).
    pub renderer_settings: RendererSettingsHandle,
    /// Path for persisting HUD/config from the debug overlay.
    pub config_save_path: PathBuf,
    /// When `true`, the ImGui config window must not write `config.toml` (startup extract failed).
    pub suppress_renderer_config_disk_writes: bool,
}

/// Coordinates materials, asset uploads, per-frame GPU binds, occlusion, optional deform + ImGui HUD, and the render graph.
pub struct RenderBackend {
    /// Material property store, shader routes, pipeline registry, embedded `@group(1)` binds.
    pub(crate) materials: MaterialSystem,
    /// Mesh/texture upload queues, budgets, format tables, pools, and GPU device/queue for uploads.
    pub(crate) asset_transfers: AssetTransferQueue,
    /// Optional mesh skinning / blendshape compute pipelines (after [`Self::attach`]).
    mesh_preprocess: Option<MeshPreprocessPipelines>,
    /// Compiled DAG of render passes (after [`Self::attach`]); see [`crate::render_graph`].
    frame_graph: Option<crate::render_graph::CompiledRenderGraph>,
    /// Scratch buffers for mesh deformation compute (after [`Self::attach`]).
    mesh_deform_scratch: Option<MeshDeformScratch>,
    /// MSAA depth -> R32F -> single-sample depth resolve resources when supported.
    msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// Per-frame bind groups, light staging, and debug draw slab.
    pub(crate) frame_resources: super::FrameResourceManager,
    /// Dear ImGui overlay and capture state.
    debug_hud: DebugHudBundle,
    /// Hierarchical depth pyramid, CPU readback, and temporal cull state for occlusion culling.
    pub(crate) occlusion: OcclusionSystem,
    /// Render-graph transient texture/buffer pool retained across frames.
    pub(crate) transient_pool: TransientPool,
}

/// Disjoint borrows of [`MaterialSystem`], [`AssetTransferQueue`], and the GPU skin cache for world mesh forward encoding.
///
/// Built by [`RenderBackend::world_mesh_forward_encode_refs`] so the raster encoder never holds
/// `&mut RenderBackend` while also borrowing the deform cache on [`RenderBackend`].
pub(crate) struct WorldMeshForwardEncodeRefs<'a> {
    /// Material registry, embedded binds, and property store.
    pub(crate) materials: &'a mut MaterialSystem,
    /// Mesh and texture pools (mutable for lazy extended vertex stream uploads).
    pub(crate) asset_transfers: &'a mut AssetTransferQueue,
    /// Arena-backed deformed positions and normals keyed by renderable (after [`RenderBackend::attach`]).
    pub(crate) skin_cache: Option<&'a GpuSkinCache>,
}

impl<'a> WorldMeshForwardEncodeRefs<'a> {
    /// Mutable mesh pool for lazy extended vertex stream uploads during draw recording.
    pub(crate) fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        &mut self.asset_transfers.mesh_pool
    }

    /// Pool views for embedded `@group(1)` texture resolution.
    pub(crate) fn embedded_texture_pools(&self) -> EmbeddedTexturePools<'_> {
        EmbeddedTexturePools {
            texture: &self.asset_transfers.texture_pool,
            texture3d: &self.asset_transfers.texture3d_pool,
            cubemap: &self.asset_transfers.cubemap_pool,
            render_texture: &self.asset_transfers.render_texture_pool,
        }
    }
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
            frame_graph: None,
            mesh_deform_scratch: None,
            msaa_depth_resolve: None,
            frame_resources: super::FrameResourceManager::new(),
            debug_hud: DebugHudBundle::new(),
            occlusion: OcclusionSystem::new(),
            transient_pool: TransientPool::new(),
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

    /// Arena-backed deformed vertex streams shared by mesh deform compute and mesh forward draws.
    pub fn skin_cache(&self) -> Option<&GpuSkinCache> {
        self.frame_resources.skin_cache()
    }

    /// Mutable skin cache for mesh deform compute and cache sweeps.
    pub fn skin_cache_mut(&mut self) -> Option<&mut GpuSkinCache> {
        self.frame_resources.skin_cache_mut()
    }

    /// MSAA depth → R32F → single-sample depth resolve resources when supported.
    pub(crate) fn msaa_depth_resolve(&self) -> Option<Arc<MsaaDepthResolveResources>> {
        self.msaa_depth_resolve.clone()
    }

    /// Resets per-tick light prep flags, mesh deform coalescing, and advances the skin cache frame counter.
    ///
    /// Call once per winit tick before IPC and frame work (see [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn reset_light_prep_for_tick(&mut self) {
        self.frame_resources.reset_light_prep_for_tick();
    }

    /// Borrows material and pool state disjointly from the GPU skin cache for mesh forward encoding.
    pub(crate) fn world_mesh_forward_encode_refs(&mut self) -> WorldMeshForwardEncodeRefs<'_> {
        WorldMeshForwardEncodeRefs {
            materials: &mut self.materials,
            asset_transfers: &mut self.asset_transfers,
            skin_cache: self.frame_resources.skin_cache(),
        }
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

    /// Number of schedules passes in the compiled frame graph, or `0` if none.
    pub fn frame_graph_pass_count(&self) -> usize {
        self.frame_graph.as_ref().map_or(0, |g| g.pass_count())
    }

    /// Compile-time topological wave count for the cached frame graph, or `0` if none has been built yet.
    pub fn frame_graph_topo_levels(&self) -> usize {
        self.frame_graph
            .as_ref()
            .map_or(0, |g| g.compile_stats.topo_levels)
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    ///
    /// Wires device/queue into uploads, allocates frame binds and materials, and builds the default graph.
    /// `shm` flushes pending mesh/texture payloads that require shared-memory reads; omit when none is
    /// available yet (uploads stay queued).
    ///
    /// On error, CPU-side asset queues may already be partially configured; GPU draws must not run until
    /// a successful attach.
    pub fn attach(
        &mut self,
        desc: RenderBackendAttachDesc,
        shm: Option<&mut crate::ipc::SharedMemoryAccessor>,
    ) -> Result<(), RenderBackendAttachError> {
        let RenderBackendAttachDesc {
            device,
            queue,
            gpu_limits,
            surface_format,
            renderer_settings,
            config_save_path,
            suppress_renderer_config_disk_writes,
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
        self.mesh_deform_scratch = Some(MeshDeformScratch::new(
            device.as_ref(),
            gpu_limits.max_buffer_size(),
        ));
        self.frame_resources.attach(device.as_ref(), gpu_limits)?;
        self.debug_hud.attach(
            device.as_ref(),
            queue.as_ref(),
            surface_format,
            renderer_settings,
            config_save_path,
            suppress_renderer_config_disk_writes,
        );
        match MeshPreprocessPipelines::new(device.as_ref()) {
            Ok(p) => self.mesh_preprocess = Some(p),
            Err(e) => {
                logger::warn!("mesh preprocess compute pipelines not created: {e}");
                self.mesh_preprocess = None;
            }
        }
        self.materials.try_attach_gpu(device.clone(), &queue)?;
        asset_uploads::attach_flush_pending_asset_uploads(&mut self.asset_transfers, &device, shm);

        self.msaa_depth_resolve = MsaaDepthResolveResources::try_new(device.as_ref()).map(Arc::new);

        self.frame_graph = match crate::render_graph::build_default_main_graph() {
            Ok(g) => Some(g),
            Err(e) => {
                logger::warn!("default render graph build failed: {e}");
                None
            }
        };
        Ok(())
    }

    /// Updates whether main HUD diagnostics run (mirrors [`crate::config::DebugSettings::debug_hud_enabled`]).
    pub fn set_debug_hud_main_enabled(&mut self, enabled: bool) {
        self.debug_hud.set_main_enabled(enabled);
    }

    /// Whether main debug HUD is on (mesh-draw stats for [`crate::render_graph::passes::WorldMeshForwardOpaquePass`]).
    pub(crate) fn debug_hud_main_enabled(&self) -> bool {
        self.debug_hud.main_enabled()
    }

    /// Updates whether texture HUD diagnostics run.
    pub(crate) fn set_debug_hud_textures_enabled(&mut self, enabled: bool) {
        self.debug_hud.set_textures_enabled(enabled);
    }

    /// Whether texture debug HUD capture is on.
    pub(crate) fn debug_hud_textures_enabled(&self) -> bool {
        self.debug_hud.textures_enabled()
    }

    /// Clears the current-view Texture2D set before collecting this frame's submitted draws.
    pub(crate) fn clear_debug_hud_current_view_texture_2d_asset_ids(&mut self) {
        self.debug_hud.clear_current_view_texture_2d_asset_ids();
    }

    /// Adds Texture2D ids used by submitted world draws for the current view.
    pub(crate) fn note_debug_hud_current_view_texture_2d_asset_ids(
        &mut self,
        asset_ids: impl IntoIterator<Item = i32>,
    ) {
        self.debug_hud
            .note_current_view_texture_2d_asset_ids(asset_ids);
    }

    /// Texture2D ids used by submitted world draws for the current view.
    pub(crate) fn debug_hud_current_view_texture_2d_asset_ids(&self) -> &BTreeSet<i32> {
        self.debug_hud.current_view_texture_2d_asset_ids()
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

    pub(crate) fn set_last_world_mesh_draw_state_rows(&mut self, rows: Vec<WorldMeshDrawStateRow>) {
        self.debug_hud.set_last_world_mesh_draw_state_rows(rows);
    }

    pub(crate) fn last_world_mesh_draw_state_rows(&self) -> Vec<WorldMeshDrawStateRow> {
        self.debug_hud.last_world_mesh_draw_state_rows()
    }

    /// Updates the **Scene transforms** Dear ImGui window payload for the next composite pass.
    pub(crate) fn set_debug_hud_scene_transforms_snapshot(
        &mut self,
        snapshot: SceneTransformsSnapshot,
    ) {
        self.debug_hud.set_scene_transforms_snapshot(snapshot);
    }

    /// Updates the **Textures** Dear ImGui window payload for the next composite pass.
    pub(crate) fn set_debug_hud_texture_debug_snapshot(
        &mut self,
        snapshot: crate::diagnostics::TextureDebugSnapshot,
    ) {
        self.debug_hud.set_texture_debug_snapshot(snapshot);
    }

    /// Clears the **Textures** HUD payload.
    pub(crate) fn clear_debug_hud_texture_debug_snapshot(&mut self) {
        self.debug_hud.clear_texture_debug_snapshot();
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

    /// Mutable render-graph transient resource pool.
    pub(crate) fn transient_pool_mut(&mut self) -> &mut TransientPool {
        &mut self.transient_pool
    }

    /// Scratch buffers for mesh deformation (`MeshDeformPass`).
    pub fn mesh_deform_scratch_mut(&mut self) -> Option<&mut MeshDeformScratch> {
        self.mesh_deform_scratch.as_mut()
    }

    /// Compute preprocess pipelines + deform scratch (`MeshDeformPass`) as one disjoint borrow.
    pub fn mesh_deform_pre_and_scratch(
        &mut self,
    ) -> Option<(&MeshPreprocessPipelines, &mut MeshDeformScratch)> {
        let pre = self.mesh_preprocess.as_ref()?;
        let scratch = self.mesh_deform_scratch.as_mut()?;
        Some((pre, scratch))
    }

    /// Preprocess pipelines, deform scratch, and GPU skin cache as one disjoint borrow for [`MeshDeformPass`].
    ///
    /// Bundles [`Self::mesh_preprocess`], [`Self::mesh_deform_scratch`], and
    /// [`FrameResourceManager::skin_cache_mut`](super::FrameResourceManager::skin_cache_mut) so callers
    /// avoid splitting borrows with raw pointers.
    pub fn mesh_deform_pre_scratch_and_skin_cache(
        &mut self,
    ) -> Option<(
        &MeshPreprocessPipelines,
        &mut MeshDeformScratch,
        &mut GpuSkinCache,
    )> {
        let pre = self.mesh_preprocess.as_ref()?;
        let scratch = self.mesh_deform_scratch.as_mut()?;
        let skin = self.frame_resources.skin_cache_mut()?;
        Some((pre, scratch, skin))
    }
}
