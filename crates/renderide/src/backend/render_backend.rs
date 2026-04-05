//! [`RenderBackend`] implementation.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::assets::material::{
    parse_materials_update_batch_into_store, MaterialPropertyStore, ParseMaterialBatchOptions,
    PropertyIdRegistry,
};
use crate::assets::mesh::try_upload_mesh_from_raw;
use crate::assets::texture::write_texture2d_mips;
use crate::gpu::{GpuContext, MeshPreprocessPipelines};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::MaterialFamilyId;
use crate::render_graph::{CompiledRenderGraph, GraphExecuteError};
use crate::resources::{GpuTexture2d, MeshPool, TexturePool};
use crate::scene::SceneCoordinator;

use super::light_gpu::{order_lights_for_clustered_shading, GpuLight};
use crate::shared::{
    MaterialsUpdateBatch, MaterialsUpdateBatchResult, MeshUnload, MeshUploadData, MeshUploadResult,
    RendererCommand, SetTexture2DData, SetTexture2DFormat, SetTexture2DProperties,
    SetTexture2DResult, TextureUpdateResultType, UnloadTexture2D,
};
use winit::window::Window;

/// Max queued [`MeshUploadData`] when GPU is not ready yet (host data stays in shared memory).
pub const MAX_PENDING_MESH_UPLOADS: usize = 256;

/// Max queued texture data commands when GPU or format is not ready.
pub const MAX_PENDING_TEXTURE_UPLOADS: usize = 256;

/// Max queued [`MaterialsUpdateBatch`] when shared memory is not available.
pub const MAX_PENDING_MATERIAL_BATCHES: usize = 256;

/// GPU resource pools, material property data, and asset upload paths.
pub struct RenderBackend {
    /// Host material property batches (`MaterialsUpdateBatch`); separate maps for materials vs blocks.
    material_property_store: MaterialPropertyStore,
    /// Stable ids for [`crate::shared::MaterialPropertyIdRequest`] / batch `property_id` keys.
    property_id_registry: PropertyIdRegistry,
    pending_material_batches: VecDeque<MaterialsUpdateBatch>,
    mesh_pool: MeshPool,
    texture_pool: TexturePool,
    /// Latest [`SetTexture2DFormat`] per asset (required before data upload).
    texture_formats: HashMap<i32, SetTexture2DFormat>,
    /// Latest [`SetTexture2DProperties`] per asset (sampler metadata on [`GpuTexture2d`]).
    texture_properties: HashMap<i32, SetTexture2DProperties>,
    gpu_device: Option<Arc<wgpu::Device>>,
    gpu_queue: Option<Arc<Mutex<wgpu::Queue>>>,
    pending_mesh_uploads: VecDeque<MeshUploadData>,
    pending_texture_uploads: VecDeque<SetTexture2DData>,
    /// GPU material families, router, and pipeline cache (after [`Self::attach`]).
    material_registry: Option<crate::materials::MaterialRegistry>,
    /// Shader asset id → family when uploads arrive before GPU attach.
    pending_shader_routes: HashMap<i32, MaterialFamilyId>,
    /// Optional mesh skinning / blendshape compute pipelines (after [`Self::attach`]).
    mesh_preprocess: Option<MeshPreprocessPipelines>,
    /// Compiled DAG of render passes (after [`Self::attach`]); see [`crate::render_graph`].
    frame_graph: Option<CompiledRenderGraph>,
    /// Last packed lights for the frame (after [`Self::prepare_lights_from_scene`]).
    light_scratch: Vec<GpuLight>,
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
            material_property_store: MaterialPropertyStore::new(),
            property_id_registry: PropertyIdRegistry::new(),
            pending_material_batches: VecDeque::new(),
            mesh_pool: MeshPool::default_pool(),
            texture_pool: TexturePool::default_pool(),
            texture_formats: HashMap::new(),
            texture_properties: HashMap::new(),
            gpu_device: None,
            gpu_queue: None,
            pending_mesh_uploads: VecDeque::new(),
            pending_texture_uploads: VecDeque::new(),
            material_registry: None,
            pending_shader_routes: HashMap::new(),
            mesh_preprocess: None,
            frame_graph: None,
            light_scratch: Vec::new(),
        }
    }

    /// Packed GPU lights from the last [`Self::prepare_lights_from_scene`] call.
    pub fn frame_lights(&self) -> &[GpuLight] {
        &self.light_scratch
    }

    /// Fills [`Self::light_scratch`] from [`SceneCoordinator`] (all spaces, clustered ordering, cap [`crate::backend::MAX_LIGHTS`]).
    pub fn prepare_lights_from_scene(&mut self, scene: &SceneCoordinator) {
        self.light_scratch.clear();
        let mut all = Vec::new();
        for id in scene.render_space_ids() {
            all.extend(scene.resolve_lights_world(id));
        }
        let ordered = order_lights_for_clustered_shading(&all);
        self.light_scratch
            .extend(ordered.iter().map(GpuLight::from_resolved));
    }

    /// Mesh deformation compute pipelines when GPU init succeeded.
    pub fn mesh_preprocess(&self) -> Option<&MeshPreprocessPipelines> {
        self.mesh_preprocess.as_ref()
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &MeshPool {
        &self.mesh_pool
    }

    /// Mutable mesh pool (eviction experiments).
    pub fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        &mut self.mesh_pool
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &TexturePool {
        &self.texture_pool
    }

    /// Mutable texture pool.
    pub fn texture_pool_mut(&mut self) -> &mut TexturePool {
        &mut self.texture_pool
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &MaterialPropertyStore {
        &self.material_property_store
    }

    /// Mutable store for tests and tooling.
    pub fn material_property_store_mut(&mut self) -> &mut MaterialPropertyStore {
        &mut self.material_property_store
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &PropertyIdRegistry {
        &self.property_id_registry
    }

    /// Mutable property id registry.
    pub fn property_id_registry_mut(&mut self) -> &mut PropertyIdRegistry {
        &mut self.property_id_registry
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.material_registry.as_ref()
    }

    /// Mutable registry (e.g.register custom [`crate::materials::MaterialPipelineFamily`]).
    pub fn material_registry_mut(&mut self) -> Option<&mut crate::materials::MaterialRegistry> {
        self.material_registry.as_mut()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    ///
    /// `shm` is used to flush pending mesh/texture payloads that require shared-memory reads; omit
    /// when none is available yet (uploads stay queued).
    pub fn attach(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<Mutex<wgpu::Queue>>,
        shm: Option<&mut SharedMemoryAccessor>,
    ) {
        self.gpu_device = Some(device.clone());
        self.gpu_queue = Some(queue);
        match MeshPreprocessPipelines::new(device.as_ref()) {
            Ok(p) => self.mesh_preprocess = Some(p),
            Err(e) => {
                logger::warn!("mesh preprocess compute pipelines not created: {e}");
                self.mesh_preprocess = None;
            }
        }
        self.material_registry = Some(crate::materials::MaterialRegistry::with_default_families(
            device.clone(),
        ));
        if let Some(reg) = self.material_registry.as_mut() {
            for (asset_id, family) in self.pending_shader_routes.drain() {
                reg.map_shader_to_family(asset_id, family);
            }
        }
        self.flush_pending_texture_allocations(&device);
        let pending_tex: Vec<SetTexture2DData> = self.pending_texture_uploads.drain(..).collect();
        let pending_mesh: Vec<MeshUploadData> = self.pending_mesh_uploads.drain(..).collect();
        if let Some(shm) = shm {
            for data in pending_tex {
                self.try_texture_upload_with_device(data, shm, None);
            }
            for data in pending_mesh {
                self.try_mesh_upload_with_device(&device, data, shm, None);
            }
        } else {
            for data in pending_tex {
                self.pending_texture_uploads.push_back(data);
            }
            for data in pending_mesh {
                self.pending_mesh_uploads.push_back(data);
            }
        }

        self.frame_graph = match crate::render_graph::build_default_main_graph() {
            Ok(g) => Some(g),
            Err(e) => {
                logger::warn!("default render graph build failed: {e}");
                None
            }
        };
    }

    /// Records and presents one frame using the compiled render graph (swapchain clear + future passes).
    ///
    /// Returns [`GraphExecuteError::NoFrameGraph`] if graph build failed during [`Self::attach`].
    pub fn execute_frame_graph(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
    ) -> Result<(), GraphExecuteError> {
        let Some(graph) = self.frame_graph.as_mut() else {
            return Err(GraphExecuteError::NoFrameGraph);
        };
        graph.execute(gpu, window)
    }

    /// Maps shader asset to material family, or defers until [`Self::attach`].
    pub fn register_shader_route(&mut self, asset_id: i32, family: MaterialFamilyId) {
        if let Some(reg) = self.material_registry.as_mut() {
            reg.map_shader_to_family(asset_id, family);
        } else {
            self.pending_shader_routes.insert(asset_id, family);
        }
    }

    /// Removes shader routing for `asset_id`.
    pub fn unregister_shader_route(&mut self, asset_id: i32) {
        self.pending_shader_routes.remove(&asset_id);
        if let Some(reg) = self.material_registry.as_mut() {
            reg.unmap_shader(asset_id);
        }
    }

    /// Drain pending material batches using the given shared memory and IPC.
    pub fn flush_pending_material_batches(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        let batches: Vec<MaterialsUpdateBatch> = self.pending_material_batches.drain(..).collect();
        for batch in batches {
            self.apply_materials_update_batch(batch, shm, ipc);
        }
    }

    /// Queue a materials batch when shared memory is not yet available. Returns `false` if queue full.
    pub fn enqueue_materials_batch_no_shm(&mut self, batch: MaterialsUpdateBatch) -> bool {
        if self.pending_material_batches.len() >= MAX_PENDING_MATERIAL_BATCHES {
            logger::warn!(
                "materials update batch {} dropped: pending queue full (no shared memory)",
                batch.update_batch_id
            );
            return false;
        }
        self.pending_material_batches.push_back(batch);
        true
    }

    /// Apply one host materials batch (shared memory must be valid for the batch descriptors).
    pub fn apply_materials_update_batch(
        &mut self,
        batch: MaterialsUpdateBatch,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        let update_batch_id = batch.update_batch_id;
        let opts = ParseMaterialBatchOptions::default();
        parse_materials_update_batch_into_store(
            shm,
            &batch,
            &mut self.material_property_store,
            &opts,
        );
        ipc.send_background(RendererCommand::materials_update_batch_result(
            MaterialsUpdateBatchResult { update_batch_id },
        ));
    }

    fn flush_pending_texture_allocations(&mut self, device: &Arc<wgpu::Device>) {
        let ids: Vec<i32> = self.texture_formats.keys().copied().collect();
        for id in ids {
            if self.texture_pool.get_texture(id).is_some() {
                continue;
            }
            let Some(fmt) = self.texture_formats.get(&id).cloned() else {
                continue;
            };
            let props = self.texture_properties.get(&id);
            let Some(tex) = GpuTexture2d::new_from_format(device.as_ref(), &fmt, props) else {
                logger::warn!("texture {id}: failed to allocate GPU texture on attach");
                continue;
            };
            let _ = self.texture_pool.insert_texture(tex);
        }
    }

    fn send_texture_2d_result(
        ipc: Option<&mut DualQueueIpc>,
        asset_id: i32,
        update: i32,
        instance_changed: bool,
    ) {
        let Some(ipc) = ipc else {
            return;
        };
        ipc.send_background(RendererCommand::set_texture_2d_result(SetTexture2DResult {
            asset_id,
            r#type: TextureUpdateResultType(update),
            instance_changed,
        }));
    }

    /// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
    pub fn on_set_texture_2d_format(
        &mut self,
        f: SetTexture2DFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        let id = f.asset_id;
        self.texture_formats.insert(id, f.clone());
        let props = self.texture_properties.get(&id);
        let Some(device) = self.gpu_device.clone() else {
            Self::send_texture_2d_result(
                ipc,
                id,
                TextureUpdateResultType::FORMAT_SET,
                self.texture_pool.get_texture(id).is_none(),
            );
            return;
        };
        let Some(tex) = GpuTexture2d::new_from_format(device.as_ref(), &f, props) else {
            logger::warn!("texture {id}: SetTexture2DFormat rejected (bad size or device)");
            return;
        };
        let existed_before = self.texture_pool.insert_texture(tex);
        Self::send_texture_2d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            !existed_before,
        );
        logger::info!(
            "texture {} format {:?} {}×{} mips={} (resident_bytes≈{})",
            id,
            f.format,
            f.width,
            f.height,
            f.mipmap_count,
            self.texture_pool.accounting().texture_resident_bytes()
        );
    }

    /// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
    pub fn on_set_texture_2d_properties(
        &mut self,
        p: SetTexture2DProperties,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        let id = p.asset_id;
        self.texture_properties.insert(id, p.clone());
        if let Some(t) = self.texture_pool.get_texture_mut(id) {
            t.apply_properties(&p);
        }
        Self::send_texture_2d_result(ipc, id, TextureUpdateResultType::PROPERTIES_SET, false);
    }

    /// Handle [`SetTexture2DData`](crate::shared::SetTexture2DData). Pass shared memory when available
    /// so mips can be read from the host buffer; if GPU or texture is not ready, data is queued.
    pub fn on_set_texture_2d_data(
        &mut self,
        d: SetTexture2DData,
        shm: Option<&mut SharedMemoryAccessor>,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        if d.data.length <= 0 {
            return;
        }
        if !self.texture_formats.contains_key(&d.asset_id) {
            logger::warn!(
                "texture {}: SetTexture2DData before format; ignored",
                d.asset_id
            );
            return;
        }
        if self.gpu_device.is_none() || self.gpu_queue.is_none() {
            if self.pending_texture_uploads.len() >= MAX_PENDING_TEXTURE_UPLOADS {
                logger::warn!(
                    "texture {}: pending texture upload queue full; dropping",
                    d.asset_id
                );
                return;
            }
            self.pending_texture_uploads.push_back(d);
            return;
        }
        let Some(ref device) = self.gpu_device.clone() else {
            return;
        };
        if self.texture_pool.get_texture(d.asset_id).is_none() {
            self.flush_pending_texture_allocations(device);
        }
        if self.texture_pool.get_texture(d.asset_id).is_none() {
            if self.pending_texture_uploads.len() >= MAX_PENDING_TEXTURE_UPLOADS {
                logger::warn!(
                    "texture {}: no GPU texture and pending full; dropping data",
                    d.asset_id
                );
                return;
            }
            self.pending_texture_uploads.push_back(d);
            return;
        }
        let Some(shm) = shm else {
            logger::warn!(
                "texture {}: SetTexture2DData needs shared memory for upload",
                d.asset_id
            );
            return;
        };
        self.try_texture_upload_with_device(d, shm, ipc);
    }

    /// Upload texture mips from shared memory and optionally notify the host on the background queue.
    pub fn try_texture_upload_with_device(
        &mut self,
        data: SetTexture2DData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        let id = data.asset_id;
        let Some(fmt) = self.texture_formats.get(&id).cloned() else {
            logger::warn!("texture {id}: missing format");
            return;
        };
        let (tex_arc, wgpu_fmt) = match self.texture_pool.get_texture(id) {
            Some(t) => (t.texture.clone(), t.wgpu_format),
            None => {
                logger::warn!("texture {id}: missing GPU texture");
                return;
            }
        };
        let Some(queue_arc) = self.gpu_queue.as_ref() else {
            return;
        };
        let upload_out = shm.with_read_bytes(&data.data, |raw| {
            let q = queue_arc.lock().expect("queue mutex poisoned");
            Some(write_texture2d_mips(
                &q,
                tex_arc.as_ref(),
                &fmt,
                wgpu_fmt,
                &data,
                raw,
            ))
        });
        match upload_out {
            Some(Ok(())) => {
                if let Some(t) = self.texture_pool.get_texture_mut(id) {
                    let uploaded_mips = data.mip_map_sizes.len() as u32;
                    let start = data.start_mip_level.max(0) as u32;
                    let end_exclusive = start.saturating_add(uploaded_mips).min(t.mip_levels_total);
                    t.mip_levels_resident = t.mip_levels_resident.max(end_exclusive);
                }
                Self::send_texture_2d_result(ipc, id, TextureUpdateResultType::DATA_UPLOAD, false);
                logger::trace!("texture {id}: data upload ok");
            }
            Some(Err(e)) => {
                logger::warn!("texture {id}: upload failed: {e}");
            }
            None => {
                logger::warn!("texture {id}: shared memory slice missing");
            }
        }
    }

    /// Remove a texture asset from CPU tables and the pool.
    pub fn on_unload_texture_2d(&mut self, u: UnloadTexture2D) {
        let id = u.asset_id;
        self.texture_formats.remove(&id);
        self.texture_properties.remove(&id);
        if self.texture_pool.remove_texture(id) {
            logger::info!(
                "texture {id} unloaded (mesh≈{} tex≈{} total≈{})",
                self.mesh_pool.accounting().mesh_resident_bytes(),
                self.texture_pool.accounting().texture_resident_bytes(),
                self.mesh_pool.accounting().total_resident_bytes()
            );
        }
    }

    /// Ingest mesh bytes from shared memory; notifies host when `ipc` is set.
    pub fn try_process_mesh_upload(
        &mut self,
        data: MeshUploadData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        if data.buffer.length <= 0 {
            return;
        }
        let Some(device) = self.gpu_device.clone() else {
            if self.pending_mesh_uploads.len() >= MAX_PENDING_MESH_UPLOADS {
                logger::warn!(
                    "mesh upload pending queue full; dropping asset {}",
                    data.asset_id
                );
                return;
            }
            self.pending_mesh_uploads.push_back(data);
            return;
        };
        self.try_mesh_upload_with_device(&device, data, shm, ipc);
    }

    fn try_mesh_upload_with_device(
        &mut self,
        device: &Arc<wgpu::Device>,
        data: MeshUploadData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        let upload_result = shm.with_read_bytes(&data.buffer, |raw| {
            try_upload_mesh_from_raw(device.as_ref(), raw, &data)
        });
        let Some(mesh) = upload_result else {
            logger::warn!("mesh {}: upload failed or rejected", data.asset_id);
            return;
        };
        let existed_before = self.mesh_pool.insert_mesh(mesh);
        if let Some(ipc) = ipc {
            ipc.send_background(RendererCommand::mesh_upload_result(MeshUploadResult {
                asset_id: data.asset_id,
                instance_changed: !existed_before,
            }));
        }
        logger::info!(
            "mesh {} uploaded (replaced={} resident_bytes≈{})",
            data.asset_id,
            existed_before,
            self.mesh_pool.accounting().total_resident_bytes()
        );
    }

    /// Remove a mesh from the pool.
    pub fn on_mesh_unload(&mut self, u: MeshUnload) {
        if self.mesh_pool.remove_mesh(u.asset_id) {
            logger::info!(
                "mesh {} unloaded (resident_bytes≈{})",
                u.asset_id,
                self.mesh_pool.accounting().total_resident_bytes()
            );
        }
    }

    /// Remove material / property-block entries from the host store.
    pub fn on_unload_material(&mut self, asset_id: i32) {
        self.material_property_store.remove_material(asset_id);
    }

    /// Remove a property block from the host store.
    pub fn on_unload_material_property_block(&mut self, asset_id: i32) {
        self.material_property_store.remove_property_block(asset_id);
    }
}
