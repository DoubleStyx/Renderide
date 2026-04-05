//! Renderer orchestration: IPC polling, init lifecycle, lock-step frame gating, mesh + texture ingest.
//!
//! Phase order is aligned with `RenderingManager.HandleUpdate`: optionally send
//! [`FrameStartData`](crate::shared::FrameStartData), drain integration-style work (stub here), then
//! process incoming commands.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::assets::mesh::try_upload_mesh_from_raw;
use crate::assets::texture::{supported_host_formats_for_init, write_texture2d_mips};
use crate::assets::AssetSubsystem;
use crate::connection::{ConnectionParams, InitError};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::resources::{GpuTexture2d, MeshPool, TexturePool};
use crate::shared::{
    FrameStartData, FrameSubmitData, HeadOutputDevice, MeshUnload, MeshUploadData,
    MeshUploadResult, RendererCommand, RendererInitData, RendererInitResult, SetTexture2DData,
    SetTexture2DFormat, SetTexture2DProperties, SetTexture2DResult, TextureUpdateResultType,
    UnloadTexture2D,
};

/// Max queued [`MeshUploadData`] when GPU is not ready yet (host data stays in shared memory).
const MAX_PENDING_MESH_UPLOADS: usize = 256;

/// Max queued texture data commands when GPU or format is not ready.
const MAX_PENDING_TEXTURE_UPLOADS: usize = 256;

/// Host init sequence state (replaces paired booleans such as `init_received` / `init_finalized`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InitState {
    /// Waiting for [`RendererCommand::renderer_init_data`].
    #[default]
    Uninitialized,
    /// `renderer_init_data` received; waiting for [`RendererCommand::renderer_init_finalize_data`].
    InitReceived,
    /// Normal operation (or standalone mode).
    Finalized,
}

impl InitState {
    /// Whether host init handshake is complete.
    pub fn is_finalized(self) -> bool {
        matches!(self, InitState::Finalized)
    }
}

/// Owns IPC (optional), lock-step flags, shared memory, and GPU resource pools.
pub struct RendererRuntime {
    ipc: Option<DualQueueIpc>,
    params: Option<ConnectionParams>,
    init_state: InitState,
    /// After a successful [`FrameSubmitData`] application, host may expect another begin-frame.
    pub last_frame_data_processed: bool,
    pub last_frame_index: i32,
    sent_bootstrap_frame_start: bool,
    pub shutdown_requested: bool,
    pub fatal_error: bool,
    assets: AssetSubsystem,
    pending_init: Option<RendererInitData>,
    shared_memory: Option<SharedMemoryAccessor>,
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
}

impl RendererRuntime {
    /// Builds a runtime; does not open IPC yet (see [`Self::connect_ipc`]).
    pub fn new(params: Option<ConnectionParams>) -> Self {
        let standalone = params.is_none();
        let init_state = if standalone {
            InitState::Finalized
        } else {
            InitState::default()
        };
        Self {
            ipc: None,
            params,
            init_state,
            last_frame_data_processed: standalone,
            last_frame_index: -1,
            sent_bootstrap_frame_start: false,
            shutdown_requested: false,
            fatal_error: false,
            assets: AssetSubsystem::default(),
            pending_init: None,
            shared_memory: None,
            mesh_pool: MeshPool::default_pool(),
            texture_pool: TexturePool::default_pool(),
            texture_formats: HashMap::new(),
            texture_properties: HashMap::new(),
            gpu_device: None,
            gpu_queue: None,
            pending_mesh_uploads: VecDeque::new(),
            pending_texture_uploads: VecDeque::new(),
        }
    }

    /// Opens Primary/Background queues when [`Self::new`] was given connection parameters.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        let Some(ref p) = self.params.clone() else {
            return Ok(());
        };
        self.ipc = Some(DualQueueIpc::connect(p)?);
        Ok(())
    }

    /// Whether IPC queues are open.
    pub fn is_ipc_connected(&self) -> bool {
        self.ipc.is_some()
    }

    pub fn init_state(&self) -> InitState {
        self.init_state
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

    /// Exposes asset subsystem hooks (upload queues, handle table) for future workers.
    pub fn assets_mut(&mut self) -> &mut AssetSubsystem {
        &mut self.assets
    }

    /// Applies pending init once a GPU/window stack exists (e.g. window title).
    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.pending_init.take()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    pub fn attach_gpu(&mut self, device: Arc<wgpu::Device>, queue: Arc<Mutex<wgpu::Queue>>) {
        self.gpu_device = Some(device.clone());
        self.gpu_queue = Some(queue);
        self.flush_pending_texture_allocations(&device);
        let pending_tex: Vec<SetTexture2DData> = self.pending_texture_uploads.drain(..).collect();
        for data in pending_tex {
            self.try_texture_upload_with_device(data);
        }
        let pending: Vec<MeshUploadData> = self.pending_mesh_uploads.drain(..).collect();
        for data in pending {
            self.try_mesh_upload_with_device(&device, data);
        }
    }

    /// If connected and init is complete, sends [`FrameStartData`] when we are ready for the next
    /// host frame (Unity: `_lastFrameDataProcessed` or bootstrap), then clears the processed flag.
    pub fn pre_frame(&mut self) {
        if !self.init_state.is_finalized() || self.fatal_error || self.ipc.is_none() {
            return;
        }

        let bootstrap = self.last_frame_index < 0 && !self.sent_bootstrap_frame_start;
        let should_send = self.last_frame_data_processed || bootstrap;
        if !should_send {
            return;
        }

        let frame_start = FrameStartData {
            last_frame_index: self.last_frame_index,
            ..Default::default()
        };
        if let Some(ref mut ipc) = self.ipc {
            ipc.send_primary(RendererCommand::frame_start_data(frame_start));
        }
        self.last_frame_data_processed = false;
        if bootstrap {
            self.sent_bootstrap_frame_start = true;
        }
    }

    /// Placeholder for bounded asset integration between begin-frame and frame processing (Unity:
    /// `RunAssetIntegration`).
    pub fn run_asset_integration_stub(&mut self, _budget: Duration) {
        let _ = self.assets.drain_pending_meta();
    }

    /// Drains IPC and dispatches commands. Frame submissions are sorted before other commands from
    /// the same [`Self::poll`] batch.
    pub fn poll_ipc(&mut self) {
        let Some(ref mut ipc) = self.ipc else {
            return;
        };
        let mut batch = ipc.poll();
        batch.sort_by_key(|c| !matches!(c, RendererCommand::frame_submit_data(_)));
        for cmd in batch {
            self.handle_command(cmd);
        }
    }

    fn handle_command(&mut self, cmd: RendererCommand) {
        match self.init_state {
            InitState::Uninitialized => match cmd {
                RendererCommand::keep_alive(_) => {}
                RendererCommand::renderer_init_data(d) => self.on_init_data(d),
                _ => {
                    logger::error!("IPC: expected RendererInitData first");
                    self.fatal_error = true;
                }
            },
            InitState::InitReceived => match cmd {
                RendererCommand::keep_alive(_) => {}
                RendererCommand::renderer_init_finalize_data(_) => {
                    self.init_state = InitState::Finalized;
                }
                RendererCommand::renderer_init_progress_update(_) => {}
                RendererCommand::renderer_engine_ready(_) => {}
                _ => {
                    logger::trace!("IPC: deferring command until init finalized (skeleton)");
                }
            },
            InitState::Finalized => self.handle_running_command(cmd),
        }
    }

    fn on_init_data(&mut self, d: RendererInitData) {
        if let Some(ref prefix) = d.shared_memory_prefix {
            self.shared_memory = Some(SharedMemoryAccessor::new(prefix.clone()));
            logger::info!("Shared memory prefix: {}", prefix);
        }
        self.pending_init = Some(d.clone());
        if let Some(ref mut ipc) = self.ipc {
            send_renderer_init_result(ipc, d.output_device);
        }
        self.init_state = InitState::InitReceived;
        self.last_frame_data_processed = true;
    }

    fn handle_running_command(&mut self, cmd: RendererCommand) {
        match cmd {
            RendererCommand::keep_alive(_) => {}
            RendererCommand::renderer_shutdown(_)
            | RendererCommand::renderer_shutdown_request(_) => {
                self.shutdown_requested = true;
            }
            RendererCommand::frame_submit_data(data) => self.on_frame_submit(data),
            RendererCommand::mesh_upload_data(d) => self.try_process_mesh_upload(d),
            RendererCommand::mesh_unload(u) => self.on_mesh_unload(u),
            RendererCommand::set_texture_2d_format(f) => self.on_set_texture_2d_format(f),
            RendererCommand::set_texture_2d_properties(p) => self.on_set_texture_2d_properties(p),
            RendererCommand::set_texture_2d_data(d) => self.on_set_texture_2d_data(d),
            RendererCommand::unload_texture_2d(u) => self.on_unload_texture_2d(u),
            RendererCommand::free_shared_memory_view(f) => {
                if let Some(shm) = self.shared_memory.as_mut() {
                    shm.release_view(f.buffer_id);
                }
            }
            _ => {
                logger::trace!("runtime: unhandled RendererCommand (expand handlers here)");
            }
        }
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

    fn send_texture_2d_result(&mut self, asset_id: i32, update: i32, instance_changed: bool) {
        let Some(ref mut ipc) = self.ipc else {
            return;
        };
        ipc.send_background(RendererCommand::set_texture_2d_result(SetTexture2DResult {
            asset_id,
            r#type: TextureUpdateResultType(update),
            instance_changed,
        }));
    }

    fn on_set_texture_2d_format(&mut self, f: SetTexture2DFormat) {
        let id = f.asset_id;
        self.texture_formats.insert(id, f.clone());
        let props = self.texture_properties.get(&id);
        let Some(device) = self.gpu_device.clone() else {
            self.send_texture_2d_result(
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
        self.send_texture_2d_result(id, TextureUpdateResultType::FORMAT_SET, !existed_before);
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

    fn on_set_texture_2d_properties(&mut self, p: SetTexture2DProperties) {
        let id = p.asset_id;
        self.texture_properties.insert(id, p.clone());
        if let Some(t) = self.texture_pool.get_texture_mut(id) {
            t.apply_properties(&p);
        }
        self.send_texture_2d_result(id, TextureUpdateResultType::PROPERTIES_SET, false);
    }

    fn on_set_texture_2d_data(&mut self, d: SetTexture2DData) {
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
        self.try_texture_upload_with_device(d);
    }

    fn try_texture_upload_with_device(&mut self, data: SetTexture2DData) {
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
        let Some(shm) = self.shared_memory.as_mut() else {
            logger::warn!("texture {id}: no shared memory accessor");
            return;
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
                self.send_texture_2d_result(id, TextureUpdateResultType::DATA_UPLOAD, false);
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

    fn on_unload_texture_2d(&mut self, u: UnloadTexture2D) {
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

    fn try_process_mesh_upload(&mut self, data: MeshUploadData) {
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
        self.try_mesh_upload_with_device(&device, data);
    }

    fn try_mesh_upload_with_device(&mut self, device: &Arc<wgpu::Device>, data: MeshUploadData) {
        let Some(shm) = self.shared_memory.as_mut() else {
            logger::warn!(
                "mesh {}: no shared memory accessor (standalone or missing prefix)",
                data.asset_id
            );
            return;
        };
        let upload_result = shm.with_read_bytes(&data.buffer, |raw| {
            try_upload_mesh_from_raw(device.as_ref(), raw, &data)
        });
        let Some(mesh) = upload_result else {
            logger::warn!("mesh {}: upload failed or rejected", data.asset_id);
            return;
        };
        let existed_before = self.mesh_pool.insert_mesh(mesh);
        if let Some(ref mut ipc) = self.ipc {
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

    fn on_mesh_unload(&mut self, u: MeshUnload) {
        if self.mesh_pool.remove_mesh(u.asset_id) {
            logger::info!(
                "mesh {} unloaded (resident_bytes≈{})",
                u.asset_id,
                self.mesh_pool.accounting().total_resident_bytes()
            );
        }
    }

    fn on_frame_submit(&mut self, data: FrameSubmitData) {
        self.last_frame_index = data.frame_index;
        self.last_frame_data_processed = true;
        let start = Instant::now();
        self.run_asset_integration_stub(Duration::from_millis(2));
        logger::trace!(
            "frame_submit frame_index={} stub_integration_ms={:.3}",
            data.frame_index,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn send_renderer_init_result(ipc: &mut DualQueueIpc, output_device: HeadOutputDevice) {
    let result = RendererInitResult {
        actual_output_device: output_device,
        renderer_identifier: Some("Renderide 0.1.0 (wgpu skeleton)".to_string()),
        main_window_handle_ptr: 0,
        stereo_rendering_mode: Some("None".to_string()),
        max_texture_size: 8192,
        is_gpu_texture_pot_byte_aligned: true,
        supported_texture_formats: supported_host_formats_for_init(),
    };
    ipc.send_primary(RendererCommand::renderer_init_result(result));
}
