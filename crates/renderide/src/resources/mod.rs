//! GPU resource pools and VRAM hooks (meshes and Texture2D).

mod budget;
mod render_texture_pool;
mod texture_pool;

pub use budget::{
    MeshResidencyMeta, NoopStreamingPolicy, ResidencyTier, StreamingPolicy, TextureResidencyMeta,
    VramAccounting, VramResourceKind,
};
pub use render_texture_pool::{GpuRenderTexture, RenderTexturePool};
pub use texture_pool::{GpuTexture2d, Texture2dSamplerState, TexturePool};

use std::collections::HashMap;

use crate::assets::mesh::{GpuMesh, MeshBufferLayout};

/// Common surface for resident GPU resources (extend for textures, buffers, etc.).
pub trait GpuResource {
    /// Approximate GPU memory for accounting.
    fn resident_bytes(&self) -> u64;
    /// Host asset id.
    fn asset_id(&self) -> i32;
}

impl GpuResource for GpuMesh {
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn asset_id(&self) -> i32 {
        self.asset_id
    }
}

/// Insert / remove pool for meshes; evictions call [`VramAccounting`] and optional [`StreamingPolicy`].
pub struct MeshPool {
    meshes: HashMap<i32, GpuMesh>,
    /// Last successful [`MeshBufferLayout`] for [`mesh_upload_input_fingerprint`](crate::assets::mesh::mesh_upload_input_fingerprint) (skips `compute_mesh_buffer_layout` on hot uploads).
    layout_cache: HashMap<i32, (u64, MeshBufferLayout)>,
    accounting: VramAccounting,
    streaming: Box<dyn StreamingPolicy>,
}

impl MeshPool {
    /// Creates an empty pool with the given streaming policy.
    pub fn new(streaming: Box<dyn StreamingPolicy>) -> Self {
        Self {
            meshes: HashMap::new(),
            layout_cache: HashMap::new(),
            accounting: VramAccounting::default(),
            streaming,
        }
    }

    /// Default pool with [`NoopStreamingPolicy`].
    pub fn default_pool() -> Self {
        Self::new(Box::new(NoopStreamingPolicy))
    }

    /// VRAM accounting totals for resident meshes.
    pub fn accounting(&self) -> &VramAccounting {
        &self.accounting
    }

    /// Mutable VRAM accounting (evictions and uploads update totals).
    pub fn accounting_mut(&mut self) -> &mut VramAccounting {
        &mut self.accounting
    }

    /// Streaming policy hook for mip / eviction suggestions.
    pub fn streaming_mut(&mut self) -> &mut dyn StreamingPolicy {
        self.streaming.as_mut()
    }

    /// Inserts or replaces a mesh; returns `existed_before` (true if an entry was replaced).
    pub fn insert_mesh(&mut self, mesh: GpuMesh) -> bool {
        let id = mesh.asset_id;
        let existed_before = self.meshes.contains_key(&id);
        let bytes = mesh.resident_bytes;
        if let Some(old) = self.meshes.insert(id, mesh) {
            self.accounting
                .on_resident_removed(VramResourceKind::Mesh, old.resident_bytes);
        }
        self.accounting
            .on_resident_added(VramResourceKind::Mesh, bytes);
        self.streaming.note_mesh_access(id);
        existed_before
    }

    /// Cached [`MeshBufferLayout`] when [`crate::assets::mesh::mesh_upload_input_fingerprint`] matches.
    pub fn get_cached_mesh_layout(&self, asset_id: i32, input_fp: u64) -> Option<MeshBufferLayout> {
        self.layout_cache
            .get(&asset_id)
            .filter(|(fp, _)| *fp == input_fp)
            .map(|(_, l)| *l)
    }

    /// Stores layout for [`crate::assets::mesh::mesh_upload_input_fingerprint`] after a successful compute.
    pub fn set_cached_mesh_layout(
        &mut self,
        asset_id: i32,
        input_fp: u64,
        layout: MeshBufferLayout,
    ) {
        self.layout_cache.insert(asset_id, (input_fp, layout));
    }

    /// Removes a mesh by host id; returns `true` if it was present.
    pub fn remove_mesh(&mut self, asset_id: i32) -> bool {
        self.layout_cache.remove(&asset_id);
        if let Some(old) = self.meshes.remove(&asset_id) {
            self.accounting
                .on_resident_removed(VramResourceKind::Mesh, old.resident_bytes);
            return true;
        }
        false
    }

    /// Borrows a resident mesh by host asset id.
    pub fn get_mesh(&self, asset_id: i32) -> Option<&GpuMesh> {
        self.meshes.get(&asset_id)
    }

    /// Borrows the map for iteration (read-only draw prep).
    pub fn meshes(&self) -> &HashMap<i32, GpuMesh> {
        &self.meshes
    }
}
