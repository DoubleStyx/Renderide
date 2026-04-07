//! GPU resource pools and VRAM hooks (meshes and Texture2D).

mod budget;
mod texture_pool;

pub use budget::{
    MeshResidencyMeta, NoopStreamingPolicy, ResidencyTier, StreamingPolicy, TextureResidencyMeta,
    VramAccounting, VramResourceKind,
};
pub use texture_pool::{GpuTexture2d, Texture2dSamplerState, TexturePool};

use std::collections::HashMap;

use crate::assets::mesh::GpuMesh;

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
    accounting: VramAccounting,
    streaming: Box<dyn StreamingPolicy>,
}

impl MeshPool {
    /// Creates an empty pool with the given streaming policy.
    pub fn new(streaming: Box<dyn StreamingPolicy>) -> Self {
        Self {
            meshes: HashMap::new(),
            accounting: VramAccounting::default(),
            streaming,
        }
    }

    /// Default pool with [`NoopStreamingPolicy`].
    pub fn default_pool() -> Self {
        Self::new(Box::new(NoopStreamingPolicy))
    }

    pub fn accounting(&self) -> &VramAccounting {
        &self.accounting
    }

    pub fn accounting_mut(&mut self) -> &mut VramAccounting {
        &mut self.accounting
    }

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

    /// Removes a mesh by host id; returns `true` if it was present.
    pub fn remove_mesh(&mut self, asset_id: i32) -> bool {
        if let Some(old) = self.meshes.remove(&asset_id) {
            self.accounting
                .on_resident_removed(VramResourceKind::Mesh, old.resident_bytes);
            return true;
        }
        false
    }

    pub fn get_mesh(&self, asset_id: i32) -> Option<&GpuMesh> {
        self.meshes.get(&asset_id)
    }

    /// Borrows the map for iteration (read-only draw prep).
    pub fn meshes(&self) -> &HashMap<i32, GpuMesh> {
        &self.meshes
    }
}
