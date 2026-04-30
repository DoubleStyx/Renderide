//! Resident [`GpuMesh`] table with optional layout fingerprint cache and VRAM accounting.

use hashbrown::HashMap;

use crate::assets::mesh::{GpuMesh, MeshBufferLayout};

use super::resource_pool::{GpuResourcePool, MeshPoolAccess};
use super::{GpuResource, StreamingPolicy, VramAccounting};

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
    /// Shared resident GPU resource table.
    inner: GpuResourcePool<GpuMesh, MeshPoolAccess>,
    /// Last successful [`MeshBufferLayout`] for [`mesh_upload_input_fingerprint`](crate::assets::mesh::mesh_upload_input_fingerprint) (skips `compute_mesh_buffer_layout` on hot uploads).
    layout_cache: HashMap<i32, (u64, MeshBufferLayout)>,
}

impl MeshPool {
    /// Creates an empty pool with the given streaming policy.
    pub fn new(streaming: Box<dyn StreamingPolicy>) -> Self {
        Self {
            inner: GpuResourcePool::new(MeshPoolAccess::new(streaming)),
            layout_cache: HashMap::new(),
        }
    }

    /// Default pool with [`NoopStreamingPolicy`].
    pub fn default_pool() -> Self {
        Self {
            inner: GpuResourcePool::new(MeshPoolAccess::noop()),
            layout_cache: HashMap::new(),
        }
    }

    /// VRAM accounting totals for resident meshes.
    pub fn accounting(&self) -> &VramAccounting {
        self.inner.accounting()
    }

    /// Mutable VRAM accounting (evictions and uploads update totals).
    pub fn accounting_mut(&mut self) -> &mut VramAccounting {
        self.inner.accounting_mut()
    }

    /// Streaming policy hook for mip / eviction suggestions.
    pub fn streaming_mut(&mut self) -> &mut dyn StreamingPolicy {
        self.inner.access_mut().streaming_mut()
    }

    /// Inserts or replaces a mesh; returns `existed_before` (true if an entry was replaced).
    pub fn insert_mesh(&mut self, mesh: GpuMesh) -> bool {
        self.inner.insert(mesh)
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
        self.inner.remove(asset_id)
    }

    /// Lazily creates tangent / UV1-3 buffers for meshes drawn by extended embedded shaders.
    pub fn ensure_extended_vertex_streams(&mut self, device: &wgpu::Device, asset_id: i32) -> bool {
        let (ok, before, after) = {
            let Some(mesh) = self.inner.get_mut(asset_id) else {
                return false;
            };
            let before = mesh.resident_bytes();
            let ok = mesh.ensure_extended_vertex_streams(device);
            let after = mesh.resident_bytes();
            (ok, before, after)
        };
        if ok {
            self.inner.account_resident_delta(before, after);
            self.inner.note_access(asset_id);
        }
        ok
    }

    /// Borrows a resident mesh by host asset id.
    #[inline]
    pub fn get_mesh(&self, asset_id: i32) -> Option<&GpuMesh> {
        self.inner.get(asset_id)
    }

    /// Borrows the map for iteration (read-only draw prep).
    #[inline]
    pub fn meshes(&self) -> &HashMap<i32, GpuMesh> {
        self.inner.resources()
    }
}

#[cfg(test)]
mod layout_cache_tests {
    //! [`MeshPool`] layout fingerprint cache tests (no GPU handles).

    use super::MeshPool;
    use crate::assets::mesh::MeshBufferLayout;

    fn layout_with_vertex_size(vertex_size: usize) -> MeshBufferLayout {
        MeshBufferLayout {
            vertex_size,
            index_buffer_start: 0,
            index_buffer_length: 0,
            bone_counts_start: 0,
            bone_counts_length: 0,
            bone_weights_start: 0,
            bone_weights_length: 0,
            bind_poses_start: 0,
            bind_poses_length: 0,
            blendshape_data_start: 0,
            blendshape_data_length: 0,
            total_buffer_length: vertex_size,
        }
    }

    #[test]
    fn get_cached_mesh_layout_returns_layout_on_fingerprint_hit() {
        let mut pool = MeshPool::default_pool();
        let id = 42;
        let fp = 0xdead_beef_u64;
        let layout = layout_with_vertex_size(128);
        pool.set_cached_mesh_layout(id, fp, layout);
        assert_eq!(pool.get_cached_mesh_layout(id, fp), Some(layout));
    }

    #[test]
    fn get_cached_mesh_layout_misses_when_fingerprint_changes() {
        let mut pool = MeshPool::default_pool();
        let id = 1;
        pool.set_cached_mesh_layout(id, 100, layout_with_vertex_size(64));
        assert_eq!(pool.get_cached_mesh_layout(id, 101), None);
    }

    #[test]
    fn get_cached_mesh_layout_misses_for_unknown_asset_id() {
        let pool = MeshPool::default_pool();
        assert_eq!(pool.get_cached_mesh_layout(999, 0), None);
    }
}
