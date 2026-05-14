//! Resident [`GpuMesh`] table with layout fingerprint cache and VRAM accounting.

use hashbrown::HashMap;

use crate::assets::mesh::{GpuMesh, MeshBufferLayout};
use crate::materials::EmbeddedTangentFallbackMode;

use crate::gpu_pools::resource_pool::{GpuResourcePool, StreamingAccess};
use crate::gpu_pools::{GpuResource, VramAccounting, impl_gpu_resource};

impl_gpu_resource!(GpuMesh);

/// Insert / remove pool for meshes; insert / remove update [`VramAccounting`] and notify the
/// wired [`StreamingPolicy`].
pub struct MeshPool {
    /// Shared resident GPU resource table.
    inner: GpuResourcePool<GpuMesh, StreamingAccess>,
    /// Last successful [`MeshBufferLayout`] for [`mesh_upload_input_fingerprint`](crate::assets::mesh::mesh_upload_input_fingerprint) (skips `compute_mesh_buffer_layout` on hot uploads).
    layout_cache: HashMap<i32, (u64, MeshBufferLayout)>,
    /// Monotonic generation bumped whenever resident mesh membership or contents change.
    mutation_generation: u64,
}

impl MeshPool {
    /// Default pool with [`crate::gpu_pools::NoopStreamingPolicy`].
    pub fn default_pool() -> Self {
        Self {
            inner: GpuResourcePool::new(StreamingAccess::mesh_noop()),
            layout_cache: HashMap::new(),
            mutation_generation: 0,
        }
    }

    /// VRAM accounting for resident meshes.
    #[inline]
    pub fn accounting(&self) -> &VramAccounting {
        self.inner.accounting()
    }

    /// Inserts or replaces a mesh; returns `true` if a previous entry was replaced.
    #[inline]
    pub fn insert(&mut self, mesh: GpuMesh) -> bool {
        self.mutation_generation = self.mutation_generation.wrapping_add(1);
        self.inner.insert(mesh)
    }

    /// Removes a mesh by host id; returns `true` if it was present. Also clears any cached
    /// layout for the asset.
    pub fn remove(&mut self, asset_id: i32) -> bool {
        self.layout_cache.remove(&asset_id);
        let removed = self.inner.remove(asset_id);
        if removed {
            self.mutation_generation = self.mutation_generation.wrapping_add(1);
        }
        removed
    }

    /// Removes and returns a mesh by host id when it was present. Also clears any cached layout for
    /// the asset.
    pub(crate) fn take(&mut self, asset_id: i32) -> Option<GpuMesh> {
        self.layout_cache.remove(&asset_id);
        let mesh = self.inner.take(asset_id)?;
        self.mutation_generation = self.mutation_generation.wrapping_add(1);
        Some(mesh)
    }

    /// Monotonic generation for resident mesh insert/remove/replace events.
    #[inline]
    pub fn mutation_generation(&self) -> u64 {
        self.mutation_generation
    }

    /// Borrows a resident mesh by host asset id.
    #[inline]
    pub fn get(&self, asset_id: i32) -> Option<&GpuMesh> {
        self.inner.get(asset_id)
    }

    /// Number of resident meshes.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the pool has no resident meshes.
    #[cfg(test)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
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

    /// Lazily creates tangent / UV1-3 buffers for meshes drawn by extended embedded shaders.
    pub fn ensure_extended_vertex_streams(
        &mut self,
        device: &wgpu::Device,
        asset_id: i32,
        tangent_fallback_mode: EmbeddedTangentFallbackMode,
    ) -> bool {
        self.ensure_stream(asset_id, |mesh| {
            mesh.ensure_extended_vertex_streams(device, tangent_fallback_mode)
        })
    }

    /// Lazily creates the UV1 buffer for meshes drawn by UV1-only embedded shaders.
    pub fn ensure_uv1_vertex_stream(&mut self, device: &wgpu::Device, asset_id: i32) -> bool {
        self.ensure_stream(asset_id, |mesh| mesh.ensure_uv1_vertex_stream(device))
    }

    /// Lazily creates the tangent buffer for meshes drawn by shaders declaring `@location(4)`.
    pub fn ensure_tangent_vertex_stream(
        &mut self,
        device: &wgpu::Device,
        asset_id: i32,
        tangent_fallback_mode: EmbeddedTangentFallbackMode,
    ) -> bool {
        self.ensure_stream(asset_id, |mesh| {
            mesh.ensure_tangent_vertex_stream(device, tangent_fallback_mode)
        })
    }

    /// Lazily creates the raw tangent payload buffer for UI shaders declaring `@location(4)`.
    pub fn ensure_raw_tangent_vertex_stream(
        &mut self,
        device: &wgpu::Device,
        asset_id: i32,
    ) -> bool {
        self.ensure_stream(asset_id, |mesh| {
            mesh.ensure_raw_tangent_vertex_stream(device)
        })
    }

    /// Lazily creates the UV2 buffer for meshes drawn by shaders declaring `@location(6)`.
    pub fn ensure_uv2_vertex_stream(&mut self, device: &wgpu::Device, asset_id: i32) -> bool {
        self.ensure_stream(asset_id, |mesh| mesh.ensure_uv2_vertex_stream(device))
    }

    /// Lazily creates the UV3 buffer for meshes drawn by shaders declaring `@location(7)`.
    pub fn ensure_uv3_vertex_stream(&mut self, device: &wgpu::Device, asset_id: i32) -> bool {
        self.ensure_stream(asset_id, |mesh| mesh.ensure_uv3_vertex_stream(device))
    }

    /// Lazily creates the packed UV0-UV7 buffer for shaders with wide UV inputs.
    pub fn ensure_wide_uv_vertex_stream(&mut self, device: &wgpu::Device, asset_id: i32) -> bool {
        self.ensure_stream(asset_id, |mesh| mesh.ensure_wide_uv_vertex_stream(device))
    }

    /// Runs `op` against the resident mesh for `asset_id` (if any), then
    /// reconciles VRAM accounting and notifies the streaming policy when the
    /// operation succeeds. Does not touch `mutation_generation`: vertex-stream
    /// additions are content-internal, not membership changes.
    fn ensure_stream<F>(&mut self, asset_id: i32, op: F) -> bool
    where
        F: FnOnce(&mut GpuMesh) -> bool,
    {
        let (ok, before, after) = {
            let Some(mesh) = self.inner.get_mut(asset_id) else {
                return false;
            };
            let before = mesh.resident_bytes();
            let ok = op(mesh);
            let after = mesh.resident_bytes();
            (ok, before, after)
        };
        if ok {
            self.inner.account_resident_delta(before, after);
            self.inner.note_access(asset_id);
        }
        ok
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
