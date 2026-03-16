//! Asset registry: stores mesh, texture, and other assets by handle.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::shared::MeshUploadData;

use super::manager::AssetManager;
use super::mesh::{self, MeshAsset};
use super::texture::TextureAsset;

/// Stores assets by handle using generic per-type managers.
/// Extensible for textures, materials, video, etc.
pub struct AssetRegistry {
    meshes: AssetManager<MeshAsset>,
    textures: AssetManager<TextureAsset>,
    upload_count: u64,
    unload_count: u64,
}

impl AssetRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            meshes: AssetManager::new(),
            textures: AssetManager::new(),
            upload_count: 0,
            unload_count: 0,
        }
    }

    /// Returns a mesh by handle.
    pub fn get_mesh(&self, handle: i32) -> Option<&MeshAsset> {
        self.meshes.get(handle)
    }

    /// Number of mesh assets in the registry.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Handles a mesh upload from shared memory.
    /// Returns `(success, existed_before)` where `existed_before` is true if the asset was replaced.
    pub fn handle_mesh_upload(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        data: MeshUploadData,
    ) -> (bool, bool) {
        if data.buffer.length <= 0 {
            return (false, false);
        }
        let vertex_stride = mesh::compute_vertex_stride(&data.vertex_attributes);
        let index_count = data
            .submeshes
            .iter()
            .map(|s| s.index_start + s.index_count)
            .max()
            .unwrap_or(0);
        let index_bytes = match data.index_buffer_format {
            crate::shared::IndexBufferFormat::u_int16 => 2,
            crate::shared::IndexBufferFormat::u_int32 => 4,
        };
        let layout = mesh::compute_mesh_buffer_layout(
            vertex_stride,
            data.vertex_count,
            index_count,
            index_bytes,
            data.bone_count,
            data.bone_weight_count,
        );

        let raw = match shm.access_copy::<u8>(&data.buffer) {
            Some(r) => r,
            None => return (false, false),
        };
        let min_len = layout.bind_poses_start + layout.bind_poses_length;
        if raw.len() < min_len {
            return (false, false);
        }

        let vertex_data = raw[..layout.vertex_size].to_vec();
        let index_data = raw[layout.index_buffer_start
            ..layout.index_buffer_start + layout.index_buffer_length]
            .to_vec();

        let (bind_poses, bone_counts, bone_weights) = if data.bone_count > 0 {
            let bind_poses = mesh::extract_bind_poses(
                &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length],
                data.bone_count as usize,
            );
            let bone_counts = raw[layout.bone_counts_start
                ..layout.bone_counts_start + layout.bone_counts_length]
                .to_vec();
            let bone_weights = raw[layout.bone_weights_start
                ..layout.bone_weights_start + layout.bone_weights_length]
                .to_vec();
            (bind_poses, Some(bone_counts), Some(bone_weights))
        } else {
            (None, None, None)
        };

        let existed_before = self.meshes.contains_key(data.asset_id);
        let asset = MeshAsset {
            id: data.asset_id,
            vertex_data,
            index_data,
            vertex_count: data.vertex_count,
            index_count,
            index_format: data.index_buffer_format,
            submeshes: data.submeshes,
            vertex_attributes: data.vertex_attributes,
            bounds: data.bounds,
            bind_poses,
            bone_counts,
            bone_weights,
        };
        self.meshes.insert(asset);
        self.upload_count += 1;
        (true, existed_before)
    }

    /// Handles a texture upload from shared memory.
    /// Stub: does nothing yet. Returns `(false, false)`.
    pub fn handle_texture_upload(&mut self, _shm: &mut SharedMemoryAccessor) -> (bool, bool) {
        (false, false)
    }

    /// Handles a mesh unload.
    pub fn handle_mesh_unload(&mut self, asset_id: i32) {
        self.meshes.remove(asset_id);
        self.unload_count += 1;
    }
}

impl Default for AssetRegistry {
    fn default() -> Self {
        Self::new()
    }
}
