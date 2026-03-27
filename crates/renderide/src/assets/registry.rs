//! Asset registry: stores mesh, texture, shader, and other assets by handle.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::shared::{MeshUploadData, SetTexture2DData, SetTexture2DFormat, ShaderUpload};

use super::manager::AssetManager;
use super::material_properties::MaterialPropertyStore;
use super::mesh::{self, MeshAsset, compute_index_count, index_bytes_per_element};
use super::shader::ShaderAsset;
use super::shader_logical_name::resolve_logical_shader_name_from_upload;
use super::texture::{TextureAsset, decode_texture_mip0_to_rgba8};

/// Stores assets by handle using generic per-type managers.
/// Extensible for textures, materials, video, etc.
pub struct AssetRegistry {
    meshes: AssetManager<MeshAsset>,
    /// Host `Texture2D` assets after `SetTexture2DFormat` / `SetTexture2DData`.
    textures: AssetManager<TextureAsset>,
    shaders: AssetManager<ShaderAsset>,
    /// Material property values per block (from MaterialsUpdateBatch).
    pub material_property_store: MaterialPropertyStore,
    /// Next value assigned to [`TextureAsset::data_version`] on texture format/data updates.
    texture_data_version_seq: u64,
    upload_count: u64,
    unload_count: u64,
}

impl AssetRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            meshes: AssetManager::new(),
            textures: AssetManager::new(),
            shaders: AssetManager::new(),
            material_property_store: MaterialPropertyStore::new(),
            texture_data_version_seq: 0,
            upload_count: 0,
            unload_count: 0,
        }
    }

    /// Returns a mesh by handle.
    pub fn get_mesh(&self, handle: i32) -> Option<&MeshAsset> {
        self.meshes.get(handle)
    }

    /// Returns a shader by handle.
    pub fn get_shader(&self, handle: i32) -> Option<&ShaderAsset> {
        self.shaders.get(handle)
    }

    /// Returns a texture by handle. Reserved for future texture support.
    pub fn get_texture(&self, handle: i32) -> Option<&TextureAsset> {
        self.textures.get(handle)
    }

    /// Number of mesh assets in the registry.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Number of host `Texture2D` assets (after `SetTexture2DFormat`; may still lack mip0 pixels).
    pub fn texture_2d_count(&self) -> usize {
        self.textures.len()
    }

    /// Textures with decoded mip0 ready for [`crate::gpu::GpuState::ensure_texture2d_gpu`].
    pub fn texture_2d_ready_for_gpu_count(&self) -> usize {
        self.textures.values().filter(|t| t.ready_for_gpu()).count()
    }

    /// Handles a mesh upload from shared memory.
    ///
    /// Layout must match host's MeshBuffer.ComputeBufferLayout (vertices, indices, bone_counts,
    /// bone_weights, bind_poses, blendshape_data). When `upload_hint.blendshapes()` and
    /// `blendshape_buffers` is non-empty, extracts blendshape offsets into `MeshAsset.blendshape_offsets`.
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
        let index_count = compute_index_count(&data.submeshes);
        let index_bytes = index_bytes_per_element(data.index_buffer_format);
        let use_blendshapes =
            data.upload_hint.flags.blendshapes() && !data.blendshape_buffers.is_empty();
        let layout = match mesh::compute_mesh_buffer_layout(
            vertex_stride,
            data.vertex_count,
            index_count,
            index_bytes,
            data.bone_count,
            data.bone_weight_count,
            Some(&data.blendshape_buffers),
        ) {
            Ok(l) => l,
            Err(e) => {
                logger::error!("Mesh upload rejected: {}", e);
                return (false, false);
            }
        };

        let raw = match shm.access_copy::<u8>(&data.buffer) {
            Some(r) => r,
            None => return (false, false),
        };
        let expected_bone_weights_len = (data.bone_weight_count.max(0) * 8) as usize;
        let expected_bind_poses_len = (data.bone_count.max(0) * 64) as usize;
        if layout.bone_weights_length != expected_bone_weights_len {
            logger::error!(
                "Mesh upload rejected: bone_weights layout mismatch (expected {} got {})",
                expected_bone_weights_len,
                layout.bone_weights_length
            );
            return (false, false);
        }
        if layout.bind_poses_length != expected_bind_poses_len {
            logger::error!(
                "Mesh upload rejected: bind_poses layout mismatch (expected {} got {})",
                expected_bind_poses_len,
                layout.bind_poses_length
            );
            return (false, false);
        }
        let min_len = layout.total_buffer_length;
        if raw.len() < min_len {
            logger::error!(
                "Mesh upload rejected: buffer too short (need {} bytes, got {})",
                min_len,
                raw.len()
            );
            return (false, false);
        }

        let vertex_data = raw[..layout.vertex_size].to_vec();
        let index_data = raw
            [layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length]
            .to_vec();

        let (bind_poses, bone_counts, bone_weights) = if data.bone_count > 0 {
            let bind_poses = mesh::extract_bind_poses(
                &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length],
                data.bone_count as usize,
            );
            let bone_counts = raw
                [layout.bone_counts_start..layout.bone_counts_start + layout.bone_counts_length]
                .to_vec();
            let bone_weights = raw
                [layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length]
                .to_vec();
            (bind_poses, Some(bone_counts), Some(bone_weights))
        } else if use_blendshapes && data.vertex_count > 0 {
            let (bp, bc, bw) = mesh::synthetic_bone_data_for_blendshape_only(data.vertex_count);
            (Some(bp), Some(bc), Some(bw))
        } else {
            (None, None, None)
        };

        let (blendshape_offsets, num_blendshapes) = if use_blendshapes {
            mesh::extract_blendshape_offsets(
                raw.as_slice(),
                &layout,
                &data.blendshape_buffers,
                data.vertex_count,
            )
            .map(|(offsets, n)| (Some(offsets), n))
            .unwrap_or((None, 0))
        } else {
            (None, 0)
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
            blendshape_offsets,
            num_blendshapes,
        };
        self.meshes.insert(asset);
        self.upload_count += 1;
        (true, existed_before)
    }

    /// Applies `SetTexture2DFormat`: allocates or replaces metadata for `asset_id` (clears pixel data).
    ///
    /// Returns `(success, existed_before)`.
    pub fn set_texture_2d_format(&mut self, fmt: SetTexture2DFormat) -> (bool, bool) {
        let id = fmt.asset_id;
        if fmt.width <= 0 || fmt.height <= 0 {
            logger::warn!(
                "Texture2D format rejected: non-positive size (asset_id={} {}x{})",
                id,
                fmt.width,
                fmt.height
            );
            return (false, false);
        }
        let existed_before = self.textures.contains_key(id);
        self.texture_data_version_seq = self.texture_data_version_seq.saturating_add(1);
        let asset = TextureAsset {
            id,
            width: fmt.width as u32,
            height: fmt.height as u32,
            format: fmt.format,
            rgba8_mip0: Vec::new(),
            data_version: self.texture_data_version_seq,
        };
        self.textures.insert(asset);
        self.upload_count += 1;
        (true, existed_before)
    }

    /// Applies `SetTexture2DData`: decodes mip0 from shared memory into [`TextureAsset::rgba8_mip0`].
    ///
    /// Returns `(success, existed_before)` where `existed_before` refers to the texture row before insert.
    pub fn set_texture_2d_data(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        data: &SetTexture2DData,
    ) -> (bool, bool) {
        let id = data.asset_id;
        let Some(tex) = self.textures.get(id) else {
            logger::warn!("Texture2D data for asset_id={} without prior format", id);
            return (false, false);
        };
        if data.data.length <= 0 {
            return (false, false);
        }
        let raw_u8 = match shm.access_copy::<u8>(&data.data) {
            Some(r) => r,
            None => return (false, false),
        };
        let start = data.mip_starts.first().copied().unwrap_or(0).max(0) as usize;
        let sub = raw_u8.get(start..).unwrap_or(raw_u8.as_slice());
        let (mw, mh) = if let Some(s) = data.mip_map_sizes.first() {
            (s.x.max(1) as u32, s.y.max(1) as u32)
        } else {
            (tex.width.max(1), tex.height.max(1))
        };
        let fmt = tex.format;
        let was_empty = tex.rgba8_mip0.is_empty();
        let Some(rgba) = decode_texture_mip0_to_rgba8(fmt, mw, mh, data.flip_y, sub) else {
            logger::warn!(
                "Texture2D decode failed or unsupported format (asset_id={} format={:?})",
                id,
                fmt
            );
            return (false, false);
        };
        self.texture_data_version_seq = self.texture_data_version_seq.saturating_add(1);
        self.textures.insert(TextureAsset {
            id,
            width: mw,
            height: mh,
            format: fmt,
            rgba8_mip0: rgba,
            data_version: self.texture_data_version_seq,
        });
        self.upload_count += 1;
        (true, was_empty)
    }

    /// Removes a 2D texture. Called on `unload_texture_2d`.
    pub fn unload_texture_2d(&mut self, asset_id: i32) {
        self.textures.remove(asset_id);
        self.unload_count += 1;
    }

    /// Removes a shader asset. Called when the host sends `shader_unload`.
    pub fn handle_shader_unload(&mut self, asset_id: i32) {
        self.shaders.remove(asset_id);
        self.unload_count += 1;
    }

    /// Handles a shader upload. Stores id, optional `file` (path, logical stem label, or inline source), and resolved Unity shader name.
    /// Returns `(success, existed_before)`.
    pub fn handle_shader_upload(&mut self, data: ShaderUpload) -> (bool, bool) {
        let existed_before = self.shaders.contains_key(data.asset_id);
        let unity_shader_name = resolve_logical_shader_name_from_upload(&data);
        let asset = ShaderAsset {
            id: data.asset_id,
            wgsl_source: data.file,
            unity_shader_name,
        };
        self.shaders.insert(asset);
        self.upload_count += 1;
        (true, existed_before)
    }

    /// Handles a mesh unload.
    pub fn handle_mesh_unload(&mut self, asset_id: i32) {
        self.meshes.remove(asset_id);
        self.unload_count += 1;
    }

    /// Inserts a mesh for unit tests (bypasses IPC upload layout).
    #[cfg(test)]
    pub(crate) fn insert_mesh_for_tests(&mut self, mesh: MeshAsset) {
        let _ = self.meshes.insert(mesh);
    }
}

impl Default for AssetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod texture_version_tests {
    use super::AssetRegistry;
    use crate::shared::{ColorProfile, SetTexture2DFormat, TextureFormat};

    fn fmt_cmd(asset_id: i32, w: i32, h: i32) -> SetTexture2DFormat {
        SetTexture2DFormat {
            asset_id,
            width: w,
            height: h,
            mipmap_count: 1,
            format: TextureFormat::rgba32,
            profile: ColorProfile::default(),
        }
    }

    /// Each `SetTexture2DFormat` for the same id bumps [`crate::assets::TextureAsset::data_version`]
    /// so GPU uploads are not skipped after metadata-only changes.
    #[test]
    fn texture_data_version_increments_on_repeated_format() {
        let mut reg = AssetRegistry::new();
        reg.set_texture_2d_format(fmt_cmd(7, 64, 64));
        let v1 = reg.get_texture(7).expect("texture 7").data_version;
        reg.set_texture_2d_format(fmt_cmd(7, 64, 64));
        let v2 = reg.get_texture(7).expect("texture 7").data_version;
        assert!(v2 > v1, "expected data_version to bump on second format");
    }

    /// Sequential host operations assign monotonically increasing versions across assets.
    #[test]
    fn texture_data_version_monotonic_across_assets() {
        let mut reg = AssetRegistry::new();
        reg.set_texture_2d_format(fmt_cmd(1, 8, 8));
        let a = reg.get_texture(1).unwrap().data_version;
        reg.set_texture_2d_format(fmt_cmd(2, 8, 8));
        let b = reg.get_texture(2).unwrap().data_version;
        assert!(b > a);
    }
}
