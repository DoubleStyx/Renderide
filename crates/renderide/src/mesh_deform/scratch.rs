//! Reusable GPU buffers for per-frame mesh deformation (bone palette, blendshape uniforms).

/// CPU-reserved caps; buffers grow when exceeded.
const INITIAL_MAX_BONES: u32 = 256;
const INITIAL_MAX_BLENDSHAPES: u32 = 256;
/// Initial staging for packed blendshape `Params` (32 bytes x chunks).
const INITIAL_BLENDSHAPE_PARAMS_STAGING: u64 = 4096;
/// Initial number of 256-byte slots for per-dispatch `SkinDispatchParams` (32 B payload each).
const INITIAL_SKIN_DISPATCH_SLOTS: u64 = 16;

/// Bytes per skinning palette matrix (column-major `mat4`).
const BONE_MATRIX_BYTES: u64 = 64;
/// Bytes per blendshape weight (`f32`).
const BLENDSHAPE_WEIGHT_BYTES: u64 = 4;

/// Pads to the per-draw slab stride (matches [`crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE`]).
///
/// The device's `min_storage_buffer_offset_alignment` is verified to be `<= 256` in
/// [`crate::gpu::GpuLimits::try_new`], so this constant satisfies dynamic-offset alignment for
/// every supported adapter. Use 256 here (not the device alignment) because the slab payload
/// stride is a fixed CPU/GPU contract, not a per-device value.
#[inline]
fn align256(n: u64) -> u64 {
    (n + 255) & !255
}

/// Static description of a growable scratch buffer: label, usage, and minimum size floor.
///
/// Centralises the per-buffer recipe so [`MeshDeformScratch::new`] and the [`Self::ensure`]
/// growth path share one buffer descriptor and one log message format.
struct GrowableBuffer {
    label: &'static str,
    usage: wgpu::BufferUsages,
    /// Floor below which the buffer is never sized. Matches the `.max(N)` literals from the
    /// per-call buffer descriptors.
    min_size: u64,
}

impl GrowableBuffer {
    fn create(&self, device: &wgpu::Device, requested: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: requested.max(self.min_size),
            usage: self.usage,
            mapped_at_creation: false,
        })
    }

    /// Ensures `buf` is at least `need_bytes` long, growing to the next power of two.
    ///
    /// Returns `false` (and logs a warning) when growth would exceed `max_buffer_size`. The buffer
    /// is left unchanged in that case so the caller can fall back to a smaller dispatch.
    fn ensure(
        &self,
        device: &wgpu::Device,
        buf: &mut wgpu::Buffer,
        need_bytes: u64,
        max_buffer_size: u64,
    ) -> bool {
        if need_bytes <= buf.size() {
            return true;
        }
        let next = need_bytes.next_power_of_two().max(self.min_size);
        if next > max_buffer_size {
            logger::warn!(
                "mesh deform scratch: {} would need {} bytes (max_buffer_size={})",
                self.label,
                next,
                max_buffer_size
            );
            return false;
        }
        *buf = self.create(device, next);
        true
    }
}

const BONE_MATRICES: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_bone_palette",
    usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST),
    min_size: BONE_MATRIX_BYTES,
};

const BLENDSHAPE_PARAMS: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_blendshape_params",
    usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
    min_size: 32,
};

const BLENDSHAPE_PARAMS_STAGING: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_blendshape_params_staging",
    usage: wgpu::BufferUsages::COPY_SRC.union(wgpu::BufferUsages::COPY_DST),
    min_size: INITIAL_BLENDSHAPE_PARAMS_STAGING,
};

const BLENDSHAPE_WEIGHTS: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_blendshape_weights",
    usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST),
    min_size: 16,
};

const SKIN_DISPATCH: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_skin_dispatch",
    usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
    min_size: 256,
};

const DUMMY_VEC4_READ: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_dummy_vec4_read",
    usage: wgpu::BufferUsages::STORAGE,
    min_size: 16,
};

const DUMMY_VEC4_WRITE: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_dummy_vec4_write",
    usage: wgpu::BufferUsages::STORAGE,
    min_size: 16,
};

const DUMMY_VEC4_WRITE_ALT: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_dummy_vec4_write_alt",
    usage: wgpu::BufferUsages::STORAGE,
    min_size: 16,
};

/// Scratch storage written each frame before compute dispatches.
pub struct MeshDeformScratch {
    /// Linear blend skinning bone palette (`mat4` column-major, 64 bytes each); subranges use 256-byte-aligned offsets.
    pub bone_matrices: wgpu::Buffer,
    /// 32-byte uniform for sparse blendshape scatter (`shaders/passes/compute/mesh_blendshape.wgsl` `Params`).
    pub blendshape_params: wgpu::Buffer,
    /// Upload + copy source slab for packed scatter `Params` before `copy_buffer_to_buffer` into `blendshape_params`.
    pub blendshape_params_staging: wgpu::Buffer,
    /// `f32` weight per blendshape; subranges use 256-byte-aligned offsets between meshes.
    pub blendshape_weights: wgpu::Buffer,
    /// Slab of `mesh_skinning.wgsl` [`SkinDispatchParams`] (32 bytes per dispatch at 256-byte-aligned offsets).
    pub skin_dispatch: wgpu::Buffer,
    /// Dummy read-only storage used for optional shader input bindings when an attribute path is disabled.
    pub dummy_vec4_read: wgpu::Buffer,
    /// Dummy writable storage used for optional shader output bindings when an attribute path is disabled.
    pub dummy_vec4_write: wgpu::Buffer,
    /// Alternate dummy writable storage for a second optional output binding in the same dispatch.
    pub dummy_vec4_write_alt: wgpu::Buffer,
    /// Reusable byte buffer for one mesh's blendshape weight binding before [`crate::render_graph::frame_upload_batch::FrameUploadBatch::write_buffer`].
    ///
    /// Cleared (length-only, capacity retained) at the start of each blendshape record call.
    pub blend_weight_bytes: Vec<u8>,
    /// Reusable byte buffer for one skinning palette before it is copied into the frame upload batch.
    ///
    /// Cleared (length-only, capacity retained) at the start of each skinning record call.
    pub bone_palette_bytes: Vec<u8>,
    /// Reusable byte buffer for packed scatter `Params` per mesh; one entry per dispatch chunk.
    ///
    /// Cleared (length-only, capacity retained) at the start of each blendshape record call.
    pub packed_scatter_params: Vec<u8>,
    /// Reusable workgroup count per scatter dispatch chunk; parallels [`Self::packed_scatter_params`].
    ///
    /// Cleared (length-only, capacity retained) at the start of each blendshape record call.
    pub scatter_dispatch_wgs: Vec<u32>,
    max_bones: u32,
    max_shapes: u32,
    /// [`wgpu::Limits::max_buffer_size`]; growth refuses past this cap.
    max_buffer_size: u64,
}

impl MeshDeformScratch {
    /// Allocates initial scratch buffers on `device`.
    ///
    /// `max_buffer_size` must be [`wgpu::Device::limits`].`max_buffer_size` (see [`crate::gpu::GpuLimits::max_buffer_size`]).
    pub fn new(device: &wgpu::Device, max_buffer_size: u64) -> Self {
        let bone_bytes = u64::from(INITIAL_MAX_BONES) * BONE_MATRIX_BYTES;
        let weight_bytes = u64::from(INITIAL_MAX_BLENDSHAPES) * BLENDSHAPE_WEIGHT_BYTES;
        let skin_dispatch_bytes = INITIAL_SKIN_DISPATCH_SLOTS.saturating_mul(256);
        Self {
            bone_matrices: BONE_MATRICES.create(device, bone_bytes),
            blendshape_params: BLENDSHAPE_PARAMS.create(device, BLENDSHAPE_PARAMS.min_size),
            blendshape_params_staging: BLENDSHAPE_PARAMS_STAGING
                .create(device, BLENDSHAPE_PARAMS_STAGING.min_size),
            blendshape_weights: BLENDSHAPE_WEIGHTS.create(device, weight_bytes),
            skin_dispatch: SKIN_DISPATCH.create(device, skin_dispatch_bytes),
            dummy_vec4_read: DUMMY_VEC4_READ.create(device, DUMMY_VEC4_READ.min_size),
            dummy_vec4_write: DUMMY_VEC4_WRITE.create(device, DUMMY_VEC4_WRITE.min_size),
            dummy_vec4_write_alt: DUMMY_VEC4_WRITE_ALT
                .create(device, DUMMY_VEC4_WRITE_ALT.min_size),
            blend_weight_bytes: Vec::new(),
            bone_palette_bytes: Vec::new(),
            packed_scatter_params: Vec::new(),
            scatter_dispatch_wgs: Vec::new(),
            max_bones: INITIAL_MAX_BONES,
            max_shapes: INITIAL_MAX_BLENDSHAPES,
            max_buffer_size,
        }
    }

    /// Ensures the bone palette buffer fits at least `need_bones` matrices for a single-mesh dispatch.
    pub fn ensure_bone_capacity(&mut self, device: &wgpu::Device, need_bones: u32) {
        if need_bones <= self.max_bones {
            return;
        }
        let next = need_bones.next_power_of_two().max(INITIAL_MAX_BONES);
        let bone_bytes = u64::from(next) * BONE_MATRIX_BYTES;
        if BONE_MATRICES.ensure(
            device,
            &mut self.bone_matrices,
            bone_bytes,
            self.max_buffer_size,
        ) {
            self.max_bones = next;
        }
    }

    /// Ensures the bone buffer is large enough for byte range `[0, end_exclusive)`.
    pub fn ensure_bone_byte_capacity(&mut self, device: &wgpu::Device, end_exclusive: u64) {
        BONE_MATRICES.ensure(
            device,
            &mut self.bone_matrices,
            end_exclusive,
            self.max_buffer_size,
        );
    }

    /// Ensures the blendshape weight buffer fits at least `need_shapes` floats for a single-mesh dispatch.
    pub fn ensure_shape_weight_capacity(&mut self, device: &wgpu::Device, need_shapes: u32) {
        if need_shapes <= self.max_shapes {
            return;
        }
        let next = need_shapes.next_power_of_two().max(INITIAL_MAX_BLENDSHAPES);
        let weight_bytes = u64::from(next) * BLENDSHAPE_WEIGHT_BYTES;
        if BLENDSHAPE_WEIGHTS.ensure(
            device,
            &mut self.blendshape_weights,
            weight_bytes,
            self.max_buffer_size,
        ) {
            self.max_shapes = next;
        }
    }

    /// Ensures the weight slab can address bytes `[0, end_exclusive)`.
    pub fn ensure_blend_weight_byte_capacity(&mut self, device: &wgpu::Device, end_exclusive: u64) {
        BLENDSHAPE_WEIGHTS.ensure(
            device,
            &mut self.blendshape_weights,
            end_exclusive,
            self.max_buffer_size,
        );
    }

    /// Ensures staging holds at least `need_bytes` for packed blendshape chunk params.
    pub fn ensure_blendshape_params_staging(&mut self, device: &wgpu::Device, need_bytes: u64) {
        BLENDSHAPE_PARAMS_STAGING.ensure(
            device,
            &mut self.blendshape_params_staging,
            need_bytes,
            self.max_buffer_size,
        );
    }

    /// Ensures the skin-dispatch uniform slab can address byte range `[0, end_exclusive)`.
    ///
    /// Each skinning dispatch writes 32 bytes at a 256-byte-aligned cursor; callers advance with
    /// [`advance_slab_cursor`].
    pub fn ensure_skin_dispatch_byte_capacity(
        &mut self,
        device: &wgpu::Device,
        end_exclusive: u64,
    ) {
        SKIN_DISPATCH.ensure(
            device,
            &mut self.skin_dispatch,
            end_exclusive,
            self.max_buffer_size,
        );
    }
}

/// Returns the next slab cursor after placing `byte_len` bytes at `cursor`, padding to 256-byte
/// boundaries so subsequent storage/uniform bindings meet typical WebGPU offset alignment.
pub fn advance_slab_cursor(cursor: u64, byte_len: u64) -> u64 {
    if byte_len == 0 {
        return cursor;
    }
    cursor + align256(byte_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skin_dispatch_cursor_advances_by_256_per_32_byte_payload() {
        assert_eq!(advance_slab_cursor(0, 32), 256);
        assert_eq!(advance_slab_cursor(256, 32), 512);
    }
}
