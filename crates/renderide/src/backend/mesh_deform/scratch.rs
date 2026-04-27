//! Reusable GPU buffers for per-frame mesh deformation (bone palette, blendshape uniforms).

/// CPU-reserved caps; buffers grow when exceeded.
const INITIAL_MAX_BONES: u32 = 256;
const INITIAL_MAX_BLENDSHAPES: u32 = 256;
/// Initial staging for packed blendshape `Params` (32 bytes × chunks).
const INITIAL_BLENDSHAPE_PARAMS_STAGING: u64 = 4096;
/// Initial number of 256-byte slots for per-dispatch `SkinDispatchParams` (32 B payload each).
const INITIAL_SKIN_DISPATCH_SLOTS: u64 = 16;

/// Pads to the per-draw slab stride (matches [`crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE`]).
///
/// The device's `min_storage_buffer_offset_alignment` is verified to be `<= 256` in
/// [`crate::gpu::GpuLimits::try_new`], so this constant satisfies dynamic-offset alignment for
/// every supported adapter. Use 256 here (not the device alignment) because the slab payload
/// stride is a fixed CPU/GPU contract, not a per-device value.
#[inline]
fn align256(n: u64) -> u64 {
    (n + 255) & !255
}

/// Scratch storage written each frame before compute dispatches.
pub struct MeshDeformScratch {
    /// Linear blend skinning bone palette (`mat4` column-major, 64 bytes each); subranges use 256-byte-aligned offsets.
    pub bone_matrices: wgpu::Buffer,
    /// 32-byte uniform for sparse blendshape scatter (`source/compute/mesh_blendshape.wgsl` `Params`).
    pub blendshape_params: wgpu::Buffer,
    /// Upload + copy source for packed scatter `Params` before `copy_buffer_to_buffer` into `blendshape_params`.
    pub blendshape_params_staging: wgpu::Buffer,
    /// `f32` weight per blendshape; subranges use 256-byte-aligned offsets between meshes.
    pub blendshape_weights: wgpu::Buffer,
    /// Slab of `mesh_skinning.wgsl` [`SkinDispatchParams`] (32 bytes per dispatch at 256-byte-aligned offsets).
    pub skin_dispatch: wgpu::Buffer,
    /// Reusable byte buffer for one mesh's blendshape weight binding before [`crate::render_graph::frame_upload_batch::FrameUploadBatch::write_buffer`].
    ///
    /// Cleared (length-only, capacity retained) at the start of each blendshape record call.
    pub blend_weight_bytes: Vec<u8>,
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
        let bone_bytes = (INITIAL_MAX_BONES as u64) * 64;
        let weight_bytes = (INITIAL_MAX_BLENDSHAPES as u64) * 4;
        let skin_dispatch_bytes = INITIAL_SKIN_DISPATCH_SLOTS.saturating_mul(256);
        Self {
            bone_matrices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_deform_bone_palette"),
                size: bone_bytes.max(64),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blendshape_params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_deform_blendshape_params"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blendshape_params_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_deform_blendshape_params_staging"),
                size: INITIAL_BLENDSHAPE_PARAMS_STAGING,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blendshape_weights: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_deform_blendshape_weights"),
                size: weight_bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            skin_dispatch: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_deform_skin_dispatch"),
                size: skin_dispatch_bytes.max(256),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blend_weight_bytes: Vec::new(),
            packed_scatter_params: Vec::new(),
            scatter_dispatch_wgs: Vec::new(),
            max_bones: INITIAL_MAX_BONES,
            max_shapes: INITIAL_MAX_BLENDSHAPES,
            max_buffer_size,
        }
    }

    fn grow_allowed(&self, next_size: u64, label: &'static str) -> bool {
        if next_size <= self.max_buffer_size {
            return true;
        }
        logger::warn!(
            "mesh deform scratch: {label} would need {next_size} bytes (max_buffer_size={})",
            self.max_buffer_size
        );
        false
    }

    /// Ensures the bone palette buffer fits at least `need_bones` matrices for a single-mesh dispatch.
    pub fn ensure_bone_capacity(&mut self, device: &wgpu::Device, need_bones: u32) {
        if need_bones <= self.max_bones {
            return;
        }
        let next = need_bones.next_power_of_two().max(INITIAL_MAX_BONES);
        let bone_bytes = (next as u64) * 64;
        if !self.grow_allowed(bone_bytes, "bone_matrices") {
            return;
        }
        self.bone_matrices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_bone_palette"),
            size: bone_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.max_bones = next;
    }

    /// Ensures the bone buffer is large enough for byte range `[0, end_exclusive)`.
    pub fn ensure_bone_byte_capacity(&mut self, device: &wgpu::Device, end_exclusive: u64) {
        if end_exclusive <= self.bone_matrices.size() {
            return;
        }
        let next_size = end_exclusive.next_power_of_two().max(64);
        if !self.grow_allowed(next_size, "bone_matrices") {
            return;
        }
        self.bone_matrices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_bone_palette"),
            size: next_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    /// Ensures the blendshape weight buffer fits at least `need_shapes` floats for a single-mesh dispatch.
    pub fn ensure_shape_weight_capacity(&mut self, device: &wgpu::Device, need_shapes: u32) {
        if need_shapes <= self.max_shapes {
            return;
        }
        let next = need_shapes.next_power_of_two().max(INITIAL_MAX_BLENDSHAPES);
        let weight_bytes = (next as u64) * 4;
        if !self.grow_allowed(weight_bytes, "blendshape_weights") {
            return;
        }
        self.blendshape_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_blendshape_weights"),
            size: weight_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.max_shapes = next;
    }

    /// Ensures the weight slab can address bytes `[0, end_exclusive)`.
    pub fn ensure_blend_weight_byte_capacity(&mut self, device: &wgpu::Device, end_exclusive: u64) {
        if end_exclusive <= self.blendshape_weights.size() {
            return;
        }
        let next_size = end_exclusive.next_power_of_two().max(16);
        if !self.grow_allowed(next_size, "blendshape_weights") {
            return;
        }
        self.blendshape_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_blendshape_weights"),
            size: next_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    /// Ensures staging holds at least `need_bytes` for packed blendshape chunk params.
    pub fn ensure_blendshape_params_staging(&mut self, device: &wgpu::Device, need_bytes: u64) {
        if need_bytes <= self.blendshape_params_staging.size() {
            return;
        }
        let next = need_bytes
            .next_power_of_two()
            .max(INITIAL_BLENDSHAPE_PARAMS_STAGING);
        if !self.grow_allowed(next, "blendshape_params_staging") {
            return;
        }
        self.blendshape_params_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_blendshape_params_staging"),
            size: next,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        if end_exclusive <= self.skin_dispatch.size() {
            return;
        }
        let next_size = end_exclusive.next_power_of_two().max(256);
        if !self.grow_allowed(next_size, "skin_dispatch") {
            return;
        }
        self.skin_dispatch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_skin_dispatch"),
            size: next_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
