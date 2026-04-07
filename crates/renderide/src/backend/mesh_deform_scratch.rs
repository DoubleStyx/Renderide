//! Reusable GPU buffers for per-frame mesh deformation (bone palette, blendshape uniforms).

/// CPU-reserved caps; buffers grow when exceeded.
const INITIAL_MAX_BONES: u32 = 256;
const INITIAL_MAX_BLENDSHAPES: u32 = 256;
/// Initial staging for packed blendshape `Params` (32 bytes × chunks).
const INITIAL_BLENDSHAPE_PARAMS_STAGING: u64 = 4096;

/// WebGPU storage / uniform dynamic offset alignment (typical 256).
#[inline]
fn align256(n: u64) -> u64 {
    (n + 255) & !255
}

/// Scratch storage written each frame before compute dispatches.
pub struct MeshDeformScratch {
    /// Linear blend skinning bone palette (`mat4` column-major, 64 bytes each); subranges use 256-byte-aligned offsets.
    pub bone_matrices: wgpu::Buffer,
    /// 32-byte uniform for chunked blendshape compute (`mesh_blendshape.wgsl` `Params`).
    pub blendshape_params: wgpu::Buffer,
    /// Upload + copy source for packed per-chunk `Params` before `copy_buffer_to_buffer` into `blendshape_params`.
    pub blendshape_params_staging: wgpu::Buffer,
    /// `f32` weight per blendshape; subranges use 256-byte-aligned offsets between meshes.
    pub blendshape_weights: wgpu::Buffer,
    max_bones: u32,
    max_shapes: u32,
}

impl MeshDeformScratch {
    /// Allocates initial scratch buffers on `device`.
    pub fn new(device: &wgpu::Device) -> Self {
        let bone_bytes = (INITIAL_MAX_BONES as u64) * 64;
        let weight_bytes = (INITIAL_MAX_BLENDSHAPES as u64) * 4;
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
            max_bones: INITIAL_MAX_BONES,
            max_shapes: INITIAL_MAX_BLENDSHAPES,
        }
    }

    /// Ensures the bone palette buffer fits at least `need_bones` matrices (legacy single-mesh helper).
    pub fn ensure_bone_capacity(&mut self, device: &wgpu::Device, need_bones: u32) {
        if need_bones <= self.max_bones {
            return;
        }
        let next = need_bones.next_power_of_two().max(INITIAL_MAX_BONES);
        let bone_bytes = (next as u64) * 64;
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
        self.bone_matrices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_bone_palette"),
            size: next_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    /// Ensures the blendshape weight buffer fits at least `need_shapes` floats (legacy helper).
    pub fn ensure_shape_weight_capacity(&mut self, device: &wgpu::Device, need_shapes: u32) {
        if need_shapes <= self.max_shapes {
            return;
        }
        let next = need_shapes.next_power_of_two().max(INITIAL_MAX_BLENDSHAPES);
        let weight_bytes = (next as u64) * 4;
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
        self.blendshape_params_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_blendshape_params_staging"),
            size: next,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
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
