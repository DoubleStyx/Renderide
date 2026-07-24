//! Reusable GPU buffers for per-frame mesh deformation (bone palette, blendshape uniforms).

mod bind_group_cache;
mod growable_buffer;

use std::sync::Arc;

use self::bind_group_cache::BindGroupCaches;
use self::growable_buffer::{
    BLENDSHAPE_PARAMS, BONE_MATRICES, BONE_MATRIX_BYTES, DUMMY_VEC4_READ, DUMMY_VEC4_WRITE,
    GrowableBuffer, SKIN_DISPATCH,
};

pub use self::bind_group_cache::{BlendshapeBindGroupKey, SkinningBindGroupKey, buffer_identity};

/// CPU-reserved cap on the bone palette; the GPU buffer grows when this is exceeded.
const INITIAL_MAX_BONES: u32 = 256;
/// Initial number of 256-byte slots for per-dispatch `SkinDispatchParams`.
const INITIAL_SKIN_DISPATCH_SLOTS: u64 = 16;

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

/// Scratch storage written each frame before compute dispatches.
pub struct MeshDeformScratch {
    /// Linear blend skinning bone palette (`mat4` column-major, 64 bytes each); subranges use 256-byte-aligned offsets.
    pub bone_matrices: wgpu::Buffer,
    /// Dynamic-uniform slab for sparse blendshape scatter (`mesh_blendshape.wgsl` `Params`).
    pub blendshape_params: wgpu::Buffer,
    /// Slab of `mesh_skinning.wgsl` [`SkinDispatchParams`] at 256-byte-aligned offsets.
    pub skin_dispatch: wgpu::Buffer,
    /// Dummy read-only storage used for optional shader input bindings when an attribute path is disabled.
    pub dummy_vec4_read: wgpu::Buffer,
    /// Dummy writable storage used for optional shader output bindings when an attribute path is disabled.
    pub dummy_vec4_write: wgpu::Buffer,
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
    /// Reusable blendshape output channel per scatter dispatch chunk.
    ///
    /// Cleared (length-only, capacity retained) at the start of each blendshape record call.
    pub scatter_dispatch_targets: Vec<u32>,
    max_bones: u32,
    /// [`wgpu::Limits::max_buffer_size`]; growth refuses past this cap.
    max_buffer_size: u64,
    /// Grow operations observed since the previous diagnostics drain.
    frame_grow_count: u64,
    /// Incremented whenever a scratch GPU buffer is replaced.
    resource_generation: u64,
    bind_group_caches: BindGroupCaches,
}

impl MeshDeformScratch {
    /// Allocates initial scratch buffers on `device`.
    ///
    /// `max_buffer_size` must be [`wgpu::Device::limits`].`max_buffer_size` (see [`crate::gpu::GpuLimits::max_buffer_size`]).
    pub fn new(device: &wgpu::Device, max_buffer_size: u64) -> Self {
        let bone_bytes = u64::from(INITIAL_MAX_BONES) * BONE_MATRIX_BYTES;
        let skin_dispatch_bytes = INITIAL_SKIN_DISPATCH_SLOTS.saturating_mul(256);
        Self {
            bone_matrices: BONE_MATRICES.create(device, bone_bytes),
            blendshape_params: BLENDSHAPE_PARAMS.create(device, BLENDSHAPE_PARAMS.min_size),
            skin_dispatch: SKIN_DISPATCH.create(device, skin_dispatch_bytes),
            dummy_vec4_read: DUMMY_VEC4_READ.create(device, DUMMY_VEC4_READ.min_size),
            dummy_vec4_write: DUMMY_VEC4_WRITE.create(device, DUMMY_VEC4_WRITE.min_size),
            bone_palette_bytes: Vec::new(),
            packed_scatter_params: Vec::new(),
            scatter_dispatch_wgs: Vec::new(),
            scatter_dispatch_targets: Vec::new(),
            max_bones: INITIAL_MAX_BONES,
            max_buffer_size,
            frame_grow_count: 0,
            resource_generation: 0,
            bind_group_caches: BindGroupCaches::default(),
        }
    }

    fn note_gpu_buffer_grow(&mut self) {
        self.frame_grow_count = self.frame_grow_count.saturating_add(1);
        self.resource_generation = self.resource_generation.wrapping_add(1);
        self.bind_group_caches.clear_on_grow();
    }

    /// Drops cached deform bind groups after the skin-cache arenas were recreated.
    ///
    /// Bind-group keys identify buffers by address xor size, which aliases when a recreated arena
    /// buffer reuses a prior capacity, so stale bind groups must be cleared here.
    pub fn invalidate_after_arena_reset(&mut self) {
        self.resource_generation = self.resource_generation.wrapping_add(1);
        self.bind_group_caches.clear_on_grow();
    }

    /// Current scratch-resource generation for bind-group cache keys.
    #[inline]
    pub fn resource_generation(&self) -> u64 {
        self.resource_generation
    }

    /// Looks up a cached blendshape scatter bind group.
    #[inline]
    pub fn blendshape_bind_group(
        &self,
        key: BlendshapeBindGroupKey,
    ) -> Option<Arc<wgpu::BindGroup>> {
        self.bind_group_caches.blendshape(key)
    }

    /// Inserts a blendshape scatter bind group into the scratch cache.
    #[inline]
    pub fn insert_blendshape_bind_group(
        &mut self,
        key: BlendshapeBindGroupKey,
        bind_group: Arc<wgpu::BindGroup>,
    ) {
        self.bind_group_caches.insert_blendshape(key, bind_group);
    }

    /// Looks up a cached skinning bind group.
    #[inline]
    pub fn skinning_bind_group(&self, key: SkinningBindGroupKey) -> Option<Arc<wgpu::BindGroup>> {
        self.bind_group_caches.skinning(key)
    }

    /// Inserts a skinning bind group into the scratch cache.
    #[inline]
    pub fn insert_skinning_bind_group(
        &mut self,
        key: SkinningBindGroupKey,
        bind_group: Arc<wgpu::BindGroup>,
    ) {
        self.bind_group_caches.insert_skinning(key, bind_group);
    }

    /// Grows `which` to fit byte range `[0, end_exclusive)` on demand and tracks the growth.
    ///
    /// Returns `true` when `buf` is now (or already was) large enough. When growth actually happens
    /// the bind-group caches are cleared so stale bind groups referencing the freed buffer are not
    /// reused. Returns `false` (leaving `buf` unchanged) when growth would exceed the device's
    /// `max_buffer_size`.
    fn ensure_growable_capacity(
        &mut self,
        device: &wgpu::Device,
        which: &GrowableBuffer,
        buf_selector: fn(&mut Self) -> &mut wgpu::Buffer,
        end_exclusive: u64,
    ) -> bool {
        let max_buffer_size = self.max_buffer_size;
        let old_size = buf_selector(self).size();
        let ok = which.ensure(device, buf_selector(self), end_exclusive, max_buffer_size);
        if ok && buf_selector(self).size() > old_size {
            self.note_gpu_buffer_grow();
        }
        ok
    }

    /// Ensures the bone palette buffer fits at least `need_bones` matrices for a single-mesh dispatch.
    pub fn ensure_bone_capacity(&mut self, device: &wgpu::Device, need_bones: u32) {
        if need_bones <= self.max_bones {
            return;
        }
        let next = need_bones.next_power_of_two().max(INITIAL_MAX_BONES);
        let bone_bytes = u64::from(next) * BONE_MATRIX_BYTES;
        if self.ensure_growable_capacity(
            device,
            &BONE_MATRICES,
            |s| &mut s.bone_matrices,
            bone_bytes,
        ) {
            self.max_bones = next;
        }
    }

    /// Ensures the bone buffer is large enough for byte range `[0, end_exclusive)`.
    pub fn ensure_bone_byte_capacity(&mut self, device: &wgpu::Device, end_exclusive: u64) {
        self.ensure_growable_capacity(
            device,
            &BONE_MATRICES,
            |s| &mut s.bone_matrices,
            end_exclusive,
        );
    }

    /// Ensures the blendshape params uniform slab can address byte range `[0, end_exclusive)`.
    pub fn ensure_blendshape_param_byte_capacity(
        &mut self,
        device: &wgpu::Device,
        end_exclusive: u64,
    ) {
        self.ensure_growable_capacity(
            device,
            &BLENDSHAPE_PARAMS,
            |s| &mut s.blendshape_params,
            end_exclusive,
        );
    }

    /// Ensures the skin-dispatch uniform slab can address byte range `[0, end_exclusive)`.
    ///
    /// Each skinning dispatch writes a small uniform payload at a 256-byte-aligned cursor; callers advance with
    /// [`advance_slab_cursor`].
    pub fn ensure_skin_dispatch_byte_capacity(
        &mut self,
        device: &wgpu::Device,
        end_exclusive: u64,
    ) {
        self.ensure_growable_capacity(
            device,
            &SKIN_DISPATCH,
            |s| &mut s.skin_dispatch,
            end_exclusive,
        );
    }

    /// Returns and clears the grow count accumulated since the previous diagnostics drain.
    pub fn take_frame_grow_count(&mut self) -> u64 {
        let count = self.frame_grow_count;
        self.frame_grow_count = 0;
        count
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
    fn dispatch_cursor_advances_by_256_for_small_payloads() {
        assert_eq!(advance_slab_cursor(0, 32), 256);
        assert_eq!(advance_slab_cursor(256, 48), 512);
    }
}
