//! CPU layout for `shaders/source/modules/globals.wgsl` (`FrameGlobals` at `@group(0) @binding(0)`).

use bytemuck::{Pod, Zeroable};

/// Uniform block matching WGSL `FrameGlobals` (32-byte size, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameGpuUniforms {
    /// World-space camera position (`.w` unused).
    pub camera_world_pos: [f32; 4],
    /// Number of valid elements in the bound lights storage buffer.
    pub light_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl FrameGpuUniforms {
    /// Builds frame uniforms from a camera world position and active light count.
    pub fn new(camera_world_pos: glam::Vec3, light_count: u32) -> Self {
        Self {
            camera_world_pos: [
                camera_world_pos.x,
                camera_world_pos.y,
                camera_world_pos.z,
                0.0,
            ],
            light_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_globals_size_32() {
        assert_eq!(std::mem::size_of::<FrameGpuUniforms>(), 32);
    }
}
