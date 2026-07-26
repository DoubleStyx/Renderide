//! Growable indirect command buffers for arena-resident indexed draws.
//!
/// One `DrawIndexedIndirect` command; layout matches `wgpu::util::DrawIndexedIndirectArgs`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct IndexedIndirectCommand {
    /// Indices drawn for this mesh.
    pub index_count: u32,
    /// Instances drawn (per-draw slab covers the instance range).
    pub instance_count: u32,
    /// First index in the shared index buffer (index-range offset / index size).
    pub first_index: u32,
    /// Added to each vertex index (vertex-range offset / vertex stride).
    pub base_vertex: i32,
    /// First instance; non-zero requires `Features::INDIRECT_FIRST_INSTANCE`.
    pub first_instance: u32,
}

/// Bytes per indirect command (five 32-bit words).
const INDIRECT_COMMAND_BYTES: u64 = size_of::<IndexedIndirectCommand>() as u64;

/// Growable GPU buffer of indirect draw commands.
pub(crate) struct IndirectDrawBuffer {
    buffer: wgpu::Buffer,
    capacity_commands: u32,
    len_commands: u32,
    /// Replaced buffers retained until the driver thread submits their commands.
    retired_buffers: Vec<wgpu::Buffer>,
}

impl IndirectDrawBuffer {
    /// Creates a buffer sized for `initial_commands` (at least one).
    pub(crate) fn new(device: &wgpu::Device, initial_commands: u32) -> Self {
        let capacity_commands = initial_commands.max(1);
        Self {
            buffer: create_indirect_buffer(device, capacity_commands),
            capacity_commands,
            len_commands: 0,
            retired_buffers: Vec::new(),
        }
    }

    /// Buffer to pass to `multi_draw_indexed_indirect`.
    #[inline]
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Prepares capacity and length without uploading command bytes.
    pub(crate) fn prepare_len(&mut self, device: &wgpu::Device, count: u32) {
        if count > self.capacity_commands {
            let capacity = grow_command_capacity(self.capacity_commands, count);
            replace_and_retire(
                &mut self.buffer,
                create_indirect_buffer(device, capacity),
                &mut self.retired_buffers,
            );
            self.capacity_commands = capacity;
        }
        self.len_commands = count;
    }

    /// Retains current and replaced buffers for deferred submission.
    pub(crate) fn retain_submit_resources(
        &mut self,
        resources: &mut crate::gpu::GpuRetainedResources,
    ) {
        resources.retain_buffer(self.buffer.clone());
        resources.retain_buffers(self.retired_buffers.drain(..));
    }

    /// Records a clamped indirect range after the caller binds its shared state.
    pub(crate) fn draw_range(
        &self,
        rpass: &mut wgpu::RenderPass<'_>,
        first_command: u32,
        count: u32,
    ) {
        let end = first_command.saturating_add(count).min(self.len_commands);
        let count = end.saturating_sub(first_command);
        if count == 0 {
            return;
        }
        let offset = u64::from(first_command) * INDIRECT_COMMAND_BYTES;
        rpass.multi_draw_indexed_indirect(&self.buffer, offset, count);
    }
}

/// Replaces a live resource while preserving the previous value for deferred-submit retention.
fn replace_and_retire<T>(current: &mut T, replacement: T, retired: &mut Vec<T>) {
    retired.push(std::mem::replace(current, replacement));
}

/// Doubles `current` until it holds at least `needed` commands.
fn grow_command_capacity(current: u32, needed: u32) -> u32 {
    let mut capacity = current.max(1);
    while capacity < needed {
        capacity = capacity.saturating_mul(2);
    }
    capacity
}

fn create_indirect_buffer(device: &wgpu::Device, capacity_commands: u32) -> wgpu::Buffer {
    let size = u64::from(capacity_commands.max(1)) * INDIRECT_COMMAND_BYTES;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("indirect_draw_commands"),
        size,
        usage: wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    crate::profiling::note_resource_churn!(Buffer, "gpu::indirect_draw_commands");
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_matches_wgpu_indirect_layout() {
        assert_eq!(size_of::<IndexedIndirectCommand>(), 20);
        assert_eq!(INDIRECT_COMMAND_BYTES, 20);
    }

    #[test]
    fn grow_capacity_doubles_until_needed() {
        assert_eq!(grow_command_capacity(1, 1), 1);
        assert_eq!(grow_command_capacity(1, 3), 4);
        assert_eq!(grow_command_capacity(4, 5), 8);
        assert_eq!(grow_command_capacity(4, 4), 4);
    }

    #[test]
    fn command_is_pod_castable() {
        let commands = [
            IndexedIndirectCommand {
                index_count: 36,
                instance_count: 1,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            },
            IndexedIndirectCommand {
                index_count: 12,
                instance_count: 4,
                first_index: 36,
                base_vertex: 24,
                first_instance: 2,
            },
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&commands);
        assert_eq!(bytes.len(), 40);
        // The second command starts at byte 20; its first_index (third word) lands at byte 28.
        assert_eq!(bytes[28..32], 36u32.to_ne_bytes());
        // base_vertex (fourth word) follows at byte 32.
        assert_eq!(bytes[32..36], 24i32.to_ne_bytes());
    }

    #[test]
    fn replacement_is_retired_in_order() {
        let mut current = 1u32;
        let mut retired = Vec::new();

        replace_and_retire(&mut current, 2, &mut retired);
        replace_and_retire(&mut current, 3, &mut retired);

        assert_eq!(current, 3);
        assert_eq!(retired, vec![1, 2]);
    }
}
