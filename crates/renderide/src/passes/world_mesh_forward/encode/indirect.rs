//! Small indirect-submission helpers shared by forward encoding paths.

use crate::gpu::indirect_buffer::IndexedIndirectCommand;

/// Issues a one-command indirect run directly.
///
/// A one-command `multi_draw_indexed_indirect` buys no batching and still makes the driver fetch
/// its parameters from GPU memory. The arena streams are already bound before this helper runs, so
/// the direct draw preserves every binding while dropping the indirect read. This is the common
/// path: it covered 75% of forward indirect runs in the measured scene.
pub(super) fn draw_single_indirect_command(
    rpass: &mut wgpu::RenderPass<'_>,
    command: IndexedIndirectCommand,
) {
    rpass.draw_indexed(
        command.first_index..command.first_index.saturating_add(command.index_count),
        command.base_vertex,
        command.first_instance
            ..command
                .first_instance
                .saturating_add(command.instance_count),
    );
}
