//! Shared-memory task-row parsing and answer writing for reflection-probe SH2 requests.

use crate::shared::memory_packable::MemoryPackable;
use crate::shared::memory_packer::MemoryPacker;
use crate::shared::{ComputeResult, ReflectionProbeSH2Task, RenderSH2, RENDER_SH2_HOST_ROW_BYTES};

/// Compact task header parsed out of a shared-memory row.
#[derive(Clone, Copy, Debug)]
pub(super) struct TaskHeader {
    /// Host renderable index for the SH2 component.
    pub(super) renderable_index: i32,
    /// Reflection-probe renderable index referenced by this SH2 task.
    pub(super) reflection_probe_renderable_index: i32,
}

/// Immediate task answer to write into shared memory.
pub(super) struct TaskAnswer {
    /// Result status.
    result: ComputeResult,
    /// Optional SH2 payload for computed rows.
    data: Option<RenderSH2>,
}

impl TaskAnswer {
    /// Creates a status-only answer.
    pub(super) fn status(result: ComputeResult) -> Self {
        Self { result, data: None }
    }

    /// Creates a computed answer with SH2 data.
    pub(super) fn computed(data: RenderSH2) -> Self {
        Self {
            result: ComputeResult::Computed,
            data: Some(data),
        }
    }
}

/// Stride of a host SH2 task row.
pub(super) fn task_stride() -> usize {
    std::mem::size_of::<ReflectionProbeSH2Task>()
}

/// Reads the two index fields from one SH2 task row.
pub(super) fn read_task_header(bytes: &[u8], offset: usize) -> Option<TaskHeader> {
    let renderable_index = read_i32_le(bytes.get(offset..offset + 4)?)?;
    let probe_index = read_i32_le(bytes.get(offset + 4..offset + 8)?)?;
    Some(TaskHeader {
        renderable_index,
        reflection_probe_renderable_index: probe_index,
    })
}

/// Reads a little-endian `i32` from a four-byte slice.
pub(super) fn read_i32_le(bytes: &[u8]) -> Option<i32> {
    let arr: [u8; 4] = bytes.try_into().ok()?;
    Some(i32::from_le_bytes(arr))
}

/// Writes a task answer into a shared-memory row.
pub(super) fn write_task_answer(bytes: &mut [u8], offset: usize, answer: TaskAnswer) {
    const RESULT_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result);
    const DATA_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result_data);
    if let Some(mut data) = answer.data {
        let Some(slot) =
            bytes.get_mut(offset + DATA_OFFSET..offset + DATA_OFFSET + RENDER_SH2_HOST_ROW_BYTES)
        else {
            return;
        };
        let mut packer = MemoryPacker::new(slot);
        data.pack(&mut packer);
    }
    if let Some(slot) = bytes.get_mut(offset + RESULT_OFFSET..offset + RESULT_OFFSET + 4) {
        slot.copy_from_slice(&(answer.result as i32).to_le_bytes());
    }
}

/// Debug helper that asserts every active row has been moved out of `Scheduled`.
pub(super) fn debug_assert_no_scheduled_rows(bytes: &[u8]) {
    #[cfg(debug_assertions)]
    {
        const RESULT_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result);
        let mut offset = 0usize;
        while offset + task_stride() <= bytes.len() {
            let Some(task) = read_task_header(bytes, offset) else {
                break;
            };
            if task.renderable_index < 0 {
                break;
            }
            let Some(result_bytes) = bytes.get(offset + RESULT_OFFSET..offset + RESULT_OFFSET + 4)
            else {
                break;
            };
            let Some(result) = read_i32_le(result_bytes) else {
                break;
            };
            debug_assert_ne!(result, ComputeResult::Scheduled as i32);
            offset += task_stride();
        }
    }
}
