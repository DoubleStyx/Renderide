//! Reflection probe SH2 task handling.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::shared::{ReflectionProbeSH2Task, ReflectionProbeSH2Tasks};

/// Writes `ComputeResult::Failed` to each ReflectionProbeSH2Task in shared memory.
/// The host expects the renderer to update the result field before frame finalization;
/// otherwise it panics with "Invalid compute result: Scheduled". We do not compute SH2
/// (spherical harmonics from reflection probes), so we mark all tasks as failed.
pub(crate) fn apply_reflection_probe_sh2_tasks(
    shm: &mut SharedMemoryAccessor,
    sh2_tasks: &ReflectionProbeSH2Tasks,
) {
    if sh2_tasks.tasks.length <= 0 {
        return;
    }
    const TASK_STRIDE: usize = std::mem::size_of::<ReflectionProbeSH2Task>();
    const RESULT_OFFSET: usize = 8; // after renderable_index (4) + reflection_probe_renderable_index (4)
    const COMPUTE_RESULT_FAILED: i32 = 3;
    if !shm.access_mut_bytes(&sh2_tasks.tasks, |bytes| {
        let mut offset = 0;
        while offset + TASK_STRIDE <= bytes.len() {
            let renderable_index =
                i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap_or([0; 4]));
            if renderable_index < 0 {
                break;
            }
            bytes[offset + RESULT_OFFSET..offset + RESULT_OFFSET + 4]
                .copy_from_slice(&COMPUTE_RESULT_FAILED.to_le_bytes());
            offset += TASK_STRIDE;
        }
    }) {}
}
