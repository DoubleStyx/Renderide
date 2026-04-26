//! Reflection probe SH2 task completion in shared memory.
//!
//! The host queues per-probe spherical-harmonics work with [`ComputeResult::Scheduled`] in
//! [`ReflectionProbeSH2Task`](crate::shared::ReflectionProbeSH2Task). The managed renderer clears
//! each task to [`ComputeResult::Computed`] or [`ComputeResult::Failed`] before finalizing the
//! frame; leaving [`ComputeResult::Scheduled`] can trigger host errors (for example invalid
//! compute result while scheduled).
//!
//! This renderer does not implement SH2 extraction yet. Until it does, we mark every task
//! [`ComputeResult::Failed`] so the host can proceed without waiting on GPU SH2 work.

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{ComputeResult, ReflectionProbeSH2Task, ReflectionProbeSH2Tasks};

/// Writes [`ComputeResult::Failed`] to the `result` field of each active task in `tasks.tasks`.
///
/// Tasks are laid out as a dense array of [`ReflectionProbeSH2Task`] in shared memory; iteration
/// stops at the first entry whose `renderable_index` is negative (host terminator convention).
pub(crate) fn mark_reflection_probe_sh2_tasks_failed(
    shm: &mut SharedMemoryAccessor,
    tasks: &ReflectionProbeSH2Tasks,
) {
    if tasks.tasks.length <= 0 {
        return;
    }

    const STRIDE: usize = std::mem::size_of::<ReflectionProbeSH2Task>();
    const RESULT_OFFSET: usize = std::mem::offset_of!(ReflectionProbeSH2Task, result);
    let failed_le = (ComputeResult::Failed as i32).to_le_bytes();

    let ok = shm.access_mut_bytes(&tasks.tasks, |bytes| {
        let mut offset = 0usize;
        while offset + STRIDE <= bytes.len() {
            let Some(renderable_bytes) = bytes.get(offset..offset + 4) else {
                break;
            };
            let Ok(arr) = renderable_bytes.try_into() else {
                break;
            };
            let renderable_index = i32::from_le_bytes(arr);
            if renderable_index < 0 {
                break;
            }
            let Some(slot) = bytes.get_mut(offset + RESULT_OFFSET..offset + RESULT_OFFSET + 4)
            else {
                break;
            };
            slot.copy_from_slice(&failed_le);
            offset += STRIDE;
        }
    });

    if !ok {
        logger::warn!(
            "reflection_probe_sh2: could not write ComputeResult::Failed (shared memory buffer)"
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::ReflectionProbeSH2Task;

    #[test]
    fn result_field_fits_inside_task_stride() {
        assert!(
            std::mem::offset_of!(ReflectionProbeSH2Task, result) + 4
                <= std::mem::size_of::<ReflectionProbeSH2Task>()
        );
    }
}
