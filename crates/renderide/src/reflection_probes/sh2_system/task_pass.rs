//! Per-frame answering of SH2 task rows queued by the host.

use super::super::source_resolution::resolve_task_source;
use super::super::task_rows::{
    TaskAnswer, TaskHeader, debug_assert_no_scheduled_rows, read_task_header, task_stride,
    write_task_answer,
};
use super::ReflectionProbeSh2System;
use crate::ipc::SharedMemoryAccessor;
use crate::profiling;
use crate::reflection_probes::ReflectionProbeCubemapAssets;
use crate::reflection_probes::specular::RuntimeReflectionProbeCaptureStore;
use crate::scene::SceneCoordinator;
use crate::shared::{ComputeResult, ReflectionProbeSH2Tasks};

/// Borrow bundle for resolving SH2 task rows against host scene/material/asset state.
pub(super) struct Sh2TaskSourceContext<'a> {
    pub(super) scene: &'a SceneCoordinator,
    pub(super) assets: &'a dyn ReflectionProbeCubemapAssets,
    pub(super) captures: &'a RuntimeReflectionProbeCaptureStore,
    pub(super) render_space_id: i32,
}

impl ReflectionProbeSh2System {
    /// Answers all rows in one shared-memory task descriptor.
    pub(super) fn answer_task_buffer(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        source_ctx: Sh2TaskSourceContext<'_>,
        tasks: &ReflectionProbeSH2Tasks,
    ) {
        profiling::scope!("reflection_probe_sh2::answer_task_buffer");
        if tasks.tasks.length <= 0 {
            return;
        }

        let ok = shm.access_mut_bytes(&tasks.tasks, |bytes| {
            profiling::scope!("reflection_probe_sh2::task_buffer_scan");
            let mut offset = 0usize;
            while offset + task_stride() <= bytes.len() {
                let Some(task) = read_task_header(bytes, offset) else {
                    break;
                };
                if task.renderable_index < 0 {
                    break;
                }
                let answer = self.answer_for_task(&source_ctx, task);
                write_task_answer(bytes, offset, answer);
                offset += task_stride();
            }
            debug_assert_no_scheduled_rows(bytes);
        });

        if !ok {
            logger::warn!(
                "reflection_probe_sh2: could not write SH2 task results (shared memory buffer)"
            );
        }
    }

    /// Resolves one host task into an immediate answer.
    fn answer_for_task(
        &mut self,
        source_ctx: &Sh2TaskSourceContext<'_>,
        task: TaskHeader,
    ) -> TaskAnswer {
        let Some((key, source)) = resolve_task_source(
            source_ctx.scene,
            source_ctx.assets,
            source_ctx.captures,
            source_ctx.render_space_id,
            task,
        ) else {
            return TaskAnswer::status(ComputeResult::Failed);
        };

        let key_failed = self.failed.contains(&key);
        match self.ensure_resolved_source(key, source) {
            Some(sh) => TaskAnswer::computed(sh),
            None if key_failed => TaskAnswer::status(ComputeResult::Failed),
            None => TaskAnswer::status(ComputeResult::Postpone),
        }
    }
}
