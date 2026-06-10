//! Runtime-owned queues for host-submit completion work that must drain before begin-frame.

use crate::scene::RenderSpaceId;
use crate::shared::{CameraRenderTask, FrameSubmitData, ReflectionProbeRenderTask};

/// Reflection-probe bake task plus the render space that carried it.
#[derive(Clone, Debug)]
pub(in crate::runtime) struct QueuedReflectionProbeRenderTask {
    /// Host render space containing the reflection probe.
    pub(in crate::runtime) render_space_id: RenderSpaceId,
    /// Host bake task payload.
    pub(in crate::runtime) task: ReflectionProbeRenderTask,
}

/// Host-submit completion work that must drain before the renderer requests the next host frame.
#[derive(Default)]
pub(in crate::runtime) struct SubmitCompletionWorkQueue {
    camera_render_tasks: Vec<CameraRenderTask>,
    reflection_probe_render_tasks: Vec<QueuedReflectionProbeRenderTask>,
}

impl SubmitCompletionWorkQueue {
    /// Creates an empty submit-completion queue.
    pub(in crate::runtime) fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when no host-submit blocker remains queued.
    pub(in crate::runtime) fn is_empty(&self) -> bool {
        self.camera_render_tasks.is_empty() && self.reflection_probe_render_tasks.is_empty()
    }

    /// Number of host camera readback tasks waiting for GPU processing.
    pub(in crate::runtime) fn camera_count(&self) -> usize {
        self.camera_render_tasks.len()
    }

    /// Number of host reflection-probe bake tasks waiting for GPU processing.
    pub(in crate::runtime) fn reflection_probe_count(&self) -> usize {
        self.reflection_probe_render_tasks.len()
    }

    /// Appends host camera render tasks to the pre-begin-frame GPU readback queue.
    pub(in crate::runtime) fn queue_camera_tasks(&mut self, tasks: &[CameraRenderTask]) {
        self.camera_render_tasks.extend(tasks.iter().cloned());
    }

    /// Appends host reflection-probe cubemap bake tasks from a frame submit.
    pub(in crate::runtime) fn queue_reflection_probe_tasks_from_submit(
        &mut self,
        data: &FrameSubmitData,
    ) -> usize {
        let initial = self.reflection_probe_render_tasks.len();
        for space in &data.render_spaces {
            let render_space_id = RenderSpaceId(space.id);
            self.reflection_probe_render_tasks.extend(
                space
                    .reflection_probe_render_tasks
                    .iter()
                    .cloned()
                    .map(|task| QueuedReflectionProbeRenderTask {
                        render_space_id,
                        task,
                    }),
            );
        }
        self.reflection_probe_render_tasks
            .len()
            .saturating_sub(initial)
    }

    /// Takes all queued camera render tasks in FIFO order.
    pub(in crate::runtime) fn take_camera_tasks(&mut self) -> Vec<CameraRenderTask> {
        std::mem::take(&mut self.camera_render_tasks)
    }

    /// Takes all queued reflection-probe render tasks in FIFO order.
    pub(in crate::runtime) fn take_reflection_probe_tasks(
        &mut self,
    ) -> Vec<QueuedReflectionProbeRenderTask> {
        std::mem::take(&mut self.reflection_probe_render_tasks)
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::{CameraRenderTask, FrameSubmitData, ReflectionProbeRenderTask};

    use super::SubmitCompletionWorkQueue;

    #[test]
    fn empty_queue_has_no_submit_blockers() {
        let queue = SubmitCompletionWorkQueue::new();

        assert!(queue.is_empty());
        assert_eq!(queue.camera_count(), 0);
        assert_eq!(queue.reflection_probe_count(), 0);
    }

    #[test]
    fn queued_camera_tasks_block_submit_completion_until_taken() {
        let mut queue = SubmitCompletionWorkQueue::new();

        queue.queue_camera_tasks(&[CameraRenderTask::default()]);

        assert!(!queue.is_empty());
        assert_eq!(queue.camera_count(), 1);
        assert_eq!(queue.take_camera_tasks().len(), 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn queued_reflection_probe_tasks_preserve_render_space_scope() {
        let mut queue = SubmitCompletionWorkQueue::new();
        let data = FrameSubmitData {
            render_spaces: vec![crate::shared::RenderSpaceUpdate {
                id: 7,
                reflection_probe_render_tasks: vec![ReflectionProbeRenderTask {
                    render_task_id: 99,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };

        assert_eq!(queue.queue_reflection_probe_tasks_from_submit(&data), 1);

        let queued = queue.take_reflection_probe_tasks();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].render_space_id.0, 7);
        assert_eq!(queued[0].task.render_task_id, 99);
        assert!(queue.is_empty());
    }
}
