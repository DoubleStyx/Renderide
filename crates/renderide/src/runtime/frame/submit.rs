//! Host [`crate::shared::FrameSubmitData`] application on [`super::RendererRuntime`].

use std::time::Instant;

use super::super::offscreen_tasks::camera::zero_camera_render_task_results;
use super::super::offscreen_tasks::reflection_probe::reflection_probe_render_task_count;
use super::super::{RendererRuntime, lockstep};
use crate::diagnostics::crash_context::{self, TickPhase};
use crate::shared::FrameSubmitData;

impl RendererRuntime {
    /// Applies a host frame submit to lock-step, output state, camera fields, scene caches, and
    /// head-output transform.
    pub(crate) fn apply_frame_submit_data(&mut self, data: FrameSubmitData) {
        profiling::scope!("scene::apply_frame_submit");
        let prev_frame_index = self.host_camera.frame_index;
        if prev_frame_index >= 0 {
            let delta = i64::from(data.frame_index) - i64::from(prev_frame_index);
            if !(0..=1).contains(&delta) {
                logger::warn!(
                    "host frame index jump: previous={} current={} delta={}",
                    prev_frame_index,
                    data.frame_index,
                    delta
                );
            }
        }
        lockstep::trace_duplicate_frame_index_if_interesting(data.frame_index, prev_frame_index);
        self.process_frame_submit(data);
    }

    fn process_frame_submit(&mut self, data: FrameSubmitData) {
        profiling::scope!("scene::frame_submit");
        crash_context::set_tick_phase(TickPhase::FrameSubmit);
        crash_context::set_last_host_frame_index(i64::from(data.frame_index));
        let frame_index = data.frame_index;
        let submitted_render_spaces = data.render_spaces.len();
        let submitted_render_tasks = data.render_tasks.len();
        let shared_memory_available = self.frontend.shared_memory().is_some();
        self.begin_frame_submit_application(&data);

        let start = Instant::now();
        let mut apply_failed = false;
        let mut rendered_reflection_probes = Vec::new();
        let mut onchanges_reflection_probe_requests = Vec::new();
        let mut scene_apply_report = None;
        let mut queue_camera_tasks = false;
        let reflection_probe_task_count = reflection_probe_render_task_count(&data);
        let mut queue_reflection_probe_tasks = false;
        let mut failed_reflection_probe_tasks = false;
        let mut failed_camera_tasks = 0u64;
        if let Some(ref mut shm) = self.frontend.shared_memory_mut() {
            {
                profiling::scope!("scene::frame_submit_apply_scene");
                match self.scene.apply_frame_submit(shm, &data) {
                    Ok(report) => {
                        self.backend.note_scene_apply_report(&report);
                        scene_apply_report = Some(report);
                    }
                    Err(e) => {
                        logger::error!(
                            "scene apply_frame_submit failed: {e}; frame_index={frame_index} render_spaces={submitted_render_spaces} render_tasks={submitted_render_tasks} shared_memory_available={shared_memory_available}"
                        );
                        apply_failed = true;
                    }
                }
            }
            {
                profiling::scope!("scene::frame_submit_flush_world_caches");
                match self.scene.flush_world_caches() {
                    Ok(report) => self.backend.note_scene_cache_flush_report(&report),
                    Err(e) => {
                        logger::error!(
                            "scene flush_world_caches failed: {e}; frame_index={frame_index} render_spaces={} mesh_renderables={}",
                            self.scene.render_space_count(),
                            self.scene.total_mesh_renderable_count()
                        );
                        apply_failed = true;
                    }
                }
            }
            if !apply_failed {
                profiling::scope!("scene::frame_submit_reflection_probes");
                self.backend
                    .answer_reflection_probe_sh2_tasks(shm, &self.scene, &data);
                let mut changes = self.scene.take_reflection_probe_render_changes();
                rendered_reflection_probes.append(&mut changes.completed);
                onchanges_reflection_probe_requests.append(&mut changes.scene_captures);
                queue_camera_tasks = !data.render_tasks.is_empty();
                queue_reflection_probe_tasks = reflection_probe_task_count > 0;
            } else if !data.render_tasks.is_empty() {
                let zero_failed = zero_camera_render_task_results(shm, &data.render_tasks);
                logger::warn!(
                    "zero-filled {} CameraRenderTask readback(s) after failed frame submit apply (zero_fill_failed={zero_failed})",
                    data.render_tasks.len()
                );
                failed_camera_tasks =
                    failed_camera_tasks.saturating_add(data.render_tasks.len() as u64);
                failed_reflection_probe_tasks = reflection_probe_task_count > 0;
            } else if reflection_probe_task_count > 0 {
                failed_reflection_probe_tasks = true;
            }
        } else if !data.render_tasks.is_empty() {
            logger::warn!(
                "dropping {} CameraRenderTask readback(s): frame submit has no shared memory accessor",
                data.render_tasks.len()
            );
            failed_camera_tasks =
                failed_camera_tasks.saturating_add(data.render_tasks.len() as u64);
            failed_reflection_probe_tasks = reflection_probe_task_count > 0;
        } else if reflection_probe_task_count > 0 {
            failed_reflection_probe_tasks = true;
        }
        if !apply_failed && let Some(report) = scene_apply_report.as_ref() {
            self.log_successful_scene_apply(&data, report);
        }
        self.finish_frame_submit_readback_queues(FrameSubmitReadbackQueueDecision {
            data: &data,
            reflection_probe_task_count,
            queue_camera_tasks,
            queue_reflection_probe_tasks,
            failed_reflection_probe_tasks,
            failed_camera_tasks,
        });
        self.queue_onchanges_reflection_probe_requests(onchanges_reflection_probe_requests);
        self.frontend
            .enqueue_rendered_reflection_probes(rendered_reflection_probes);
        if apply_failed {
            self.finish_failed_frame_submit_apply();
        }
        self.derive_host_camera_after_frame_submit();
        self.trace_frame_submit_processed(&data, reflection_probe_task_count, start);
    }

    fn begin_frame_submit_application(&mut self, data: &FrameSubmitData) {
        {
            profiling::scope!("scene::frame_submit_frontend_bookkeeping");
            self.frontend.note_frame_submit_processed(data.frame_index);
            self.frontend
                .apply_frame_submit_output(data.output_state.clone());
            self.set_last_submit_render_task_count(data.render_tasks.len());
        };

        {
            profiling::scope!("scene::frame_submit_camera_fields");
            crate::camera::apply_frame_submit_fields(&mut self.host_camera, data);
        };
    }

    fn finish_failed_frame_submit_apply(&mut self) {
        self.note_frame_submit_apply_failure();
        logger::error!("{}", crash_context::format_snapshot());
        self.frontend.set_fatal_error(true);
    }

    fn derive_host_camera_after_frame_submit(&mut self) {
        profiling::scope!("scene::frame_submit_host_camera_derive");
        self.host_camera.head_output_transform =
            crate::camera::head_output_from_active_main_space(&self.scene);
        self.host_camera.eye_world_position =
            crate::camera::eye_world_position_from_active_main_space(&self.scene);
    }

    fn log_successful_scene_apply(
        &mut self,
        data: &FrameSubmitData,
        report: &crate::scene::SceneApplyReport,
    ) {
        let render_spaces = self.scene.render_space_count();
        let mesh_renderables = self.scene.total_mesh_renderable_count();
        if !self.diagnostics.logged_first_frame_submit {
            logger::info!(
                "first FrameSubmitData applied: frame_index={} submitted_spaces={} tracked_spaces={} mesh_renderables={} render_tasks={} changed_spaces={} removed_spaces={}",
                data.frame_index,
                data.render_spaces.len(),
                render_spaces,
                mesh_renderables,
                data.render_tasks.len(),
                report.changed_spaces.len(),
                report.removed_spaces.len(),
            );
            self.diagnostics.logged_first_frame_submit = true;
        }
        self.diagnostics.last_scene_render_space_count = render_spaces;
        self.diagnostics.last_scene_mesh_renderable_count = mesh_renderables;
    }

    fn finish_frame_submit_readback_queues(
        &mut self,
        decision: FrameSubmitReadbackQueueDecision<'_>,
    ) {
        if decision.queue_camera_tasks {
            self.queue_camera_render_tasks(&decision.data.render_tasks);
        }
        if decision.queue_reflection_probe_tasks {
            self.queue_reflection_probe_render_tasks(decision.data);
        }
        if decision.failed_reflection_probe_tasks {
            logger::warn!(
                "queueing {} failed ReflectionProbeRenderTask result(s) after frame submit rejection",
                decision.reflection_probe_task_count
            );
            self.queue_failed_reflection_probe_render_task_results(decision.data);
            self.flush_reflection_probe_render_results();
        }
        if decision.failed_camera_tasks > 0 {
            self.note_camera_readback_results(0, decision.failed_camera_tasks);
        }
    }

    fn trace_frame_submit_processed(
        &self,
        data: &FrameSubmitData,
        reflection_probe_task_count: usize,
        start: Instant,
    ) {
        logger::trace!(
            "frame_submit frame_index={} render_spaces={} render_tasks={} reflection_probe_render_tasks={} output_state={} debug_log={} near_clip={} far_clip={} desktop_fov_deg={} vr_active={} scene_apply_ms={:.3}",
            data.frame_index,
            data.render_spaces.len(),
            data.render_tasks.len(),
            reflection_probe_task_count,
            data.output_state.is_some(),
            data.debug_log,
            self.host_camera.clip.near,
            self.host_camera.clip.far,
            self.host_camera.desktop_fov_degrees,
            self.host_camera.vr_active,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

struct FrameSubmitReadbackQueueDecision<'a> {
    data: &'a FrameSubmitData,
    reflection_probe_task_count: usize,
    queue_camera_tasks: bool,
    queue_reflection_probe_tasks: bool,
    failed_reflection_probe_tasks: bool,
    failed_camera_tasks: u64,
}

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use crate::shared::{
        CameraRenderParameters, CameraRenderTask, FrameSubmitData, ReflectionProbeRenderTask,
        RenderSpaceUpdate, TextureFormat,
    };

    use super::super::super::RendererRuntime;

    #[test]
    fn successful_frame_submit_queues_camera_render_tasks() {
        let mut runtime = RendererRuntime::new(
            Option::<crate::connection::ConnectionParams>::None,
            std::sync::Arc::new(std::sync::RwLock::new(
                crate::config::RendererSettings::default(),
            )),
            std::path::PathBuf::from("test_config.toml"),
        );
        runtime.test_set_shared_memory("renderide-test-camera-queue");
        let data = FrameSubmitData {
            render_tasks: vec![CameraRenderTask {
                parameters: Some(CameraRenderParameters {
                    resolution: IVec2 { x: 2, y: 2 },
                    texture_format: TextureFormat::RGBA32,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };

        runtime.apply_frame_submit_data(data);

        assert_eq!(runtime.pending_camera_render_task_count(), 1);
    }

    #[test]
    fn successful_frame_submit_queues_reflection_probe_render_tasks() {
        let mut runtime = RendererRuntime::new(
            Option::<crate::connection::ConnectionParams>::None,
            std::sync::Arc::new(std::sync::RwLock::new(
                crate::config::RendererSettings::default(),
            )),
            std::path::PathBuf::from("test_config.toml"),
        );
        runtime.test_set_shared_memory("renderide-test-reflection-probe-queue");
        let data = FrameSubmitData {
            render_spaces: vec![RenderSpaceUpdate {
                id: 7,
                is_active: true,
                reflection_probe_render_tasks: vec![ReflectionProbeRenderTask {
                    render_task_id: 99,
                    size: 4,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };

        runtime.apply_frame_submit_data(data);

        assert_eq!(runtime.pending_reflection_probe_render_task_count(), 1);
    }
}
