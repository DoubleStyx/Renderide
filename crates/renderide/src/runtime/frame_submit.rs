//! Host [`crate::shared::FrameSubmitData`] application: scene caches, HUD counters, and camera fields.

use std::time::Instant;

use super::host_camera_apply;
use super::RendererRuntime;
use crate::shared::FrameSubmitData;

/// Applies a host frame submit: lock-step note, output state, camera fields, scene caches, head-output transform.
pub(crate) fn process_frame_submit(runtime: &mut RendererRuntime, data: FrameSubmitData) {
    profiling::scope!("scene::frame_submit");
    runtime
        .frontend
        .note_frame_submit_processed(data.frame_index);
    runtime
        .frontend
        .apply_frame_submit_output(data.output_state.clone());
    runtime.last_submit_render_task_count = data.render_tasks.len();

    host_camera_apply::apply_frame_submit_fields(&mut runtime.host_camera, &data);

    let start = Instant::now();
    let mut apply_failed = false;
    let mut rendered_reflection_probes = Vec::new();
    if let Some(ref mut shm) = runtime.frontend.shared_memory_mut() {
        if let Err(e) = runtime.scene.apply_frame_submit(shm, &data) {
            logger::error!("scene apply_frame_submit failed: {e}");
            apply_failed = true;
        }
        if let Err(e) = runtime.scene.flush_world_caches() {
            logger::error!("scene flush_world_caches failed: {e}");
            apply_failed = true;
        }
        if !apply_failed {
            runtime
                .backend
                .answer_reflection_probe_sh2_tasks(shm, &runtime.scene, &data);
            rendered_reflection_probes = runtime
                .scene
                .take_supported_reflection_probe_render_results();
        }
    }
    runtime
        .frontend
        .enqueue_rendered_reflection_probes(rendered_reflection_probes);
    if apply_failed {
        runtime.frame_submit_apply_failures = runtime.frame_submit_apply_failures.saturating_add(1);
        runtime.frontend.set_fatal_error(true);
    }
    runtime.host_camera.head_output_transform =
        host_camera_apply::head_output_from_active_main_space(&runtime.scene);
    runtime.host_camera.eye_world_position =
        host_camera_apply::eye_world_position_from_active_main_space(&runtime.scene);

    logger::trace!(
        "frame_submit frame_index={} near_clip={} far_clip={} desktop_fov_deg={} vr_active={} scene_apply_ms={:.3}",
        data.frame_index,
        runtime.host_camera.near_clip,
        runtime.host_camera.far_clip,
        runtime.host_camera.desktop_fov_degrees,
        runtime.host_camera.vr_active,
        start.elapsed().as_secs_f64() * 1000.0
    );
}
