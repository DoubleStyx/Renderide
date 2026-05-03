//! Host [`crate::shared::FrameSubmitData`] application: scene caches, HUD counters, and camera fields.

use std::time::Instant;

use rayon::prelude::*;

use super::host_camera_apply;
use crate::ipc::SharedMemoryAccessor;
use crate::runtime::RendererRuntime;
use crate::shared::FrameSubmitData;

/// Buffers at or above this size are zeroed via rayon `par_chunks_mut`; smaller
/// buffers fall back to a single SIMD memset to avoid rayon dispatch overhead.
/// 256 KiB roughly matches glibc's memset NT-store crossover and is large
/// enough that splitting across cores wins on multi-channel DRAM.
const PAR_ZERO_THRESHOLD: usize = 256 * 1024;
/// Per-thread chunk size for parallel zeroing. 64 KiB keeps each chunk
/// L2-resident on most desktop CPUs while staying large enough that
/// memset's non-temporal store path is selected per chunk.
const PAR_ZERO_CHUNK: usize = 64 * 1024;

/// Applies a host frame submit: lock-step note, output state, camera fields, scene caches, head-output transform.
pub(crate) fn process_frame_submit(runtime: &mut RendererRuntime, data: FrameSubmitData) {
    profiling::scope!("scene::frame_submit");
    {
        profiling::scope!("scene::frame_submit_frontend_bookkeeping");
        runtime
            .frontend
            .note_frame_submit_processed(data.frame_index);
        runtime
            .frontend
            .apply_frame_submit_output(data.output_state.clone());
        runtime.set_last_submit_render_task_count(data.render_tasks.len());
    };

    {
        profiling::scope!("scene::frame_submit_camera_fields");
        host_camera_apply::apply_frame_submit_fields(&mut runtime.host_camera, &data);
    };

    let start = Instant::now();
    let mut apply_failed = false;
    let mut rendered_reflection_probes = Vec::new();
    if let Some(ref mut shm) = runtime.frontend.shared_memory_mut() {
        {
            profiling::scope!("scene::frame_submit_apply_scene");
            if let Err(e) = runtime.scene.apply_frame_submit(shm, &data) {
                logger::error!("scene apply_frame_submit failed: {e}");
                apply_failed = true;
            }
        }
        {
            profiling::scope!("scene::frame_submit_flush_world_caches");
            if let Err(e) = runtime.scene.flush_world_caches() {
                logger::error!("scene flush_world_caches failed: {e}");
                apply_failed = true;
            }
        }
        if !apply_failed {
            profiling::scope!("scene::frame_submit_reflection_probes");
            runtime
                .backend
                .answer_reflection_probe_sh2_tasks(shm, &runtime.scene, &data);
            rendered_reflection_probes = runtime
                .scene
                .take_supported_reflection_probe_render_results();
            clear_unimplemented_camera_render_tasks(shm, &data);
        }
    }
    runtime
        .frontend
        .enqueue_rendered_reflection_probes(rendered_reflection_probes);
    if apply_failed {
        runtime.note_frame_submit_apply_failure();
        runtime.frontend.set_fatal_error(true);
    }
    {
        profiling::scope!("scene::frame_submit_host_camera_derive");
        runtime.host_camera.head_output_transform =
            host_camera_apply::head_output_from_active_main_space(&runtime.scene);
        runtime.host_camera.eye_world_position =
            host_camera_apply::eye_world_position_from_active_main_space(&runtime.scene);
    };

    logger::trace!(
        "frame_submit frame_index={} render_spaces={} render_tasks={} output_state={} debug_log={} near_clip={} far_clip={} desktop_fov_deg={} vr_active={} scene_apply_ms={:.3}",
        data.frame_index,
        data.render_spaces.len(),
        data.render_tasks.len(),
        data.output_state.is_some(),
        data.debug_log,
        runtime.host_camera.clip.near,
        runtime.host_camera.clip.far,
        runtime.host_camera.desktop_fov_degrees,
        runtime.host_camera.vr_active,
        start.elapsed().as_secs_f64() * 1000.0
    );
}

/// Zeroes `bytes` in-place using the platform-vectorized memset (AVX2/AVX-512
/// on x86_64 glibc, NEON on aarch64, vectorized CRT memset on Windows). Large
/// buffers are split into chunks and zeroed in parallel through rayon so a
/// 4K photo result clears in well under a frame.
fn zero_bytes_simd(bytes: &mut [u8]) {
    if bytes.len() >= PAR_ZERO_THRESHOLD {
        bytes
            .par_chunks_mut(PAR_ZERO_CHUNK)
            .for_each(|chunk| chunk.fill(0));
    } else {
        bytes.fill(0);
    }
}

/// Stopgap for unimplemented camera readback: zeros every
/// [`crate::shared::CameraRenderTask`] result buffer in `data`.
///
/// FrooxEngine pre-allocates each `CameraRenderTask.result_data` from a
/// recycled shared-memory pool, so an unwritten buffer surfaces stale bytes
/// from the host's previous lease as a glitchy photo. Clearing here makes the
/// host's awaited `Bitmap2D` deterministic (solid black) until the renderer
/// implements full camera rendering.
fn clear_unimplemented_camera_render_tasks(shm: &mut SharedMemoryAccessor, data: &FrameSubmitData) {
    profiling::scope!("scene::frame_submit_clear_camera_tasks");
    if data.render_tasks.is_empty() {
        return;
    }
    let mut cleared = 0usize;
    let mut failed = 0usize;
    for task in &data.render_tasks {
        if shm.access_mut_bytes(&task.result_data, zero_bytes_simd) {
            cleared += 1;
        } else {
            failed += 1;
        }
    }
    logger::debug!(
        "cleared {cleared} unimplemented CameraRenderTask result buffers ({failed} failed)"
    );
}
