//! Host [`crate::shared::FrameSubmitData`] application: scene caches, HUD counters, and camera fields.

use std::time::Instant;

use rayon::prelude::*;

use super::host_camera_apply;
use crate::ipc::SharedMemoryAccessor;
use crate::runtime::RendererRuntime;
use crate::shared::FrameSubmitData;

/// Buffers at or above this size are filled via rayon `par_chunks_mut`; smaller
/// buffers fall back to a single SIMD memset to avoid rayon dispatch overhead.
/// 128 KiB is large enough to amortize rayon dispatch while letting medium photo
/// buffers use available memory bandwidth sooner.
const PAR_FILL_THRESHOLD: usize = 128 * 1024;
/// Per-thread chunk size for parallel fills. 64 KiB keeps each chunk
/// L2-resident on most desktop CPUs while staying large enough that
/// memset's non-temporal store path is selected per chunk.
const PAR_FILL_CHUNK: usize = 64 * 1024;
/// Byte value written into every unimplemented `CameraRenderTask` result
/// buffer: `0xFF` renders as opaque white in 8-bit BGRA/RGBA photo formats
/// (RGBA8/ARGB32), which is the format FrooxEngine asks for in `Photo` capture.
/// HDR/float pixel formats would interpret this as NaN; revisit when those
/// camera paths land.
const CAMERA_TASK_FILL_BYTE: u8 = 0xFF;

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
            match runtime.scene.apply_frame_submit(shm, &data) {
                Ok(report) => runtime.backend.note_scene_apply_report(&report),
                Err(e) => {
                    logger::error!("scene apply_frame_submit failed: {e}");
                    apply_failed = true;
                }
            }
        }
        {
            profiling::scope!("scene::frame_submit_flush_world_caches");
            match runtime.scene.flush_world_caches() {
                Ok(report) => runtime.backend.note_scene_cache_flush_report(&report),
                Err(e) => {
                    logger::error!("scene flush_world_caches failed: {e}");
                    apply_failed = true;
                }
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

/// Fills `bytes` with `value` using the platform-vectorized memset (AVX2/AVX-512
/// on x86_64 glibc, NEON on aarch64, vectorized CRT memset on Windows). Large
/// buffers are split into chunks and filled in parallel through rayon so a
/// 4K photo result completes in well under a frame.
fn fill_bytes_simd(bytes: &mut [u8], value: u8) {
    if bytes.len() >= PAR_FILL_THRESHOLD {
        bytes
            .par_chunks_mut(PAR_FILL_CHUNK)
            .for_each(|chunk| chunk.fill(value));
    } else {
        bytes.fill(value);
    }
}

/// Stopgap for unimplemented camera readback: fills every
/// [`crate::shared::CameraRenderTask`] result buffer in `data` with
/// [`CAMERA_TASK_FILL_BYTE`].
///
/// FrooxEngine pre-allocates each `CameraRenderTask.result_data` from a
/// recycled shared-memory pool, so an unwritten buffer surfaces stale bytes
/// from the host's previous lease as a glitchy photo. Filling here makes the
/// host's awaited `Bitmap2D` deterministic (opaque white in 8-bit RGBA/BGRA)
/// until the renderer implements full camera rendering.
fn clear_unimplemented_camera_render_tasks(shm: &mut SharedMemoryAccessor, data: &FrameSubmitData) {
    profiling::scope!("scene::frame_submit_clear_camera_tasks");
    if data.render_tasks.is_empty() {
        return;
    }
    let mut filled = 0usize;
    let mut failed = 0usize;
    for task in &data.render_tasks {
        if shm.access_mut_bytes(&task.result_data, |bytes| {
            fill_bytes_simd(bytes, CAMERA_TASK_FILL_BYTE);
        }) {
            filled += 1;
        } else {
            failed += 1;
        }
    }
    logger::debug!(
        "filled {filled} unimplemented CameraRenderTask result buffers with 0x{CAMERA_TASK_FILL_BYTE:02X} ({failed} failed)"
    );
}
