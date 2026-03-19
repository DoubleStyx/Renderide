//! Render task executor: runs [`CameraRenderTask`] offscreen renders and copies to shared memory.
//!
//! Render and texture→buffer copy share one queue submit via [`RenderLoop::render_to_target`]. Mapping
//! and shared-memory writes are deferred to [`RenderLoop::drain_pending_camera_task_readbacks`] so
//! multiple tasks can pipeline on the GPU without draining the full queue per task.

use nalgebra::Vector3;

use super::SpaceDrawBatch;
use super::r#loop::{PendingCameraTaskReadback, RenderLoop};
use super::pass::projection_for_params;
use super::target::RenderTarget;
use crate::gpu::GpuState;
use crate::session::Session;
use crate::shared::{CameraRenderTask, RenderTransform, TextureFormat};

/// Maps shared TextureFormat to wgpu texture format for offscreen targets.
fn texture_format_to_wgpu(format: TextureFormat) -> Option<wgpu::TextureFormat> {
    match format {
        TextureFormat::rgba32 => Some(wgpu::TextureFormat::Rgba8Unorm),
        TextureFormat::bgra32 => Some(wgpu::TextureFormat::Bgra8Unorm),
        _ => None,
    }
}

/// Executes CameraRenderTasks: offscreen render and copy to shared memory.
pub struct RenderTaskExecutor;

impl RenderTaskExecutor {
    /// Executes each task with valid parameters. Skips tasks without parameters or invalid resolution.
    ///
    /// Completed readbacks are flushed by [`RenderLoop::drain_pending_camera_task_readbacks`]; the
    /// caller should invoke that after this returns (and earlier in the tick for stale completions).
    pub fn execute(
        gpu: &mut GpuState,
        render_loop: &mut RenderLoop,
        session: &mut Session,
        tasks: Vec<CameraRenderTask>,
    ) {
        for task in tasks {
            let Some(ref params) = task.parameters else {
                continue;
            };
            let w = params.resolution.x.max(0) as u32;
            let h = params.resolution.y.max(0) as u32;
            if w == 0 || h == 0 {
                continue;
            }
            let Some(wgpu_format) = texture_format_to_wgpu(params.texture_format) else {
                continue;
            };
            if task.result_data.is_empty() {
                continue;
            }
            let expected_bytes = (w as usize).saturating_mul(h as usize).saturating_mul(4);
            if (task.result_data.length as usize) < expected_bytes {
                continue;
            }

            let target = RenderTarget::create_offscreen(&gpu.device, w, h, wgpu_format);
            let Some(texture) = target.color_texture() else {
                continue;
            };

            let bytes_per_pixel = 4u32;
            let row_bytes = w * bytes_per_pixel;
            let bytes_per_row = row_bytes.div_ceil(256) * 256;
            let buffer_size = bytes_per_row * h;

            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("camera render task readback"),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let batches = session.collect_draw_batches_for_task(
                task.render_space_id,
                &task.only_render_list,
                &task.exclude_render_list,
                params.render_private_ui,
                None,
            );

            let camera_transform = RenderTransform {
                position: task.position,
                rotation: task.rotation,
                scale: Vector3::new(1.0, 1.0, 1.0),
            };
            let batches_with_view: Vec<SpaceDrawBatch> = batches
                .into_iter()
                .map(|mut b| {
                    b.view_transform = camera_transform;
                    b
                })
                .collect();

            let aspect = w as f32 / h.max(1) as f32;
            let proj = projection_for_params(params, aspect);

            let source = texture.as_image_copy();
            let destination = wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            };
            let extent = wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            };

            let mut copy_to_readback = move |encoder: &mut wgpu::CommandEncoder| {
                encoder.copy_texture_to_buffer(source, destination, extent);
            };

            if let Err(e) = render_loop.render_to_target(
                gpu,
                session,
                &batches_with_view,
                &target,
                proj,
                Some(&mut copy_to_readback),
            ) {
                logger::error!("Render task render_to_target failed: {:?}", e);
                continue;
            }

            let slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });

            render_loop.enqueue_pending_camera_task_readback(PendingCameraTaskReadback {
                buffer,
                rx,
                task,
                width: w,
                height: h,
                bytes_per_row,
                row_bytes,
            });
        }
    }
}
