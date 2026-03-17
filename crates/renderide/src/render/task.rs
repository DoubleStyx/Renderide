//! Render task executor: runs CameraRenderTask offscreen renders and copies to shared memory.
//!
//! Mirrors CameraRenderer.cs flow: create offscreen target, collect batches, render, copy pixels.

use nalgebra::Vector3;

use super::pass::projection_for_params;
use super::r#loop::RenderLoop;
use super::target::RenderTarget;
use super::SpaceDrawBatch;
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
            // Task proj is passed as ctx.proj; overlay batches use it when overlay_projection_override
            // is None. For orthographic tasks, overlay batches correctly use the task's orthographic proj.

            if let Err(e) = render_loop.render_to_target(
                gpu,
                session,
                &batches_with_view,
                &target,
                proj,
            ) {
                logger::error!("Render task render_to_target failed: {:?}", e);
                continue;
            }

            let Some(texture) = target.color_texture() else {
                continue;
            };
            let bytes_per_pixel = 4u32;
            let row_bytes = w * bytes_per_pixel;
            let bytes_per_row = ((row_bytes + 255) / 256) * 256;
            let buffer_size = bytes_per_row * h;

            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("camera render task readback"),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let source = texture.as_image_copy();
            let destination = wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            };
            encoder.copy_texture_to_buffer(
                source,
                destination,
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            gpu.queue.submit(std::iter::once(encoder.finish()));

            let slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            if gpu.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
                continue;
            }
            if rx.recv().ok().and_then(|r| r.ok()).is_none() {
                continue;
            }

            let mapped = slice.get_mapped_range();
            let Some(ref mut shm) = session.shared_memory_mut() else {
                continue;
            };
            let dest_len = task.result_data.length as usize;
            let row_len = (w as usize) * 4;
            let bytes_per_row_u64 = bytes_per_row as u64;
            if bytes_per_row_u64 == row_bytes as u64 {
                shm.access_mut_bytes(&task.result_data, |dest: &mut [u8]| {
                    let copy_len = dest_len.min(mapped.len());
                    dest[..copy_len].copy_from_slice(&mapped[..copy_len]);
                });
            } else {
                shm.access_mut_bytes(&task.result_data, |dest: &mut [u8]| {
                    let rows = (dest_len / row_len).min(h as usize);
                    for row in 0..rows {
                        let src_start = (row as u64 * bytes_per_row_u64) as usize;
                        let dst_start = row * row_len;
                        let copy_len = row_len.min(dest_len.saturating_sub(dst_start));
                        dest[dst_start..dst_start + copy_len]
                            .copy_from_slice(&mapped[src_start..src_start + copy_len]);
                    }
                });
            }
            drop(mapped);
            buffer.unmap();
        }
    }
}
