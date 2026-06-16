//! Host-format conversion and shared-memory writing for completed camera render tasks.

use crate::ipc::SharedMemoryAccessor;
use crate::shared::CameraRenderTask;

use super::super::readback::par_fill_zeros;
use super::{CameraReadbackError, CameraTaskExtent, CameraTaskOutputFormat, RGBA8_BYTES_PER_PIXEL};

const MAX_CAMERA_RESULT_DESCRIPTOR_BYTES: i32 = 256 * 1024 * 1024;

pub(super) fn write_camera_task_result(
    shm: &mut SharedMemoryAccessor,
    task: &CameraRenderTask,
    output_format: CameraTaskOutputFormat,
    extent: CameraTaskExtent,
    rgba: &[u8],
) -> Result<(), CameraReadbackError> {
    profiling::scope!("camera_task::shared_memory_write");
    let required = output_byte_count(extent, output_format)?;
    if task.result_data.length > MAX_CAMERA_RESULT_DESCRIPTOR_BYTES {
        return Err(CameraReadbackError::ResultDescriptorTooSmall {
            required,
            actual: 0,
        });
    }
    let mut result = Err(CameraReadbackError::SharedMemoryMapFailed);
    let mapped = shm.access_mut_bytes(&task.result_data, |bytes| {
        if bytes.len() < required {
            result = Err(CameraReadbackError::ResultDescriptorTooSmall {
                required,
                actual: bytes.len(),
            });
            return;
        }
        par_fill_zeros(&mut bytes[..required]);
        result = pack_rgba8_to_host_buffer(rgba, extent, output_format, &mut bytes[..required]);
    });
    if mapped {
        result
    } else {
        Err(CameraReadbackError::SharedMemoryMapFailed)
    }
}

pub(super) fn output_byte_count(
    extent: CameraTaskExtent,
    output_format: CameraTaskOutputFormat,
) -> Result<usize, CameraReadbackError> {
    (extent.width as usize)
        .checked_mul(extent.height as usize)
        .and_then(|pixels| pixels.checked_mul(output_format.bytes_per_pixel()))
        .ok_or(CameraReadbackError::OutputByteCountOverflow)
}

pub(super) fn pack_rgba8_to_host_buffer(
    rgba: &[u8],
    extent: CameraTaskExtent,
    output_format: CameraTaskOutputFormat,
    dst: &mut [u8],
) -> Result<(), CameraReadbackError> {
    let src_required = output_byte_count(extent, CameraTaskOutputFormat::Rgba32)?;
    let dst_required = output_byte_count(extent, output_format)?;
    if rgba.len() < src_required {
        return Err(CameraReadbackError::ResultDescriptorTooSmall {
            required: src_required,
            actual: rgba.len(),
        });
    }
    if dst.len() < dst_required {
        return Err(CameraReadbackError::ResultDescriptorTooSmall {
            required: dst_required,
            actual: dst.len(),
        });
    }

    let width = extent.width as usize;
    let height = extent.height as usize;
    let src_row_bytes = width * RGBA8_BYTES_PER_PIXEL;
    let dst_pixel_bytes = output_format.bytes_per_pixel();
    let dst_row_bytes = width * dst_pixel_bytes;
    for dst_row in 0..height {
        let src_row_start = dst_row * src_row_bytes;
        let dst_row_start = dst_row * dst_row_bytes;
        for x in 0..width {
            let src = src_row_start + x * RGBA8_BYTES_PER_PIXEL;
            let dst_i = dst_row_start + x * dst_pixel_bytes;
            let r = rgba[src];
            let g = rgba[src + 1];
            let b = rgba[src + 2];
            let a = rgba[src + 3];
            match output_format {
                CameraTaskOutputFormat::Argb32 => {
                    dst[dst_i] = a;
                    dst[dst_i + 1] = r;
                    dst[dst_i + 2] = g;
                    dst[dst_i + 3] = b;
                }
                CameraTaskOutputFormat::Rgba32 => {
                    dst[dst_i] = r;
                    dst[dst_i + 1] = g;
                    dst[dst_i + 2] = b;
                    dst[dst_i + 3] = a;
                }
                CameraTaskOutputFormat::Bgra32 => {
                    dst[dst_i] = b;
                    dst[dst_i + 1] = g;
                    dst[dst_i + 2] = r;
                    dst[dst_i + 3] = a;
                }
                CameraTaskOutputFormat::Rgb24 => {
                    dst[dst_i] = r;
                    dst[dst_i + 1] = g;
                    dst[dst_i + 2] = b;
                }
            }
        }
    }
    Ok(())
}

pub(in crate::runtime) fn zero_camera_render_task_results(
    shm: &mut SharedMemoryAccessor,
    tasks: &[CameraRenderTask],
) -> usize {
    profiling::scope!("camera_task::zero_results");
    tasks
        .iter()
        .filter(|task| !zero_task_result(shm, task))
        .count()
}

pub(super) fn zero_task_result(shm: &mut SharedMemoryAccessor, task: &CameraRenderTask) -> bool {
    profiling::scope!("camera_task::zero_result");
    if task.result_data.length > MAX_CAMERA_RESULT_DESCRIPTOR_BYTES {
        logger::warn!(
            "CameraRenderTask zero-fill rejected oversized result buffer_id={} offset={} length={} cap={}",
            task.result_data.buffer_id,
            task.result_data.offset,
            task.result_data.length,
            MAX_CAMERA_RESULT_DESCRIPTOR_BYTES
        );
        return false;
    }
    let ok = shm.access_mut_bytes(&task.result_data, par_fill_zeros);
    if !ok {
        logger::warn!(
            "CameraRenderTask zero-fill failed for result buffer_id={} offset={} length={}",
            task.result_data.buffer_id,
            task.result_data.offset,
            task.result_data.length
        );
    }
    ok
}
