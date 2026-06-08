//! Readback layout, GPU copy, and host packing for reflection-probe bake tasks.

use std::time::Duration;

use crate::gpu::GpuContext;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::ReflectionProbeRenderTask;
use crate::skybox::ibl_cache::{SkyboxIblConvolver, mip_extent};

use super::super::readback::{AwaitBufferMapError, align_u32_up, align_u64_up, await_buffer_map};
use super::{
    CUBE_FACE_COUNT, ProbeCubeFace, ProbeMipReadback, ProbeOutputFormat, ProbeReadbackLayout,
    ProbeTaskExtent, RGBA16F_BYTES_PER_PIXEL, ReflectionProbeBakeError,
};

const PROBE_READBACK_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_REFLECTION_PROBE_RESULT_DESCRIPTOR_BYTES: i32 = 256 * 1024 * 1024;

impl From<AwaitBufferMapError> for ReflectionProbeBakeError {
    fn from(err: AwaitBufferMapError) -> Self {
        match err {
            AwaitBufferMapError::DeviceLost(s) => Self::DeviceLost(s),
            AwaitBufferMapError::Timeout => Self::ReadbackTimeout,
            AwaitBufferMapError::Map(s) => Self::Map(s),
        }
    }
}

pub(in crate::runtime) fn compute_probe_readback_layout(
    task: &ReflectionProbeRenderTask,
    extent: ProbeTaskExtent,
    output_format: ProbeOutputFormat,
    max_buffer_size: u64,
) -> Result<ProbeReadbackLayout, ReflectionProbeBakeError> {
    if task.mip_origins.len() != CUBE_FACE_COUNT {
        return Err(ReflectionProbeBakeError::InvalidMipOriginFaces {
            actual: task.mip_origins.len(),
        });
    }

    let mut subresources = Vec::with_capacity(CUBE_FACE_COUNT * extent.mip_levels as usize);
    let mut buffer_size = 0u64;
    let mut required_host_bytes = 0usize;
    for face in ProbeCubeFace::ALL {
        let face_index = face.index();
        let origins = &task.mip_origins[face_index];
        if origins.len() != extent.mip_levels as usize {
            return Err(ReflectionProbeBakeError::InvalidMipOriginCount {
                face: face_index,
                expected: extent.mip_levels as usize,
                actual: origins.len(),
            });
        }
        for mip in 0..extent.mip_levels {
            let mip_index = mip as usize;
            let origin = origins[mip_index];
            if origin < 0 {
                return Err(ReflectionProbeBakeError::NegativeMipOrigin {
                    face: face_index,
                    mip: mip_index,
                    origin,
                });
            }
            let edge = mip_extent(extent.size, mip);
            let bytes_per_row_tight = edge
                .checked_mul(RGBA16F_BYTES_PER_PIXEL as u32)
                .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
            let bytes_per_row_padded =
                align_u32_up(bytes_per_row_tight, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                    .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
            let buffer_offset = align_u64_up(buffer_size, wgpu::COPY_BUFFER_ALIGNMENT)
                .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
            let copy_byte_count = u64::from(bytes_per_row_padded)
                .checked_mul(u64::from(edge))
                .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
            buffer_size = buffer_offset
                .checked_add(copy_byte_count)
                .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
            let host_origin = usize::try_from(origin)
                .map_err(|_err| ReflectionProbeBakeError::OutputByteCountOverflow)?;
            let host_byte_count = (edge as usize)
                .checked_mul(edge as usize)
                .and_then(|pixels| pixels.checked_mul(output_format.bytes_per_pixel()))
                .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
            let host_end = host_origin
                .checked_add(host_byte_count)
                .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
            required_host_bytes = required_host_bytes.max(host_end);
            subresources.push(ProbeMipReadback {
                face,
                mip,
                extent: edge,
                bytes_per_row_tight,
                bytes_per_row_padded,
                buffer_offset,
                host_origin,
                host_byte_count,
            });
        }
    }

    let actual_host_bytes = if task.result_data.length > 0 {
        task.result_data.length as usize
    } else {
        0
    };
    if actual_host_bytes < required_host_bytes {
        return Err(ReflectionProbeBakeError::ResultDescriptorTooSmall {
            required: required_host_bytes,
            actual: actual_host_bytes,
        });
    }
    if buffer_size > max_buffer_size {
        return Err(ReflectionProbeBakeError::ReadbackBufferTooLarge {
            size: buffer_size,
            max: max_buffer_size,
        });
    }

    Ok(ProbeReadbackLayout {
        subresources,
        buffer_size,
        output_format,
    })
}

pub(in crate::runtime) fn readback_reflection_probe_cube(
    gpu: &mut GpuContext,
    convolver: &mut SkyboxIblConvolver,
    cube_texture: &wgpu::Texture,
    extent: ProbeTaskExtent,
    layout: &ProbeReadbackLayout,
) -> Result<Vec<u8>, ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_task::gpu_copy_and_map");
    let readback = create_probe_readback_buffer(gpu, layout);
    gpu.flush_driver();
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("renderide-reflection-probe-task-readback"),
        });
    let _convolve_resources = convolver
        .encode_existing_cube_mips(
            gpu,
            &mut encoder,
            cube_texture,
            extent.size,
            extent.mip_levels,
            gpu.gpu_profiler(),
        )
        .map_err(|err| ReflectionProbeBakeError::Convolve(err.to_string()))?;
    let copy_query = gpu
        .gpu_profiler_mut()
        .map(|p| p.begin_query("reflection_probe_task::cube_readback_copy", &mut encoder));
    encode_probe_cube_readback_copy(&mut encoder, cube_texture, layout, &readback);
    if let Some(query) = copy_query
        && let Some(prof) = gpu.gpu_profiler_mut()
    {
        prof.end_query(&mut encoder, query);
        prof.resolve_queries(&mut encoder);
    }
    let command_buffer = {
        profiling::scope!("CommandEncoder::finish::reflection_probe_task_readback");
        encoder.finish()
    };
    gpu.queue().submit(std::iter::once(command_buffer));
    let slice = readback.slice(..);
    {
        profiling::scope!("reflection_probe_task::map_readback");
        await_buffer_map(slice, gpu.device(), PROBE_READBACK_TIMEOUT)?;
    }
    let mapped = {
        profiling::scope!("reflection_probe_task::copy_mapped_readback");
        let view = slice.get_mapped_range();
        let required = usize::try_from(layout.buffer_size)
            .map_err(|_err| ReflectionProbeBakeError::OutputByteCountOverflow)?;
        if view.len() < required {
            return Err(ReflectionProbeBakeError::MappedReadbackTooSmall {
                required,
                actual: view.len(),
            });
        }
        view[..required].to_vec()
    };
    readback.unmap();
    Ok(mapped)
}

fn create_probe_readback_buffer(gpu: &GpuContext, layout: &ProbeReadbackLayout) -> wgpu::Buffer {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("renderide-reflection-probe-task-readback"),
        size: layout.buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    crate::profiling::note_resource_churn!(
        Buffer,
        "runtime::reflection_probe_task_readback_buffer"
    );
    buffer
}

fn encode_probe_cube_readback_copy(
    encoder: &mut wgpu::CommandEncoder,
    cube_texture: &wgpu::Texture,
    layout: &ProbeReadbackLayout,
    readback: &wgpu::Buffer,
) {
    profiling::scope!("reflection_probe_task::gpu_copy");
    for subresource in &layout.subresources {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: cube_texture,
                mip_level: subresource.mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: subresource.face.layer(),
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: subresource.buffer_offset,
                    bytes_per_row: Some(subresource.bytes_per_row_padded),
                    rows_per_image: Some(subresource.extent),
                },
            },
            wgpu::Extent3d {
                width: subresource.extent,
                height: subresource.extent,
                depth_or_array_layers: 1,
            },
        );
    }
}

pub(in crate::runtime) fn write_probe_task_result(
    shm: &mut SharedMemoryAccessor,
    task: &ReflectionProbeRenderTask,
    layout: &ProbeReadbackLayout,
    mapped: &[u8],
) -> Result<(), ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_task::shared_memory_write");
    let required = probe_result_required_byte_count(layout)?;
    if task.result_data.length > MAX_REFLECTION_PROBE_RESULT_DESCRIPTOR_BYTES {
        return Err(ReflectionProbeBakeError::ResultDescriptorTooSmall {
            required,
            actual: 0,
        });
    }
    let mut result = Err(ReflectionProbeBakeError::SharedMemoryMapFailed);
    let mapped_shm = shm.access_mut_bytes(&task.result_data, |bytes| {
        if bytes.len() < required {
            result = Err(ReflectionProbeBakeError::ResultDescriptorTooSmall {
                required,
                actual: bytes.len(),
            });
            return;
        }
        bytes[..required].fill(0);
        result = pack_probe_readback_to_host(mapped, layout, &mut bytes[..required]);
    });
    if mapped_shm {
        result
    } else {
        Err(ReflectionProbeBakeError::SharedMemoryMapFailed)
    }
}

fn probe_result_required_byte_count(
    layout: &ProbeReadbackLayout,
) -> Result<usize, ReflectionProbeBakeError> {
    layout
        .subresources
        .iter()
        .try_fold(0usize, |required, sub| {
            let end = sub
                .host_origin
                .checked_add(sub.host_byte_count)
                .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
            Ok(required.max(end))
        })
}

fn pack_probe_readback_to_host(
    mapped: &[u8],
    layout: &ProbeReadbackLayout,
    dst: &mut [u8],
) -> Result<(), ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_task::pack_host_result");
    let required = probe_result_required_byte_count(layout)?;
    if dst.len() < required {
        return Err(ReflectionProbeBakeError::ResultDescriptorTooSmall {
            required,
            actual: dst.len(),
        });
    }

    for subresource in &layout.subresources {
        let source_start = usize::try_from(subresource.buffer_offset)
            .map_err(|_err| ReflectionProbeBakeError::OutputByteCountOverflow)?;
        let source_byte_count = (subresource.bytes_per_row_padded as usize)
            .checked_mul(subresource.extent as usize)
            .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
        let source_end = source_start
            .checked_add(source_byte_count)
            .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
        if mapped.len() < source_end {
            return Err(ReflectionProbeBakeError::MappedReadbackTooSmall {
                required: source_end,
                actual: mapped.len(),
            });
        }
        let dst_end = subresource
            .host_origin
            .checked_add(subresource.host_byte_count)
            .ok_or(ReflectionProbeBakeError::OutputByteCountOverflow)?;
        if dst.len() < dst_end {
            return Err(ReflectionProbeBakeError::ResultDescriptorTooSmall {
                required: dst_end,
                actual: dst.len(),
            });
        }
        let source = &mapped[source_start..source_end];
        let destination = &mut dst[subresource.host_origin..dst_end];
        match layout.output_format {
            ProbeOutputFormat::Rgba16Float => {
                copy_rgba16f_rows(source, subresource, destination);
            }
            ProbeOutputFormat::Rgba8 => {
                encode_rgba16f_rows_to_linear_rgba8(source, subresource, destination);
            }
        }
    }
    Ok(())
}

fn copy_rgba16f_rows(src: &[u8], subresource: &ProbeMipReadback, dst: &mut [u8]) {
    let extent = subresource.extent as usize;
    let src_row_bytes = subresource.bytes_per_row_padded as usize;
    let tight_row_bytes = subresource.bytes_per_row_tight as usize;
    for row in 0..extent {
        let src_row = unity_bitmap_cube_source_row(extent, row);
        let src_start = src_row * src_row_bytes;
        let src_end = src_start + tight_row_bytes;
        let dst_start = row * tight_row_bytes;
        let dst_end = dst_start + tight_row_bytes;
        dst[dst_start..dst_end].copy_from_slice(&src[src_start..src_end]);
    }
}

/// Encodes linear HDR probe pixels into the host's linear `RGBA32` bitmap payload.
fn encode_rgba16f_rows_to_linear_rgba8(src: &[u8], subresource: &ProbeMipReadback, dst: &mut [u8]) {
    let extent = subresource.extent as usize;
    let src_row_bytes = subresource.bytes_per_row_padded as usize;
    let dst_row_bytes = extent * super::RGBA8_BYTES_PER_PIXEL;
    for row in 0..extent {
        let src_row = unity_bitmap_cube_source_row(extent, row);
        let src_row_start = src_row * src_row_bytes;
        let dst_row_start = row * dst_row_bytes;
        for x in 0..extent {
            let src_i = src_row_start + x * RGBA16F_BYTES_PER_PIXEL;
            let dst_i = dst_row_start + x * super::RGBA8_BYTES_PER_PIXEL;
            dst[dst_i] = linear_f32_to_unorm8(f16_bits_to_f32(read_u16_le(src, src_i)));
            dst[dst_i + 1] = linear_f32_to_unorm8(f16_bits_to_f32(read_u16_le(src, src_i + 2)));
            dst[dst_i + 2] = linear_f32_to_unorm8(f16_bits_to_f32(read_u16_le(src, src_i + 4)));
            dst[dst_i + 3] = linear_f32_to_unorm8(f16_bits_to_f32(read_u16_le(src, src_i + 6)));
        }
    }
}

fn unity_bitmap_cube_source_row(extent: usize, row: usize) -> usize {
    extent - 1 - row
}

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x03ff;
    let float_bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mantissa = u32::from(frac);
                let mut exponent = -14i32;
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x03ff;
                sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(frac) << 13),
        _ => sign | (u32::from(exp + 112) << 23) | (u32::from(frac) << 13),
    };
    f32::from_bits(float_bits)
}

fn linear_f32_to_unorm8(value: f32) -> u8 {
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= 1.0 {
        u8::MAX
    } else {
        (value * f32::from(u8::MAX)).round() as u8
    }
}

pub(in crate::runtime) fn zero_probe_task_result(
    shm: &mut SharedMemoryAccessor,
    task: &ReflectionProbeRenderTask,
) -> bool {
    profiling::scope!("reflection_probe_task::zero_result");
    if task.result_data.length > MAX_REFLECTION_PROBE_RESULT_DESCRIPTOR_BYTES {
        logger::warn!(
            "ReflectionProbeRenderTask zero-fill rejected oversized result buffer_id={} offset={} length={} cap={}",
            task.result_data.buffer_id,
            task.result_data.offset,
            task.result_data.length,
            MAX_REFLECTION_PROBE_RESULT_DESCRIPTOR_BYTES
        );
        return false;
    }
    let ok = shm.access_mut_bytes(&task.result_data, |bytes| bytes.fill(0));
    if !ok {
        logger::warn!(
            "ReflectionProbeRenderTask zero-fill failed for result buffer_id={} offset={} length={}",
            task.result_data.buffer_id,
            task.result_data.offset,
            task.result_data.length
        );
    }
    ok
}

#[cfg(test)]
mod tests {
    use renderide_shared::buffer::SharedMemoryBufferDescriptor;

    use super::super::{RGBA8_BYTES_PER_PIXEL, mip_levels_for_edge};
    use super::*;

    fn task_with_origins(size: i32, hdr: bool) -> ReflectionProbeRenderTask {
        let size_u32 = u32::try_from(size).expect("test size must fit u32");
        let mips = mip_levels_for_edge(size_u32) as usize;
        let bytes_per_pixel = if hdr {
            RGBA16F_BYTES_PER_PIXEL
        } else {
            RGBA8_BYTES_PER_PIXEL
        };
        let mut offset = 0i32;
        let mut mip_origins = Vec::new();
        for _face in 0..CUBE_FACE_COUNT {
            let mut face = Vec::new();
            for mip in 0..mips {
                face.push(offset);
                let edge = mip_extent(size_u32, mip as u32) as i32;
                offset += edge * edge * bytes_per_pixel as i32;
            }
            mip_origins.push(face);
        }
        ReflectionProbeRenderTask {
            size,
            hdr,
            mip_origins,
            result_data: SharedMemoryBufferDescriptor {
                length: offset,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn readback_layout_accepts_valid_bitmap_cube_layout() {
        let task = task_with_origins(4, true);
        let extent = ProbeTaskExtent::from_task(&task).expect("extent");
        let layout = compute_probe_readback_layout(
            &task,
            extent,
            ProbeOutputFormat::from_hdr(task.hdr),
            1_000_000,
        )
        .expect("layout");

        assert_eq!(layout.subresources.len(), CUBE_FACE_COUNT * 3);
        assert_eq!(layout.subresources[0].face, ProbeCubeFace::PosX);
        assert_eq!(layout.subresources[3].face, ProbeCubeFace::NegX);
        assert!(layout.buffer_size > 0);
    }

    #[test]
    fn readback_layout_rejects_missing_faces() {
        let mut task = task_with_origins(4, false);
        task.mip_origins.pop();
        let extent = ProbeTaskExtent::from_task(&task).expect("extent");

        let error = compute_probe_readback_layout(
            &task,
            extent,
            ProbeOutputFormat::from_hdr(task.hdr),
            1_000_000,
        )
        .expect_err("missing face must fail");

        assert!(matches!(
            error,
            ReflectionProbeBakeError::InvalidMipOriginFaces { actual: 5 }
        ));
    }

    #[test]
    fn readback_layout_rejects_wrong_mip_count() {
        let mut task = task_with_origins(4, false);
        task.mip_origins[0].pop();
        let extent = ProbeTaskExtent::from_task(&task).expect("extent");

        let error = compute_probe_readback_layout(
            &task,
            extent,
            ProbeOutputFormat::from_hdr(task.hdr),
            1_000_000,
        )
        .expect_err("wrong mip count must fail");

        assert!(matches!(
            error,
            ReflectionProbeBakeError::InvalidMipOriginCount {
                face: 0,
                expected: 3,
                actual: 2
            }
        ));
    }

    #[test]
    fn readback_layout_rejects_negative_mip_origin() {
        let mut task = task_with_origins(4, false);
        task.mip_origins[2][1] = -5;
        let extent = ProbeTaskExtent::from_task(&task).expect("extent");

        let error = compute_probe_readback_layout(
            &task,
            extent,
            ProbeOutputFormat::from_hdr(task.hdr),
            1_000_000,
        )
        .expect_err("negative origin must fail");

        assert!(matches!(
            error,
            ReflectionProbeBakeError::NegativeMipOrigin {
                face: 2,
                mip: 1,
                origin: -5
            }
        ));
    }

    #[test]
    fn readback_layout_rejects_small_descriptor() {
        let mut task = task_with_origins(4, false);
        task.result_data.length = 1;
        let extent = ProbeTaskExtent::from_task(&task).expect("extent");

        let error = compute_probe_readback_layout(
            &task,
            extent,
            ProbeOutputFormat::from_hdr(task.hdr),
            1_000_000,
        )
        .expect_err("small descriptor must fail");

        assert!(matches!(
            error,
            ReflectionProbeBakeError::ResultDescriptorTooSmall { .. }
        ));
    }

    #[test]
    fn readback_layout_rejects_buffer_above_device_limit() {
        let task = task_with_origins(4, true);
        let extent = ProbeTaskExtent::from_task(&task).expect("extent");

        let error =
            compute_probe_readback_layout(&task, extent, ProbeOutputFormat::from_hdr(task.hdr), 1)
                .expect_err("readback buffer limit");

        assert!(matches!(
            error,
            ReflectionProbeBakeError::ReadbackBufferTooLarge { size, max: 1 }
                if size > 1
        ));
    }

    #[test]
    fn probe_result_required_byte_count_uses_sparse_host_origins() {
        let layout = ProbeReadbackLayout {
            subresources: vec![
                ProbeMipReadback {
                    face: ProbeCubeFace::PosX,
                    mip: 0,
                    extent: 1,
                    bytes_per_row_tight: 8,
                    bytes_per_row_padded: 8,
                    buffer_offset: 0,
                    host_origin: 12,
                    host_byte_count: 4,
                },
                ProbeMipReadback {
                    face: ProbeCubeFace::NegX,
                    mip: 0,
                    extent: 1,
                    bytes_per_row_tight: 8,
                    bytes_per_row_padded: 8,
                    buffer_offset: 8,
                    host_origin: 0,
                    host_byte_count: 2,
                },
            ],
            buffer_size: 16,
            output_format: ProbeOutputFormat::Rgba16Float,
        };

        let required = probe_result_required_byte_count(&layout).expect("required bytes");

        assert_eq!(required, 16);
    }

    #[test]
    fn pack_rgba16f_to_linear_rgba8_writes_unity_bitmap_cube_rows_and_clamps() {
        let subresource = ProbeMipReadback {
            face: ProbeCubeFace::PosX,
            mip: 0,
            extent: 2,
            bytes_per_row_tight: 16,
            bytes_per_row_padded: 16,
            buffer_offset: 0,
            host_origin: 0,
            host_byte_count: 16,
        };
        let layout = ProbeReadbackLayout {
            subresources: vec![subresource],
            buffer_size: 32,
            output_format: ProbeOutputFormat::Rgba8,
        };
        let mapped = [
            0x00, 0x3c, 0x00, 0x38, 0x00, 0x34, 0x00, 0x3c, 0x00, 0x40, 0x00, 0xc0, 0x00, 0x00,
            0x00, 0x3c, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x7c, 0x00, 0x3c, 0x00, 0x7e, 0x00, 0xfc,
            0x00, 0x38, 0x00, 0x3c,
        ];
        let mut dst = [0u8; 16];

        pack_probe_readback_to_host(&mapped, &layout, &mut dst).expect("pack");

        assert_eq!(
            dst,
            [
                0, 255, 255, 255, 0, 0, 128, 255, 255, 128, 64, 255, 255, 0, 0, 255
            ]
        );
    }

    #[test]
    fn pack_probe_readback_rejects_short_mapped_buffer() {
        let layout = ProbeReadbackLayout {
            subresources: vec![ProbeMipReadback {
                face: ProbeCubeFace::PosX,
                mip: 0,
                extent: 1,
                bytes_per_row_tight: 8,
                bytes_per_row_padded: 8,
                buffer_offset: 0,
                host_origin: 0,
                host_byte_count: 8,
            }],
            buffer_size: 8,
            output_format: ProbeOutputFormat::Rgba16Float,
        };
        let mut dst = [0u8; 8];

        let error = pack_probe_readback_to_host(&[0u8; 7], &layout, &mut dst)
            .expect_err("short mapped buffer");

        assert!(matches!(
            error,
            ReflectionProbeBakeError::MappedReadbackTooSmall {
                required: 8,
                actual: 7
            }
        ));
    }

    #[test]
    fn pack_probe_readback_rejects_short_destination() {
        let layout = ProbeReadbackLayout {
            subresources: vec![ProbeMipReadback {
                face: ProbeCubeFace::PosX,
                mip: 0,
                extent: 1,
                bytes_per_row_tight: 8,
                bytes_per_row_padded: 8,
                buffer_offset: 0,
                host_origin: 0,
                host_byte_count: 8,
            }],
            buffer_size: 8,
            output_format: ProbeOutputFormat::Rgba16Float,
        };
        let mut dst = [0u8; 7];

        let error = pack_probe_readback_to_host(&[0u8; 8], &layout, &mut dst)
            .expect_err("short destination");

        assert!(matches!(
            error,
            ReflectionProbeBakeError::ResultDescriptorTooSmall {
                required: 8,
                actual: 7
            }
        ));
    }

    #[test]
    fn pack_rgba16f_writes_unity_bitmap_cube_rows_and_omits_padding() {
        let subresource = ProbeMipReadback {
            face: ProbeCubeFace::PosX,
            mip: 0,
            extent: 2,
            bytes_per_row_tight: 16,
            bytes_per_row_padded: 24,
            buffer_offset: 0,
            host_origin: 0,
            host_byte_count: 32,
        };
        let layout = ProbeReadbackLayout {
            subresources: vec![subresource],
            buffer_size: 48,
            output_format: ProbeOutputFormat::Rgba16Float,
        };
        let mapped = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 200, 201, 202, 203, 204, 205,
            206, 207, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 208, 209,
            210, 211, 212, 213, 214, 215,
        ];
        let mut dst = [0u8; 32];

        pack_probe_readback_to_host(&mapped, &layout, &mut dst).expect("pack");

        assert_eq!(
            dst,
            [
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            ]
        );
    }

    #[test]
    fn half_float_decode_handles_zero_normal_infinity_and_nan() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xc000), -2.0);
        assert!(f16_bits_to_f32(0x7c00).is_infinite());
        assert!(f16_bits_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn linear_f32_to_unorm8_clamps_and_rounds() {
        assert_eq!(linear_f32_to_unorm8(f32::NAN), 0);
        assert_eq!(linear_f32_to_unorm8(-0.01), 0);
        assert_eq!(linear_f32_to_unorm8(0.0), 0);
        assert_eq!(linear_f32_to_unorm8(0.5), 128);
        assert_eq!(linear_f32_to_unorm8(1.0), 255);
        assert_eq!(linear_f32_to_unorm8(2.0), 255);
    }

    #[test]
    fn unity_bitmap_cube_row_mapping_flips_each_mip_extent() {
        assert_eq!(unity_bitmap_cube_source_row(1, 0), 0);
        assert_eq!(unity_bitmap_cube_source_row(4, 0), 3);
        assert_eq!(unity_bitmap_cube_source_row(4, 1), 2);
        assert_eq!(unity_bitmap_cube_source_row(4, 2), 1);
        assert_eq!(unity_bitmap_cube_source_row(4, 3), 0);
    }
}
