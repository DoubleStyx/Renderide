//! Headless offscreen driver loop (no window; adapter chosen by wgpu, often GPU or lavapipe on CI).
//!
//! Mirrors the desktop `RenderideApp` driver as closely as possible: builds a [`GpuContext`]
//! (with no surface) via [`GpuContext::new_headless`], attaches it to the runtime, and then runs
//! the same per-frame phases as desktop. **Full frames** ([`RendererRuntime::tick_one_frame`],
//! including `render_frame`) run on a wall-clock cadence (`--headless-interval-ms`, default
//! 1000 ms). Between those ticks the loop only runs
//! [`RendererRuntime::tick_one_frame_lockstep_only`] so IPC and asset integration stay responsive
//! on slow adapters (e.g. software Vulkan). After each full frame, the GpuContext's primary offscreen color
//! texture is optionally read back to a PNG.
//!
//! This loop has **no winit / OpenXR involvement** and never threads a window through any
//! render-path API — the renderer's window lives inside [`GpuContext`] in windowed mode and is
//! `None` here, with the same render graph executing against `OffscreenRt` views substituted for
//! `Swapchain` views inside `render_frame` (see [`crate::runtime::RendererRuntime`]).

use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::gpu::GpuContext;
use crate::ipc::HeadlessParams;
use crate::run_error::RunError;
use crate::runtime::RendererRuntime;
use crate::shared::InputState;

use super::startup::ExternalShutdownCoordinator;

/// Sleep granularity inside the headless tick loop.
///
/// Keeps IPC poll latency low (host's lock-step `FrameSubmitData` is processed within this window)
/// while not pegging a CPU core. Full renders are gated by [`HeadlessParams::interval_ms`], not by
/// this sleep.
const HEADLESS_TICK_SLEEP: Duration = Duration::from_millis(5);

/// Runs the renderer in headless offscreen mode until external shutdown or runtime exit.
pub fn run_headless(
    runtime: &mut RendererRuntime,
    params: HeadlessParams,
    external_shutdown: Option<ExternalShutdownCoordinator>,
    initial_gpu_validation: bool,
    initial_power_preference: wgpu::PowerPreference,
) -> Result<Option<i32>, RunError> {
    logger::info!(
        "Headless mode: output={} size={}x{} interval_ms={}",
        params.output_path.display(),
        params.width,
        params.height,
        params.interval_ms,
    );

    let mut gpu = pollster::block_on(GpuContext::new_headless(
        params.width,
        params.height,
        initial_gpu_validation,
        initial_power_preference,
    ))?;
    runtime.attach_gpu(&gpu);

    let render_interval = Duration::from_millis(params.interval_ms.max(1));
    // Next wall-clock instant for a full frame (render + optional PNG). Advanced after every
    // render attempt so we do not spin `tick_one_frame` every `HEADLESS_TICK_SLEEP` while waiting
    // for an offscreen target or a slow first frame (that starved IPC on lavapipe).
    let mut next_full_frame_at = Instant::now();
    let mut frames_written: u64 = 0;

    loop {
        if let Some(coord) = external_shutdown.as_ref() {
            if coord.requested.load(Ordering::Relaxed) {
                logger::info!("Headless: external shutdown requested, exiting");
                break;
            }
        }

        let now = Instant::now();
        runtime.tick_frame_wall_clock_begin(now);
        let due_for_full_frame = now >= next_full_frame_at;

        // Per-frame work split:
        //
        // - Most ticks only do `tick_one_frame_lockstep_only`: `poll_ipc + asset_integration +
        //   pre_frame`. No `render_frame`. This is cheap (microseconds) and keeps the IPC
        //   queue drained so the host's `FrameSubmitData` is consumed promptly.
        // - On the wall-clock schedule (`render_interval`, default 1s), run the full
        //   `tick_one_frame` which includes `render_frame`. Advance the clock **after** that
        //   attempt so we do not re-enter a full render on every 5ms tick while lavapipe is still
        //   busy or before the offscreen texture exists.
        let outcome = if due_for_full_frame {
            profiling::scope!("headless::full_frame");
            let o = runtime.tick_one_frame(&mut gpu, InputState::default());
            next_full_frame_at = Instant::now() + render_interval;
            o
        } else {
            profiling::scope!("headless::lockstep_tick");
            runtime.tick_one_frame_lockstep_only(Some(&gpu), InputState::default())
        };
        if outcome.shutdown_requested {
            logger::info!("Headless: host shutdown requested, exiting");
            break;
        }
        if outcome.fatal_error {
            logger::error!("Headless: fatal IPC error, exiting");
            crate::profiling::emit_frame_mark();
            return Ok(Some(4));
        }
        if let Some(err) = outcome.graph_error {
            logger::warn!("Headless: render graph error this tick: {err:?}");
        }

        if due_for_full_frame && gpu.headless_color_texture().is_some() {
            if let Err(e) = readback_and_write_png_atomically(&gpu, &params.output_path) {
                logger::warn!("Headless PNG write failed: {e}");
            } else {
                frames_written = frames_written.saturating_add(1);
                logger::trace!("Headless wrote PNG #{frames_written}");
            }
        }

        runtime.tick_frame_render_time_end(gpu.last_completed_gpu_render_time_seconds());
        crate::profiling::emit_frame_mark();
        std::thread::sleep(HEADLESS_TICK_SLEEP);
    }

    Ok(Some(0))
}

/// Copies the headless primary color texture to a CPU buffer and writes it as a PNG via an
/// atomic `*.tmp` rename so the test harness never reads a torn file.
fn readback_and_write_png_atomically(
    gpu: &GpuContext,
    output_path: &Path,
) -> Result<(), HeadlessReadbackError> {
    let color_texture = gpu
        .headless_color_texture()
        .ok_or(HeadlessReadbackError::NoOffscreenTexture)?;
    let extent = color_texture.size();
    let width = extent.width;
    let height = extent.height;
    if width == 0 || height == 0 {
        return Err(HeadlessReadbackError::EmptyExtent);
    }

    let bytes_per_row_tight = width * 4;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let bytes_per_row_padded = bytes_per_row_tight.div_ceil(alignment) * alignment;
    let buffer_size = (bytes_per_row_padded as u64) * (height as u64);

    let limits = gpu.limits();
    if !limits.buffer_size_fits(buffer_size) {
        return Err(HeadlessReadbackError::BufferSizeExceedsLimit {
            size: buffer_size,
            max: limits.max_buffer_size(),
        });
    }

    let readback = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("renderide-headless-readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Ensure the frame's main submit has been executed by the driver thread before we
    // enqueue a texture-to-buffer copy that reads from the offscreen color texture. The
    // driver thread's ring is FIFO, so observing the flush sentinel's completion implies
    // every earlier batch's `Queue::submit` has already run.
    gpu.flush_driver();

    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("renderide-headless-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row_padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue().submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| HeadlessReadbackError::DeviceLost(format!("{e:?}")))?;
    #[expect(
        clippy::map_err_ignore,
        reason = "ReadbackTimeout is the specific case we map; `RecvTimeoutError::Timeout` adds no detail"
    )]
    receiver
        .recv_timeout(Duration::from_secs(5))
        .map_err(|_| HeadlessReadbackError::ReadbackTimeout)?
        .map_err(|e| HeadlessReadbackError::Map(format!("{e:?}")))?;

    let mut tight = vec![0u8; (bytes_per_row_tight as usize) * (height as usize)];
    {
        let view = slice.get_mapped_range();
        for row in 0..(height as usize) {
            let src_start = row * bytes_per_row_padded as usize;
            let src_end = src_start + bytes_per_row_tight as usize;
            let dst_start = row * bytes_per_row_tight as usize;
            let dst_end = dst_start + bytes_per_row_tight as usize;
            tight[dst_start..dst_end].copy_from_slice(&view[src_start..src_end]);
        }
    }
    readback.unmap();

    write_png_atomically(&tight, width, height, output_path)?;
    Ok(())
}

fn write_png_atomically(
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), HeadlessReadbackError> {
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(HeadlessReadbackError::Io)?;
        }
    }
    let buffer = image::RgbaImage::from_raw(width, height, rgba_bytes.to_vec())
        .ok_or(HeadlessReadbackError::EncodeBufferSize)?;
    let tmp_path = output_path.with_extension("png.tmp");
    buffer
        .save_with_format(&tmp_path, image::ImageFormat::Png)
        .map_err(|e| HeadlessReadbackError::Encode(format!("{e:?}")))?;
    std::fs::rename(&tmp_path, output_path).map_err(HeadlessReadbackError::Io)?;
    Ok(())
}

/// Failures while copying the offscreen color texture back to CPU and writing the PNG.
#[derive(Debug, thiserror::Error)]
enum HeadlessReadbackError {
    /// The headless [`crate::gpu::PrimaryOffscreenTargets`] have not been allocated.
    #[error("no headless offscreen color texture allocated")]
    NoOffscreenTexture,
    /// The offscreen target had a zero-sized extent (misconfiguration).
    #[error("headless offscreen target has empty extent")]
    EmptyExtent,
    /// `device.poll` reported a device loss while waiting on `map_async`.
    #[error("device lost during readback poll: {0}")]
    DeviceLost(String),
    /// `map_async` callback never delivered a result before the timeout.
    #[error("buffer.map_async timed out")]
    ReadbackTimeout,
    /// `map_async` returned an error.
    #[error("map_async failed: {0}")]
    Map(String),
    /// The pixel buffer dimensions did not match the produced byte count.
    #[error("readback dimensions invalid for image::RgbaImage construction")]
    EncodeBufferSize,
    /// Encoding to PNG via the `image` crate failed.
    #[error("png encode: {0}")]
    Encode(String),
    /// Filesystem operation (mkdir, rename) failed.
    #[error("io: {0}")]
    Io(#[source] std::io::Error),
    /// The readback buffer size would exceed [`wgpu::Limits::max_buffer_size`].
    #[error("readback buffer {size} bytes exceeds device max_buffer_size={max}")]
    BufferSizeExceedsLimit {
        /// Requested readback buffer size.
        size: u64,
        /// Device cap.
        max: u64,
    },
}

#[cfg(test)]
mod readback_error_tests {
    use super::HeadlessReadbackError;

    /// Variants with constant or mostly-constant messages have stable [`std::fmt::Display`] output
    /// used in the headless warn line, so CI log scrapers can match them.
    #[test]
    fn constant_variants_render_stable_messages() {
        assert_eq!(
            HeadlessReadbackError::NoOffscreenTexture.to_string(),
            "no headless offscreen color texture allocated"
        );
        assert_eq!(
            HeadlessReadbackError::EmptyExtent.to_string(),
            "headless offscreen target has empty extent"
        );
        assert_eq!(
            HeadlessReadbackError::ReadbackTimeout.to_string(),
            "buffer.map_async timed out"
        );
        assert_eq!(
            HeadlessReadbackError::EncodeBufferSize.to_string(),
            "readback dimensions invalid for image::RgbaImage construction"
        );
    }

    /// String-carrying variants interpolate their payload into the message without truncating.
    #[test]
    fn payload_variants_interpolate_inner_string() {
        let lost = HeadlessReadbackError::DeviceLost("adapter went away".into()).to_string();
        assert!(lost.contains("adapter went away"), "got {lost:?}");

        let mapped = HeadlessReadbackError::Map("OOM".into()).to_string();
        assert!(mapped.contains("OOM"), "got {mapped:?}");

        let encoded = HeadlessReadbackError::Encode("IO(BrokenPipe)".into()).to_string();
        assert!(encoded.contains("IO(BrokenPipe)"), "got {encoded:?}");
    }
}
