use crate::assets::video::WgpuGstVideoSink;
use glam::IVec2;
use gstreamer::prelude::ElementExt;
use gstreamer_app::{AppSink, AppSinkCallbacks};
use std::sync::{Arc, Mutex};

/// Internal state shared between [`CpuCopyVideoSink`] and the appsink callback.
struct SinkState {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// The texture currently being written into by the callback.
    write_texture: Option<Arc<wgpu::Texture>>,
    width: u32,
    height: u32,
    /// Set to `Some` when a new texture is created, consumed by [`CpuCopyVideoSink::poll_texture_change`].
    pending_view: Option<Arc<wgpu::TextureView>>,
}

impl SinkState {
    /// Reallocates the write texture when the video size changes.
    /// Returns `true` if a new texture was created.
    fn resize_if_needed(&mut self, asset_id: i32, width: u32, height: u32) -> bool {
        if self.width == width && self.height == height && self.write_texture.is_some() {
            return false;
        }

        let texture = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("VideoTexture {asset_id}")),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));

        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor::default()));

        self.write_texture = Some(texture);
        self.width = width;
        self.height = height;
        self.pending_view = Some(view);

        true
    }
}

/// Owns the video [`AppSink`], the wgpu texture it writes into.
pub struct CpuCopyVideoSink {
    sink: AppSink,
    state: Arc<Mutex<SinkState>>,
}

impl CpuCopyVideoSink {
    pub fn new(asset_id: i32, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let sink = AppSink::builder()
            .caps(
                &gstreamer::Caps::builder("video/x-raw")
                    .field("format", "RGBA")
                    .build(),
            )
            .max_buffers(1)
            .drop(true)
            .build();

        let state = Arc::new(Mutex::new(SinkState {
            device,
            queue,
            write_texture: None,
            width: 0,
            height: 0,
            pending_view: None,
        }));

        let state_cb = Arc::clone(&state);

        sink.set_callbacks(
            AppSinkCallbacks::builder()
                .new_sample(move |appsink| {
                    let sample = match appsink.pull_sample() {
                        Ok(s) => s,
                        Err(e) => {
                            logger::warn!("CpuCopyVideoSink {asset_id}: failed to pull sample: {e}");
                            return Err(gstreamer::FlowError::Eos);
                        }
                    };

                    let caps = match sample.caps() {
                        Some(c) => c,
                        None => {
                            logger::warn!("CpuCopyVideoSink {asset_id}: sample without caps");
                            return Ok(gstreamer::FlowSuccess::Ok);
                        }
                    };

                    let structure = match caps.structure(0) {
                        Some(s) => s,
                        None => {
                            logger::warn!("CpuCopyVideoSink {asset_id}: caps without structure");
                            return Ok(gstreamer::FlowSuccess::Ok);
                        }
                    };

                    let (width, height) = match (
                        structure.get::<i32>("width"),
                        structure.get::<i32>("height"),
                    ) {
                        (Ok(w), Ok(h)) if w > 0 && h > 0 => (w as u32, h as u32),
                        _ => {
                            logger::warn!("CpuCopyVideoSink {asset_id}: invalid dimensions in caps: {:?}",
                                structure);
                            return Ok(gstreamer::FlowSuccess::Ok);
                        }
                    };

                    let buffer = match sample.buffer() {
                        Some(b) => b,
                        None => {
                            logger::warn!("CpuCopyVideoSink {asset_id}: sample without buffer");
                            return Ok(gstreamer::FlowSuccess::Ok);
                        }
                    };

                    let map = match buffer.map_readable() {
                        Ok(m) => m,
                        Err(e) => {
                            logger::warn!("CpuCopyVideoSink {asset_id}: failed to map buffer: {e}");
                            return Ok(gstreamer::FlowSuccess::Ok);
                        }
                    };

                    let Ok(mut state) = state_cb.lock() else {
                        return Ok(gstreamer::FlowSuccess::Ok);
                    };

                    state.resize_if_needed(asset_id, width, height);

                    let Some(texture) = state.write_texture.as_ref() else {
                        logger::warn!("CpuCopyVideoSink {asset_id}: no texture available after resize");
                        return Ok(gstreamer::FlowSuccess::Ok);
                    };

                    let expected = width as usize * height as usize * 4;
                    if map.len() != expected {
                        logger::warn!(
                            "CpuCopyVideoSink {asset_id}: frame size mismatch (got {} bytes, expected {expected})",
                            map.len()
                        );
                        return Ok(gstreamer::FlowSuccess::Ok);
                    }

                    state.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        map.as_slice(),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(width * 4),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );

                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );

        Self { sink, state }
    }
}

impl WgpuGstVideoSink for CpuCopyVideoSink {
    fn name(&self) -> &str {
        "CpuCopyVideoSink"
    }

    fn appsink(&self) -> &AppSink {
        &self.sink
    }

    fn poll_texture_change(&mut self) -> Option<(Arc<wgpu::TextureView>, u32, u32, u64)> {
        let mut state = self.state.lock().ok()?;
        let view = state.pending_view.take()?;
        let w = state.width;
        let h = state.height;
        let bytes = w as u64 * h as u64 * 4;
        Some((view, w, h, bytes))
    }

    fn size(&self) -> Option<IVec2> {
        use gstreamer::prelude::PadExt;
        let pad = self.sink.static_pad("sink")?;
        let caps = pad.current_caps()?;
        let structure = caps.structure(0)?;
        let width = structure.get::<i32>("width").ok()?;
        let height = structure.get::<i32>("height").ok()?;
        (width > 0 && height > 0).then_some(IVec2::new(width, height))
    }
}
