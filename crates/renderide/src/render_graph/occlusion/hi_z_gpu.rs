//! GPU hierarchical depth pyramid build and CPU readback for occlusion tests.

use std::num::NonZeroU64;
use std::sync::mpsc;
use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};

use crate::render_graph::{
    hi_z_pyramid_dimensions, hi_z_snapshot_from_linear_linear, mip_dimensions,
    mip_levels_for_extent, unpack_linear_rows_to_mips, HiZCpuSnapshot, HiZStereoCpuSnapshot,
    HiZTemporalState, OutputDepthMode,
};

const HIZ_MAX_MIPS: u32 = 8;

/// Triple-buffered staging so a slot is not reused until prior `map_async` completes (non-blocking).
const HIZ_STAGING_RING: usize = 3;

type MapRecv = mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>;

const fn pending_none_array<T>() -> [Option<T>; HIZ_STAGING_RING] {
    [None, None, None]
}

const MIP0_DESKTOP_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/hi_z_mip0_desktop.wgsl"
));
const MIP0_STEREO_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/hi_z_mip0_stereo.wgsl"
));
const DOWNSAMPLE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/hi_z_downsample_max.wgsl"
));

/// GPU + CPU Hi-Z state owned by [`crate::backend::OcclusionSystem`].
pub struct HiZGpuState {
    /// Last successfully read desktop pyramid (previous frame).
    pub desktop: Option<HiZCpuSnapshot>,
    /// Last successfully read stereo pyramids (previous frame).
    pub stereo: Option<HiZStereoCpuSnapshot>,
    /// View/projection snapshot for the frame that produced [`Self::desktop`] / [`Self::stereo`].
    pub temporal: Option<HiZTemporalState>,
    scratch: Option<HiZGpuScratch>,
    last_extent: (u32, u32),
    last_mode: OutputDepthMode,
    /// Next ring index for [`encode_hi_z_build`] copy targets (0..[`HIZ_STAGING_RING`]).
    write_idx: usize,
    /// Staging slot written in the current encode (consumed by [`Self::on_frame_submitted`]).
    hi_z_encoded_slot: Option<usize>,
    /// Pending `map_async` callbacks per desktop / left-eye staging buffer.
    desktop_pending: [Option<MapRecv>; HIZ_STAGING_RING],
    /// Pending `map_async` per right-eye buffer when stereo; `None` when desktop-only.
    right_pending: Option<[Option<MapRecv>; HIZ_STAGING_RING]>,
    /// Partial stereo CPU bytes until both eyes for the same ring slot complete.
    stereo_left_stash: [Option<Vec<u8>>; HIZ_STAGING_RING],
    stereo_right_stash: [Option<Vec<u8>>; HIZ_STAGING_RING],
}

impl Default for HiZGpuState {
    fn default() -> Self {
        Self {
            desktop: None,
            stereo: None,
            temporal: None,
            scratch: None,
            last_extent: (0, 0),
            last_mode: OutputDepthMode::DesktopSingle,
            write_idx: 0,
            hi_z_encoded_slot: None,
            desktop_pending: pending_none_array(),
            right_pending: None,
            stereo_left_stash: pending_none_array(),
            stereo_right_stash: pending_none_array(),
        }
    }
}

impl HiZGpuState {
    /// Drops GPU scratch and CPU snapshots when resolution or depth mode changes.
    pub fn invalidate_if_needed(&mut self, extent: (u32, u32), mode: OutputDepthMode) {
        if self.last_extent != extent || self.last_mode != mode {
            self.desktop = None;
            self.stereo = None;
            self.temporal = None;
            self.scratch = None;
            self.write_idx = 0;
            self.hi_z_encoded_slot = None;
            self.desktop_pending = pending_none_array();
            self.right_pending = None;
            self.stereo_left_stash = pending_none_array();
            self.stereo_right_stash = pending_none_array();
        }
        self.last_extent = extent;
        self.last_mode = mode;
    }

    /// Clears ring readback state without mapping (e.g. device loss).
    pub fn clear_pending(&mut self) {
        self.write_idx = 0;
        self.hi_z_encoded_slot = None;
        self.desktop_pending = pending_none_array();
        self.right_pending = None;
        self.stereo_left_stash = pending_none_array();
        self.stereo_right_stash = pending_none_array();
    }

    /// Drains completed `map_async` work into [`Self::desktop`] / [`Self::stereo`] without blocking.
    ///
    /// Call at the **start** of each frame (before encoding the render graph). Uses at most one
    /// [`wgpu::Device::poll`] to advance callbacks; if a read is not ready, prior snapshots are kept.
    pub fn begin_frame_readback(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);

        let Some(scratch) = self.scratch.as_ref() else {
            return;
        };
        let extent = scratch.extent;
        let mip_levels = scratch.mip_levels;
        let stereo = scratch.staging_r.is_some();

        for i in 0..HIZ_STAGING_RING {
            if let Some(recv) = self.desktop_pending[i].as_mut() {
                match recv.try_recv() {
                    Ok(Ok(())) => {
                        let buf = &scratch.staging_desktop[i];
                        let Some(raw) = read_mapped_buffer(buf) else {
                            self.desktop_pending[i] = None;
                            continue;
                        };
                        self.desktop_pending[i] = None;
                        if stereo {
                            self.stereo_left_stash[i] = Some(raw);
                        } else if let Some(snap) = unpack_desktop_snapshot(extent, mip_levels, &raw)
                        {
                            self.desktop = Some(snap);
                            self.stereo = None;
                        }
                    }
                    Ok(Err(_)) => {
                        scratch.staging_desktop[i].unmap();
                        self.desktop_pending[i] = None;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.desktop_pending[i] = None;
                    }
                }
            }
        }

        if stereo {
            if let Some(ref staging_r) = scratch.staging_r {
                if self.right_pending.is_none() {
                    self.right_pending = Some(pending_none_array());
                }
                let right_pending = self.right_pending.as_mut().expect("stereo right pending");
                for i in 0..HIZ_STAGING_RING {
                    if let Some(recv) = right_pending[i].as_mut() {
                        match recv.try_recv() {
                            Ok(Ok(())) => {
                                let buf = &staging_r[i];
                                let Some(raw) = read_mapped_buffer(buf) else {
                                    right_pending[i] = None;
                                    continue;
                                };
                                right_pending[i] = None;
                                self.stereo_right_stash[i] = Some(raw);
                            }
                            Ok(Err(_)) => {
                                staging_r[i].unmap();
                                right_pending[i] = None;
                            }
                            Err(mpsc::TryRecvError::Empty) => {}
                            Err(mpsc::TryRecvError::Disconnected) => {
                                right_pending[i] = None;
                            }
                        }
                    }
                }
            }
        }

        if stereo {
            for i in 0..HIZ_STAGING_RING {
                if self.stereo_left_stash[i].is_some() && self.stereo_right_stash[i].is_some() {
                    let left_raw = self.stereo_left_stash[i].take();
                    let right_raw = self.stereo_right_stash[i].take();
                    if let (Some(left_raw), Some(right_raw)) = (left_raw, right_raw) {
                        if let Some(stereo_snap) =
                            unpack_stereo_snapshot(extent, mip_levels, &left_raw, &right_raw)
                        {
                            self.stereo = Some(stereo_snap);
                            self.desktop = None;
                        }
                    }
                }
            }
        }
    }

    /// Starts `map_async` on the staging buffer(s) written this frame. Call **after**
    /// [`wgpu::Queue::submit`] for the command buffer that contains the Hi-Z copies.
    pub fn on_frame_submitted(&mut self, _device: &wgpu::Device) {
        let Some(ws) = self.hi_z_encoded_slot.take() else {
            return;
        };
        debug_assert!(ws < HIZ_STAGING_RING);
        let Some(scratch) = self.scratch.as_ref() else {
            return;
        };
        debug_assert!(self.desktop_pending[ws].is_none());

        let slice = scratch.staging_desktop[ws].slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.desktop_pending[ws] = Some(rx);

        if let Some(ref staging_r) = scratch.staging_r {
            if self.right_pending.is_none() {
                self.right_pending = Some(pending_none_array());
            }
            if let Some(rp) = self.right_pending.as_mut() {
                debug_assert!(rp[ws].is_none());
                let slice_r = staging_r[ws].slice(..);
                let (tx_r, rx_r) = mpsc::channel();
                slice_r.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx_r.send(r);
                });
                rp[ws] = Some(rx_r);
            }
        }

        self.write_idx = (self.write_idx + 1) % HIZ_STAGING_RING;
    }

    fn can_encode_hi_z(&self, scratch: &HiZGpuScratch) -> bool {
        let idx = self.write_idx;
        if self.desktop_pending[idx].is_some() {
            return false;
        }
        if scratch.staging_r.is_some() {
            if let Some(ref rp) = self.right_pending {
                if rp[idx].is_some() {
                    return false;
                }
            }
        }
        true
    }
}

fn read_mapped_buffer(buf: &wgpu::Buffer) -> Option<Vec<u8>> {
    let range = buf.slice(..).get_mapped_range().to_vec();
    buf.unmap();
    Some(range)
}

fn unpack_desktop_snapshot(
    extent: (u32, u32),
    mip_levels: u32,
    raw: &[u8],
) -> Option<HiZCpuSnapshot> {
    let mips = match unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, raw) {
        Some(m) => m,
        None => {
            logger::warn!("Hi-Z desktop readback unpack failed");
            return None;
        }
    };
    match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips) {
        Some(s) => Some(s),
        None => {
            logger::warn!("Hi-Z desktop snapshot validation failed");
            None
        }
    }
}

fn unpack_stereo_snapshot(
    extent: (u32, u32),
    mip_levels: u32,
    left_raw: &[u8],
    right_raw: &[u8],
) -> Option<HiZStereoCpuSnapshot> {
    let mips_l = match unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, left_raw) {
        Some(m) => m,
        None => {
            logger::warn!("Hi-Z stereo left readback unpack failed");
            return None;
        }
    };
    let left = match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips_l) {
        Some(s) => s,
        None => {
            logger::warn!("Hi-Z stereo left snapshot validation failed");
            return None;
        }
    };
    let mips_r = match unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, right_raw) {
        Some(m) => m,
        None => {
            logger::warn!("Hi-Z stereo right readback unpack failed");
            return None;
        }
    };
    let right = match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips_r) {
        Some(s) => s,
        None => {
            logger::warn!("Hi-Z stereo right snapshot validation failed");
            return None;
        }
    };
    Some(HiZStereoCpuSnapshot { left, right })
}

/// Transient GPU resources reused while extent and mip count stay stable.
struct HiZGpuScratch {
    extent: (u32, u32),
    mip_levels: u32,
    pyramid: wgpu::Texture,
    views: Vec<wgpu::TextureView>,
    pyramid_r: Option<(wgpu::Texture, Vec<wgpu::TextureView>)>,
    /// Triple-buffered staging for async readback (see [`HiZGpuState::write_idx`]).
    staging_desktop: [wgpu::Buffer; HIZ_STAGING_RING],
    staging_r: Option<[wgpu::Buffer; HIZ_STAGING_RING]>,
    layer_uniform: wgpu::Buffer,
    downsample_uniform: wgpu::Buffer,
}

struct HiZPipelines {
    mip0_desktop: wgpu::ComputePipeline,
    mip0_stereo: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    bgl_mip0_desktop: wgpu::BindGroupLayout,
    bgl_mip0_stereo: wgpu::BindGroupLayout,
    bgl_downsample: wgpu::BindGroupLayout,
}

impl HiZPipelines {
    fn get(device: &wgpu::Device) -> &'static Self {
        static CACHE: OnceLock<HiZPipelines> = OnceLock::new();
        CACHE.get_or_init(|| Self::new(device))
    }

    fn new(device: &wgpu::Device) -> Self {
        let bgl_mip0_desktop = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hi_z_mip0_desktop"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let bgl_mip0_stereo = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hi_z_mip0_stereo"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let bgl_downsample = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hi_z_downsample"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
            ],
        });

        let layout_mip0_d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_mip0_desktop_layout"),
            bind_group_layouts: &[Some(&bgl_mip0_desktop)],
            immediate_size: 0,
        });
        let layout_mip0_s = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_mip0_stereo_layout"),
            bind_group_layouts: &[Some(&bgl_mip0_stereo)],
            immediate_size: 0,
        });
        let layout_ds = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_downsample_layout"),
            bind_group_layouts: &[Some(&bgl_downsample)],
            immediate_size: 0,
        });

        let shader_m0d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_mip0_desktop"),
            source: wgpu::ShaderSource::Wgsl(MIP0_DESKTOP_SRC.into()),
        });
        let shader_m0s = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_mip0_stereo"),
            source: wgpu::ShaderSource::Wgsl(MIP0_STEREO_SRC.into()),
        });
        let shader_ds = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_downsample"),
            source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_SRC.into()),
        });

        let mip0_desktop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hi_z_mip0_desktop"),
            layout: Some(&layout_mip0_d),
            module: &shader_m0d,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mip0_stereo = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hi_z_mip0_stereo"),
            layout: Some(&layout_mip0_s),
            module: &shader_m0s,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let downsample = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hi_z_downsample"),
            layout: Some(&layout_ds),
            module: &shader_ds,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            mip0_desktop,
            mip0_stereo,
            downsample,
            bgl_mip0_desktop,
            bgl_mip0_stereo,
            bgl_downsample,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerUniform {
    layer: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DownsampleUniform {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

fn staging_size_pyramid(base_w: u32, base_h: u32, mip_levels: u32) -> u64 {
    let mut total = 0u64;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap_or((0, 0));
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as u64;
        total += row_pitch * u64::from(h);
    }
    total
}

fn make_staging_ring(
    device: &wgpu::Device,
    staging_size: u64,
    label_prefix: &str,
) -> [wgpu::Buffer; HIZ_STAGING_RING] {
    std::array::from_fn(|i| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}_{i}")),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    })
}

impl HiZGpuScratch {
    fn new(device: &wgpu::Device, extent: (u32, u32), stereo: bool) -> Option<Self> {
        let (bw, bh) = extent;
        if bw == 0 || bh == 0 {
            return None;
        }
        let mip_levels = mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS);
        if mip_levels == 0 {
            return None;
        }

        let make_pyramid = || -> (wgpu::Texture, Vec<wgpu::TextureView>) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("hi_z_pyramid"),
                size: wgpu::Extent3d {
                    width: bw,
                    height: bh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mip_levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let mut views = Vec::with_capacity(mip_levels as usize);
            for m in 0..mip_levels {
                let v = tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("hi_z_pyramid_mip"),
                    format: Some(wgpu::TextureFormat::R32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                    ..Default::default()
                });
                views.push(v);
            }
            (tex, views)
        };

        let (pyramid, views) = make_pyramid();
        let staging_size = staging_size_pyramid(bw, bh, mip_levels);
        let staging_desktop = make_staging_ring(device, staging_size, "hi_z_staging_desktop");

        let (pyramid_r, staging_r) = if stereo {
            let (t, v) = make_pyramid();
            let buf = make_staging_ring(device, staging_size, "hi_z_staging_r");
            (Some((t, v)), Some(buf))
        } else {
            (None, None)
        };

        let layer_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_layer_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let downsample_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_downsample_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            extent: (bw, bh),
            mip_levels,
            pyramid,
            views,
            pyramid_r,
            staging_desktop,
            staging_r,
            layer_uniform,
            downsample_uniform,
        })
    }
}

#[derive(Clone, Copy)]
enum DepthBinding {
    D2,
    D2Array { layer: u32 },
}

/// Records Hi-Z build + copy-to-staging into [`HiZGpuState::write_idx`].
///
/// Call [`HiZGpuState::on_frame_submitted`] after [`wgpu::Queue::submit`]. Call
/// [`HiZGpuState::begin_frame_readback`] at the **start** of the next frame to drain completed maps.
pub fn encode_hi_z_build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    depth_view: &wgpu::TextureView,
    extent: (u32, u32),
    mode: OutputDepthMode,
    state: &mut HiZGpuState,
) {
    state.hi_z_encoded_slot = None;
    state.invalidate_if_needed(extent, mode);

    let (full_w, full_h) = extent;
    if full_w == 0 || full_h == 0 {
        return;
    }

    let (bw, bh) = hi_z_pyramid_dimensions(full_w, full_h);
    if bw == 0 || bh == 0 {
        return;
    }

    let stereo = matches!(mode, OutputDepthMode::StereoArray { .. });
    let mip_levels = mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS);
    if state.scratch.as_ref().map(|s| (s.extent, s.mip_levels)) != Some(((bw, bh), mip_levels))
        || state.scratch.as_ref().map(|s| s.pyramid_r.is_some()) != Some(stereo)
    {
        state.scratch = HiZGpuScratch::new(device, (bw, bh), stereo);
        state.desktop_pending = pending_none_array();
        state.stereo_left_stash = pending_none_array();
        state.stereo_right_stash = pending_none_array();
        state.write_idx = 0;
        state.hi_z_encoded_slot = None;
        if stereo {
            state.right_pending = Some(pending_none_array());
        } else {
            state.right_pending = None;
        }
    }
    let Some(scratch_ref) = state.scratch.as_ref() else {
        return;
    };

    if stereo && state.right_pending.is_none() {
        state.right_pending = Some(pending_none_array());
    }
    if !stereo {
        state.right_pending = None;
    }

    if !state.can_encode_hi_z(scratch_ref) {
        return;
    }

    let Some(scratch) = state.scratch.as_mut() else {
        return;
    };

    let ws = state.write_idx;
    let pipes = HiZPipelines::get(device);

    let dispatch_mip0_and_downsample =
        |encoder: &mut wgpu::CommandEncoder,
         pyramid_views: &[wgpu::TextureView],
         depth_bind: DepthBinding| {
            match depth_bind {
                DepthBinding::D2 => {
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hi_z_mip0_d_bg"),
                        layout: &pipes.bgl_mip0_desktop,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&pyramid_views[0]),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hi_z_mip0_desktop"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipes.mip0_desktop);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            scratch.extent.0.div_ceil(8),
                            scratch.extent.1.div_ceil(8),
                            1,
                        );
                    }
                }
                DepthBinding::D2Array { layer } => {
                    let layer_u = LayerUniform {
                        layer,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    queue.write_buffer(&scratch.layer_uniform, 0, bytemuck::bytes_of(&layer_u));
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hi_z_mip0_s_bg"),
                        layout: &pipes.bgl_mip0_stereo,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: scratch.layer_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&pyramid_views[0]),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hi_z_mip0_stereo"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipes.mip0_stereo);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            scratch.extent.0.div_ceil(8),
                            scratch.extent.1.div_ceil(8),
                            1,
                        );
                    }
                }
            }

            for mip in 0..scratch.mip_levels.saturating_sub(1) {
                let (sw, sh) = mip_dimensions(bw, bh, mip).unwrap_or((1, 1));
                let (dw, dh) = mip_dimensions(bw, bh, mip + 1).unwrap_or((1, 1));
                let du = DownsampleUniform {
                    src_w: sw,
                    src_h: sh,
                    dst_w: dw,
                    dst_h: dh,
                };
                queue.write_buffer(&scratch.downsample_uniform, 0, bytemuck::bytes_of(&du));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hi_z_ds_bg"),
                    layout: &pipes.bgl_downsample,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &pyramid_views[mip as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &pyramid_views[mip as usize + 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: scratch.downsample_uniform.as_entire_binding(),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("hi_z_downsample"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipes.downsample);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
                }
            }
        };

    match mode {
        OutputDepthMode::DesktopSingle => {
            dispatch_mip0_and_downsample(encoder, &scratch.views, DepthBinding::D2);
            copy_pyramid_to_staging(
                encoder,
                &scratch.pyramid,
                bw,
                bh,
                scratch.mip_levels,
                &scratch.staging_desktop[ws],
            );
        }
        OutputDepthMode::StereoArray { .. } => {
            let Some((ref pyr_r, ref views_r)) = scratch.pyramid_r else {
                return;
            };
            dispatch_mip0_and_downsample(
                encoder,
                &scratch.views,
                DepthBinding::D2Array { layer: 0 },
            );
            dispatch_mip0_and_downsample(encoder, views_r, DepthBinding::D2Array { layer: 1 });
            copy_pyramid_to_staging(
                encoder,
                &scratch.pyramid,
                bw,
                bh,
                scratch.mip_levels,
                &scratch.staging_desktop[ws],
            );
            copy_pyramid_to_staging(
                encoder,
                pyr_r,
                bw,
                bh,
                scratch.mip_levels,
                &scratch.staging_r.as_ref().expect("stereo staging")[ws],
            );
        }
    }

    state.hi_z_encoded_slot = Some(ws);
}

fn copy_pyramid_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    base_w: u32,
    base_h: u32,
    mip_levels: u32,
    staging: &wgpu::Buffer,
) {
    let mut offset = 0u64;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap_or((1, 1));
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as u32;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        offset += u64::from(row_pitch) * u64::from(h);
    }
}
