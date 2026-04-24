//! Per-frame `@group(0)` resources: scene uniform, lights storage, shared cluster buffers, and
//! scene snapshot textures.
//!
//! Cluster buffers ([`ClusterBufferCache`]) and the `@group(0)` layout live here and are
//! **shared across every view**; per-view uniform buffers and bind groups live in
//! [`crate::backend::frame_resource_manager::PerViewFrameState`] and reference these shared
//! cluster buffers (safe under single-submit ordering — see [`ClusterBufferCache`]).

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::backend::cluster_gpu::{ClusterBufferCache, ClusterBufferRefs, CLUSTER_COUNT_Z};
use crate::backend::light_gpu::{GpuLight, MAX_LIGHTS};
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::GpuLimits;

use super::frame_gpu_error::FrameGpuInitError;

/// GPU buffers and bind groups for `@group(0)` frame globals (camera, lights, cluster lists,
/// and sampled scene snapshots).
///
/// `@group(0)` bind groups are per-view and are owned by
/// [`crate::backend::frame_resource_manager::PerViewFrameState`], keyed by
/// [`crate::render_graph::OcclusionViewId`], and built using
/// [`Self::build_per_view_bind_group`]. Every per-view bind group references the **same**
/// shared cluster buffers from [`Self::cluster_cache`].
pub struct FrameGpuResources {
    /// Uniform buffer for [`FrameGpuUniforms`] (global fallback; per-view uniforms are in
    /// [`crate::backend::frame_resource_manager::PerViewFrameState`]).
    pub frame_uniform: wgpu::Buffer,
    /// Storage buffer holding up to [`MAX_LIGHTS`] [`GpuLight`] records (scene-global; shared
    /// across all views).
    pub lights_buffer: wgpu::Buffer,
    /// Shared cluster buffers for the whole frame; every view's `@group(0)` bind group
    /// references this one cache (see [`ClusterBufferCache`] for the ordering argument that
    /// makes sharing safe under single-submit semantics).
    pub cluster_cache: ClusterBufferCache,
    /// Sampled single-view scene depth snapshot for materials that need `_CameraDepthTexture`.
    scene_depth_2d: (wgpu::Texture, wgpu::TextureView),
    scene_depth_2d_extent_px: (u32, u32),
    scene_depth_2d_format: wgpu::TextureFormat,
    /// Sampled multiview scene depth snapshot (`D2Array`, 2 layers).
    scene_depth_array: (wgpu::Texture, wgpu::TextureView),
    scene_depth_array_extent_px: (u32, u32),
    scene_depth_array_format: wgpu::TextureFormat,
    /// Sampled single-view scene color snapshot (grab pass) for materials like `filters_blur_perobject`.
    scene_color_2d: (wgpu::Texture, wgpu::TextureView),
    scene_color_2d_extent_px: (u32, u32),
    scene_color_2d_format: wgpu::TextureFormat,
    /// Sampled multiview scene color snapshot (`D2Array`, 2 layers).
    scene_color_array: (wgpu::Texture, wgpu::TextureView),
    scene_color_array_extent_px: (u32, u32),
    scene_color_array_format: wgpu::TextureFormat,
    /// Shared linear-clamp sampler for sampling [`Self::scene_color_2d`] / [`Self::scene_color_array`].
    scene_color_sampler: wgpu::Sampler,
    /// Global `@group(0)` bind group (global frame uniform + shared lights/snapshots).
    ///
    /// Per-view passes bind the per-view bind group from
    /// [`crate::backend::frame_resource_manager::PerViewFrameState`] instead.
    pub bind_group: Arc<wgpu::BindGroup>,
    cluster_bind_version: u64,
    limits: Arc<GpuLimits>,
    /// Monotonically increasing counter; incremented each time a scene snapshot texture is
    /// recreated due to a size or format change.
    ///
    /// [`crate::backend::frame_resource_manager::PerViewFrameState`] tracks the version at which
    /// it last rebuilt its `@group(0)` bind group and rebuilds when this diverges.
    pub(super) snapshot_version: u64,
}

/// Default scene color snapshot format used when no grab pass has been triggered yet.
const DEFAULT_SCENE_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Sampled scene depth/color snapshot views and sampler for [`FrameGpuResources`] `@group(0)` bindings 4–8.
pub struct FrameSceneSnapshotTextureViews<'a> {
    /// Single-view depth snapshot (`binding(4)`).
    pub scene_depth_2d: &'a wgpu::TextureView,
    /// Multiview depth snapshot (`binding(5)`).
    pub scene_depth_array: &'a wgpu::TextureView,
    /// Single-view color snapshot (`binding(6)`).
    pub scene_color_2d: &'a wgpu::TextureView,
    /// Multiview color snapshot (`binding(7)`).
    pub scene_color_array: &'a wgpu::TextureView,
    /// Shared sampler for scene color (`binding(8)`).
    pub scene_color_sampler: &'a wgpu::Sampler,
}

/// Viewport and view layout for [`FrameGpuResources::copy_scene_color_snapshot`].
pub struct SceneColorSnapshotCopyParams {
    /// Extent in pixels used for snapshot textures and copy.
    pub viewport: (u32, u32),
    /// When true, copy into the `D2Array` scene color snapshot (two layers).
    pub multiview: bool,
    /// Cluster buffer stereo layout passed to [`FrameGpuResources::rebuild_bind_group`].
    pub stereo_cluster: bool,
}

/// [`wgpu::TextureAspect`] for [`wgpu::CommandEncoder::copy_texture_to_texture`] when copying depth
/// textures.
///
/// Combined depth-stencil formats require [`wgpu::TextureAspect::All`] on both source and destination
/// so the copy covers the full plane (wgpu: copy source aspects must refer to all aspects of the
/// source texture format).
fn depth_copy_aspect_for_texture_to_texture(format: wgpu::TextureFormat) -> wgpu::TextureAspect {
    if format.has_stencil_aspect() {
        wgpu::TextureAspect::All
    } else {
        wgpu::TextureAspect::DepthOnly
    }
}

impl FrameGpuResources {
    /// Layout for `@group(0)`: uniform frame + lights + cluster counts + cluster indices +
    /// single-view / multiview scene depth snapshots.
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame_globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<FrameGpuUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuLight>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        frame_uniform: &wgpu::Buffer,
        lights_buffer: &wgpu::Buffer,
        refs: ClusterBufferRefs<'_>,
        snapshots: FrameSceneSnapshotTextureViews<'_>,
    ) -> Arc<wgpu::BindGroup> {
        let layout = Self::bind_group_layout(device);
        Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame_globals_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: refs.cluster_light_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: refs.cluster_light_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(snapshots.scene_depth_2d),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(snapshots.scene_depth_array),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(snapshots.scene_color_2d),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(snapshots.scene_color_array),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(snapshots.scene_color_sampler),
                },
            ],
        }))
    }

    fn create_depth_snapshot_2d(
        device: &wgpu::Device,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame_scene_depth_2d"),
            size: wgpu::Extent3d {
                width: extent_px.0.max(1),
                height: extent_px.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("frame_scene_depth_2d_view"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        (tex, view)
    }

    fn create_depth_snapshot_array(
        device: &wgpu::Device,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame_scene_depth_array"),
            size: wgpu::Extent3d {
                width: extent_px.0.max(1),
                height: extent_px.1.max(1),
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("frame_scene_depth_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        (tex, view)
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        let Some(refs) = self.cluster_cache.current_refs() else {
            logger::warn!("FrameGpu: cluster buffers missing; skipping bind group rebuild");
            return;
        };
        self.bind_group = Self::create_bind_group(
            device,
            &self.frame_uniform,
            &self.lights_buffer,
            refs,
            FrameSceneSnapshotTextureViews {
                scene_depth_2d: &self.scene_depth_2d.1,
                scene_depth_array: &self.scene_depth_array.1,
                scene_color_2d: &self.scene_color_2d.1,
                scene_color_array: &self.scene_color_array.1,
                scene_color_sampler: &self.scene_color_sampler,
            },
        );
    }

    fn ensure_scene_depth_2d(
        &mut self,
        device: &wgpu::Device,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) {
        let want = (extent_px.0.max(1), extent_px.1.max(1));
        let max_dim = self.limits.max_texture_dimension_2d();
        if want.0 > max_dim || want.1 > max_dim {
            logger::warn!(
                "scene depth 2d snapshot: extent {}×{} exceeds max_texture_dimension_2d ({max_dim}); keeping previous texture",
                want.0,
                want.1
            );
            return;
        }
        if self.scene_depth_2d_extent_px == want && self.scene_depth_2d_format == format {
            return;
        }
        self.scene_depth_2d = Self::create_depth_snapshot_2d(device, want, format);
        self.scene_depth_2d_extent_px = want;
        self.scene_depth_2d_format = format;
        self.snapshot_version = self.snapshot_version.wrapping_add(1);
    }

    fn ensure_scene_depth_array(
        &mut self,
        device: &wgpu::Device,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) {
        let want = (extent_px.0.max(1), extent_px.1.max(1));
        let max_dim = self.limits.max_texture_dimension_2d();
        if want.0 > max_dim || want.1 > max_dim {
            logger::warn!(
                "scene depth array snapshot: extent {}×{} exceeds max_texture_dimension_2d ({max_dim}); keeping previous texture",
                want.0,
                want.1
            );
            return;
        }
        if self.scene_depth_array_extent_px == want && self.scene_depth_array_format == format {
            return;
        }
        self.scene_depth_array = Self::create_depth_snapshot_array(device, want, format);
        self.scene_depth_array_extent_px = want;
        self.scene_depth_array_format = format;
        self.snapshot_version = self.snapshot_version.wrapping_add(1);
    }

    fn create_color_snapshot_2d(
        device: &wgpu::Device,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame_scene_color_2d"),
            size: wgpu::Extent3d {
                width: extent_px.0.max(1),
                height: extent_px.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("frame_scene_color_2d_view"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });
        (tex, view)
    }

    fn create_color_snapshot_array(
        device: &wgpu::Device,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame_scene_color_array"),
            size: wgpu::Extent3d {
                width: extent_px.0.max(1),
                height: extent_px.1.max(1),
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("frame_scene_color_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            ..Default::default()
        });
        (tex, view)
    }

    fn ensure_scene_color_2d(
        &mut self,
        device: &wgpu::Device,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) {
        let want = (extent_px.0.max(1), extent_px.1.max(1));
        let max_dim = self.limits.max_texture_dimension_2d();
        if want.0 > max_dim || want.1 > max_dim {
            logger::warn!(
                "scene color 2d snapshot: extent {}×{} exceeds max_texture_dimension_2d ({max_dim}); keeping previous texture",
                want.0,
                want.1
            );
            return;
        }
        if self.scene_color_2d_extent_px == want && self.scene_color_2d_format == format {
            return;
        }
        self.scene_color_2d = Self::create_color_snapshot_2d(device, want, format);
        self.scene_color_2d_extent_px = want;
        self.scene_color_2d_format = format;
        self.snapshot_version = self.snapshot_version.wrapping_add(1);
    }

    fn ensure_scene_color_array(
        &mut self,
        device: &wgpu::Device,
        extent_px: (u32, u32),
        format: wgpu::TextureFormat,
    ) {
        let want = (extent_px.0.max(1), extent_px.1.max(1));
        let max_dim = self.limits.max_texture_dimension_2d();
        if want.0 > max_dim || want.1 > max_dim {
            logger::warn!(
                "scene color array snapshot: extent {}×{} exceeds max_texture_dimension_2d ({max_dim}); keeping previous texture",
                want.0,
                want.1
            );
            return;
        }
        if self.scene_color_array_extent_px == want && self.scene_color_array_format == format {
            return;
        }
        self.scene_color_array = Self::create_color_snapshot_array(device, want, format);
        self.scene_color_array_extent_px = want;
        self.scene_color_array_format = format;
        self.snapshot_version = self.snapshot_version.wrapping_add(1);
    }

    /// Allocates frame uniform, lights storage, minimal cluster grid `(1×1×Z)`; builds [`Self::bind_group`].
    ///
    /// Returns an error when the initial cluster buffer cache could not be populated (zero viewport or internal mismatch).
    pub fn new(device: &wgpu::Device, limits: Arc<GpuLimits>) -> Result<Self, FrameGpuInitError> {
        let lights_size = (MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as u64;
        if lights_size > limits.max_storage_buffer_binding_size()
            || lights_size > limits.max_buffer_size()
        {
            return Err(FrameGpuInitError::LightsStorageExceedsLimits { size: lights_size });
        }
        let frame_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_globals_uniform"),
            size: std::mem::size_of::<FrameGpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_lights_storage"),
            size: lights_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cluster_cache = ClusterBufferCache::new();
        cluster_cache
            .ensure_buffers(device, limits.as_ref(), (1, 1), CLUSTER_COUNT_Z, false)
            .ok_or(FrameGpuInitError::ClusterEnsureFailed)?;
        let cluster_bind_version = cluster_cache.version;
        let refs = cluster_cache
            .current_refs()
            .ok_or(FrameGpuInitError::ClusterGetBuffersFailed)?;
        let scene_depth_format =
            crate::render_graph::main_forward_depth_stencil_format(device.features());
        let scene_depth_2d = Self::create_depth_snapshot_2d(device, (1, 1), scene_depth_format);
        let scene_depth_array =
            Self::create_depth_snapshot_array(device, (1, 1), scene_depth_format);
        let scene_color_2d =
            Self::create_color_snapshot_2d(device, (1, 1), DEFAULT_SCENE_COLOR_FORMAT);
        let scene_color_array =
            Self::create_color_snapshot_array(device, (1, 1), DEFAULT_SCENE_COLOR_FORMAT);
        let scene_color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("frame_scene_color_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = Self::create_bind_group(
            device,
            &frame_uniform,
            &lights_buffer,
            refs,
            FrameSceneSnapshotTextureViews {
                scene_depth_2d: &scene_depth_2d.1,
                scene_depth_array: &scene_depth_array.1,
                scene_color_2d: &scene_color_2d.1,
                scene_color_array: &scene_color_array.1,
                scene_color_sampler: &scene_color_sampler,
            },
        );
        Ok(Self {
            frame_uniform,
            lights_buffer,
            cluster_cache,
            scene_depth_2d,
            scene_depth_2d_extent_px: (1, 1),
            scene_depth_2d_format: scene_depth_format,
            scene_depth_array,
            scene_depth_array_extent_px: (1, 1),
            scene_depth_array_format: scene_depth_format,
            scene_color_2d,
            scene_color_2d_extent_px: (1, 1),
            scene_color_2d_format: DEFAULT_SCENE_COLOR_FORMAT,
            scene_color_array,
            scene_color_array_extent_px: (1, 1),
            scene_color_array_format: DEFAULT_SCENE_COLOR_FORMAT,
            scene_color_sampler,
            bind_group,
            cluster_bind_version,
            limits,
            snapshot_version: 0,
        })
    }

    /// Grows the shared cluster cache to cover `viewport` × `stereo` if needed; rebuilds
    /// [`Self::bind_group`] when the underlying buffers were reallocated.
    ///
    /// When `stereo` is true, cluster count/index buffers are doubled for per-eye storage.
    /// Returns `true` if the bind group was recreated.
    ///
    /// Because the shared cache is grow-only (see [`ClusterBufferCache`]), calling this with
    /// a smaller viewport than a previous call is a no-op.
    pub fn sync_cluster_viewport(
        &mut self,
        device: &wgpu::Device,
        viewport: (u32, u32),
        stereo: bool,
    ) -> bool {
        profiling::scope!("render::sync_cluster_viewport");
        if self
            .cluster_cache
            .ensure_buffers(
                device,
                self.limits.as_ref(),
                viewport,
                CLUSTER_COUNT_Z,
                stereo,
            )
            .is_none()
        {
            return false;
        }
        let ver = self.cluster_cache.version;
        if ver == self.cluster_bind_version {
            return false;
        }
        self.rebuild_bind_group(device);
        self.cluster_bind_version = ver;
        true
    }

    /// Copies the main depth attachment into the sampled scene-depth snapshot used by embedded
    /// materials such as `pbsintersectspecular`, then rebuilds [`Self::bind_group`] so `@group(0)`
    /// points at the updated texture view.
    pub fn copy_scene_depth_snapshot(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
        _stereo_cluster: bool,
    ) {
        let width = viewport.0.max(1);
        let height = viewport.1.max(1);
        let max_dim = self.limits.max_texture_dimension_2d();
        if width > max_dim || height > max_dim {
            logger::warn!(
                "copy_scene_depth_snapshot: viewport {}×{} exceeds max_texture_dimension_2d ({max_dim}); skipping copy",
                width,
                height
            );
            return;
        }
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: if multiview { 2 } else { 1 },
        };
        let format = source_depth.format();
        let copy_aspect = depth_copy_aspect_for_texture_to_texture(format);
        if multiview {
            self.ensure_scene_depth_array(device, (width, height), format);
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_depth,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: copy_aspect,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.scene_depth_array.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: copy_aspect,
                },
                extent,
            );
        } else {
            self.ensure_scene_depth_2d(device, (width, height), format);
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_depth,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: copy_aspect,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.scene_depth_2d.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: copy_aspect,
                },
                extent,
            );
        }
        self.rebuild_bind_group(device);
    }

    /// Copies the main depth attachment into an already provisioned scene-depth snapshot without
    /// rebuilding any bind groups.
    ///
    /// Call this after [`Self::sync_cluster_viewport`] has already ensured the snapshot texture
    /// exists for the target `viewport` / `multiview` layout. This keeps per-view recording free of
    /// shared bind-group mutation while still encoding the per-view depth copy.
    pub fn encode_scene_depth_snapshot_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) {
        let width = viewport.0.max(1);
        let height = viewport.1.max(1);
        let format = source_depth.format();
        let copy_aspect = depth_copy_aspect_for_texture_to_texture(format);
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: if multiview { 2 } else { 1 },
        };

        if multiview {
            if self.scene_depth_array_extent_px != (width, height)
                || self.scene_depth_array_format != format
            {
                logger::warn!(
                    "scene depth snapshot copy: array target not pre-synced for {}×{} {:?}; skipping copy",
                    width,
                    height,
                    format
                );
                return;
            }
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_depth,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: copy_aspect,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.scene_depth_array.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: copy_aspect,
                },
                extent,
            );
        } else {
            if self.scene_depth_2d_extent_px != (width, height)
                || self.scene_depth_2d_format != format
            {
                logger::warn!(
                    "scene depth snapshot copy: 2d target not pre-synced for {}×{} {:?}; skipping copy",
                    width,
                    height,
                    format
                );
                return;
            }
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_depth,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: copy_aspect,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.scene_depth_2d.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: copy_aspect,
                },
                extent,
            );
        }
    }

    /// Copies the main color attachment into the sampled scene-color snapshot used by grab-pass
    /// materials such as `filters_blur_perobject`, then rebuilds [`Self::bind_group`] so `@group(0)`
    /// points at the updated texture view.
    pub fn copy_scene_color_snapshot(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        params: SceneColorSnapshotCopyParams,
    ) {
        let SceneColorSnapshotCopyParams {
            viewport,
            multiview,
            stereo_cluster: _stereo_cluster,
        } = params;
        let width = viewport.0.max(1);
        let height = viewport.1.max(1);
        let max_dim = self.limits.max_texture_dimension_2d();
        if width > max_dim || height > max_dim {
            logger::warn!(
                "copy_scene_color_snapshot: viewport {}×{} exceeds max_texture_dimension_2d ({max_dim}); skipping copy",
                width,
                height
            );
            return;
        }
        let format = source_color.format();
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: if multiview { 2 } else { 1 },
        };
        if multiview {
            self.ensure_scene_color_array(device, (width, height), format);
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.scene_color_array.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                extent,
            );
        } else {
            self.ensure_scene_color_2d(device, (width, height), format);
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.scene_color_2d.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                extent,
            );
        }
        self.rebuild_bind_group(device);
    }

    /// Builds a per-view `@group(0)` bind group using this view's own `frame_uniform` and
    /// `cluster_refs`, but sharing lights storage and scene snapshot textures from [`Self`].
    ///
    /// Called by [`crate::backend::frame_resource_manager::PerViewFrameState`] whenever the view's
    /// cluster buffers or snapshot textures change.
    pub(super) fn build_per_view_bind_group(
        &self,
        device: &wgpu::Device,
        frame_uniform: &wgpu::Buffer,
        cluster_refs: ClusterBufferRefs<'_>,
    ) -> Arc<wgpu::BindGroup> {
        Self::create_bind_group(
            device,
            frame_uniform,
            &self.lights_buffer,
            cluster_refs,
            FrameSceneSnapshotTextureViews {
                scene_depth_2d: &self.scene_depth_2d.1,
                scene_depth_array: &self.scene_depth_array.1,
                scene_color_2d: &self.scene_color_2d.1,
                scene_color_array: &self.scene_color_array.1,
                scene_color_sampler: &self.scene_color_sampler,
            },
        )
    }

    /// Uploads [`FrameGpuUniforms`] only (packed lights unchanged).
    pub fn write_frame_uniform(&self, queue: &wgpu::Queue, uniforms: &FrameGpuUniforms) {
        queue.write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(uniforms));
    }

    /// Uploads [`FrameGpuUniforms`] and packed lights for this frame.
    pub fn write_frame_uniform_and_lights(
        &self,
        queue: &wgpu::Queue,
        uniforms: &FrameGpuUniforms,
        lights: &[GpuLight],
    ) {
        self.write_frame_uniform(queue, uniforms);
        Self::write_lights_buffer_inner(queue, &self.lights_buffer, lights);
    }

    /// Uploads only the lights storage buffer (used by [`crate::render_graph::passes::ClusteredLightPass`]).
    pub fn write_lights_buffer(&self, queue: &wgpu::Queue, lights: &[GpuLight]) {
        Self::write_lights_buffer_inner(queue, &self.lights_buffer, lights);
    }

    fn write_lights_buffer_inner(
        queue: &wgpu::Queue,
        lights_buffer: &wgpu::Buffer,
        lights: &[GpuLight],
    ) {
        let n = lights.len().min(MAX_LIGHTS);
        if n > 0 {
            let bytes = bytemuck::cast_slice(&lights[..n]);
            queue.write_buffer(lights_buffer, 0, bytes);
        } else {
            let zero = [0u8; std::mem::size_of::<GpuLight>()];
            queue.write_buffer(lights_buffer, 0, &zero);
        }
    }
}

/// Empty `@group(1)` layout for materials that declare no per-material bindings yet.
pub fn empty_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty_material_slot"),
        entries: &[],
    })
}

/// Single reusable empty bind group for [`empty_material_bind_group_layout`].
pub fn empty_material_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("empty_material_bind_group"),
        layout,
        entries: &[],
    })
}

/// Cached empty material bind group layout + instance (one per device attach).
pub struct EmptyMaterialBindGroup {
    /// Shared layout for the empty `@group(1)` placeholder.
    pub layout: wgpu::BindGroupLayout,
    /// Bind group with no entries (material slot unused).
    pub bind_group: Arc<wgpu::BindGroup>,
}

impl EmptyMaterialBindGroup {
    /// Builds layout and bind group for `@group(1)` placeholder.
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = empty_material_bind_group_layout(device);
        let bind_group = Arc::new(empty_material_bind_group(device, &layout));
        Self { layout, bind_group }
    }
}
