//! Per-frame `@group(0)` resources: scene uniform, lights storage, clustered light buffers.

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::backend::cluster_gpu::{ClusterBufferCache, ClusterBufferRefs, CLUSTER_COUNT_Z};
use crate::backend::light_gpu::{GpuLight, MAX_LIGHTS};
use crate::gpu::frame_globals::FrameGpuUniforms;

/// GPU buffers and bind group for [`FrameGpuUniforms`], [`GpuLight`] storage, and cluster lists.
pub struct FrameGpuResources {
    /// Uniform buffer for [`FrameGpuUniforms`].
    pub frame_uniform: wgpu::Buffer,
    /// Storage buffer holding up to [`MAX_LIGHTS`] [`GpuLight`] records.
    pub lights_buffer: wgpu::Buffer,
    /// Cluster buffers and compute params; resized with viewport ([`Self::sync_cluster_viewport`]).
    pub cluster_cache: ClusterBufferCache,
    /// Sampled single-view scene depth snapshot for materials that need `_CameraDepthTexture`.
    scene_depth_2d: (wgpu::Texture, wgpu::TextureView),
    scene_depth_2d_extent_px: (u32, u32),
    /// Sampled multiview scene depth snapshot (`D2Array`, 2 layers).
    scene_depth_array: (wgpu::Texture, wgpu::TextureView),
    scene_depth_array_extent_px: (u32, u32),
    /// Bind group for `@group(0)` in composed mesh shaders.
    pub bind_group: Arc<wgpu::BindGroup>,
    cluster_bind_version: u64,
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
            ],
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        frame_uniform: &wgpu::Buffer,
        lights_buffer: &wgpu::Buffer,
        refs: ClusterBufferRefs<'_>,
        scene_depth_2d: &wgpu::TextureView,
        scene_depth_array: &wgpu::TextureView,
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
                    resource: wgpu::BindingResource::TextureView(scene_depth_2d),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(scene_depth_array),
                },
            ],
        }))
    }

    fn create_depth_snapshot_2d(
        device: &wgpu::Device,
        extent_px: (u32, u32),
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
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("frame_scene_depth_2d_view"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });
        (tex, view)
    }

    fn create_depth_snapshot_array(
        device: &wgpu::Device,
        extent_px: (u32, u32),
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
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("frame_scene_depth_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(2),
            ..Default::default()
        });
        (tex, view)
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device, viewport: (u32, u32), stereo: bool) {
        let Some(refs) = self
            .cluster_cache
            .get_buffers(viewport, CLUSTER_COUNT_Z, stereo)
        else {
            logger::warn!(
                "FrameGpu: cluster buffers missing for viewport {:?} stereo={}; skipping bind group rebuild",
                viewport,
                stereo
            );
            return;
        };
        self.bind_group = Self::create_bind_group(
            device,
            &self.frame_uniform,
            &self.lights_buffer,
            refs,
            &self.scene_depth_2d.1,
            &self.scene_depth_array.1,
        );
    }

    fn ensure_scene_depth_2d(&mut self, device: &wgpu::Device, extent_px: (u32, u32)) {
        let want = (extent_px.0.max(1), extent_px.1.max(1));
        if self.scene_depth_2d_extent_px == want {
            return;
        }
        self.scene_depth_2d = Self::create_depth_snapshot_2d(device, want);
        self.scene_depth_2d_extent_px = want;
    }

    fn ensure_scene_depth_array(&mut self, device: &wgpu::Device, extent_px: (u32, u32)) {
        let want = (extent_px.0.max(1), extent_px.1.max(1));
        if self.scene_depth_array_extent_px == want {
            return;
        }
        self.scene_depth_array = Self::create_depth_snapshot_array(device, want);
        self.scene_depth_array_extent_px = want;
    }

    /// Allocates frame uniform, lights storage, minimal cluster grid `(1×1×Z)`; builds [`Self::bind_group`].
    ///
    /// Returns an error when the initial cluster buffer cache could not be populated (zero viewport or internal mismatch).
    pub fn new(device: &wgpu::Device) -> Result<Self, String> {
        let frame_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_globals_uniform"),
            size: std::mem::size_of::<FrameGpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_size = (MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as u64;
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_lights_storage"),
            size: lights_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cluster_cache = ClusterBufferCache::new();
        cluster_cache
            .ensure_buffers(device, (1, 1), CLUSTER_COUNT_Z, false)
            .ok_or_else(|| "cluster buffers: ensure_buffers failed for 1x1 viewport".to_string())?;
        let cluster_bind_version = cluster_cache.version;
        let refs = cluster_cache
            .get_buffers((1, 1), CLUSTER_COUNT_Z, false)
            .ok_or_else(|| "cluster buffers: get_buffers failed for 1x1 viewport".to_string())?;
        let scene_depth_2d = Self::create_depth_snapshot_2d(device, (1, 1));
        let scene_depth_array = Self::create_depth_snapshot_array(device, (1, 1));
        let bind_group = Self::create_bind_group(
            device,
            &frame_uniform,
            &lights_buffer,
            refs,
            &scene_depth_2d.1,
            &scene_depth_array.1,
        );
        Ok(Self {
            frame_uniform,
            lights_buffer,
            cluster_cache,
            scene_depth_2d,
            scene_depth_2d_extent_px: (1, 1),
            scene_depth_array,
            scene_depth_array_extent_px: (1, 1),
            bind_group,
            cluster_bind_version,
        })
    }

    /// Resizes cluster buffers when `viewport` or `stereo` changes; rebuilds [`Self::bind_group`].
    ///
    /// When `stereo` is true, cluster count/index buffers are doubled for per-eye storage.
    /// Returns `true` if the bind group was recreated.
    pub fn sync_cluster_viewport(
        &mut self,
        device: &wgpu::Device,
        viewport: (u32, u32),
        stereo: bool,
    ) -> bool {
        if self
            .cluster_cache
            .ensure_buffers(device, viewport, CLUSTER_COUNT_Z, stereo)
            .is_none()
        {
            return false;
        }
        let ver = self.cluster_cache.version;
        if ver == self.cluster_bind_version {
            return false;
        }
        self.rebuild_bind_group(device, viewport, stereo);
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
        stereo_cluster: bool,
    ) {
        let width = viewport.0.max(1);
        let height = viewport.1.max(1);
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: if multiview { 2 } else { 1 },
        };
        if multiview {
            self.ensure_scene_depth_array(device, (width, height));
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_depth,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.scene_depth_array.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                extent,
            );
        } else {
            self.ensure_scene_depth_2d(device, (width, height));
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: source_depth,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.scene_depth_2d.0,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                extent,
            );
        }
        self.rebuild_bind_group(device, viewport, stereo_cluster);
    }

    /// Uploads [`FrameGpuUniforms`] and packed lights for this frame.
    pub fn write_frame_uniform_and_lights(
        &self,
        queue: &wgpu::Queue,
        uniforms: &FrameGpuUniforms,
        lights: &[GpuLight],
    ) {
        queue.write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(uniforms));
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
            queue.write_buffer(lights_buffer, 0, &[0u8; 4]);
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
    pub layout: wgpu::BindGroupLayout,
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
