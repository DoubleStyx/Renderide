//! Per-frame `@group(0)` resources: scene uniform, lights storage, shared cluster buffers, and
//! scene snapshot textures.
//!
//! Cluster buffers ([`ClusterBufferCache`]) and the `@group(0)` layout live here and are
//! **shared across every view**; per-view uniform buffers and bind groups live in
//! [`crate::backend::frame_resource_manager::PerViewFrameState`] and reference these shared
//! cluster buffers (safe under single-submit ordering — see [`ClusterBufferCache`]).

mod empty_material;
mod scene_snapshot;

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::backend::cluster_gpu::{ClusterBufferCache, ClusterBufferRefs, CLUSTER_COUNT_Z};
use crate::backend::light_gpu::{GpuLight, MAX_LIGHTS};
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::GpuLimits;

use super::frame_gpu_error::FrameGpuInitError;
pub use empty_material::{empty_material_bind_group_layout, EmptyMaterialBindGroup};
pub use scene_snapshot::FrameSceneSnapshotTextureViews;
use scene_snapshot::{
    SceneSnapshotKind, SceneSnapshotLayout, SceneSnapshotSet, DEFAULT_SCENE_COLOR_FORMAT,
};

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
    /// Scene depth/color snapshots sampled by embedded material shaders.
    scene_snapshots: SceneSnapshotSet,
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

/// Viewport and view layout for [`FrameGpuResources::copy_scene_color_snapshot`].
pub struct SceneColorSnapshotCopyParams {
    /// Extent in pixels used for snapshot textures and copy.
    pub viewport: (u32, u32),
    /// When true, copy into the `D2Array` scene color snapshot (two layers).
    pub multiview: bool,
    /// Cluster buffer stereo layout passed to [`FrameGpuResources::rebuild_bind_group`].
    pub stereo_cluster: bool,
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
            self.scene_snapshots.views(),
        );
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
        let scene_snapshots =
            SceneSnapshotSet::new(device, scene_depth_format, DEFAULT_SCENE_COLOR_FORMAT);
        let bind_group = Self::create_bind_group(
            device,
            &frame_uniform,
            &lights_buffer,
            refs,
            scene_snapshots.views(),
        );
        Ok(Self {
            frame_uniform,
            lights_buffer,
            cluster_cache,
            scene_snapshots,
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

    /// Ensures sampled scene snapshot textures exist for the active view layout and formats.
    ///
    /// This must run before per-view `@group(0)` bind groups are created for graph recording: the
    /// graph copy passes encode with `&self` and therefore cannot recreate texture views while
    /// recording. Returns `true` when the shared frame bind group was rebuilt.
    pub fn sync_scene_snapshot_textures(
        &mut self,
        device: &wgpu::Device,
        viewport: (u32, u32),
        depth_format: wgpu::TextureFormat,
        color_format: wgpu::TextureFormat,
        multiview: bool,
    ) -> bool {
        let layout = SceneSnapshotLayout::from_multiview(multiview);
        let depth_changed = self.scene_snapshots.ensure(
            device,
            self.limits.as_ref(),
            SceneSnapshotKind::Depth,
            layout,
            viewport,
            depth_format,
        );
        let color_changed = self.scene_snapshots.ensure(
            device,
            self.limits.as_ref(),
            SceneSnapshotKind::Color,
            layout,
            viewport,
            color_format,
        );
        if !(depth_changed || color_changed) {
            return false;
        }
        self.snapshot_version = self.snapshot_version.wrapping_add(1);
        self.rebuild_bind_group(device);
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
                "copy_scene_depth_snapshot: viewport {}x{} exceeds max_texture_dimension_2d ({max_dim}); skipping copy",
                width,
                height
            );
            return;
        }
        let format = source_depth.format();
        let layout = SceneSnapshotLayout::from_multiview(multiview);
        let changed = self.scene_snapshots.ensure(
            device,
            self.limits.as_ref(),
            SceneSnapshotKind::Depth,
            layout,
            (width, height),
            format,
        );
        self.scene_snapshots.encode_copy(
            encoder,
            source_depth,
            SceneSnapshotKind::Depth,
            layout,
            (width, height),
        );
        if changed {
            self.snapshot_version = self.snapshot_version.wrapping_add(1);
            self.rebuild_bind_group(device);
        }
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
        self.scene_snapshots.encode_copy(
            encoder,
            source_depth,
            SceneSnapshotKind::Depth,
            SceneSnapshotLayout::from_multiview(multiview),
            (width, height),
        );
    }

    /// Copies the main color attachment into an already provisioned scene-color snapshot without
    /// rebuilding any bind groups.
    ///
    /// Call this after [`Self::sync_scene_snapshot_textures`] has already ensured the snapshot
    /// texture exists for the target `viewport` / `multiview` layout. This keeps graph recording
    /// free of shared bind-group mutation while still encoding the per-view grab-pass copy.
    pub fn encode_scene_color_snapshot_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) {
        let width = viewport.0.max(1);
        let height = viewport.1.max(1);
        self.scene_snapshots.encode_copy(
            encoder,
            source_color,
            SceneSnapshotKind::Color,
            SceneSnapshotLayout::from_multiview(multiview),
            (width, height),
        );
    }

    /// Copies the main color attachment into the sampled scene-color snapshot used by grab-pass
    /// materials such as `blur_perobject`, then rebuilds [`Self::bind_group`] so `@group(0)`
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
                "copy_scene_color_snapshot: viewport {}x{} exceeds max_texture_dimension_2d ({max_dim}); skipping copy",
                width,
                height
            );
            return;
        }
        let format = source_color.format();
        let layout = SceneSnapshotLayout::from_multiview(multiview);
        let changed = self.scene_snapshots.ensure(
            device,
            self.limits.as_ref(),
            SceneSnapshotKind::Color,
            layout,
            (width, height),
            format,
        );
        self.scene_snapshots.encode_copy(
            encoder,
            source_color,
            SceneSnapshotKind::Color,
            layout,
            (width, height),
        );
        if changed {
            self.snapshot_version = self.snapshot_version.wrapping_add(1);
            self.rebuild_bind_group(device);
        }
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
            self.scene_snapshots.views(),
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
