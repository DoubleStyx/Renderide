//! Per-frame `@group(0)` resources: scene uniform, lights storage, shared cluster buffers, and
//! fallback scene snapshot textures.
//!
//! Cluster buffers ([`ClusterBufferCache`]) and the `@group(0)` layout live here and are
//! **shared across every view**; per-view uniform buffers and bind groups live in
//! [`crate::backend::frame_resource_manager::PerViewFrameState`] and reference these shared
//! cluster buffers plus view-local scene snapshots (safe under single-submit ordering — see
//! [`ClusterBufferCache`]).

mod empty_material;
mod scene_snapshot;
mod skybox_specular;

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::backend::cluster_gpu::{CLUSTER_COUNT_Z, ClusterBufferCache, ClusterBufferRefs};
use crate::backend::light_gpu::{GpuLight, MAX_LIGHTS};
use crate::gpu::GpuLimits;
use crate::gpu::frame_globals::{FrameGpuUniforms, SkyboxSpecularUniformParams};
use crate::materials::embedded::texture_resolve::{TextureBindKind, create_sampler};

use super::frame_gpu_error::FrameGpuInitError;
pub use empty_material::{EmptyMaterialBindGroup, empty_material_bind_group_layout};
pub use scene_snapshot::FrameSceneSnapshotTextureViews;
use scene_snapshot::{
    DEFAULT_SCENE_COLOR_FORMAT, SceneSnapshotKind, SceneSnapshotLayout, SceneSnapshotSet,
};
use skybox_specular::{
    SkyboxSpecularBindGroupResources, SkyboxSpecularEnvironmentKey,
    create_black_skybox_specular_equirect_fallback, create_black_skybox_specular_fallback,
};
pub use skybox_specular::{
    SkyboxSpecularCubemapSource, SkyboxSpecularEnvironmentSource, SkyboxSpecularEquirectSource,
    SkyboxSpecularGeneratedCubemapSource,
};

/// GPU buffers and bind groups for `@group(0)` frame globals (camera, lights, cluster lists,
/// fallback sampled scene snapshots, and skybox indirect specular).
///
/// `@group(0)` bind groups are per-view and are owned by
/// [`crate::backend::frame_resource_manager::PerViewFrameState`], keyed by
/// [`crate::camera::ViewId`], and built using
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
    /// Fallback scene depth/color snapshots sampled by the global bind group.
    ///
    /// Actual render views use per-view snapshots owned by
    /// [`crate::backend::frame_resource_manager::PerViewFrameState`].
    scene_snapshots: SceneSnapshotSet,
    /// Zero cubemap kept alive for frames without a resident cubemap skybox environment.
    _skybox_specular_fallback_texture: Arc<wgpu::Texture>,
    /// Zero cubemap view used by `@group(0) @binding(9)` when indirect specular is disabled.
    skybox_specular_fallback_view: Arc<wgpu::TextureView>,
    /// Fallback sampler used by `@group(0) @binding(10)` when indirect specular is disabled.
    skybox_specular_fallback_sampler: Arc<wgpu::Sampler>,
    /// Zero equirect texture kept alive for frames without a resident equirect skybox environment.
    _skybox_specular_equirect_fallback_texture: Arc<wgpu::Texture>,
    /// Zero equirect view used by `@group(0) @binding(11)` when indirect specular is disabled.
    skybox_specular_equirect_fallback_view: Arc<wgpu::TextureView>,
    /// Fallback sampler used by `@group(0) @binding(12)` when indirect specular is disabled.
    skybox_specular_equirect_fallback_sampler: Arc<wgpu::Sampler>,
    /// Current cubemap view bound as the frame-global indirect specular environment.
    skybox_specular_view: Arc<wgpu::TextureView>,
    /// Current sampler paired with [`Self::skybox_specular_view`].
    skybox_specular_sampler: Arc<wgpu::Sampler>,
    /// Current equirect view bound as the frame-global indirect specular environment.
    skybox_specular_equirect_view: Arc<wgpu::TextureView>,
    /// Current sampler paired with [`Self::skybox_specular_equirect_view`].
    skybox_specular_equirect_sampler: Arc<wgpu::Sampler>,
    /// Uniform parameters describing the currently bound skybox specular source.
    skybox_specular_params: SkyboxSpecularUniformParams,
    /// Stable key for the current skybox specular binding.
    skybox_specular_key: SkyboxSpecularEnvironmentKey,
    /// Monotonic version incremented whenever the skybox specular binding changes.
    skybox_specular_version: u64,
    /// Global `@group(0)` bind group (global frame uniform + shared lights/snapshots).
    ///
    /// Per-view passes bind the per-view bind group from
    /// [`crate::backend::frame_resource_manager::PerViewFrameState`] instead.
    pub bind_group: Arc<wgpu::BindGroup>,
    cluster_bind_version: u64,
    limits: Arc<GpuLimits>,
}

/// Per-view scene snapshot ownership for one render view.
pub(super) struct PerViewSceneSnapshots {
    /// Depth/color snapshot textures bound through this view's `@group(0)`.
    set: SceneSnapshotSet,
}

/// Requested per-view scene snapshot shape and families for pre-record synchronization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PerViewSceneSnapshotSyncParams {
    /// Extent in pixels used for any requested snapshot texture.
    pub viewport: (u32, u32),
    /// Depth snapshot format for `_CameraDepthTexture`-style material sampling.
    pub depth_format: wgpu::TextureFormat,
    /// HDR scene-color snapshot format for grab-pass material sampling.
    pub color_format: wgpu::TextureFormat,
    /// When true, synchronize the stereo-array snapshot layout instead of the mono layout.
    pub multiview: bool,
    /// Whether the depth snapshot family should be grown for this layout.
    pub needs_depth_snapshot: bool,
    /// Whether the color snapshot family should be grown for this layout.
    pub needs_color_snapshot: bool,
}

impl PerViewSceneSnapshots {
    /// Creates fallback `1x1` snapshots for one render view.
    pub(super) fn new(
        device: &wgpu::Device,
        depth_format: wgpu::TextureFormat,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            set: SceneSnapshotSet::new(device, depth_format, color_format),
        }
    }

    /// Returns the snapshot views used when building this view's `@group(0)` bind group.
    pub(super) fn views(&self) -> FrameSceneSnapshotTextureViews<'_> {
        self.set.views()
    }

    /// Ensures requested per-view snapshot textures exist before command recording starts.
    pub(super) fn sync(
        &mut self,
        device: &wgpu::Device,
        limits: &GpuLimits,
        params: PerViewSceneSnapshotSyncParams,
    ) -> bool {
        let layout = SceneSnapshotLayout::from_multiview(params.multiview);
        let depth_changed = params.needs_depth_snapshot
            && self.set.ensure(
                device,
                limits,
                SceneSnapshotKind::Depth,
                layout,
                params.viewport,
                params.depth_format,
            );
        let color_changed = params.needs_color_snapshot
            && self.set.ensure(
                device,
                limits,
                SceneSnapshotKind::Color,
                layout,
                params.viewport,
                params.color_format,
            );
        depth_changed || color_changed
    }

    /// Encodes a copy into this view's scene-depth snapshot.
    pub(super) fn encode_depth_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) {
        self.set.encode_copy(
            encoder,
            source_depth,
            SceneSnapshotKind::Depth,
            SceneSnapshotLayout::from_multiview(multiview),
            viewport,
        );
    }

    /// Encodes a copy into this view's scene-color snapshot.
    pub(super) fn encode_color_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) {
        self.set.encode_copy(
            encoder,
            source_color,
            SceneSnapshotKind::Color,
            SceneSnapshotLayout::from_multiview(multiview),
            viewport,
        );
    }
}

/// Appends uniform/storage entries that every clustered frame bind group owns.
fn append_frame_buffer_layout_entries(entries: &mut Vec<wgpu::BindGroupLayoutEntry>) {
    entries.extend([
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size_of::<FrameGpuUniforms>() as u64),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size_of::<GpuLight>() as u64),
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
    ]);
}

/// Appends per-view depth/color snapshot entries used by grab-pass material sampling.
fn append_scene_snapshot_layout_entries(entries: &mut Vec<wgpu::BindGroupLayoutEntry>) {
    entries.extend([
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
    ]);
}

/// Appends the frame-global skybox texture and filtering sampler entries for indirect specular.
fn append_skybox_specular_layout_entries(entries: &mut Vec<wgpu::BindGroupLayoutEntry>) {
    entries.extend([
        wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 11,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 12,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ]);
}

impl FrameGpuResources {
    /// Layout for `@group(0)`: uniform frame + lights + cluster counts + cluster indices +
    /// scene snapshots + skybox specular sources.
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let mut entries = Vec::with_capacity(13);
        append_frame_buffer_layout_entries(&mut entries);
        append_scene_snapshot_layout_entries(&mut entries);
        append_skybox_specular_layout_entries(&mut entries);
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame_globals"),
            entries: &entries,
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        frame_uniform: &wgpu::Buffer,
        lights_buffer: &wgpu::Buffer,
        refs: ClusterBufferRefs<'_>,
        snapshots: FrameSceneSnapshotTextureViews<'_>,
        skybox_specular: SkyboxSpecularBindGroupResources<'_>,
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
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(skybox_specular.cubemap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(skybox_specular.cubemap_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(skybox_specular.equirect_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(skybox_specular.equirect_sampler),
                },
            ],
        }))
    }

    /// Returns the currently selected skybox specular bind-group resources.
    fn skybox_specular_bind_group_resources(&self) -> SkyboxSpecularBindGroupResources<'_> {
        SkyboxSpecularBindGroupResources {
            cubemap_view: self.skybox_specular_view.as_ref(),
            cubemap_sampler: self.skybox_specular_sampler.as_ref(),
            equirect_view: self.skybox_specular_equirect_view.as_ref(),
            equirect_sampler: self.skybox_specular_equirect_sampler.as_ref(),
        }
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
            self.skybox_specular_bind_group_resources(),
        );
    }

    /// Allocates frame uniform, lights storage, minimal cluster grid `(1×1×Z)`, and fallback
    /// sampled textures; builds [`Self::bind_group`].
    ///
    /// Returns an error when the initial cluster buffer cache could not be populated (zero viewport or internal mismatch).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        limits: Arc<GpuLimits>,
    ) -> Result<Self, FrameGpuInitError> {
        let lights_size = (MAX_LIGHTS * size_of::<GpuLight>()) as u64;
        if lights_size > limits.max_storage_buffer_binding_size()
            || lights_size > limits.max_buffer_size()
        {
            return Err(FrameGpuInitError::LightsStorageExceedsLimits { size: lights_size });
        }
        let frame_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_globals_uniform"),
            size: size_of::<FrameGpuUniforms>() as u64,
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
        let scene_depth_format = crate::gpu::main_forward_depth_stencil_format(device.features());
        let scene_snapshots =
            SceneSnapshotSet::new(device, scene_depth_format, DEFAULT_SCENE_COLOR_FORMAT);
        let (
            skybox_specular_fallback_texture,
            skybox_specular_fallback_view,
            skybox_specular_fallback_sampler,
        ) = create_black_skybox_specular_fallback(device, queue);
        let (
            skybox_specular_equirect_fallback_texture,
            skybox_specular_equirect_fallback_view,
            skybox_specular_equirect_fallback_sampler,
        ) = create_black_skybox_specular_equirect_fallback(device, queue);
        let skybox_specular_view = skybox_specular_fallback_view.clone();
        let skybox_specular_sampler = skybox_specular_fallback_sampler.clone();
        let skybox_specular_equirect_view = skybox_specular_equirect_fallback_view.clone();
        let skybox_specular_equirect_sampler = skybox_specular_equirect_fallback_sampler.clone();
        let bind_group = Self::create_bind_group(
            device,
            &frame_uniform,
            &lights_buffer,
            refs,
            scene_snapshots.views(),
            SkyboxSpecularBindGroupResources {
                cubemap_view: skybox_specular_view.as_ref(),
                cubemap_sampler: skybox_specular_sampler.as_ref(),
                equirect_view: skybox_specular_equirect_view.as_ref(),
                equirect_sampler: skybox_specular_equirect_sampler.as_ref(),
            },
        );
        Ok(Self {
            frame_uniform,
            lights_buffer,
            cluster_cache,
            scene_snapshots,
            _skybox_specular_fallback_texture: skybox_specular_fallback_texture,
            skybox_specular_fallback_view,
            skybox_specular_fallback_sampler,
            _skybox_specular_equirect_fallback_texture: skybox_specular_equirect_fallback_texture,
            skybox_specular_equirect_fallback_view,
            skybox_specular_equirect_fallback_sampler,
            skybox_specular_view,
            skybox_specular_sampler,
            skybox_specular_equirect_view,
            skybox_specular_equirect_sampler,
            skybox_specular_params: SkyboxSpecularUniformParams::disabled(),
            skybox_specular_key: SkyboxSpecularEnvironmentKey::default(),
            skybox_specular_version: 0,
            bind_group,
            cluster_bind_version,
            limits,
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

    /// Builds a per-view `@group(0)` bind group using this view's own `frame_uniform` and
    /// `cluster_refs`, sharing only the scene-global lights storage from [`Self`].
    ///
    /// Called by [`crate::backend::frame_resource_manager::PerViewFrameState`] whenever the view's
    /// cluster buffers or snapshot textures change.
    pub(super) fn build_per_view_bind_group(
        &self,
        device: &wgpu::Device,
        frame_uniform: &wgpu::Buffer,
        cluster_refs: ClusterBufferRefs<'_>,
        snapshots: FrameSceneSnapshotTextureViews<'_>,
    ) -> Arc<wgpu::BindGroup> {
        Self::create_bind_group(
            device,
            frame_uniform,
            &self.lights_buffer,
            cluster_refs,
            snapshots,
            self.skybox_specular_bind_group_resources(),
        )
    }

    /// Current skybox specular environment version for per-view bind-group invalidation.
    pub fn skybox_specular_version(&self) -> u64 {
        self.skybox_specular_version
    }

    /// Uniform parameters for the currently bound skybox specular environment.
    pub fn skybox_specular_uniform_params(&self) -> SkyboxSpecularUniformParams {
        self.skybox_specular_params
    }

    /// Synchronizes the frame-global skybox specular source and rebuilds bind groups when needed.
    pub fn sync_skybox_specular_environment(
        &mut self,
        device: &wgpu::Device,
        source: Option<SkyboxSpecularEnvironmentSource>,
    ) -> bool {
        let Some(source) = source else {
            if self.skybox_specular_key == SkyboxSpecularEnvironmentKey::default() {
                return false;
            }
            self.skybox_specular_view = self.skybox_specular_fallback_view.clone();
            self.skybox_specular_sampler = self.skybox_specular_fallback_sampler.clone();
            self.skybox_specular_equirect_view =
                self.skybox_specular_equirect_fallback_view.clone();
            self.skybox_specular_equirect_sampler =
                self.skybox_specular_equirect_fallback_sampler.clone();
            self.skybox_specular_params = SkyboxSpecularUniformParams::disabled();
            self.skybox_specular_key = SkyboxSpecularEnvironmentKey::default();
            self.skybox_specular_version = self.skybox_specular_version.wrapping_add(1);
            self.rebuild_bind_group(device);
            return true;
        };

        let new_key = SkyboxSpecularEnvironmentKey::from_source(&source);
        let new_params = source.uniform_params();
        if new_key == self.skybox_specular_key {
            self.skybox_specular_params = new_params;
            return false;
        }

        match source {
            SkyboxSpecularEnvironmentSource::Cubemap(source) => {
                let sampler = Arc::new(create_sampler(
                    device,
                    &source.sampler,
                    TextureBindKind::Cube,
                    source.mip_levels_resident,
                ));
                self.skybox_specular_view = source.view;
                self.skybox_specular_sampler = sampler;
                self.skybox_specular_equirect_view =
                    self.skybox_specular_equirect_fallback_view.clone();
                self.skybox_specular_equirect_sampler =
                    self.skybox_specular_equirect_fallback_sampler.clone();
            }
            SkyboxSpecularEnvironmentSource::GeneratedCubemap(source) => {
                let sampler = Arc::new(create_sampler(
                    device,
                    &source.sampler,
                    TextureBindKind::Cube,
                    source.mip_levels_resident,
                ));
                self.skybox_specular_view = source.view;
                self.skybox_specular_sampler = sampler;
                self.skybox_specular_equirect_view =
                    self.skybox_specular_equirect_fallback_view.clone();
                self.skybox_specular_equirect_sampler =
                    self.skybox_specular_equirect_fallback_sampler.clone();
            }
            SkyboxSpecularEnvironmentSource::Projection360Equirect(source) => {
                let sampler = Arc::new(create_sampler(
                    device,
                    &source.sampler,
                    TextureBindKind::Tex2D,
                    source.mip_levels_resident,
                ));
                self.skybox_specular_view = self.skybox_specular_fallback_view.clone();
                self.skybox_specular_sampler = self.skybox_specular_fallback_sampler.clone();
                self.skybox_specular_equirect_view = source.view;
                self.skybox_specular_equirect_sampler = sampler;
            }
        }
        self.skybox_specular_params = new_params;
        self.skybox_specular_key = new_key;
        self.skybox_specular_version = self.skybox_specular_version.wrapping_add(1);
        self.rebuild_bind_group(device);
        true
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

    /// Uploads only the lights storage buffer (used by [`crate::passes::ClusteredLightPass`]).
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
            let zero = [0u8; size_of::<GpuLight>()];
            queue.write_buffer(lights_buffer, 0, &zero);
        }
    }
}
