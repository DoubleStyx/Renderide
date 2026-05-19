//! Per-frame `@group(0)` resources: fallback scene uniform/lights storage, shared cluster
//! buffers, and fallback scene snapshot textures.
//!
//! Cluster buffers ([`ClusterBufferCache`]) and the `@group(0)` layout live here and are
//! **shared across every view**; per-view uniform buffers and bind groups live in
//! [`crate::backend::frame_resource_manager::PerViewFrameState`] and reference these shared
//! cluster buffers plus view-local scene snapshots (safe under single-submit ordering -- see
//! [`ClusterBufferCache`]).

mod empty_material;
mod ibl_dfg;
mod reflection_probe_specular;
mod scene_snapshot;
mod shadows;

use std::sync::Arc;

use crate::backend::cluster_gpu::{CLUSTER_COUNT_Z, ClusterBufferCache, ClusterBufferRefs};
use crate::backend::light_gpu::GpuLight;
use crate::gpu::frame_globals::{FrameGpuUniforms, SkyboxSpecularUniformParams};
use crate::gpu::{GpuLimits, GpuShadowLight, GpuShadowView, MAX_LIGHTS, frame_bind_group_layout};
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::shared::ShadowResolutionMode;

use super::frame_gpu_error::FrameGpuInitError;
pub(crate) use crate::gpu::{
    GpuReflectionProbeMetadata, REFLECTION_PROBE_ATLAS_FORMAT,
    REFLECTION_PROBE_METADATA_BOX_PROJECTION, REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL,
};
pub(crate) use empty_material::EmptyMaterialBindGroup;
use ibl_dfg::create_ibl_dfg_lut;
pub(crate) use reflection_probe_specular::ReflectionProbeSpecularResources;
use reflection_probe_specular::{
    ReflectionProbeSpecularBindGroupResources, create_reflection_probe_specular_fallback,
};
pub(crate) use scene_snapshot::FrameSceneSnapshotTextureViews;
use scene_snapshot::{
    DEFAULT_SCENE_COLOR_FORMAT, SceneSnapshotKind, SceneSnapshotLayout, SceneSnapshotSet,
};
use shadows::{ShadowMapResources, create_shadow_light_buffer, create_shadow_view_buffer};

/// GPU buffers and bind groups for `@group(0)` frame globals (camera, lights, cluster lists,
/// fallback sampled scene snapshots, and reflection-probe specular IBL).
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
    /// Fallback storage buffer holding up to [`MAX_LIGHTS`] [`GpuLight`] records.
    ///
    /// Normal per-view rendering binds the light buffer owned by
    /// [`crate::backend::frame_resource_manager::PerViewFrameState`].
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
    /// Black atlas array kept alive for frames without resident reflection probes.
    _reflection_probe_fallback_texture: Arc<wgpu::Texture>,
    /// Current 2D-array atlas view bound for reflection-probe specular IBL.
    reflection_probe_array_view: Arc<wgpu::TextureView>,
    /// Current sampler paired with [`Self::reflection_probe_array_view`].
    reflection_probe_sampler: Arc<wgpu::Sampler>,
    /// Current metadata buffer for reflection-probe specular IBL.
    reflection_probe_metadata_buffer: Arc<wgpu::Buffer>,
    /// Monotonic version incremented whenever reflection-probe bind resources change.
    reflection_probe_version: u64,
    /// Texture backing the static DFG LUT used by split-sum IBL.
    _ibl_dfg_lut_texture: Arc<wgpu::Texture>,
    /// Frame-global DFG LUT view bound at `@group(0) @binding(11)`.
    ibl_dfg_lut_view: Arc<wgpu::TextureView>,
    /// Persistent shadow-map texture array and comparison sampler.
    shadow_maps: ShadowMapResources,
    /// Fallback per-light shadow indirection buffer.
    shadow_lights_buffer: Arc<wgpu::Buffer>,
    /// Fallback per-shadow-view matrix buffer.
    shadow_views_buffer: Arc<wgpu::Buffer>,
    /// Global `@group(0)` bind group (fallback frame uniform + fallback lights/snapshots).
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
    ) -> bool {
        self.set.encode_copy(
            encoder,
            source_depth,
            SceneSnapshotKind::Depth,
            SceneSnapshotLayout::from_multiview(multiview),
            viewport,
        )
    }

    /// Encodes a copy into this view's scene-color snapshot.
    pub(super) fn encode_color_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) -> bool {
        self.set.encode_copy(
            encoder,
            source_color,
            SceneSnapshotKind::Color,
            SceneSnapshotLayout::from_multiview(multiview),
            viewport,
        )
    }
}

/// Inputs for creating one frame `@group(0)` bind group.
struct FrameBindGroupInputs<'a> {
    /// Uniform buffer containing frame/view globals.
    frame_uniform: &'a wgpu::Buffer,
    /// Storage buffer containing visible light records.
    lights_buffer: &'a wgpu::Buffer,
    /// Shared clustered-light buffers.
    cluster_refs: ClusterBufferRefs<'a>,
    /// Scene snapshot texture views and sampler.
    snapshots: FrameSceneSnapshotTextureViews<'a>,
    /// Reflection-probe specular IBL resources.
    reflection_probes: ReflectionProbeSpecularBindGroupResources<'a>,
    /// Static DFG LUT used by split-sum IBL.
    ibl_dfg_lut_view: &'a wgpu::TextureView,
    /// Shadow-map texture array and compare sampler.
    shadows: shadows::ShadowBindGroupResources<'a>,
    /// Per-light shadow metadata buffer.
    shadow_lights_buffer: &'a wgpu::Buffer,
    /// Per-shadow-view matrix metadata buffer.
    shadow_views_buffer: &'a wgpu::Buffer,
}

/// Per-view inputs that differ from the shared frame-global bind resources.
pub(super) struct PerViewFrameBindGroupInputs<'a> {
    /// Per-view frame uniform buffer.
    pub(super) frame_uniform: &'a wgpu::Buffer,
    /// Per-view light storage buffer.
    pub(super) lights_buffer: &'a wgpu::Buffer,
    /// Shared clustered-light buffers.
    pub(super) cluster_refs: ClusterBufferRefs<'a>,
    /// Per-view scene snapshot texture views and sampler.
    pub(super) snapshots: FrameSceneSnapshotTextureViews<'a>,
    /// Per-view shadow-light metadata buffer.
    pub(super) shadow_lights_buffer: &'a wgpu::Buffer,
    /// Per-view shadow-view metadata buffer.
    pub(super) shadow_views_buffer: &'a wgpu::Buffer,
}

impl FrameGpuResources {
    /// Layout for `@group(0)`: uniform frame + lights + cluster ranges + cluster indices +
    /// scene snapshots + reflection-probe specular resources.
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        frame_bind_group_layout(device)
    }

    fn create_bind_group(
        device: &wgpu::Device,
        inputs: FrameBindGroupInputs<'_>,
    ) -> Arc<wgpu::BindGroup> {
        let layout = Self::bind_group_layout(device);
        let bind_group = Arc::new(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("frame_globals_bind_group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inputs.frame_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: inputs.lights_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: inputs.cluster_refs.cluster_light_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: inputs
                            .cluster_refs
                            .cluster_light_indices
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            inputs.snapshots.scene_depth_2d,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(
                            inputs.snapshots.scene_depth_array,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(
                            inputs.snapshots.scene_color_2d,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            inputs.snapshots.scene_color_array,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Sampler(
                            inputs.snapshots.scene_color_sampler,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(
                            inputs.reflection_probes.array_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(inputs.reflection_probes.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(inputs.ibl_dfg_lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: inputs.reflection_probes.metadata_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::TextureView(inputs.shadows.array_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: wgpu::BindingResource::Sampler(inputs.shadows.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: inputs.shadow_lights_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 16,
                        resource: inputs.shadow_views_buffer.as_entire_binding(),
                    },
                ],
            }),
        );
        crate::profiling::note_resource_churn!(BindGroup, "backend::frame_globals_bind_group");
        bind_group
    }

    /// Allocates a lights storage buffer large enough for [`MAX_LIGHTS`] rows.
    pub(in crate::backend) fn create_lights_storage_buffer(
        device: &wgpu::Device,
        label: &'static str,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (MAX_LIGHTS * size_of::<GpuLight>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Allocates a per-light shadow metadata buffer large enough for [`MAX_LIGHTS`] rows.
    pub(in crate::backend) fn create_shadow_light_storage_buffer(
        device: &wgpu::Device,
        label: &'static str,
    ) -> wgpu::Buffer {
        create_shadow_light_buffer(device, label)
    }

    /// Allocates a shadow-view metadata buffer large enough for all shadow-map layers.
    pub(in crate::backend) fn create_shadow_view_storage_buffer(
        device: &wgpu::Device,
        label: &'static str,
    ) -> wgpu::Buffer {
        create_shadow_view_buffer(device, label)
    }

    /// Returns the currently selected reflection-probe bind-group resources.
    fn reflection_probe_bind_group_resources(
        &self,
    ) -> ReflectionProbeSpecularBindGroupResources<'_> {
        ReflectionProbeSpecularBindGroupResources {
            array_view: self.reflection_probe_array_view.as_ref(),
            sampler: self.reflection_probe_sampler.as_ref(),
            metadata_buffer: self.reflection_probe_metadata_buffer.as_ref(),
        }
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        let Some(refs) = self.cluster_cache.current_refs() else {
            logger::warn!("FrameGpu: cluster buffers missing; skipping bind group rebuild");
            return;
        };
        self.bind_group = Self::create_bind_group(
            device,
            FrameBindGroupInputs {
                frame_uniform: &self.frame_uniform,
                lights_buffer: &self.lights_buffer,
                cluster_refs: refs,
                snapshots: self.scene_snapshots.views(),
                reflection_probes: self.reflection_probe_bind_group_resources(),
                ibl_dfg_lut_view: self.ibl_dfg_lut_view.as_ref(),
                shadows: self.shadow_maps.bind_group_resources(),
                shadow_lights_buffer: self.shadow_lights_buffer.as_ref(),
                shadow_views_buffer: self.shadow_views_buffer.as_ref(),
            },
        );
    }

    /// Allocates frame uniform, lights storage, minimal cluster grid `(1x1xZ)`, and fallback
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
        crate::profiling::note_resource_churn!(Buffer, "backend::frame_globals_uniform");
        let lights_buffer = Self::create_lights_storage_buffer(device, "frame_lights_storage");
        crate::profiling::note_resource_churn!(Buffer, "backend::frame_lights_storage");
        let mut cluster_cache = ClusterBufferCache::new();
        cluster_cache
            .ensure_buffers(device, limits.as_ref(), (1, 1), CLUSTER_COUNT_Z, false, 1)
            .ok_or(FrameGpuInitError::ClusterEnsureFailed)?;
        let cluster_bind_version = cluster_cache.version;
        let refs = cluster_cache
            .current_refs()
            .ok_or(FrameGpuInitError::ClusterGetBuffersFailed)?;
        let scene_depth_format = crate::gpu::main_forward_depth_stencil_format(device.features());
        let scene_snapshots =
            SceneSnapshotSet::new(device, scene_depth_format, DEFAULT_SCENE_COLOR_FORMAT);
        let (
            reflection_probe_fallback_texture,
            reflection_probe_array_view,
            reflection_probe_sampler,
            reflection_probe_metadata_buffer,
        ) = create_reflection_probe_specular_fallback(device);
        let (ibl_dfg_lut_texture, ibl_dfg_lut_view) = create_ibl_dfg_lut(device, queue);
        let shadow_maps = ShadowMapResources::new(device);
        let shadow_lights_buffer = Arc::new(create_shadow_light_buffer(
            device,
            "frame_shadow_lights_storage",
        ));
        let shadow_views_buffer = Arc::new(create_shadow_view_buffer(
            device,
            "frame_shadow_views_storage",
        ));
        let bind_group = Self::create_bind_group(
            device,
            FrameBindGroupInputs {
                frame_uniform: &frame_uniform,
                lights_buffer: &lights_buffer,
                cluster_refs: refs,
                snapshots: scene_snapshots.views(),
                reflection_probes: ReflectionProbeSpecularBindGroupResources {
                    array_view: reflection_probe_array_view.as_ref(),
                    sampler: reflection_probe_sampler.as_ref(),
                    metadata_buffer: reflection_probe_metadata_buffer.as_ref(),
                },
                ibl_dfg_lut_view: ibl_dfg_lut_view.as_ref(),
                shadows: shadow_maps.bind_group_resources(),
                shadow_lights_buffer: shadow_lights_buffer.as_ref(),
                shadow_views_buffer: shadow_views_buffer.as_ref(),
            },
        );
        Ok(Self {
            frame_uniform,
            lights_buffer,
            cluster_cache,
            scene_snapshots,
            _reflection_probe_fallback_texture: reflection_probe_fallback_texture,
            reflection_probe_array_view,
            reflection_probe_sampler,
            reflection_probe_metadata_buffer,
            reflection_probe_version: 0,
            _ibl_dfg_lut_texture: ibl_dfg_lut_texture,
            ibl_dfg_lut_view,
            shadow_maps,
            shadow_lights_buffer,
            shadow_views_buffer,
            bind_group,
            cluster_bind_version,
            limits,
        })
    }

    /// Grows the shared cluster cache to cover `viewport` x `stereo` and `index_capacity_words`
    /// if possible; rebuilds
    /// [`Self::bind_group`] when the underlying buffers were reallocated.
    ///
    /// When `stereo` is true, cluster range storage is doubled for per-eye storage.
    /// Returns [`None`] when the requested layout exceeds device limits. Otherwise returns
    /// whether the bind group was recreated.
    ///
    /// Because the shared cache is grow-only (see [`ClusterBufferCache`]), calling this with
    /// a smaller viewport than a previous call is a no-op.
    pub fn sync_cluster_viewport(
        &mut self,
        device: &wgpu::Device,
        viewport: (u32, u32),
        stereo: bool,
        index_capacity_words: u64,
    ) -> Option<bool> {
        profiling::scope!("render::sync_cluster_viewport");
        self.cluster_cache.ensure_buffers(
            device,
            self.limits.as_ref(),
            viewport,
            CLUSTER_COUNT_Z,
            stereo,
            index_capacity_words,
        )?;
        let ver = self.cluster_cache.version;
        if ver == self.cluster_bind_version {
            return Some(false);
        }
        self.rebuild_bind_group(device);
        self.cluster_bind_version = ver;
        Some(true)
    }

    /// Builds a per-view `@group(0)` bind group using this view's own frame uniform and light
    /// storage plus the shared cluster buffers from [`Self`].
    ///
    /// Called by [`crate::backend::frame_resource_manager::PerViewFrameState`] whenever the view's
    /// cluster buffers or snapshot textures change.
    pub(super) fn build_per_view_bind_group(
        &self,
        device: &wgpu::Device,
        inputs: PerViewFrameBindGroupInputs<'_>,
    ) -> Arc<wgpu::BindGroup> {
        self.build_per_view_bind_group_with_shadows(
            device,
            inputs,
            self.shadow_maps.bind_group_resources(),
        )
    }

    /// Builds a per-view `@group(0)` bind group for passes that write the live shadow atlas.
    pub(super) fn build_per_view_shadow_writer_bind_group(
        &self,
        device: &wgpu::Device,
        inputs: PerViewFrameBindGroupInputs<'_>,
    ) -> Arc<wgpu::BindGroup> {
        self.build_per_view_bind_group_with_shadows(
            device,
            inputs,
            self.shadow_maps.writer_bind_group_resources(),
        )
    }

    /// Builds a per-view frame bind group with caller-selected shadow sampling resources.
    fn build_per_view_bind_group_with_shadows(
        &self,
        device: &wgpu::Device,
        inputs: PerViewFrameBindGroupInputs<'_>,
        shadows: shadows::ShadowBindGroupResources<'_>,
    ) -> Arc<wgpu::BindGroup> {
        Self::create_bind_group(
            device,
            FrameBindGroupInputs {
                frame_uniform: inputs.frame_uniform,
                lights_buffer: inputs.lights_buffer,
                cluster_refs: inputs.cluster_refs,
                snapshots: inputs.snapshots,
                reflection_probes: self.reflection_probe_bind_group_resources(),
                ibl_dfg_lut_view: self.ibl_dfg_lut_view.as_ref(),
                shadows,
                shadow_lights_buffer: inputs.shadow_lights_buffer,
                shadow_views_buffer: inputs.shadow_views_buffer,
            },
        )
    }

    /// Ensures the shared shadow-map texture matches the current quality setting.
    pub(super) fn sync_shadow_resources(
        &mut self,
        device: &wgpu::Device,
        mode: ShadowResolutionMode,
    ) -> bool {
        let changed = self
            .shadow_maps
            .sync_resolution(device, self.limits.as_ref(), mode);
        if changed {
            self.rebuild_bind_group(device);
        }
        changed
    }

    /// Version for per-view frame bind-group invalidation when the shadow texture reallocates.
    pub(super) fn shadow_resources_version(&self) -> u64 {
        self.shadow_maps.version()
    }

    /// Current shadow-map resolution.
    pub(super) fn shadow_resolution(&self) -> u32 {
        self.shadow_maps.resolution()
    }

    /// Returns one renderable shadow-map layer view.
    pub(super) fn shadow_layer_view(&self, layer: usize) -> Option<Arc<wgpu::TextureView>> {
        self.shadow_maps.layer_view(layer)
    }

    /// Records a shadow-light metadata upload.
    pub(in crate::backend) fn write_shadow_lights_buffer_to(
        uploads: GraphUploadSink<'_>,
        shadow_lights_buffer: &wgpu::Buffer,
        lights: &[GpuShadowLight],
    ) {
        if lights.is_empty() {
            uploads.write_buffer(
                shadow_lights_buffer,
                0,
                bytemuck::bytes_of(&GpuShadowLight::default()),
            );
        } else {
            uploads.write_buffer(shadow_lights_buffer, 0, bytemuck::cast_slice(lights));
        }
    }

    /// Records a shadow-view metadata upload.
    pub(in crate::backend) fn write_shadow_views_buffer_to(
        uploads: GraphUploadSink<'_>,
        shadow_views_buffer: &wgpu::Buffer,
        views: &[GpuShadowView],
    ) {
        if views.is_empty() {
            uploads.write_buffer(
                shadow_views_buffer,
                0,
                bytemuck::bytes_of(&GpuShadowView::default()),
            );
        } else {
            uploads.write_buffer(shadow_views_buffer, 0, bytemuck::cast_slice(views));
        }
    }

    /// Current reflection-probe resource version for per-view bind-group invalidation.
    pub fn skybox_specular_version(&self) -> u64 {
        self.reflection_probe_version
    }

    /// Uniform parameters for the removed direct skybox specular path.
    pub fn skybox_specular_uniform_params(&self) -> SkyboxSpecularUniformParams {
        SkyboxSpecularUniformParams::disabled()
    }

    /// Synchronizes frame-global reflection-probe resources and rebuilds bind groups when needed.
    pub fn sync_reflection_probe_specular_resources(
        &mut self,
        device: &wgpu::Device,
        resources: Option<ReflectionProbeSpecularResources>,
    ) -> bool {
        let Some(resources) = resources else {
            return false;
        };
        if resources.version == self.reflection_probe_version {
            return false;
        }
        self.reflection_probe_array_view = resources.array_view;
        self.reflection_probe_sampler = resources.sampler;
        self.reflection_probe_metadata_buffer = resources.metadata_buffer;
        self.reflection_probe_version = resources.version;
        self.rebuild_bind_group(device);
        true
    }

    /// Records a lights storage upload into `lights_buffer`.
    pub(in crate::backend) fn write_lights_buffer_to(
        uploads: GraphUploadSink<'_>,
        lights_buffer: &wgpu::Buffer,
        lights: &[GpuLight],
    ) {
        Self::write_lights_buffer_inner(uploads, lights_buffer, lights);
    }

    fn write_lights_buffer_inner(
        uploads: GraphUploadSink<'_>,
        lights_buffer: &wgpu::Buffer,
        lights: &[GpuLight],
    ) {
        let n = lights.len().min(MAX_LIGHTS);
        if n > 0 {
            let bytes = bytemuck::cast_slice(&lights[..n]);
            uploads.write_buffer(lights_buffer, 0, bytes);
        } else {
            let zero = [0u8; size_of::<GpuLight>()];
            uploads.write_buffer(lights_buffer, 0, &zero);
        }
    }
}

#[cfg(test)]
mod tests {
    fn fragment_resource_count(
        entries: &[wgpu::BindGroupLayoutEntry],
        matches_ty: impl Fn(&wgpu::BindingType) -> bool,
    ) -> u32 {
        entries
            .iter()
            .filter(|entry| entry.visibility.contains(wgpu::ShaderStages::FRAGMENT))
            .filter(|entry| matches_ty(&entry.ty))
            .map(|entry| entry.count.map_or(1, |count| count.get()))
            .sum()
    }

    #[test]
    fn frame_layout_contributes_two_fragment_samplers() {
        let entries = crate::gpu::frame_bind_group_layout_entries();
        assert_eq!(
            fragment_resource_count(&entries, |ty| matches!(ty, wgpu::BindingType::Sampler(_))),
            3
        );
    }

    #[test]
    fn frame_layout_contributes_six_fragment_sampled_textures() {
        let entries = crate::gpu::frame_bind_group_layout_entries();
        assert_eq!(
            fragment_resource_count(&entries, |ty| matches!(
                ty,
                wgpu::BindingType::Texture { .. }
            )),
            7
        );
    }
}
