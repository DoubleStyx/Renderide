//! Persistent shadow-map atlas resources bound through frame globals.

use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::Zeroable;
use glam::Mat4;
use rayon::prelude::*;

use crate::cpu_parallelism::{
    RENDER_COMMAND_CHUNK_DRAWS, admit_render_command_items, current_reference_worker_count,
    record_parallel_admission,
};
use crate::gpu::{
    GpuLimits, GpuShadowView, MAX_SHADOW_VIEWS, SHADOW_VIEW_KIND_POINT, SHADOW_VIEW_KIND_SPOT,
};
use crate::materials::{MaterialPipelineDesc, ShaderPermutation};
use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::passes::{
    ShadowDepthDrawBatch, WorldMeshForwardEncodeRefs, WorldMeshForwardPipelineState,
    draw_shadow_depth_subset,
};
use crate::render_graph::execution_backend::ShadowAtlasEncodeParams;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::render_graph::pass::{EncoderPass, PassBuilder, PassPhase};
use crate::world_mesh::WorldMeshPhase;

use super::super::frame_gpu_error::FrameGpuInitError;
use super::super::frame_resource_manager::{ShadowCasterSet, ShadowFramePlan, ShadowRenderView};
use super::super::per_draw_resources::PerDrawResources;
use super::super::shadow_atlas_budget::clamp_shadow_atlas_resolution;
use super::super::shadow_atlas_format::{
    select_shadow_atlas_binding_format, select_shadow_atlas_format,
};
use super::{FrameGpuResources, ShadowResourceSyncResult};

/// Main-graph frame-global pass name for shadow-atlas rendering.
pub(crate) const SHADOW_ATLAS_PASS_NAME: &str = "shadow_atlas";

/// Minimum caster draws before per-layer shadow slab packing uses Rayon.
const SHADOW_SLAB_PARALLEL_MIN_DRAWS: usize = RENDER_COMMAND_CHUNK_DRAWS * 2;
/// Draws packed by one Rayon worker chunk.
const SHADOW_SLAB_PARALLEL_CHUNK_DRAWS: usize = RENDER_COMMAND_CHUNK_DRAWS;
/// Shadow slab chunks assigned to one worker leaf.
const SHADOW_SLAB_PARALLEL_CHUNKS_PER_TASK: usize = 1;

const SHADOW_NORMAL_MATRIX_IDENTITY: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PaddedShadowCasterDraw {
    model: [f32; 16],
    normal_matrix: [[f32; 4]; 3],
    _pad: [[f32; 4]; 25],
}

impl PaddedShadowCasterDraw {
    #[inline]
    fn new(item: &crate::world_mesh::WorldMeshDrawItem) -> Self {
        let model = shadow_caster_model(item);
        Self {
            model: model.to_cols_array(),
            normal_matrix: SHADOW_NORMAL_MATRIX_IDENTITY,
            _pad: [[0.0; 4]; 25],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PaddedShadowLayerUniforms {
    view_proj: [f32; 16],
    light_position_range: [f32; 4],
    shadow_params: [f32; 4],
    _pad: [[f32; 4]; 26],
}

impl PaddedShadowLayerUniforms {
    #[inline]
    fn new(view: &ShadowRenderView) -> Self {
        let radial_shadow = shadow_view_uses_radial_depth(view.kind);
        let view_proj = view.view_proj.to_cols_array();
        Self {
            view_proj,
            light_position_range: if radial_shadow {
                [
                    view.light_position.x,
                    view.light_position.y,
                    view.light_position.z,
                    view.light_range,
                ]
            } else {
                [0.0; 4]
            },
            shadow_params: [
                if radial_shadow { view.shadow_bias } else { 0.0 },
                0.0,
                0.0,
                0.0,
            ],
            _pad: [[0.0; 4]; 26],
        }
    }
}

#[inline]
fn shadow_view_uses_radial_depth(kind: u32) -> bool {
    matches!(kind, SHADOW_VIEW_KIND_POINT | SHADOW_VIEW_KIND_SPOT)
}

fn plot_shadow_atlas(plan: &ShadowFramePlan) {
    let (visible_groups, visible_group_draws) = shadow_visible_group_stats(plan);
    let upload_bytes = plan
        .requested_draw_slots
        .saturating_add(plan.render_views.len())
        .saturating_mul(PER_DRAW_UNIFORM_STRIDE);
    crate::profiling::plot_shadow_atlas(
        plan.render_views.len(),
        plan.caster_sets.len(),
        plan.requested_draw_slots,
        visible_groups,
        visible_group_draws,
        upload_bytes,
    );
}

fn shadow_visible_group_stats(plan: &ShadowFramePlan) -> (usize, usize) {
    let mut groups = 0usize;
    let mut draws = 0usize;
    for view in &plan.render_views {
        for phase in WorldMeshPhase::PRIMARY_FORWARD {
            for group in view.groups(phase) {
                groups = groups.saturating_add(1);
                draws = draws.saturating_add(
                    (group.instance_range.end - group.instance_range.start) as usize,
                );
            }
        }
    }
    (groups, draws)
}

struct ShadowLayerEncodeContext<'a, 'encoder, 'refs> {
    device: &'a wgpu::Device,
    encoder: &'encoder mut wgpu::CommandEncoder,
    pipeline: &'a WorldMeshForwardPipelineState,
    encode_refs: &'refs mut WorldMeshForwardEncodeRefs<'a>,
    gpu_limits: &'a GpuLimits,
    profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

struct ShadowLayerPlan<'a> {
    view: &'a ShadowRenderView,
    caster_set: &'a ShadowCasterSet,
}

/// Main-graph frame-global pass that renders realtime shadow atlas layers.
pub(crate) struct ShadowAtlasPass;

impl ShadowAtlasPass {
    /// Creates the shadow atlas render pass.
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl EncoderPass for ShadowAtlasPass {
    fn name(&self) -> &str {
        SHADOW_ATLAS_PASS_NAME
    }

    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed("shadows::atlas")
    }

    fn setup(
        &mut self,
        builder: &mut PassBuilder<'_>,
    ) -> Result<(), crate::render_graph::error::SetupError> {
        builder.encoder();
        builder.cull_exempt();
        builder.never_parallel();
        Ok(())
    }

    fn should_record(
        &self,
        ctx: &crate::render_graph::context::EncoderPassCtx<'_, '_, '_>,
    ) -> Result<bool, crate::render_graph::error::RenderPassError> {
        Ok(ctx
            .frame
            .systems
            .frame_resources
            .has_shadow_atlas_requests())
    }

    fn record(
        &self,
        ctx: &mut crate::render_graph::context::EncoderPassCtx<'_, '_, '_>,
    ) -> Result<(), crate::render_graph::error::RenderPassError> {
        let Some(gpu_limits) = ctx.frame.view.gpu_limits.as_deref() else {
            return Ok(());
        };
        let skin_cache = ctx
            .frame
            .systems
            .mesh_deform_skin_cache
            .as_deref()
            .or(ctx.frame.systems.skin_cache);
        ctx.frame
            .systems
            .frame_resources
            .encode_shadow_atlas(ShadowAtlasEncodeParams {
                device: ctx.device,
                encoder: ctx.encoder,
                materials: ctx.frame.systems.materials,
                asset_resources: ctx.frame.systems.asset_resources,
                skin_cache,
                gpu_limits,
                uploads: ctx.uploads,
                profiler: ctx.profiler,
            });
        Ok(())
    }

    fn phase(&self) -> PassPhase {
        PassPhase::FrameGlobal
    }
}

/// Frame-global shadow-map atlas resources.
pub(super) struct ShadowAtlasResources {
    texture: Arc<wgpu::Texture>,
    atlas_view: Arc<wgpu::TextureView>,
    layer_views: Vec<Arc<wgpu::TextureView>>,
    format: wgpu::TextureFormat,
    renderable: bool,
    sampler: Arc<wgpu::Sampler>,
    metadata_buffer: Arc<wgpu::Buffer>,
    per_draw: PerDrawResources,
    layer_uniform_buffer: Arc<wgpu::Buffer>,
    layer_uniform_bind_group: Arc<wgpu::BindGroup>,
    layer_uniform_layout: Arc<wgpu::BindGroupLayout>,
    layer_uniform_capacity: usize,
    scratch: parking_lot::Mutex<Vec<PaddedShadowCasterDraw>>,
    layer_scratch: parking_lot::Mutex<Vec<PaddedShadowLayerUniforms>>,
    resolution: u32,
    layers: u32,
    version: u64,
}

impl ShadowAtlasResources {
    /// Creates the fallback one-layer shadow atlas and metadata buffer.
    pub(super) fn new(
        device: &wgpu::Device,
        limits: Arc<GpuLimits>,
    ) -> Result<Self, FrameGpuInitError> {
        let metadata_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_view_metadata"),
            size: (MAX_SHADOW_VIEWS * size_of::<GpuShadowView>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        crate::profiling::note_resource_churn!(Buffer, "backend::shadow_view_metadata");
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_comparison_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            ..Default::default()
        }));
        let per_draw_layout = Arc::new(shadow_per_draw_layout(device));
        let per_draw =
            PerDrawResources::new_with_layout(device, per_draw_layout, Arc::clone(&limits));
        let layer_uniform_layout = Arc::new(shadow_layer_uniform_layout(device));
        let layer_uniform_capacity = 1usize;
        let (layer_uniform_buffer, layer_uniform_bind_group) = create_shadow_layer_uniforms(
            device,
            layer_uniform_layout.as_ref(),
            layer_uniform_capacity,
        );
        let resolution = initial_shadow_resolution(limits.as_ref());
        let render_format = select_shadow_atlas_format(limits.as_ref());
        let format = render_format
            .or_else(|| select_shadow_atlas_binding_format(limits.as_ref()))
            .ok_or(FrameGpuInitError::ShadowAtlasBindingFormatUnavailable)?;
        let renderable = render_format.is_some();
        let (texture, atlas_view, layer_views) =
            create_shadow_texture(device, resolution, 1, format, renderable);
        Ok(Self {
            texture,
            atlas_view,
            layer_views,
            format,
            renderable,
            sampler,
            metadata_buffer,
            per_draw,
            layer_uniform_buffer,
            layer_uniform_bind_group,
            layer_uniform_layout,
            layer_uniform_capacity,
            scratch: parking_lot::Mutex::new(Vec::new()),
            layer_scratch: parking_lot::Mutex::new(Vec::new()),
            resolution,
            layers: 1,
            version: 1,
        })
    }

    /// Grows the atlas to cover the requested full-layer resolution and layer count.
    pub(super) fn sync(
        &mut self,
        device: &wgpu::Device,
        limits: &GpuLimits,
        requested_resolution: u32,
        requested_layers: u32,
        requested_draw_slots: usize,
    ) -> ShadowResourceSyncResult {
        let _ = self
            .per_draw
            .ensure_draw_slot_capacity(device, requested_draw_slots);
        let layers = requested_layers
            .max(1)
            .min(limits.wgpu.max_texture_array_layers.max(1));
        self.ensure_layer_uniform_capacity(device, layers as usize);
        if !self.renderable {
            return self.sync_result(false);
        }
        let requested_resolution =
            clamp_shadow_texture_resolution(limits, requested_resolution, layers, self.format);
        let grown_layers = self.layers.max(layers);
        let grown_resolution = self.resolution.max(requested_resolution);
        let budgeted_grown_resolution =
            clamp_shadow_texture_resolution(limits, grown_resolution, grown_layers, self.format);
        let (next_resolution, next_layers) = if budgeted_grown_resolution == grown_resolution {
            (grown_resolution, grown_layers)
        } else {
            (requested_resolution, layers)
        };
        if next_resolution == self.resolution && next_layers == self.layers {
            return self.sync_result(false);
        }
        let (texture, atlas_view, layer_views) =
            create_shadow_texture(device, next_resolution, next_layers, self.format, true);
        self.texture = texture;
        self.atlas_view = atlas_view;
        self.layer_views = layer_views;
        self.resolution = next_resolution;
        self.layers = next_layers;
        self.version = self.version.saturating_add(1);
        self.sync_result(true)
    }

    /// Shadow metadata storage buffer.
    pub(super) fn metadata_buffer(&self) -> &wgpu::Buffer {
        self.metadata_buffer.as_ref()
    }

    /// Full atlas texture view.
    pub(super) fn atlas_view(&self) -> &wgpu::TextureView {
        self.atlas_view.as_ref()
    }

    /// Comparison sampler used by material shaders.
    pub(super) fn sampler(&self) -> &wgpu::Sampler {
        self.sampler.as_ref()
    }

    /// Single-layer render-target view for `layer`.
    pub(super) fn layer_view(&self, layer: u32) -> Option<&wgpu::TextureView> {
        self.layer_views.get(layer as usize).map(Arc::as_ref)
    }

    /// Depth format used by shadow-map render pipelines.
    pub(super) const fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Returns whether realtime shadow maps can be rendered on this adapter.
    pub(super) const fn renderable(&self) -> bool {
        self.renderable
    }

    /// Shadow-caster per-draw bind group.
    pub(super) fn per_draw_bind_group(&self) -> &wgpu::BindGroup {
        self.per_draw.bind_group.as_ref()
    }

    /// Shadow-caster per-draw storage buffer.
    pub(super) fn per_draw_storage(&self) -> &wgpu::Buffer {
        &self.per_draw.per_draw_storage
    }

    /// Shadow-layer constants bind group.
    pub(super) fn layer_uniform_bind_group(&self) -> &wgpu::BindGroup {
        self.layer_uniform_bind_group.as_ref()
    }

    /// Shadow-layer constants storage buffer.
    pub(super) fn layer_uniform_buffer(&self) -> &wgpu::Buffer {
        self.layer_uniform_buffer.as_ref()
    }

    /// Reusable CPU scratch for packing shadow caster draw rows.
    fn with_scratch(&self, f: impl FnOnce(&mut Vec<PaddedShadowCasterDraw>)) {
        f(&mut self.scratch.lock());
    }

    /// Current bind-resource version.
    pub(super) const fn version(&self) -> u64 {
        self.version
    }

    /// Retains shadow atlas resources that may be referenced by submitted command buffers.
    pub(super) fn retain_submit_resources(&self, resources: &mut crate::gpu::GpuRetainedResources) {
        resources.retain_texture(self.texture.as_ref().clone());
        resources.retain_texture_view(self.atlas_view.as_ref().clone());
        resources.retain_texture_views(self.layer_views.iter().map(|view| view.as_ref().clone()));
        resources.retain_sampler(self.sampler.as_ref().clone());
        resources.retain_buffer(self.metadata_buffer.as_ref().clone());
        self.per_draw.retain_submit_resources(resources);
        resources.retain_buffer(self.layer_uniform_buffer.as_ref().clone());
        resources.retain_bind_group(self.layer_uniform_bind_group.as_ref().clone());
    }

    fn sync_result(&self, changed: bool) -> ShadowResourceSyncResult {
        ShadowResourceSyncResult {
            changed,
            resolution: self.resolution,
        }
    }

    fn ensure_layer_uniform_capacity(&mut self, device: &wgpu::Device, need_layers: usize) {
        if need_layers <= self.layer_uniform_capacity {
            return;
        }
        let next = (4 * need_layers)
            .div_ceil(3)
            .max(16)
            .min(usize::try_from(u32::MAX).unwrap_or(usize::MAX));
        let (buffer, bind_group) =
            create_shadow_layer_uniforms(device, self.layer_uniform_layout.as_ref(), next);
        self.layer_uniform_buffer = buffer;
        self.layer_uniform_bind_group = bind_group;
        self.layer_uniform_capacity = next;
    }
}

fn shadow_per_draw_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow_caster_per_draw"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(PER_DRAW_UNIFORM_STRIDE as u64),
            },
            count: None,
        }],
    })
}

fn shadow_layer_uniform_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("shadow_caster_layer_uniform"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(PER_DRAW_UNIFORM_STRIDE as u64),
            },
            count: None,
        }],
    })
}

fn create_shadow_layer_uniforms(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    capacity: usize,
) -> (Arc<wgpu::Buffer>, Arc<wgpu::BindGroup>) {
    let size = (capacity.max(1) * PER_DRAW_UNIFORM_STRIDE) as u64;
    let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow_layer_uniforms"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    crate::profiling::note_resource_churn!(Buffer, "backend::shadow_layer_uniforms");
    let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow_layer_uniforms"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buffer.as_ref(),
                offset: 0,
                size: wgpu::BufferSize::new(PER_DRAW_UNIFORM_STRIDE as u64),
            }),
        }],
    }));
    crate::profiling::note_resource_churn!(BindGroup, "backend::shadow_layer_uniforms");
    (buffer, bind_group)
}

fn initial_shadow_resolution(limits: &GpuLimits) -> u32 {
    clamp_shadow_resolution(limits, 1)
}

fn clamp_shadow_resolution(limits: &GpuLimits, requested: u32) -> u32 {
    requested.clamp(1, limits.wgpu.max_texture_dimension_2d.max(1))
}

fn clamp_shadow_texture_resolution(
    limits: &GpuLimits,
    requested: u32,
    layers: u32,
    format: wgpu::TextureFormat,
) -> u32 {
    clamp_shadow_atlas_resolution(limits, requested, layers, format)
}

fn create_shadow_texture(
    device: &wgpu::Device,
    resolution: u32,
    layers: u32,
    format: wgpu::TextureFormat,
    renderable: bool,
) -> (
    Arc<wgpu::Texture>,
    Arc<wgpu::TextureView>,
    Vec<Arc<wgpu::TextureView>>,
) {
    let usage = if renderable {
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT
    } else {
        wgpu::TextureUsages::TEXTURE_BINDING
    };
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow_depth_atlas"),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    }));
    let atlas_view =
        Arc::new(texture.create_view(&shadow_atlas_array_view_descriptor(layers, format)));
    crate::profiling::note_resource_churn!(TextureView, "backend::shadow_depth_atlas_array");
    let layer_views = if renderable {
        let mut layer_views = Vec::with_capacity(layers as usize);
        for layer in 0..layers {
            layer_views.push(Arc::new(
                texture.create_view(&shadow_atlas_layer_view_descriptor(layer, format)),
            ));
            crate::profiling::note_resource_churn!(
                TextureView,
                "backend::shadow_depth_atlas_layer"
            );
        }
        layer_views
    } else {
        Vec::new()
    };
    (texture, atlas_view, layer_views)
}

fn shadow_atlas_array_view_descriptor(
    layers: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor {
        label: Some("shadow_depth_atlas_array"),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        aspect: wgpu::TextureAspect::DepthOnly,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(layers),
    }
}

fn shadow_atlas_layer_view_descriptor(
    layer: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor {
        label: Some("shadow_depth_atlas_layer"),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
        aspect: wgpu::TextureAspect::DepthOnly,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: layer,
        array_layer_count: Some(1),
    }
}

impl FrameGpuResources {
    /// Records all planned shadow atlas layers for this frame.
    pub(in crate::backend) fn encode_shadow_atlas(
        &self,
        plan: &ShadowFramePlan,
        params: ShadowAtlasEncodeParams<'_, '_, '_>,
    ) {
        profiling::scope!("shadows::encode_atlas");
        if plan.render_views.is_empty() || !self.shadows.renderable() {
            return;
        }
        let layer_plans = plan
            .render_views
            .iter()
            .filter_map(|view| {
                plan.caster_sets
                    .get(view.caster_set_index)
                    .map(|caster_set| ShadowLayerPlan { view, caster_set })
            })
            .collect::<Vec<_>>();
        self.pack_shadow_slabs(plan, params.gpu_limits, params.uploads);
        self.pack_shadow_layer_uniforms(plan, params.uploads);
        plot_shadow_atlas(plan);
        let mut encode_refs = WorldMeshForwardEncodeRefs {
            materials: params.materials,
            mesh_pool: params.asset_resources.mesh_pool(),
            texture_pool: params.asset_resources.texture_pool(),
            texture3d_pool: params.asset_resources.texture3d_pool(),
            cubemap_pool: params.asset_resources.cubemap_pool(),
            render_texture_pool: params.asset_resources.render_texture_pool(),
            video_texture_pool: params.asset_resources.video_texture_pool(),
            skin_cache: params.skin_cache,
        };
        let pipeline = shadow_pipeline_state(self.shadows.format());
        let mut ctx = ShadowLayerEncodeContext {
            device: params.device,
            encoder: params.encoder,
            pipeline: &pipeline,
            encode_refs: &mut encode_refs,
            gpu_limits: params.gpu_limits,
            profiler: params.profiler,
        };
        for layer in &layer_plans {
            self.encode_shadow_view(layer, &mut ctx);
        }
    }

    fn encode_shadow_view(
        &self,
        layer: &ShadowLayerPlan<'_>,
        ctx: &mut ShadowLayerEncodeContext<'_, '_, '_>,
    ) {
        let view = layer.view;
        let Some(layer_view) = self.shadow_layer_view(view.layer) else {
            return;
        };
        let caster_set = layer.caster_set;
        if caster_set.draws.is_empty() {
            clear_shadow_layer(ctx.encoder, layer_view, ctx.profiler);
            return;
        }
        let Some(layer_uniform_offset) = shadow_layer_uniform_offset(view.layer) else {
            return;
        };
        let pass_query = ctx
            .profiler
            .map(|p| p.begin_pass_query("shadows::atlas_layer", ctx.encoder));
        let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
        {
            let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_atlas_layer"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes,
                multiview_mask: None,
            });
            rpass.set_viewport(
                0.0,
                0.0,
                view.resolution as f32,
                view.resolution as f32,
                0.0,
                1.0,
            );
            rpass.set_bind_group(
                1,
                self.shadows.layer_uniform_bind_group(),
                &[layer_uniform_offset],
            );
            for phase in WorldMeshPhase::PRIMARY_FORWARD {
                draw_shadow_depth_subset(ShadowDepthDrawBatch {
                    rpass: &mut rpass,
                    groups: view.groups(phase),
                    draws: &caster_set.draws,
                    encode: &mut *ctx.encode_refs,
                    gpu_limits: ctx.gpu_limits,
                    per_draw_bind_group: self.shadow_per_draw_bind_group(),
                    slab_slot_offset: caster_set.slab_slot_offset,
                    radial_shadow: shadow_view_uses_radial_depth(view.kind),
                    supports_base_instance: ctx.gpu_limits.supports_base_instance,
                    pipeline: ctx.pipeline,
                    device: ctx.device,
                });
            }
        }
        if let Some(query) = pass_query
            && let Some(p) = ctx.profiler
        {
            p.end_query(ctx.encoder, query);
        }
    }

    fn pack_shadow_slabs(
        &self,
        plan: &ShadowFramePlan,
        gpu_limits: &GpuLimits,
        uploads: GraphUploadSink<'_>,
    ) {
        profiling::scope!("shadows::pack_slabs");
        if plan.requested_draw_slots == 0 {
            return;
        }
        self.shadows.with_scratch(|uniforms| {
            uniforms.clear();
            uniforms.resize_with(plan.requested_draw_slots, PaddedShadowCasterDraw::zeroed);
            for caster_set in &plan.caster_sets {
                let start = caster_set.slab_slot_offset;
                let Some(end) = start.checked_add(caster_set.draws.len()) else {
                    continue;
                };
                let Some(slots) = uniforms.get_mut(start..end) else {
                    continue;
                };
                pack_shadow_uniforms(slots, caster_set, gpu_limits);
            }
            uploads.write_buffer(
                self.shadow_per_draw_storage(),
                0,
                bytemuck::cast_slice(uniforms.as_slice()),
            );
        });
    }

    fn pack_shadow_layer_uniforms(&self, plan: &ShadowFramePlan, uploads: GraphUploadSink<'_>) {
        profiling::scope!("shadows::pack_layer_uniforms");
        if plan.render_views.is_empty() {
            return;
        }
        let mut layer_scratch = self.shadows.layer_scratch.lock();
        layer_scratch.clear();
        layer_scratch.resize_with(plan.render_views.len(), PaddedShadowLayerUniforms::zeroed);
        for view in &plan.render_views {
            let Some(slot) = layer_scratch.get_mut(view.layer as usize) else {
                continue;
            };
            *slot = PaddedShadowLayerUniforms::new(view);
        }
        uploads.write_buffer(
            self.shadows.layer_uniform_buffer(),
            0,
            bytemuck::cast_slice(layer_scratch.as_slice()),
        );
    }
}

fn clear_shadow_layer(
    encoder: &mut wgpu::CommandEncoder,
    layer_view: &wgpu::TextureView,
    profiler: Option<&crate::profiling::GpuProfilerHandle>,
) {
    let pass_query = profiler.map(|p| p.begin_pass_query("shadows::atlas_clear", encoder));
    let timestamp_writes = crate::profiling::render_pass_timestamp_writes(pass_query.as_ref());
    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow_atlas_clear"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: layer_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes,
            multiview_mask: None,
        });
    }
    if let Some(query) = pass_query
        && let Some(p) = profiler
    {
        p.end_query(encoder, query);
    }
}

fn pack_shadow_uniforms(
    uniforms: &mut [PaddedShadowCasterDraw],
    caster_set: &ShadowCasterSet,
    gpu_limits: &GpuLimits,
) {
    let admission =
        admit_render_command_items(caster_set.draws.len(), current_reference_worker_count());
    record_parallel_admission(
        "shadow_slab_pack",
        caster_set.draws.len(),
        caster_set.draws.len(),
        admission,
    );
    let slab_layout = &caster_set.instance_plan.slab_layout;
    let pack_one = |slot: &mut PaddedShadowCasterDraw, draw_idx: usize| {
        let item = &caster_set.draws[draw_idx];
        *slot = PaddedShadowCasterDraw::new(item);
    };
    if caster_set.draws.len() >= SHADOW_SLAB_PARALLEL_MIN_DRAWS && admission.is_parallel() {
        uniforms
            .par_chunks_mut(SHADOW_SLAB_PARALLEL_CHUNK_DRAWS)
            .with_min_len(SHADOW_SLAB_PARALLEL_CHUNKS_PER_TASK)
            .zip(
                slab_layout
                    .par_chunks(SHADOW_SLAB_PARALLEL_CHUNK_DRAWS)
                    .with_min_len(SHADOW_SLAB_PARALLEL_CHUNKS_PER_TASK),
            )
            .for_each(|(slots, layout)| {
                profiling::scope!("shadows::pack_slab::worker");
                for (slot, &draw_idx) in slots.iter_mut().zip(layout.iter()) {
                    pack_one(slot, draw_idx);
                }
            });
    } else {
        for (slot, &draw_idx) in uniforms.iter_mut().zip(slab_layout.iter()) {
            pack_one(slot, draw_idx);
        }
    }
    if !gpu_limits.supports_base_instance {
        debug_assert_eq!(
            uniforms.len(),
            slab_layout.len(),
            "downlevel shadow slabs still pack one slot per singleton draw group"
        );
    }
}

fn shadow_layer_uniform_offset(layer: u32) -> Option<u32> {
    layer.checked_mul(PER_DRAW_UNIFORM_STRIDE as u32)
}

fn shadow_caster_model(item: &crate::world_mesh::WorldMeshDrawItem) -> Mat4 {
    if item.world_space_deformed {
        Mat4::IDENTITY
    } else {
        item.rigid_world_matrix.unwrap_or(Mat4::IDENTITY)
    }
}

fn shadow_pipeline_state(format: wgpu::TextureFormat) -> WorldMeshForwardPipelineState {
    WorldMeshForwardPipelineState {
        use_multiview: false,
        pass_desc: MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
            depth_stencil_format: Some(format),
            sample_count: 1,
            multiview_mask: None,
        },
        shader_perm: ShaderPermutation::default(),
        front_face_flip: false,
    }
}

#[cfg(test)]
mod tests;
