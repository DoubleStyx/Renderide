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
use super::super::frame_resource_manager::{ShadowFramePlan, ShadowRenderView};
use super::super::per_draw_resources::PerDrawResources;
use super::super::shadow_atlas_format::{
    select_shadow_atlas_binding_format, select_shadow_atlas_format,
};
use super::FrameGpuResources;

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
struct PaddedShadowCasterUniforms {
    view_proj_left: [f32; 16],
    view_proj_right: [f32; 16],
    model: [f32; 16],
    normal_matrix: [[f32; 4]; 3],
    light_position_range: [f32; 4],
    shadow_params: [f32; 4],
    _pad: [[f32; 4]; 15],
}

impl PaddedShadowCasterUniforms {
    #[inline]
    fn new(view: &ShadowRenderView, item: &crate::world_mesh::WorldMeshDrawItem) -> Self {
        let radial_shadow = shadow_view_uses_radial_depth(view.kind);
        let model = shadow_caster_model(item);
        let view_proj = view.view_proj.to_cols_array();
        Self {
            view_proj_left: view_proj,
            view_proj_right: view_proj,
            model: model.to_cols_array(),
            normal_matrix: SHADOW_NORMAL_MATRIX_IDENTITY,
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
            _pad: [[0.0; 4]; 15],
        }
    }
}

#[inline]
fn shadow_view_uses_radial_depth(kind: u32) -> bool {
    matches!(kind, SHADOW_VIEW_KIND_POINT | SHADOW_VIEW_KIND_SPOT)
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
    instance_plan: crate::world_mesh::InstancePlan,
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
    scratch: parking_lot::Mutex<Vec<PaddedShadowCasterUniforms>>,
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
            scratch: parking_lot::Mutex::new(Vec::new()),
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
    ) -> bool {
        self.per_draw
            .ensure_draw_slot_capacity(device, requested_draw_slots);
        if !self.renderable {
            return false;
        }
        let resolution = clamp_shadow_resolution(limits, requested_resolution);
        let layers = requested_layers
            .max(1)
            .min(limits.wgpu.max_texture_array_layers.max(1));
        if resolution <= self.resolution && layers <= self.layers {
            return false;
        }
        let next_resolution = self.resolution.max(resolution);
        let next_layers = self.layers.max(layers);
        let (texture, atlas_view, layer_views) =
            create_shadow_texture(device, next_resolution, next_layers, self.format, true);
        self.texture = texture;
        self.atlas_view = atlas_view;
        self.layer_views = layer_views;
        self.resolution = next_resolution;
        self.layers = next_layers;
        self.version = self.version.saturating_add(1);
        true
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

    /// Reusable CPU scratch for packing one shadow layer's per-draw slab.
    fn with_scratch(&self, f: impl FnOnce(&mut Vec<PaddedShadowCasterUniforms>)) {
        f(&mut self.scratch.lock());
    }

    /// Current bind-resource version.
    pub(super) const fn version(&self) -> u64 {
        self.version
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

fn initial_shadow_resolution(limits: &GpuLimits) -> u32 {
    clamp_shadow_resolution(limits, 1)
}

fn clamp_shadow_resolution(limits: &GpuLimits, requested: u32) -> u32 {
    requested.clamp(1, limits.wgpu.max_texture_dimension_2d.max(1))
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
            .map(|view| ShadowLayerPlan {
                view,
                instance_plan: crate::world_mesh::build_plan_for_shader(
                    &view.draws,
                    params.gpu_limits.supports_base_instance,
                    ShaderPermutation::default(),
                ),
            })
            .collect::<Vec<_>>();
        self.pack_shadow_slabs(plan, &layer_plans, params.gpu_limits, params.uploads);
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
        if view.draws.is_empty() {
            clear_shadow_layer(ctx.encoder, layer_view, ctx.profiler);
            return;
        }
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
            for phase in WorldMeshPhase::PRIMARY_FORWARD {
                draw_shadow_depth_subset(ShadowDepthDrawBatch {
                    rpass: &mut rpass,
                    groups: layer.instance_plan.phase(phase),
                    draws: &view.draws,
                    encode: &mut *ctx.encode_refs,
                    gpu_limits: ctx.gpu_limits,
                    per_draw_bind_group: self.shadow_per_draw_bind_group(),
                    slab_slot_offset: view.slab_slot_offset,
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
        layer_plans: &[ShadowLayerPlan<'_>],
        gpu_limits: &GpuLimits,
        uploads: GraphUploadSink<'_>,
    ) {
        profiling::scope!("shadows::pack_slabs");
        if plan.requested_draw_slots == 0 {
            return;
        }
        self.shadows.with_scratch(|uniforms| {
            uniforms.clear();
            uniforms.resize_with(
                plan.requested_draw_slots,
                PaddedShadowCasterUniforms::zeroed,
            );
            for layer in layer_plans {
                let start = layer.view.slab_slot_offset;
                let Some(end) = start.checked_add(layer.view.draws.len()) else {
                    continue;
                };
                let Some(slots) = uniforms.get_mut(start..end) else {
                    continue;
                };
                pack_shadow_uniforms(
                    slots,
                    layer.view,
                    &layer.instance_plan.slab_layout,
                    gpu_limits,
                );
            }
            uploads.write_buffer(
                self.shadow_per_draw_storage(),
                0,
                bytemuck::cast_slice(uniforms.as_slice()),
            );
        });
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
    uniforms: &mut [PaddedShadowCasterUniforms],
    view: &ShadowRenderView,
    slab_layout: &[usize],
    gpu_limits: &GpuLimits,
) {
    let admission = admit_render_command_items(view.draws.len(), current_reference_worker_count());
    record_parallel_admission(
        "shadow_slab_pack",
        view.draws.len(),
        view.draws.len(),
        admission,
    );
    let pack_one = |slot: &mut PaddedShadowCasterUniforms, draw_idx: usize| {
        let item = &view.draws[draw_idx];
        *slot = PaddedShadowCasterUniforms::new(view, item);
    };
    if view.draws.len() >= SHADOW_SLAB_PARALLEL_MIN_DRAWS && admission.is_parallel() {
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
mod tests {
    use std::mem::size_of;
    use std::sync::Arc;

    use glam::{Mat4, Vec3};
    use hashbrown::HashMap;

    use super::{
        PaddedShadowCasterUniforms, clamp_shadow_resolution, shadow_atlas_array_view_descriptor,
        shadow_atlas_layer_view_descriptor, shadow_pipeline_state,
    };
    use crate::backend::frame_resource_manager::ShadowRenderView;
    use crate::gpu::{SHADOW_VIEW_KIND_DIRECTIONAL, SHADOW_VIEW_KIND_POINT, SHADOW_VIEW_KIND_SPOT};
    use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn limits(
        max_texture_dimension_2d: u32,
        max_texture_array_layers: u32,
    ) -> crate::gpu::GpuLimits {
        crate::gpu::GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_2d,
                max_texture_array_layers,
                ..Default::default()
            },
            wgpu::Features::empty(),
            HashMap::new(),
        )
    }

    fn dummy_draw_item() -> crate::world_mesh::WorldMeshDrawItem {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        })
    }

    fn shadow_view(kind: u32) -> ShadowRenderView {
        ShadowRenderView {
            layer: 0,
            kind,
            resolution: 512,
            view_proj: Mat4::from_scale(Vec3::splat(2.0)),
            light_position: Vec3::new(1.0, 2.0, 3.0),
            light_range: 12.0,
            shadow_bias: 0.25,
            slab_slot_offset: 0,
            draws: Arc::from(Vec::<crate::world_mesh::WorldMeshDrawItem>::new()),
        }
    }

    #[test]
    fn shadow_resolution_clamps_to_device_limit() {
        let limits = limits(1024, 8);
        assert_eq!(clamp_shadow_resolution(&limits, 0), 1);
        assert_eq!(clamp_shadow_resolution(&limits, 512), 512);
        assert_eq!(clamp_shadow_resolution(&limits, 2048), 1024);
    }

    #[test]
    fn shadow_atlas_array_view_is_sampled_only() {
        let format = wgpu::TextureFormat::Depth24Plus;
        let desc = shadow_atlas_array_view_descriptor(4, format);

        assert_eq!(desc.format, Some(format));
        assert_eq!(desc.dimension, Some(wgpu::TextureViewDimension::D2Array));
        assert_eq!(desc.usage, Some(wgpu::TextureUsages::TEXTURE_BINDING));
        assert_eq!(desc.aspect, wgpu::TextureAspect::DepthOnly);
        assert_eq!(desc.base_mip_level, 0);
        assert_eq!(desc.mip_level_count, Some(1));
        assert_eq!(desc.base_array_layer, 0);
        assert_eq!(desc.array_layer_count, Some(4));
    }

    #[test]
    fn shadow_atlas_layer_view_is_render_attachment_only() {
        let format = wgpu::TextureFormat::Depth16Unorm;
        let desc = shadow_atlas_layer_view_descriptor(3, format);

        assert_eq!(desc.format, Some(format));
        assert_eq!(desc.dimension, Some(wgpu::TextureViewDimension::D2));
        assert_eq!(desc.usage, Some(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert_eq!(desc.aspect, wgpu::TextureAspect::DepthOnly);
        assert_eq!(desc.base_mip_level, 0);
        assert_eq!(desc.mip_level_count, Some(1));
        assert_eq!(desc.base_array_layer, 3);
        assert_eq!(desc.array_layer_count, Some(1));
    }

    #[test]
    fn shadow_pipeline_state_uses_selected_depth_format() {
        let format = wgpu::TextureFormat::Depth24Plus;
        let pipeline = shadow_pipeline_state(format);

        assert_eq!(pipeline.pass_desc.depth_stencil_format, Some(format));
    }

    #[test]
    fn shadow_caster_uniform_stride_matches_dynamic_offset_stride() {
        assert_eq!(
            size_of::<PaddedShadowCasterUniforms>(),
            PER_DRAW_UNIFORM_STRIDE
        );
    }

    #[test]
    fn radial_shadow_caster_uniforms_pack_light_data() {
        let mut item = dummy_draw_item();
        let model = Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0));
        item.rigid_world_matrix = Some(model);

        for kind in [SHADOW_VIEW_KIND_POINT, SHADOW_VIEW_KIND_SPOT] {
            let slot = PaddedShadowCasterUniforms::new(&shadow_view(kind), &item);

            assert_eq!(
                slot.view_proj_left,
                Mat4::from_scale(Vec3::splat(2.0)).to_cols_array()
            );
            assert_eq!(
                slot.view_proj_right,
                Mat4::from_scale(Vec3::splat(2.0)).to_cols_array()
            );
            assert_eq!(slot.model, model.to_cols_array());
            assert_eq!(slot.light_position_range, [1.0, 2.0, 3.0, 12.0]);
            assert_eq!(slot.shadow_params[0], 0.25);
        }
    }

    #[test]
    fn shadow_caster_uniform_prefix_matches_forward_per_draw_layout() {
        let mut item = dummy_draw_item();
        let model = Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0));
        item.rigid_world_matrix = Some(model);

        let slot = PaddedShadowCasterUniforms::new(&shadow_view(SHADOW_VIEW_KIND_POINT), &item);
        let forward_prefix: &crate::mesh_deform::PaddedPerDrawUniforms =
            bytemuck::from_bytes(bytemuck::bytes_of(&slot));

        assert_eq!(forward_prefix.view_proj_left, slot.view_proj_left);
        assert_eq!(forward_prefix.view_proj_right, slot.view_proj_right);
        assert_eq!(forward_prefix.model, slot.model);
    }

    #[test]
    fn projected_shadow_caster_uniforms_do_not_pack_radial_bias() {
        let item = dummy_draw_item();

        let slot =
            PaddedShadowCasterUniforms::new(&shadow_view(SHADOW_VIEW_KIND_DIRECTIONAL), &item);

        assert_eq!(slot.light_position_range, [0.0; 4]);
        assert_eq!(slot.shadow_params[0], 0.0);
    }

    #[test]
    fn shadow_caster_uniforms_use_identity_model_for_world_space_positions() {
        let mut item = dummy_draw_item();
        item.world_space_deformed = true;
        item.rigid_world_matrix = Some(Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0)));

        let slot = PaddedShadowCasterUniforms::new(&shadow_view(SHADOW_VIEW_KIND_POINT), &item);

        assert_eq!(slot.model, Mat4::IDENTITY.to_cols_array());
    }
}
