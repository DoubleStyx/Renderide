//! Generic safe-opaque world-mesh depth prepass.
//!
//! This pass clears and fills the main forward depth attachment before color shading. It mirrors
//! only conservative opaque draws so missing coverage falls back to the normal forward pass.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::LazyLock;

use crate::embedded_shaders::embedded_wgsl;
use crate::gpu_resource::{OnceGpu, RenderPipelineMap};
use crate::graph_inputs::PerViewFramePlanSlot;
use crate::materials::{
    RasterFrontFace, RasterPipelineKind, RasterPrimitiveTopology, ShadowCasterPolicy,
    embedded_stem_depth_prepass_pass, materialized_pass_for_blend_mode,
    shadow_caster_policy_for_pipeline,
};
use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::stereo_mask_or_template;
use crate::render_graph::pass::{PassBuilder, RasterPass, RenderPassTemplate};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess, TextureHandle,
};
use crate::shared::ShadowCastMode;
use crate::world_mesh::{MeshPassKind, WorldMeshDrawItem};

use super::attachments::declare_forward_depth_attachment;
use super::depth_like_pipeline::{DepthLikePipelineSpec, create_depth_like_pipeline};
use super::encode::{DepthPrepassDrawBatch, draw_depth_prepass_subset};
use super::{WorldMeshForwardPipelineState, WorldMeshForwardPlanSlot};

const POSITION_ATTRIBUTES: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
    offset: 0,
    shader_location: 0,
    format: wgpu::VertexFormat::Float32x4,
}];

/// Graph handles used by [`WorldMeshForwardDepthPrepass`].
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshForwardDepthPrepassGraphResources {
    /// Imported frame depth target.
    pub depth: ImportedTextureHandle,
    /// Graph-owned forward depth target used when MSAA is active.
    pub msaa_depth: Option<TextureHandle>,
    /// Imported per-draw storage slab.
    pub per_draw_slab: ImportedBufferHandle,
}

/// Depth-only prepass for conservative opaque world-mesh draws.
#[derive(Debug)]
pub struct WorldMeshForwardDepthPrepass {
    resources: WorldMeshForwardDepthPrepassGraphResources,
    pipelines: &'static WorldMeshForwardDepthPrepassPipelineCache,
}

impl WorldMeshForwardDepthPrepass {
    /// Creates the generic world-mesh depth prepass.
    pub fn new(resources: WorldMeshForwardDepthPrepassGraphResources) -> Self {
        Self {
            resources,
            pipelines: depth_prepass_pipelines(),
        }
    }
}

/// Pipeline selectors for one depth-prepass variant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct WorldMeshForwardDepthPrepassPipelineKey {
    /// Depth/stencil format of the active forward depth target.
    pub depth_stencil_format: wgpu::TextureFormat,
    /// Active depth sample count.
    pub sample_count: u32,
    /// Multiview mask when the pass renders stereo in one draw.
    pub multiview_mask: Option<NonZeroU32>,
    /// Front-face winding selected from the draw transform.
    pub front_face: RasterFrontFace,
    /// Backface culling mode resolved for this draw.
    pub cull_mode: Option<wgpu::Face>,
    /// Primitive topology baked into the render pipeline.
    pub primitive_topology: RasterPrimitiveTopology,
    /// Depth compare function baked into the render pipeline.
    pub depth_compare: wgpu::CompareFunction,
}

/// Cached render pipelines and bind layout for the generic depth prepass.
#[derive(Debug, Default)]
pub(super) struct WorldMeshForwardDepthPrepassPipelineCache {
    per_draw_layout: OnceGpu<wgpu::BindGroupLayout>,
    pipelines: RenderPipelineMap<WorldMeshForwardDepthPrepassPipelineKey>,
}

/// Cached render pipelines and bind layouts for projected shadow casters.
#[derive(Debug, Default)]
pub(super) struct WorldMeshForwardShadowPipelineCache {
    per_draw_layout: OnceGpu<wgpu::BindGroupLayout>,
    layer_layout: OnceGpu<wgpu::BindGroupLayout>,
    pipelines: RenderPipelineMap<WorldMeshForwardDepthPrepassPipelineKey>,
}

/// Cached render pipelines and bind layout for radial point and spot shadow casters.
#[derive(Debug, Default)]
pub(super) struct WorldMeshForwardRadialShadowPipelineCache {
    per_draw_layout: OnceGpu<wgpu::BindGroupLayout>,
    layer_layout: OnceGpu<wgpu::BindGroupLayout>,
    pipelines: RenderPipelineMap<WorldMeshForwardDepthPrepassPipelineKey>,
}

#[derive(Clone, Copy)]
struct DepthPrepassCullMode(Option<wgpu::Face>);

impl WorldMeshForwardDepthPrepassPipelineCache {
    /// Returns the matching depth-prepass pipeline.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        key: WorldMeshForwardDepthPrepassPipelineKey,
    ) -> Arc<wgpu::RenderPipeline> {
        self.pipelines
            .get_or_create(key, |key| self.create_pipeline(device, *key))
    }

    fn create_pipeline(
        &self,
        device: &wgpu::Device,
        key: WorldMeshForwardDepthPrepassPipelineKey,
    ) -> wgpu::RenderPipeline {
        profiling::scope!("world_mesh_forward::depth_prepass_pipeline");
        let multiview = key.multiview_mask.is_some();
        let (label, source) = if multiview {
            (
                "world_mesh_depth_prepass_multiview",
                embedded_wgsl!("world_mesh_depth_prepass_multiview"),
            )
        } else {
            (
                "world_mesh_depth_prepass_default",
                embedded_wgsl!("world_mesh_depth_prepass_default"),
            )
        };
        logger::debug!(
            "world mesh depth prepass: building pipeline sample_count={} multiview={} topology={:?}",
            key.sample_count,
            multiview,
            key.primitive_topology
        );
        let per_draw_layout = self.per_draw_layout(device);
        let bind_group_layouts = [per_draw_layout];
        let vertex_buffers = depth_prepass_vertex_buffer_layouts();
        let pipeline = create_depth_like_pipeline(
            device,
            DepthLikePipelineSpec {
                label,
                shader_source: source,
                bind_group_layouts: &bind_group_layouts,
                vertex_buffers: &vertex_buffers,
                fragment_entry_point: None,
                color_targets: &[],
                primitive_topology: key.primitive_topology,
                front_face: key.front_face,
                cull_mode: key.cull_mode,
                depth_stencil_format: key.depth_stencil_format,
                depth_write_enabled: true,
                depth_compare: key.depth_compare,
                sample_count: key.sample_count,
                multiview_mask: key.multiview_mask,
            },
        );
        crate::profiling::note_resource_churn!(
            RenderPipeline,
            "passes::world_mesh_depth_prepass_pipeline"
        );
        pipeline
    }

    fn per_draw_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.per_draw_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("world_mesh_depth_prepass_per_draw"),
                entries: &depth_prepass_per_draw_layout_entries(),
            })
        })
    }
}

impl WorldMeshForwardShadowPipelineCache {
    /// Returns the matching projected-shadow caster pipeline.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        key: WorldMeshForwardDepthPrepassPipelineKey,
    ) -> Arc<wgpu::RenderPipeline> {
        self.pipelines
            .get_or_create(key, |key| self.create_pipeline(device, *key))
    }

    fn create_pipeline(
        &self,
        device: &wgpu::Device,
        key: WorldMeshForwardDepthPrepassPipelineKey,
    ) -> wgpu::RenderPipeline {
        profiling::scope!("world_mesh_forward::shadow_pipeline");
        let label = "world_mesh_shadow_caster";
        logger::debug!(
            "world mesh shadow caster: building pipeline sample_count={} topology={:?}",
            key.sample_count,
            key.primitive_topology
        );
        let per_draw_layout = self.per_draw_layout(device);
        let layer_layout = self.layer_layout(device);
        let bind_group_layouts = [per_draw_layout, layer_layout];
        let vertex_buffers = depth_prepass_vertex_buffer_layouts();
        let pipeline = create_depth_like_pipeline(
            device,
            DepthLikePipelineSpec {
                label,
                shader_source: embedded_wgsl!("world_mesh_shadow_caster"),
                bind_group_layouts: &bind_group_layouts,
                vertex_buffers: &vertex_buffers,
                fragment_entry_point: None,
                color_targets: &[],
                primitive_topology: key.primitive_topology,
                front_face: key.front_face,
                cull_mode: key.cull_mode,
                depth_stencil_format: key.depth_stencil_format,
                depth_write_enabled: true,
                depth_compare: key.depth_compare,
                sample_count: key.sample_count,
                multiview_mask: key.multiview_mask,
            },
        );
        crate::profiling::note_resource_churn!(
            RenderPipeline,
            "passes::world_mesh_shadow_caster_pipeline"
        );
        pipeline
    }

    fn per_draw_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.per_draw_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("world_mesh_shadow_caster_per_draw"),
                entries: &depth_prepass_per_draw_layout_entries(),
            })
        })
    }

    fn layer_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.layer_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("world_mesh_shadow_caster_layer"),
                entries: &shadow_layer_layout_entries(),
            })
        })
    }
}

impl WorldMeshForwardRadialShadowPipelineCache {
    /// Returns the matching radial-shadow caster pipeline.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        key: WorldMeshForwardDepthPrepassPipelineKey,
    ) -> Arc<wgpu::RenderPipeline> {
        self.pipelines
            .get_or_create(key, |key| self.create_pipeline(device, *key))
    }

    fn create_pipeline(
        &self,
        device: &wgpu::Device,
        key: WorldMeshForwardDepthPrepassPipelineKey,
    ) -> wgpu::RenderPipeline {
        profiling::scope!("world_mesh_forward::radial_shadow_pipeline");
        let label = "world_mesh_radial_shadow_caster";
        logger::debug!(
            "world mesh radial shadow caster: building pipeline sample_count={} topology={:?}",
            key.sample_count,
            key.primitive_topology
        );
        let per_draw_layout = self.per_draw_layout(device);
        let layer_layout = self.layer_layout(device);
        let bind_group_layouts = [per_draw_layout, layer_layout];
        let vertex_buffers = depth_prepass_vertex_buffer_layouts();
        let pipeline = create_depth_like_pipeline(
            device,
            DepthLikePipelineSpec {
                label,
                shader_source: embedded_wgsl!("world_mesh_radial_shadow_caster"),
                bind_group_layouts: &bind_group_layouts,
                vertex_buffers: &vertex_buffers,
                fragment_entry_point: Some("fs_main"),
                color_targets: &[],
                primitive_topology: key.primitive_topology,
                front_face: key.front_face,
                cull_mode: key.cull_mode,
                depth_stencil_format: key.depth_stencil_format,
                depth_write_enabled: true,
                depth_compare: key.depth_compare,
                sample_count: key.sample_count,
                multiview_mask: key.multiview_mask,
            },
        );
        crate::profiling::note_resource_churn!(
            RenderPipeline,
            "passes::world_mesh_radial_shadow_caster_pipeline"
        );
        pipeline
    }

    fn per_draw_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.per_draw_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("world_mesh_radial_shadow_caster_per_draw"),
                entries: &depth_prepass_per_draw_layout_entries(),
            })
        })
    }

    fn layer_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.layer_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("world_mesh_radial_shadow_caster_layer"),
                entries: &shadow_layer_layout_entries(),
            })
        })
    }
}

impl WorldMeshForwardDepthPrepassPipelineKey {
    /// Builds a depth-prepass pipeline key for one draw group.
    pub(crate) fn for_draw(
        item: &WorldMeshDrawItem,
        pipeline: &WorldMeshForwardPipelineState,
    ) -> Option<Self> {
        Self::for_draw_with_compare(item, pipeline, crate::gpu::MAIN_FORWARD_DEPTH_COMPARE)
    }

    /// Builds a depth-prepass pipeline key for one draw group with an explicit depth compare.
    pub(crate) fn for_draw_with_compare(
        item: &WorldMeshDrawItem,
        pipeline: &WorldMeshForwardPipelineState,
        depth_compare: wgpu::CompareFunction,
    ) -> Option<Self> {
        let depth_stencil_format = pipeline.pass_desc.depth_stencil_format?;
        let cull_mode = depth_prepass_cull_mode(item, pipeline.shader_perm)?;
        let mut front_face = item.batch_key.front_face;
        if pipeline.front_face_flip {
            front_face = front_face.flipped();
        }
        Some(Self {
            depth_stencil_format,
            sample_count: pipeline.pass_desc.sample_count,
            multiview_mask: pipeline.pass_desc.multiview_mask,
            front_face,
            cull_mode: cull_for_topology(cull_mode.0, item.batch_key.primitive_topology),
            primitive_topology: item.batch_key.primitive_topology,
            depth_compare,
        })
    }

    /// Builds a depth-only shadow-caster pipeline key for one draw group.
    pub(crate) fn for_shadow_draw(
        item: &WorldMeshDrawItem,
        pipeline: &WorldMeshForwardPipelineState,
    ) -> Option<Self> {
        let depth_stencil_format = pipeline.pass_desc.depth_stencil_format?;
        let cull_mode = shadow_caster_cull_mode(item)?;
        let cull_mode = if item.shadow_cast_mode == ShadowCastMode::DoubleSided {
            None
        } else {
            cull_for_topology(cull_mode.0, item.batch_key.primitive_topology)
        };
        Some(Self {
            depth_stencil_format,
            sample_count: pipeline.pass_desc.sample_count,
            multiview_mask: pipeline.pass_desc.multiview_mask,
            front_face: item.batch_key.front_face,
            cull_mode,
            primitive_topology: item.batch_key.primitive_topology,
            depth_compare: wgpu::CompareFunction::LessEqual,
        })
    }
}

/// Returns the generic depth-prepass pipeline key needed by `item`, when eligible.
pub(crate) fn depth_prepass_pipeline_key_for_draw(
    item: &WorldMeshDrawItem,
    pipeline: &WorldMeshForwardPipelineState,
) -> Option<WorldMeshForwardDepthPrepassPipelineKey> {
    if !crate::world_mesh::depth_prepass_item_eligible(item, pipeline.shader_perm) {
        return None;
    }
    WorldMeshForwardDepthPrepassPipelineKey::for_draw(item, pipeline)
}

/// Pre-warms the generic depth-prepass pipeline matching `key`.
pub(crate) fn pre_warm_depth_prepass_pipeline(
    device: &wgpu::Device,
    key: WorldMeshForwardDepthPrepassPipelineKey,
) {
    let _ = depth_prepass_pipelines().pipeline(device, key);
}

fn depth_prepass_cull_mode(
    item: &WorldMeshDrawItem,
    shader_perm: crate::materials::ShaderPermutation,
) -> Option<DepthPrepassCullMode> {
    match &item.batch_key.pipeline {
        RasterPipelineKind::Null => Some(DepthPrepassCullMode(Some(wgpu::Face::Back))),
        RasterPipelineKind::EmbeddedStem(stem) => {
            let pass = embedded_stem_depth_prepass_pass(stem.as_ref(), shader_perm)?;
            let pass = materialized_pass_for_blend_mode(&pass, item.batch_key.blend_mode);
            Some(DepthPrepassCullMode(
                pass.resolved_cull_mode(item.batch_key.render_state),
            ))
        }
    }
}

fn shadow_caster_cull_mode(item: &WorldMeshDrawItem) -> Option<DepthPrepassCullMode> {
    match shadow_caster_policy_for_pipeline(&item.batch_key.pipeline) {
        ShadowCasterPolicy::None => None,
        ShadowCasterPolicy::DepthOnly => Some(DepthPrepassCullMode(
            item.batch_key
                .render_state
                .resolved_cull_mode(Some(wgpu::Face::Back)),
        )),
    }
}

fn cull_for_topology(
    cull_mode: Option<wgpu::Face>,
    primitive_topology: RasterPrimitiveTopology,
) -> Option<wgpu::Face> {
    match primitive_topology {
        RasterPrimitiveTopology::TriangleList => cull_mode,
        RasterPrimitiveTopology::PointList => None,
    }
}

pub(super) fn depth_prepass_pipelines() -> &'static WorldMeshForwardDepthPrepassPipelineCache {
    static CACHE: LazyLock<WorldMeshForwardDepthPrepassPipelineCache> =
        LazyLock::new(WorldMeshForwardDepthPrepassPipelineCache::default);
    &CACHE
}

pub(super) fn radial_shadow_pipelines() -> &'static WorldMeshForwardRadialShadowPipelineCache {
    static CACHE: LazyLock<WorldMeshForwardRadialShadowPipelineCache> =
        LazyLock::new(WorldMeshForwardRadialShadowPipelineCache::default);
    &CACHE
}

pub(super) fn shadow_pipelines() -> &'static WorldMeshForwardShadowPipelineCache {
    static CACHE: LazyLock<WorldMeshForwardShadowPipelineCache> =
        LazyLock::new(WorldMeshForwardShadowPipelineCache::default);
    &CACHE
}

fn depth_prepass_vertex_buffer_layouts() -> [wgpu::VertexBufferLayout<'static>; 1] {
    [wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &POSITION_ATTRIBUTES,
    }]
}

fn depth_prepass_per_draw_layout_entries() -> [wgpu::BindGroupLayoutEntry; 1] {
    [wgpu::BindGroupLayoutEntry {
        binding: 0,
        // The depth prepass reuses the forward per-draw bind group, so this visibility must match
        // the reflected `null_per_draw` layout even though this shader reads it only in vertex.
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(PER_DRAW_UNIFORM_STRIDE as u64),
        },
        count: None,
    }]
}

fn shadow_layer_layout_entries() -> [wgpu::BindGroupLayoutEntry; 1] {
    [wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(PER_DRAW_UNIFORM_STRIDE as u64),
        },
        count: None,
    }]
}

impl RasterPass for WorldMeshForwardDepthPrepass {
    fn name(&self) -> &str {
        "WorldMeshForwardDepthPrepass"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_blackboard::<PerViewFramePlanSlot>();
        b.read_optional_blackboard::<WorldMeshForwardPlanSlot>();
        {
            let mut r = b.raster();
            let depth_ops = wgpu::Operations {
                load: wgpu::LoadOp::Clear(crate::gpu::MAIN_FORWARD_DEPTH_CLEAR),
                store: wgpu::StoreOp::Store,
            };
            declare_forward_depth_attachment(
                &mut r,
                self.resources.depth,
                self.resources.msaa_depth,
                depth_ops,
            );
        }
        b.import_buffer(
            self.resources.per_draw_slab,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::VERTEX,
                access: StorageAccess::ReadOnly,
            },
        );
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        let use_multiview = ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .is_some_and(|prepared| prepared.pipeline.use_multiview);
        stereo_mask_or_template(use_multiview, template.multiview_mask)
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::depth_prepass_record");
        let frame = &ctx.frame;

        let Some(prepared) = ctx.blackboard.get::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        let Some(per_draw_bg) = frame
            .systems
            .frame_resources
            .per_view_per_draw_bind_group(frame.view.view_id)
        else {
            return Ok(());
        };
        let Some(gpu_limits) = frame.view.gpu_limits.clone() else {
            return Ok(());
        };
        let mut encode_refs = super::WorldMeshForwardEncodeRefs::from_pass_frame(frame);
        draw_depth_prepass_subset(DepthPrepassDrawBatch {
            rpass,
            groups: prepared
                .plan
                .phase(MeshPassKind::DepthPrepass.first_phase()),
            slab_layout: &prepared.plan.slab_layout,
            draws: &prepared.draws,
            encode: &mut encode_refs,
            gpu_limits: gpu_limits.as_ref(),
            per_draw_bind_group: per_draw_bg.as_ref(),
            supports_base_instance: prepared.supports_base_instance,
            pipeline: &prepared.pipeline,
            device: ctx.device,
            depth_pipelines: self.pipelines,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        WorldMeshForwardDepthPrepassPipelineKey, cull_for_topology,
        depth_prepass_per_draw_layout_entries,
    };
    use crate::materials::{RasterPipelineKind, RasterPrimitiveTopology, ShaderPermutation};
    use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
    use crate::passes::world_mesh_forward::WorldMeshForwardPipelineState;
    use crate::shared::ShadowCastMode;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn pipeline_state() -> WorldMeshForwardPipelineState {
        WorldMeshForwardPipelineState {
            use_multiview: false,
            pass_desc: crate::materials::MaterialPipelineDesc {
                surface_format: wgpu::TextureFormat::Rgba16Float,
                depth_stencil_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
                sample_count: 4,
                multiview_mask: std::num::NonZeroU32::new(3),
            },
            shader_perm: ShaderPermutation(0),
            front_face_flip: false,
        }
    }

    #[test]
    fn depth_prepass_key_preserves_pipeline_axes() {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.front_face = crate::materials::RasterFrontFace::CounterClockwise;

        let key = WorldMeshForwardDepthPrepassPipelineKey::for_draw(&item, &pipeline_state())
            .expect("depth key");

        assert_eq!(
            key.depth_stencil_format,
            wgpu::TextureFormat::Depth24PlusStencil8
        );
        assert_eq!(key.sample_count, 4);
        assert_eq!(key.multiview_mask, std::num::NonZeroU32::new(3));
        assert_eq!(
            key.front_face,
            crate::materials::RasterFrontFace::CounterClockwise
        );
        assert_eq!(
            key.primitive_topology,
            RasterPrimitiveTopology::TriangleList
        );
    }

    #[test]
    fn point_list_depth_prepass_disables_culling() {
        assert_eq!(
            cull_for_topology(Some(wgpu::Face::Back), RasterPrimitiveTopology::PointList),
            None
        );
        assert_eq!(
            cull_for_topology(
                Some(wgpu::Face::Back),
                RasterPrimitiveTopology::TriangleList
            ),
            Some(wgpu::Face::Back)
        );
    }

    #[test]
    fn depth_prepass_per_draw_layout_matches_forward_visibility() {
        let [entry] = depth_prepass_per_draw_layout_entries();
        assert_eq!(
            entry.visibility,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        );
        assert_eq!(
            entry.ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(PER_DRAW_UNIFORM_STRIDE as u64),
            }
        );
    }

    #[test]
    fn shadow_key_accepts_pbsmetallic_without_depth_prepass_metadata() {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.pipeline =
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsmetallic_default"));

        let key =
            WorldMeshForwardDepthPrepassPipelineKey::for_shadow_draw(&item, &pipeline_state())
                .expect("pbsmetallic casts shadows even when generic depth prepass is unavailable");

        assert_eq!(
            key.depth_stencil_format,
            wgpu::TextureFormat::Depth24PlusStencil8
        );
        assert_eq!(key.depth_compare, wgpu::CompareFunction::LessEqual);
    }

    #[test]
    fn double_sided_shadow_key_disables_culling() {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.shadow_cast_mode = ShadowCastMode::DoubleSided;
        item.batch_key.pipeline =
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsmetallic_default"));

        let key =
            WorldMeshForwardDepthPrepassPipelineKey::for_shadow_draw(&item, &pipeline_state())
                .expect("double-sided PBS shadows should still cast");

        assert_eq!(key.cull_mode, None);
    }

    #[test]
    fn shadow_key_rejects_common_unlit_without_unity_caster() {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.pipeline = RasterPipelineKind::EmbeddedStem(Arc::from("unlit_default"));

        assert!(
            WorldMeshForwardDepthPrepassPipelineKey::for_shadow_draw(&item, &pipeline_state())
                .is_none()
        );
    }
}
