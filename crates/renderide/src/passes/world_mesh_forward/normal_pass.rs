//! GTAO-only world-mesh normal prepass.
//!
//! The pass runs after opaque forward depth has been written and renders smooth vertex normals
//! into an `Rgba16Float` view-space normal target. GTAO samples that target instead of deriving
//! normals from depth planes, which avoids polygon-edge discontinuities on smooth-shaded meshes.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::LazyLock;

use crate::embedded_shaders::{GTAO_VIEW_NORMALS_DEFAULT_WGSL, GTAO_VIEW_NORMALS_MULTIVIEW_WGSL};
use crate::materials::{RasterFrontFace, RasterPrimitiveTopology};
use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::render_graph::compiled::{DepthAttachmentTemplate, RenderPassTemplate};
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::{
    OnceGpu, RenderPipelineMap, create_wgsl_shader_module, stereo_mask_or_template,
};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess, TextureHandle,
};

use super::execute_helpers::{record_world_mesh_forward_normal_graph_raster, stencil_load_ops};
use super::{WorldMeshForwardPipelineState, WorldMeshForwardPlanSlot};

/// GTAO view-space normal target format.
pub(crate) const GTAO_VIEW_NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const POSITION_ATTRIBUTES: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
    offset: 0,
    shader_location: 0,
    format: wgpu::VertexFormat::Float32x4,
}];
const NORMAL_ATTRIBUTES: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
    offset: 0,
    shader_location: 1,
    format: wgpu::VertexFormat::Float32x4,
}];

/// Graph handles used by [`WorldMeshForwardNormalPass`].
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshForwardNormalGraphResources {
    /// Single-sample view-space normal target sampled by GTAO.
    pub normals: TextureHandle,
    /// Multisampled view-space normal target used when frame MSAA is active.
    pub normals_msaa: TextureHandle,
    /// Imported frame depth target.
    pub depth: ImportedTextureHandle,
    /// Graph-owned forward depth target used when MSAA is active.
    pub msaa_depth: TextureHandle,
    /// Imported per-draw storage slab.
    pub per_draw_slab: ImportedBufferHandle,
}

/// Renders smooth view-space normals for GTAO.
#[derive(Debug)]
pub struct WorldMeshForwardNormalPass {
    resources: WorldMeshForwardNormalGraphResources,
    pipelines: &'static WorldMeshForwardNormalPipelineCache,
}

impl WorldMeshForwardNormalPass {
    /// Creates the GTAO normal prepass.
    pub fn new(resources: WorldMeshForwardNormalGraphResources) -> Self {
        Self {
            resources,
            pipelines: normal_pipelines(),
        }
    }
}

/// Pipeline selectors for one GTAO normal prepass variant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct WorldMeshForwardNormalPipelineKey {
    /// Depth/stencil format of the active forward depth target.
    pub depth_stencil_format: wgpu::TextureFormat,
    /// Active color/depth sample count.
    pub sample_count: u32,
    /// Multiview mask when the pass renders stereo in one draw.
    pub multiview_mask: Option<NonZeroU32>,
    /// Front-face winding selected from the draw transform.
    pub front_face: RasterFrontFace,
    /// Primitive topology baked into the render pipeline.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// Cached render pipelines and bind layout for the GTAO normal prepass.
#[derive(Debug, Default)]
pub(super) struct WorldMeshForwardNormalPipelineCache {
    per_draw_layout: OnceGpu<wgpu::BindGroupLayout>,
    pipelines: RenderPipelineMap<WorldMeshForwardNormalPipelineKey>,
}

impl WorldMeshForwardNormalPipelineCache {
    /// Returns the matching normal prepass pipeline.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        key: WorldMeshForwardNormalPipelineKey,
    ) -> Arc<wgpu::RenderPipeline> {
        self.pipelines
            .get_or_create(key, |key| self.create_pipeline(device, *key))
    }

    fn create_pipeline(
        &self,
        device: &wgpu::Device,
        key: WorldMeshForwardNormalPipelineKey,
    ) -> wgpu::RenderPipeline {
        profiling::scope!("world_mesh_forward::normal_pipeline");
        let multiview = key.multiview_mask.is_some();
        let (label, source) = if multiview {
            (
                "gtao_view_normals_multiview",
                GTAO_VIEW_NORMALS_MULTIVIEW_WGSL,
            )
        } else {
            ("gtao_view_normals_default", GTAO_VIEW_NORMALS_DEFAULT_WGSL)
        };
        logger::debug!(
            "world mesh normal prepass: building pipeline sample_count={} multiview={} topology={:?}",
            key.sample_count,
            multiview,
            key.primitive_topology
        );
        let shader = create_wgsl_shader_module(device, label, source);
        let per_draw_layout = self.per_draw_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(per_draw_layout)],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &normal_vertex_buffer_layouts(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: GTAO_VIEW_NORMAL_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: key.primitive_topology.to_wgpu(),
                front_face: key.front_face.to_wgpu(),
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: key.depth_stencil_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Equal),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: key.sample_count.max(1),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: key.multiview_mask,
            cache: None,
        })
    }

    fn per_draw_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.per_draw_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gtao_view_normals_per_draw"),
                entries: &normal_per_draw_layout_entries(),
            })
        })
    }
}

fn normal_per_draw_layout_entries() -> [wgpu::BindGroupLayoutEntry; 1] {
    [wgpu::BindGroupLayoutEntry {
        binding: 0,
        // The normal prepass reuses the forward per-draw bind group, so this visibility must match
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

impl WorldMeshForwardNormalPipelineKey {
    /// Builds a normal-prepass pipeline key for one draw group.
    pub(super) fn for_draw(
        pipeline: &WorldMeshForwardPipelineState,
        front_face: RasterFrontFace,
        primitive_topology: RasterPrimitiveTopology,
    ) -> Option<Self> {
        let depth_stencil_format = pipeline.pass_desc.depth_stencil_format?;
        triangle_normal_topology(primitive_topology).map(|primitive_topology| Self {
            depth_stencil_format,
            sample_count: pipeline.pass_desc.sample_count,
            multiview_mask: pipeline.pass_desc.multiview_mask,
            front_face,
            primitive_topology,
        })
    }
}

fn normal_pipelines() -> &'static WorldMeshForwardNormalPipelineCache {
    static CACHE: LazyLock<WorldMeshForwardNormalPipelineCache> =
        LazyLock::new(WorldMeshForwardNormalPipelineCache::default);
    &CACHE
}

fn normal_vertex_buffer_layouts() -> [wgpu::VertexBufferLayout<'static>; 2] {
    [
        wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &POSITION_ATTRIBUTES,
        },
        wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &NORMAL_ATTRIBUTES,
        },
    ]
}

fn triangle_normal_topology(
    primitive_topology: RasterPrimitiveTopology,
) -> Option<RasterPrimitiveTopology> {
    match primitive_topology {
        RasterPrimitiveTopology::TriangleList | RasterPrimitiveTopology::TriangleStrip => {
            Some(primitive_topology)
        }
        RasterPrimitiveTopology::PointList
        | RasterPrimitiveTopology::LineList
        | RasterPrimitiveTopology::LineStrip => None,
    }
}

impl RasterPass for WorldMeshForwardNormalPass {
    fn name(&self) -> &str {
        "WorldMeshForwardNormals"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            r.frame_sampled_color(
                self.resources.normals,
                self.resources.normals_msaa,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: -1.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                Some(self.resources.normals),
            );
            r.frame_sampled_depth(
                self.resources.depth,
                self.resources.msaa_depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                None,
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

    fn stencil_ops_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        depth: &DepthAttachmentTemplate,
    ) -> Option<wgpu::Operations<u32>> {
        let Some(format) = ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .and_then(|prepared| prepared.pipeline.pass_desc.depth_stencil_format)
        else {
            return depth.stencil;
        };
        stencil_load_ops(Some(format))
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::normal_record");
        let frame = &mut *ctx.pass_frame;

        let Some(prepared) = ctx.blackboard.take::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        if prepared.opaque_recorded {
            record_world_mesh_forward_normal_graph_raster(
                rpass,
                ctx.device,
                frame,
                &prepared,
                self.pipelines,
            );
        }
        ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{normal_per_draw_layout_entries, triangle_normal_topology};
    use crate::materials::RasterPrimitiveTopology;
    use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;

    #[test]
    fn normal_prepass_only_supports_triangle_topologies() {
        assert_eq!(
            triangle_normal_topology(RasterPrimitiveTopology::TriangleList),
            Some(RasterPrimitiveTopology::TriangleList)
        );
        assert_eq!(
            triangle_normal_topology(RasterPrimitiveTopology::TriangleStrip),
            Some(RasterPrimitiveTopology::TriangleStrip)
        );
        assert_eq!(
            triangle_normal_topology(RasterPrimitiveTopology::LineList),
            None
        );
        assert_eq!(
            triangle_normal_topology(RasterPrimitiveTopology::LineStrip),
            None
        );
        assert_eq!(
            triangle_normal_topology(RasterPrimitiveTopology::PointList),
            None
        );
    }

    #[test]
    fn normal_prepass_per_draw_layout_matches_forward_visibility() {
        let [entry] = normal_per_draw_layout_entries();
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
}
