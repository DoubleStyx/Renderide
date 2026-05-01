//! GTAO normal prepass for graph-managed world-mesh forward rendering.
//!
//! The pass runs after the final forward depth resolve and before Hi-Z/post-processing. It follows
//! the forward pass's active sample tier: single-sample views write the GTAO normals texture
//! directly, while MSAA views write a multisampled normal attachment against the forward MSAA
//! depth and resolve it into the single-sample texture sampled by GTAO. Alpha stays zero where no
//! matching mesh sample writes a normal, letting GTAO fall back to depth-derived reconstruction.

use std::num::{NonZeroU32, NonZeroU64};
use std::sync::{Arc, LazyLock};

use crate::backend::WorldMeshForwardEncodeRefs;
use crate::embedded_shaders::{
    GTAO_NORMAL_PREPASS_DEFAULT_WGSL, GTAO_NORMAL_PREPASS_MULTIVIEW_WGSL,
};
use crate::gpu::GpuLimits;
use crate::materials::RasterFrontFace;
use crate::materials::raster_pipeline::mesh_forward_position_normal_vertex_buffer_layouts;
use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::{
    OnceGpu, RenderPipelineMap, create_wgsl_shader_module, stereo_mask_or_template,
};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess, TextureHandle,
};
use crate::world_mesh::{DrawGroup, InstancePlan};

use super::PreparedWorldMeshForwardFrame;
use super::WorldMeshForwardPlanSlot;
use super::encode::{GeometryDrawGroupBatch, GeometryDrawState, draw_geometry_group};

/// Graph resources consumed by [`WorldMeshGtaoNormalPrepass`].
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshGtaoNormalPrepassGraphResources {
    /// Transient normal target sampled by GTAO.
    pub normals: TextureHandle,
    /// Multisampled normal target used when the forward view renders with MSAA.
    pub msaa_normals: TextureHandle,
    /// Imported single-sample frame depth after forward depth resolve.
    pub depth: ImportedTextureHandle,
    /// Multisampled forward depth used when the forward view renders with MSAA.
    pub msaa_depth: TextureHandle,
    /// Imported per-draw storage slab used by the mesh vertex shader.
    pub per_draw_slab: ImportedBufferHandle,
}

/// Graph-managed world-mesh normal prepass for GTAO.
#[derive(Debug)]
pub struct WorldMeshGtaoNormalPrepass {
    /// Graph handles for this pass instance.
    resources: WorldMeshGtaoNormalPrepassGraphResources,
    /// Process-wide cached normal-prepass pipelines and layouts.
    pipelines: &'static GtaoNormalPrepassPipelineCache,
}

impl WorldMeshGtaoNormalPrepass {
    /// Creates a GTAO normal prepass instance.
    pub fn new(resources: WorldMeshGtaoNormalPrepassGraphResources) -> Self {
        Self {
            resources,
            pipelines: gtao_normal_prepass_pipelines(),
        }
    }
}

impl RasterPass for WorldMeshGtaoNormalPrepass {
    fn name(&self) -> &str {
        "WorldMeshGtaoNormalPrepass"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            r.frame_sampled_color(
                self.resources.normals,
                self.resources.msaa_normals,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
                stages: PER_DRAW_SLAB_SHADER_STAGES,
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
        stereo_mask_or_template(
            ctx.pass_frame.view.multiview_stereo,
            template.multiview_mask,
        )
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::gtao_normal_prepass_record");
        let frame = &*ctx.pass_frame;
        let Some(prepared) = ctx.blackboard.get::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        if !prepared.opaque_recorded {
            return Ok(());
        }
        if !normal_prepass_needed(&prepared.plan) {
            return Ok(());
        }

        let Some(normals) = ctx
            .graph_resources
            .transient_texture(self.resources.normals)
        else {
            return Err(missing_frame_resource(
                self.name(),
                "gtao normal target was not resolved",
            ));
        };
        let Some(per_draw_bg) = frame
            .shared
            .frame_resources
            .per_view_per_draw(frame.view.view_id)
            .map(|d| d.lock().bind_group.clone())
        else {
            return Err(missing_frame_resource(
                self.name(),
                "per-view per-draw bind group was not available",
            ));
        };

        let encode_refs = frame.world_mesh_forward_encode_refs();
        let target = NormalPrepassTarget {
            normal_format: normals.texture.format(),
            depth_format: frame.view.depth_texture.format(),
            sample_count: frame.view.sample_count.max(1),
            multiview_stereo: frame.view.multiview_stereo,
        };
        record_normal_groups(NormalPrepassRecordInputs {
            rpass,
            prepared,
            encode: &encode_refs,
            device: ctx.device,
            gpu_limits: ctx.gpu_limits,
            per_draw_bind_group: per_draw_bg.as_ref(),
            target,
            pipelines: self.pipelines,
        });
        Ok(())
    }
}

/// Returns whether the prepared forward plan contains any mesh group visible to GTAO.
fn normal_prepass_needed(plan: &InstancePlan) -> bool {
    !plan.regular_groups.is_empty()
        || !plan.intersect_groups.is_empty()
        || !plan.transparent_groups.is_empty()
}

/// Attachment formats and view mode for the currently recording normal pass.
#[derive(Clone, Copy)]
struct NormalPrepassTarget {
    /// Normal target color format.
    normal_format: wgpu::TextureFormat,
    /// Depth attachment format.
    depth_format: wgpu::TextureFormat,
    /// Color/depth sample count for the active normal prepass attachment.
    sample_count: u32,
    /// Whether this pass records as a two-eye multiview pass.
    multiview_stereo: bool,
}

/// Draw inputs for one normal-prepass render pass.
struct NormalPrepassRecordInputs<'a, 'rp> {
    /// Active render pass.
    rpass: &'a mut wgpu::RenderPass<'rp>,
    /// Prepared forward draw plan.
    prepared: &'a PreparedWorldMeshForwardFrame,
    /// Mesh pool and skin cache references.
    encode: &'a WorldMeshForwardEncodeRefs<'a>,
    /// GPU device used to resolve cached pipelines.
    device: &'a wgpu::Device,
    /// Device limits for per-draw dynamic offsets.
    gpu_limits: &'a GpuLimits,
    /// Per-draw storage slab bind group.
    per_draw_bind_group: &'a wgpu::BindGroup,
    /// Attachment formats and view mode.
    target: NormalPrepassTarget,
    /// Pipeline cache.
    pipelines: &'a GtaoNormalPrepassPipelineCache,
}

/// Records all forward draw groups into the normal target using geometry-only pipelines.
fn record_normal_groups(inputs: NormalPrepassRecordInputs<'_, '_>) {
    profiling::scope!("world_mesh_forward::gtao_normal_groups");
    let NormalPrepassRecordInputs {
        rpass,
        prepared,
        encode,
        device,
        gpu_limits,
        per_draw_bind_group,
        target,
        pipelines,
    } = inputs;
    let mut draw_state = GeometryDrawState::new();
    let mut bound_front_face: Option<RasterFrontFace> = None;

    for group in prepared
        .plan
        .regular_groups
        .iter()
        .chain(prepared.plan.intersect_groups.iter())
        .chain(prepared.plan.transparent_groups.iter())
    {
        let Some(item) = prepared.draws.get(group.representative_draw_idx) else {
            continue;
        };
        let front_face = item.batch_key.front_face;
        if bound_front_face != Some(front_face) {
            let pipeline = pipelines.pipeline(
                device,
                GtaoNormalPrepassPipelineKey {
                    normal_format: target.normal_format,
                    depth_format: target.depth_format,
                    sample_count: target.sample_count,
                    multiview_stereo: target.multiview_stereo,
                    front_face,
                },
            );
            rpass.set_pipeline(pipeline.as_ref());
            bound_front_face = Some(front_face);
        }
        draw_normal_group(
            rpass,
            group,
            prepared,
            encode,
            gpu_limits,
            per_draw_bind_group,
            &mut draw_state,
        );
    }
}

/// Records one forward draw group into the normal target.
fn draw_normal_group(
    rpass: &mut wgpu::RenderPass<'_>,
    group: &DrawGroup,
    prepared: &PreparedWorldMeshForwardFrame,
    encode: &WorldMeshForwardEncodeRefs<'_>,
    gpu_limits: &GpuLimits,
    per_draw_bind_group: &wgpu::BindGroup,
    state: &mut GeometryDrawState,
) {
    draw_geometry_group(GeometryDrawGroupBatch {
        rpass,
        group,
        draws: &prepared.draws,
        encode,
        gpu_limits,
        per_draw_bind_group,
        supports_base_instance: prepared.supports_base_instance,
        state,
    });
}

/// Builds a pass-specific missing-resource error.
fn missing_frame_resource(pass: &str, detail: &str) -> RenderPassError {
    RenderPassError::MissingFrameParams {
        pass: format!("{pass} ({detail})"),
    }
}

/// Process-wide pipeline cache shared by every GTAO normal prepass instance.
fn gtao_normal_prepass_pipelines() -> &'static GtaoNormalPrepassPipelineCache {
    static CACHE: LazyLock<GtaoNormalPrepassPipelineCache> =
        LazyLock::new(GtaoNormalPrepassPipelineCache::default);
    &CACHE
}

/// Shader visibility for the shared mesh-forward per-draw slab layout.
///
/// The live per-view bind group is created from the reflected null-material `@group(2)` layout,
/// whose storage binding is visible to both vertex and fragment stages. Pipeline layouts that bind
/// that shared group must use the same visibility even when a specific shader only reads the slab
/// from its vertex stage.
const PER_DRAW_SLAB_SHADER_STAGES: wgpu::ShaderStages =
    wgpu::ShaderStages::VERTEX.union(wgpu::ShaderStages::FRAGMENT);

/// Cache key for a normal-prepass render pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GtaoNormalPrepassPipelineKey {
    /// Normal target color format.
    normal_format: wgpu::TextureFormat,
    /// Depth attachment format.
    depth_format: wgpu::TextureFormat,
    /// Raster sample count.
    sample_count: u32,
    /// Whether this pipeline records as a two-eye multiview pass.
    multiview_stereo: bool,
    /// Front-face winding for the current draw group.
    front_face: RasterFrontFace,
}

/// Cached layouts and pipelines for the GTAO normal prepass.
#[derive(Debug, Default)]
struct GtaoNormalPrepassPipelineCache {
    /// Empty `@group(0)` layout; the shader only consumes `@group(2)`.
    empty_frame_layout: OnceGpu<wgpu::BindGroupLayout>,
    /// Empty `@group(1)` layout; the shader is material-independent.
    empty_material_layout: OnceGpu<wgpu::BindGroupLayout>,
    /// `@group(2)` per-draw storage layout.
    per_draw_layout: OnceGpu<wgpu::BindGroupLayout>,
    /// Render pipelines keyed by target/depth formats, view mode, and front-face winding.
    pipelines: RenderPipelineMap<GtaoNormalPrepassPipelineKey>,
}

impl GtaoNormalPrepassPipelineCache {
    /// Returns or builds a normal-prepass render pipeline.
    fn pipeline(
        &self,
        device: &wgpu::Device,
        key: GtaoNormalPrepassPipelineKey,
    ) -> Arc<wgpu::RenderPipeline> {
        self.pipelines
            .get_or_create(key, |key| self.create_pipeline(device, *key))
    }

    /// Empty layout used for unused `@group(0)` and `@group(1)` slots.
    fn empty_bind_group_layout<'a>(
        slot: &'a OnceGpu<wgpu::BindGroupLayout>,
        device: &wgpu::Device,
        label: &'static str,
    ) -> &'a wgpu::BindGroupLayout {
        slot.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[],
            })
        })
    }

    /// Per-draw storage slab layout used by the normal-prepass vertex shader.
    fn per_draw_bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.per_draw_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gtao_normal_prepass_per_draw"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: PER_DRAW_SLAB_SHADER_STAGES,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(PER_DRAW_UNIFORM_STRIDE as u64),
                    },
                    count: None,
                }],
            })
        })
    }

    /// Builds one concrete normal-prepass render pipeline.
    fn create_pipeline(
        &self,
        device: &wgpu::Device,
        key: GtaoNormalPrepassPipelineKey,
    ) -> wgpu::RenderPipeline {
        logger::debug!(
            "gtao normal prepass: building pipeline (normal={:?}, depth={:?}, samples={}, multiview={}, front_face={:?})",
            key.normal_format,
            key.depth_format,
            key.sample_count,
            key.multiview_stereo,
            key.front_face
        );
        let (shader_label, shader_source) = if key.multiview_stereo {
            (
                "gtao_normal_prepass_multiview",
                GTAO_NORMAL_PREPASS_MULTIVIEW_WGSL,
            )
        } else {
            (
                "gtao_normal_prepass_default",
                GTAO_NORMAL_PREPASS_DEFAULT_WGSL,
            )
        };
        let shader = create_wgsl_shader_module(device, shader_label, shader_source);
        let layout_label = if key.multiview_stereo {
            "gtao_normal_prepass_multiview_layout"
        } else {
            "gtao_normal_prepass_default_layout"
        };
        let frame_layout = Self::empty_bind_group_layout(
            &self.empty_frame_layout,
            device,
            "gtao_normal_prepass_empty_frame",
        );
        let material_layout = Self::empty_bind_group_layout(
            &self.empty_material_layout,
            device,
            "gtao_normal_prepass_empty_material",
        );
        let per_draw_layout = self.per_draw_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(layout_label),
            bind_group_layouts: &[
                Some(frame_layout),
                Some(material_layout),
                Some(per_draw_layout),
            ],
            immediate_size: 0,
        });
        let vertex_buffers = mesh_forward_position_normal_vertex_buffer_layouts();
        let pipeline_label = if key.multiview_stereo {
            "gtao_normal_prepass_multiview"
        } else {
            "gtao_normal_prepass_default"
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(pipeline_label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: key.normal_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: key.front_face.to_wgpu(),
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: key.depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Equal),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: key.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: key.multiview_stereo.then(|| NonZeroU32::new(3)).flatten(),
            cache: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty instance plans do not need a normal prepass.
    #[test]
    fn normal_prepass_needed_requires_a_draw_group() {
        assert!(!normal_prepass_needed(&InstancePlan::default()));

        let mut plan = InstancePlan::default();
        plan.transparent_groups.push(DrawGroup {
            representative_draw_idx: 0,
            instance_range: 0..1,
        });
        assert!(normal_prepass_needed(&plan));
    }

    /// The normal prepass binds the same per-draw bind group as forward materials.
    #[test]
    fn per_draw_slab_visibility_matches_forward_layout_contract() {
        assert_eq!(
            PER_DRAW_SLAB_SHADER_STAGES,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        );
    }
}
