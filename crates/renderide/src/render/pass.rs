//! Render graph: RenderPass trait, contexts, and pass implementations.
//!
//! Extension point for shadows, post-processing, UI, probes.

use nalgebra::{Matrix4, Vector3};

use super::SpaceDrawBatch;
use crate::gpu::{
    GpuMeshBuffers, GpuState, PipelineKey, PipelineManager, PipelineVariant, UniformData,
};
use crate::scene::render_transform_to_matrix;
use crate::session::Session;

/// Errors that can occur during render pass execution.
#[derive(Debug)]
pub enum RenderPassError {
    /// Wrapper for wgpu surface errors when acquiring the current texture.
    Surface(wgpu::SurfaceError),
}

impl From<wgpu::SurfaceError> for RenderPassError {
    fn from(e: wgpu::SurfaceError) -> Self {
        RenderPassError::Surface(e)
    }
}

/// Color and optional depth texture views for the current render target.
pub struct RenderTarget<'a> {
    /// Color attachment view.
    pub color_view: &'a wgpu::TextureView,
    /// Optional depth attachment view.
    pub depth_view: Option<&'a wgpu::TextureView>,
}

/// Per-pass context passed to `RenderPass::execute`.
pub struct RenderPassContext<'a> {
    /// GPU state including device, queue, mesh cache, and depth texture.
    pub gpu: &'a mut GpuState,
    /// Session for scene, assets, and view state.
    pub session: &'a Session,
    /// Draw batches for this frame.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager for mesh pipelines.
    pub pipeline_manager: &'a mut PipelineManager,
    /// Viewport dimensions (width, height).
    pub viewport: (u32, u32),
    /// Primary projection matrix; passes build view-proj per batch as needed.
    pub proj: Matrix4<f32>,
    /// Current color and depth attachments.
    pub render_target: RenderTarget<'a>,
    /// Command encoder for this frame; pass records into this.
    pub encoder: &'a mut wgpu::CommandEncoder,
}

/// Frame-level context created at the start of `render_frame`.
pub struct RenderGraphContext<'a> {
    /// GPU state.
    pub gpu: &'a mut GpuState,
    /// Session.
    pub session: &'a Session,
    /// Draw batches.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager.
    pub pipeline_manager: &'a mut PipelineManager,
    /// Viewport (width, height).
    pub viewport: (u32, u32),
    /// Color attachment view.
    pub color_view: &'a wgpu::TextureView,
    /// Optional depth attachment view.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Primary projection matrix.
    pub proj: Matrix4<f32>,
}

/// Trait for render passes that can be executed by the render graph.
pub trait RenderPass {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Executes the pass, recording commands into the context's encoder.
    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError>;
}

/// Graph of render passes executed in sequence each frame.
pub struct RenderGraph {
    passes: Vec<Box<dyn RenderPass>>,
}

impl RenderGraph {
    /// Creates an empty render graph.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
        }
    }

    /// Adds a pass to the graph. Passes execute in insertion order.
    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        self.passes.push(pass);
    }

    /// Executes all passes in order, recording into a new command encoder.
    pub fn execute(&mut self, ctx: &mut RenderGraphContext) -> Result<(), RenderPassError> {
        let mut encoder = ctx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let render_target = RenderTarget {
            color_view: ctx.color_view,
            depth_view: ctx.depth_view,
        };

        let mut pass_ctx = RenderPassContext {
            gpu: ctx.gpu,
            session: ctx.session,
            draw_batches: ctx.draw_batches,
            pipeline_manager: ctx.pipeline_manager,
            viewport: ctx.viewport,
            proj: ctx.proj,
            render_target,
            encoder: &mut encoder,
        };

        for pass in &mut self.passes {
            pass.execute(&mut pass_ctx)?;
        }

        ctx.gpu.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Mesh render pass: draws meshes from draw batches using normal, UV, and skinned pipelines.
pub struct MeshRenderPass {
    /// When true, use UV debug pipeline for meshes that have UVs.
    use_debug_uv: bool,
}

impl MeshRenderPass {
    /// Creates a new mesh render pass.
    pub fn new() -> Self {
        Self {
            use_debug_uv: false,
        }
    }
}

impl Default for MeshRenderPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for MeshRenderPass {
    fn name(&self) -> &str {
        "mesh"
    }

    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        let mesh_assets = ctx.session.asset_registry();

        for batch in ctx.draw_batches {
            for (_, mesh_asset_id, _is_skinned, _material_id, _) in &batch.draws {
                if *mesh_asset_id < 0 {
                    continue;
                }
                let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) else {
                    continue;
                };
                if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                    continue;
                }
                if !ctx.gpu.mesh_buffer_cache.contains_key(mesh_asset_id) {
                    let stride =
                        crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                    let stride = if stride > 0 {
                        stride
                    } else {
                        crate::gpu::compute_vertex_stride_from_mesh(mesh)
                    };
                    if let Some(b) = crate::gpu::create_mesh_buffers(&ctx.gpu.device, mesh, stride)
                    {
                        ctx.gpu.mesh_buffer_cache.insert(*mesh_asset_id, b);
                    }
                }
            }
        }

        struct BatchedDraw<'a> {
            buffers: &'a GpuMeshBuffers,
            mvp: Matrix4<f32>,
            model: Matrix4<f32>,
            use_uv_pipeline: bool,
        }
        let mut normal_draws: Vec<BatchedDraw<'_>> = Vec::new();
        let mut uv_draws: Vec<BatchedDraw<'_>> = Vec::new();
        let scene_graph = ctx.session.scene_graph();

        let frame_index = ctx.pipeline_manager.advance_frame();

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.render_target.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.8,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: ctx.render_target.depth_view.map(|dv| {
                wgpu::RenderPassDepthStencilAttachment {
                    view: dv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        for batch in ctx.draw_batches {
            let mut batch_vt = batch.view_transform;
            batch_vt.scale = filter_scale(batch_vt.scale);
            let view_mat = render_transform_to_matrix(&batch_vt)
                .try_inverse()
                .unwrap_or_else(Matrix4::identity);
            let view_mat = apply_view_handedness_fix(view_mat);
            let view_proj = ctx.proj * view_mat;

            for (model, mesh_asset_id, is_skinned, _material_id, bone_transform_ids) in
                &batch.draws
            {
                let (buffers_ref, mesh) = if *mesh_asset_id >= 0 {
                    let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) else {
                        continue;
                    };
                    if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                        continue;
                    }
                    let Some(b) = ctx.gpu.mesh_buffer_cache.get(mesh_asset_id) else {
                        continue;
                    };
                    (b, mesh)
                } else {
                    continue;
                };

                let model_mvp = view_proj * model;
                let skinned_mvp = view_proj;

                if *is_skinned {
                    let Some(bind_poses) = mesh.bind_poses.as_ref() else {
                        continue;
                    };
                    let Some(ids) = bone_transform_ids.as_deref() else {
                        continue;
                    };
                    let Some(_) = buffers_ref.vertex_buffer_skinned.as_ref() else {
                        continue;
                    };
                    let Some(skinned) = ctx.pipeline_manager.get_pipeline(
                        PipelineKey(None, PipelineVariant::Skinned),
                        &ctx.gpu.device,
                        &ctx.gpu.config,
                    ) else {
                        continue;
                    };
                    let bone_matrices =
                        scene_graph.compute_bone_matrices(batch.space_id, ids, bind_poses);
                    skinned.upload_skinned(&ctx.gpu.queue, skinned_mvp, &bone_matrices);
                    skinned.bind(&mut pass, None, frame_index);
                    skinned.draw_skinned(
                        &mut pass,
                        buffers_ref,
                        &UniformData::Skinned {
                            mvp: skinned_mvp,
                            bone_matrices: &bone_matrices,
                        },
                    );
                    continue;
                }

                let use_uv_pipeline = self.use_debug_uv && buffers_ref.has_uvs;
                let batched = BatchedDraw {
                    buffers: buffers_ref,
                    mvp: model_mvp,
                    model: *model,
                    use_uv_pipeline,
                };
                if use_uv_pipeline {
                    uv_draws.push(batched);
                } else {
                    normal_draws.push(batched);
                }
            }
        }

        let mvp_models_normal: Vec<_> = normal_draws.iter().map(|d| (d.mvp, d.model)).collect();
        let mvp_models_uv: Vec<_> = uv_draws.iter().map(|d| (d.mvp, d.model)).collect();

        let Some(normal_debug) = ctx.pipeline_manager.get_pipeline(
            PipelineKey(None, PipelineVariant::NormalDebug),
            &ctx.gpu.device,
            &ctx.gpu.config,
        ) else {
            return Ok(());
        };
        let Some(uv_debug) = ctx.pipeline_manager.get_pipeline(
            PipelineKey(None, PipelineVariant::UvDebug),
            &ctx.gpu.device,
            &ctx.gpu.config,
        ) else {
            return Ok(());
        };

        normal_debug.upload_batch(&ctx.gpu.queue, &mvp_models_normal, frame_index);
        uv_debug.upload_batch(&ctx.gpu.queue, &mvp_models_uv, frame_index);

        for (i, d) in normal_draws.iter().enumerate() {
            normal_debug.bind(&mut pass, Some(i as u32), frame_index);
            normal_debug.draw_mesh(
                &mut pass,
                d.buffers,
                &UniformData::Simple {
                    mvp: d.mvp,
                    model: d.model,
                },
            );
        }
        for (i, d) in uv_draws.iter().enumerate() {
            uv_debug.bind(&mut pass, Some(i as u32), frame_index);
            uv_debug.draw_mesh(
                &mut pass,
                d.buffers,
                &UniformData::Simple {
                    mvp: d.mvp,
                    model: d.model,
                },
            );
        }

        Ok(())
    }
}

fn filter_scale(scale: Vector3<f32>) -> Vector3<f32> {
    const MIN_SCALE: f32 = 1e-8;
    if scale.x.abs() < MIN_SCALE || scale.y.abs() < MIN_SCALE || scale.z.abs() < MIN_SCALE {
        Vector3::new(1.0, 1.0, 1.0)
    } else {
        scale
    }
}

fn apply_view_handedness_fix(view: Matrix4<f32>) -> Matrix4<f32> {
    let z_flip = Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0, -1.0));
    z_flip * view
}

/// Reverse-Z projection matrix for the given aspect and frustum.
pub fn reverse_z_projection(
    aspect: f32,
    vertical_fov: f32,
    near: f32,
    far: f32,
) -> Matrix4<f32> {
    let vertical_half = vertical_fov / 2.0;
    let tan_vertical_half = vertical_half.tan();
    let horizontal_fov = (tan_vertical_half * aspect)
        .atan()
        .clamp(0.1, std::f32::consts::FRAC_PI_2 - 0.1)
        * 2.0;
    let tan_horizontal_half = (horizontal_fov / 2.0).tan();
    let f_x = 1.0 / tan_horizontal_half;
    let f_y = 1.0 / tan_vertical_half;

    Matrix4::new(
        f_x,
        0.0,
        0.0,
        0.0,
        0.0,
        f_y,
        0.0,
        0.0,
        0.0,
        0.0,
        near / (far - near),
        (far * near) / (far - near),
        0.0,
        0.0,
        -1.0,
        0.0,
    )
}
