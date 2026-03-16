//! Render graph: RenderPass trait, contexts, and pass implementations.
//!
//! Extension point for shadows, post-processing, UI, probes.

use nalgebra::{Matrix4, Vector3};

use super::target::RenderTarget;
use super::view::ViewParams;
use super::SpaceDrawBatch;
use crate::shared::{CameraProjection, CameraRenderParameters};
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

/// Color and optional depth texture views for the current render pass.
pub struct RenderTargetViews<'a> {
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
    /// Frame index for ring buffer offset; advanced once per frame by the graph.
    pub frame_index: u64,
    /// Viewport dimensions (width, height).
    pub viewport: (u32, u32),
    /// Primary projection matrix; passes build view-proj per batch as needed.
    pub proj: Matrix4<f32>,
    /// Optional overlay projection override. When `Some`, overlay batches use this instead of
    /// `proj` (e.g. orthographic for screen-space UI). Future: set from RenderConfig or host data.
    pub overlay_projection_override: Option<ViewParams>,
    /// Current color and depth attachments.
    pub render_target: RenderTargetViews<'a>,
    /// Command encoder for this frame; pass records into this.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Optional timestamp query set for GPU pass timing.
    pub timestamp_query_set: Option<&'a wgpu::QuerySet>,
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
    /// Render target (surface or offscreen).
    pub target: &'a RenderTarget,
    /// Depth view for Surface targets; Offscreen provides its own. Dimensions must match target.
    pub depth_view_override: Option<&'a wgpu::TextureView>,
    /// Viewport (width, height); must match target dimensions.
    pub viewport: (u32, u32),
    /// Primary projection matrix.
    pub proj: Matrix4<f32>,
    /// Optional overlay projection override. When `Some`, overlay pass uses this instead of `proj`.
    pub overlay_projection_override: Option<ViewParams>,
    /// Optional timestamp query set for GPU pass timing.
    pub timestamp_query_set: Option<&'a wgpu::QuerySet>,
    /// Optional resolve buffer for timestamp readback.
    pub timestamp_resolve_buffer: Option<&'a wgpu::Buffer>,
    /// Optional staging buffer for timestamp readback.
    pub timestamp_staging_buffer: Option<&'a wgpu::Buffer>,
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

        let color_view = ctx.target.color_view();
        let depth_view = ctx.target.depth_view().or(ctx.depth_view_override);
        let render_target = RenderTargetViews {
            color_view,
            depth_view,
        };

        let frame_index = ctx.pipeline_manager.advance_frame();
        let mut pass_ctx = RenderPassContext {
            gpu: ctx.gpu,
            session: ctx.session,
            draw_batches: ctx.draw_batches,
            pipeline_manager: ctx.pipeline_manager,
            frame_index,
            viewport: ctx.viewport,
            proj: ctx.proj,
            overlay_projection_override: ctx.overlay_projection_override.clone(),
            render_target,
            encoder: &mut encoder,
            timestamp_query_set: ctx.timestamp_query_set,
        };

        for pass in &mut self.passes {
            pass.execute(&mut pass_ctx)?;
        }

        if let (Some(query_set), Some(resolve_buffer), Some(staging_buffer)) = (
            ctx.timestamp_query_set,
            ctx.timestamp_resolve_buffer,
            ctx.timestamp_staging_buffer,
        ) {
            encoder.resolve_query_set(query_set, 0..2, resolve_buffer, 0);
            encoder.copy_buffer_to_buffer(resolve_buffer, 0, staging_buffer, 0, resolve_buffer.size());
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

/// Collected non-skinned draw for batch upload.
/// Uses mesh_asset_id for buffer lookup to avoid borrowing ctx across pass boundaries.
struct BatchedDraw {
    mesh_asset_id: i32,
    mvp: Matrix4<f32>,
    model: Matrix4<f32>,
    pipeline_variant: PipelineVariant,
    is_overlay: bool,
    /// Per-draw stencil for GraphicsChunk masking. When `Some`, overlay uses stencil pipeline.
    stencil_state: Option<crate::stencil::StencilState>,
}

/// Collected skinned draw for batch upload.
/// Uses mesh_asset_id for buffer lookup to avoid borrowing ctx across pass boundaries.
struct SkinnedBatchedDraw {
    mesh_asset_id: i32,
    mvp: Matrix4<f32>,
    bone_matrices: Vec<[[f32; 4]; 4]>,
    blendshape_weights: Option<Vec<f32>>,
    num_vertices: u32,
    is_overlay: bool,
    /// Pipeline variant (Skinned or OverlayStencilSkinned).
    pipeline_variant: crate::gpu::PipelineVariant,
    /// Per-draw stencil for GraphicsChunk masking. When `Some`, overlay uses stencil pipeline.
    stencil_state: Option<crate::stencil::StencilState>,
}

/// Ensures all meshes referenced by draw batches are in the GPU mesh buffer cache.
fn ensure_mesh_buffers(ctx: &mut RenderPassContext) {
    let mesh_assets = ctx.session.asset_registry();
    for batch in ctx.draw_batches {
        for d in &batch.draws {
            if d.mesh_asset_id < 0 {
                continue;
            }
            let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                continue;
            };
            if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                continue;
            }
            if !ctx.gpu.mesh_buffer_cache.contains_key(&d.mesh_asset_id) {
                let stride =
                    crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                let stride = if stride > 0 {
                    stride
                } else {
                    crate::gpu::compute_vertex_stride_from_mesh(mesh)
                };
                if let Some(b) = crate::gpu::create_mesh_buffers(&ctx.gpu.device, mesh, stride) {
                    ctx.gpu.mesh_buffer_cache.insert(d.mesh_asset_id, b);
                }
            }
        }
    }
}

/// Collects mesh draws from batches and partitions by overlay flag.
///
/// Returns (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned).
fn collect_mesh_draws(ctx: &RenderPassContext) -> (
    Vec<SkinnedBatchedDraw>,
    Vec<SkinnedBatchedDraw>,
    Vec<BatchedDraw>,
    Vec<BatchedDraw>,
) {
    let mesh_assets = ctx.session.asset_registry();
    let scene_graph = ctx.session.scene_graph();
    let debug_skinned = ctx.session.render_config().debug_skinned;
    let mut first_skinned_logged = false;

    let mut non_skinned_draws: Vec<BatchedDraw> = Vec::new();
    let mut skinned_draws: Vec<SkinnedBatchedDraw> = Vec::new();

    for batch in ctx.draw_batches {
        let mut batch_vt = batch.view_transform;
        batch_vt.scale = filter_scale(batch_vt.scale);
        let view_mat = render_transform_to_matrix(&batch_vt)
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);
        let view_mat = apply_view_handedness_fix(view_mat);
        let proj = batch
            .is_overlay
            .then(|| ctx.overlay_projection_override.as_ref())
            .flatten()
            .map(|v| v.to_projection_matrix())
            .unwrap_or(ctx.proj);
        let view_proj = proj * view_mat;

        for d in &batch.draws {
            let (buffers_ref, mesh) = if d.mesh_asset_id >= 0 {
                let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                    continue;
                };
                if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                    continue;
                }
                let Some(b) = ctx.gpu.mesh_buffer_cache.get(&d.mesh_asset_id) else {
                    continue;
                };
                (b, mesh)
            } else {
                continue;
            };

            let model_mvp = view_proj * d.model_matrix;

            if d.is_skinned {
                let Some(bind_poses) = mesh.bind_poses.as_ref() else {
                    logger::trace!(
                        "Skinned draw skipped: mesh missing bind_poses (mesh={})",
                        d.mesh_asset_id
                    );
                    continue;
                };
                let Some(ids) = d.bone_transform_ids.as_deref() else {
                    logger::trace!(
                        "Skinned draw skipped: bone_transform_ids missing or empty (mesh={})",
                        d.mesh_asset_id
                    );
                    continue;
                };
                if ids.is_empty() {
                    logger::trace!(
                        "Skinned draw skipped: bone_transform_ids missing or empty (mesh={})",
                        d.mesh_asset_id
                    );
                    continue;
                }
                if ids.len() > bind_poses.len() {
                    logger::trace!(
                        "Skinned draw skipped: bone_transform_ids.len()={} > bind_poses.len()={} (mesh={})",
                        ids.len(),
                        bind_poses.len(),
                        d.mesh_asset_id
                    );
                    continue;
                }
                let Some(_) = buffers_ref.vertex_buffer_skinned.as_ref() else {
                    logger::trace!(
                        "Skinned draw skipped: vertex_buffer_skinned missing (mesh={})",
                        d.mesh_asset_id
                    );
                    continue;
                };
                if debug_skinned && !first_skinned_logged {
                    first_skinned_logged = true;
                    let first_3_ids: Vec<i32> = ids.iter().take(3).copied().collect();
                    let first_bind = bind_poses.first().map(|b| format!("{:?}", b)).unwrap_or_else(|| "none".to_string());
                    let (first_vert_indices, first_vert_weights) = if let (Some(bc), Some(bw)) = (mesh.bone_counts.as_ref(), mesh.bone_weights.as_ref()) {
                        let n = bc.first().copied().unwrap_or(0) as usize;
                        let n = n.min(4);
                        let mut indices = [0i32; 4];
                        let mut weights = [0.0f32; 4];
                        for j in 0..n {
                            if j * 8 + 8 <= bw.len() {
                                let idx = i32::from_le_bytes(bw[j * 8 + 4..j * 8 + 8].try_into().unwrap_or([0; 4]));
                                let w = f32::from_le_bytes(bw[j * 8..j * 8 + 4].try_into().unwrap_or([0; 4]));
                                indices[j] = idx;
                                weights[j] = w;
                            }
                        }
                        (format!("{:?}", indices), format!("{:?}", weights))
                    } else {
                        ("n/a".to_string(), "n/a".to_string())
                    };
                    logger::debug!(
                        "skinned draw: mesh={} node_id={} bone_ids_len={} first_3_ids={:?} first_bind={} first_vert_indices={} first_vert_weights={} has_skinned_vb={}",
                        d.mesh_asset_id,
                        d.node_id,
                        ids.len(),
                        first_3_ids,
                        first_bind,
                        first_vert_indices,
                        first_vert_weights,
                        buffers_ref.vertex_buffer_skinned.is_some()
                    );
                }
                let mut skinned_mvp = if ctx.session.render_config().skinned_use_root_bone {
                    let root_id = d.root_bone_transform_id.filter(|&id| id >= 0);
                    match root_id.and_then(|id| {
                        scene_graph.get_world_matrix(batch.space_id, id as usize)
                    }) {
                        Some(root_world) => view_proj * root_world,
                        None => view_proj,
                    }
                } else {
                    view_proj
                };
                if ctx.session.render_config().skinned_flip_handedness {
                    let z_flip = Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0, -1.0));
                    skinned_mvp *= z_flip;
                }
                let root_bone = if ctx.session.render_config().skinned_use_root_bone {
                    d.root_bone_transform_id
                } else {
                    None
                };
                let bone_matrices = scene_graph.compute_bone_matrices(
                    batch.space_id,
                    ids,
                    bind_poses,
                    root_bone,
                );
                skinned_draws.push(SkinnedBatchedDraw {
                    mesh_asset_id: d.mesh_asset_id,
                    mvp: skinned_mvp,
                    bone_matrices,
                    blendshape_weights: d.blendshape_weights.clone(),
                    num_vertices: mesh.vertex_count.max(0) as u32,
                    is_overlay: batch.is_overlay,
                    pipeline_variant: d.pipeline_variant.clone(),
                    stencil_state: d.stencil_state,
                });
                continue;
            }

            non_skinned_draws.push(BatchedDraw {
                mesh_asset_id: d.mesh_asset_id,
                mvp: model_mvp,
                model: d.model_matrix,
                pipeline_variant: d.pipeline_variant.clone(),
                is_overlay: batch.is_overlay,
                stencil_state: d.stencil_state,
            });
        }
    }

    let (non_overlay_skinned, overlay_skinned): (Vec<_>, Vec<_>) =
        skinned_draws.into_iter().partition(|d| !d.is_overlay);
    let (non_overlay_non_skinned, overlay_non_skinned): (Vec<_>, Vec<_>) =
        non_skinned_draws.into_iter().partition(|d| !d.is_overlay);

    (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned)
}

/// Parameters for recording mesh draws; used to avoid borrowing ctx while encoder is active.
struct MeshDrawParams<'a> {
    pipeline_manager: &'a mut PipelineManager,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    config: &'a wgpu::SurfaceConfiguration,
    frame_index: u64,
    mesh_buffer_cache: &'a std::collections::HashMap<i32, GpuMeshBuffers>,
}

/// Records skinned mesh draws into the render pass.
fn record_skinned_draws(
    pass: &mut wgpu::RenderPass,
    params: &mut MeshDrawParams,
    draws: &[SkinnedBatchedDraw],
    debug_blendshapes: bool,
) {
    if draws.is_empty() {
        return;
    }
    let mut i = 0;
    while i < draws.len() {
        let variant = draws[i].pipeline_variant.clone();
        let group_end = draws[i..]
            .iter()
            .take_while(|d| d.pipeline_variant == variant)
            .count();
        let group = &draws[i..i + group_end];

        let Some(skinned) = params.pipeline_manager.get_pipeline(
            PipelineKey(None, variant),
            params.device,
            params.config,
        ) else {
            i += group_end;
            continue;
        };
    let items: Vec<_> = group
        .iter()
        .map(|d| {
            (
                d.mvp,
                d.bone_matrices.as_slice(),
                d.blendshape_weights.as_deref(),
                d.num_vertices,
            )
        })
        .collect();
    if debug_blendshapes {
        let count = group.len();
        let first_with_weights = group
            .iter()
            .find(|d| d.blendshape_weights.as_ref().is_some_and(|w| !w.is_empty()));
        if let Some(d) = first_with_weights {
            let w = d.blendshape_weights.as_ref().unwrap();
            let preview: Vec<_> = w.iter().take(8).copied().collect();
            logger::debug!(
                "blendshape batch_count={} first_draw_weights_len={} preview={:?}",
                count,
                w.len(),
                preview
            );
        } else {
            logger::debug!("blendshape batch_count={} first_draw_weights_len=0", count);
        }
        }
        skinned.upload_skinned_batch(params.queue, &items, params.frame_index);
        for (j, d) in group.iter().enumerate() {
            let Some(buffers) = params.mesh_buffer_cache.get(&d.mesh_asset_id) else {
                continue;
            };
            let draw_bind_group = skinned
                .create_skinned_draw_bind_group(params.device, buffers)
                .expect("skinned pipeline must create draw bind groups");
            skinned.bind(pass, Some(j as u32), params.frame_index, Some(&draw_bind_group));
            if let Some(ref stencil) = d.stencil_state {
                pass.set_stencil_reference(stencil.reference as u32);
            }
            skinned.draw_skinned(
                pass,
                buffers,
                &UniformData::Skinned {
                    mvp: d.mvp,
                    bone_matrices: &d.bone_matrices,
                },
            );
        }
        i += group_end;
    }
}

/// Records non-skinned mesh draws into the render pass.
fn record_non_skinned_draws(
    pass: &mut wgpu::RenderPass,
    params: &mut MeshDrawParams,
    draws: &[BatchedDraw],
) {
    let mut i = 0;
    while i < draws.len() {
        let variant = draws[i].pipeline_variant.clone();
        let group_end = draws[i..]
            .iter()
            .take_while(|d| d.pipeline_variant == variant)
            .count();
        let group = &draws[i..i + group_end];

        let pipeline_key = PipelineKey(None, variant.clone());
        let Some(pipeline) = params.pipeline_manager.get_pipeline(
            pipeline_key,
            params.device,
            params.config,
        ) else {
            i += group_end;
            continue;
        };

        let use_overlay_upload = matches!(
            variant,
            crate::gpu::PipelineVariant::OverlayStencilContent
        );
        if use_overlay_upload {
            let items: Vec<_> = group
                .iter()
                .map(|d| {
                    let clip = d.stencil_state.and_then(|s| s.clip_rect).map(|r| {
                        [r.x, r.y, r.width, r.height]
                    });
                    (d.mvp, d.model, clip)
                })
                .collect();
            pipeline.upload_batch_overlay(params.queue, &items, params.frame_index);
        } else {
            let mvp_models: Vec<_> = group.iter().map(|d| (d.mvp, d.model)).collect();
            pipeline.upload_batch(params.queue, &mvp_models, params.frame_index);
        }

        for (j, d) in group.iter().enumerate() {
            let Some(buffers) = params.mesh_buffer_cache.get(&d.mesh_asset_id) else {
                continue;
            };
            pipeline.bind(pass, Some(j as u32), params.frame_index, None);
            if let Some(ref stencil) = d.stencil_state {
                pass.set_stencil_reference(stencil.reference as u32);
            }
            pipeline.draw_mesh(
                pass,
                buffers,
                &UniformData::Simple {
                    mvp: d.mvp,
                    model: d.model,
                },
            );
        }

        i += group_end;
    }
}

/// Mesh render pass: draws non-overlay meshes from draw batches.
///
/// Uses `LoadOp::Clear` for color and depth; draws all non-overlay batches (skinned and
/// non-skinned). Batches are pre-sorted (non-overlay first, then overlay) by the session.
pub struct MeshRenderPass;

impl MeshRenderPass {
    /// Creates a new mesh render pass.
    pub fn new() -> Self {
        Self
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
        ensure_mesh_buffers(ctx);
        let (non_overlay_skinned, _, non_overlay_non_skinned, _) = collect_mesh_draws(ctx);

        let mut draw_params = MeshDrawParams {
            pipeline_manager: &mut ctx.pipeline_manager,
            device: &ctx.gpu.device,
            queue: &ctx.gpu.queue,
            config: &ctx.gpu.config,
            frame_index: ctx.frame_index,
            mesh_buffer_cache: &ctx.gpu.mesh_buffer_cache,
        };

        let timestamp_writes = ctx.timestamp_query_set.map(|query_set| {
            wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }
        });

        // Non-overlay pass: Clear framebuffer, draw all non-overlay batches.
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh pass (non-overlay)"),
                timestamp_writes: timestamp_writes.clone(),
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
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Store,
                        }),
                    }
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let debug_blendshapes = ctx.session.render_config().debug_blendshapes;
            record_skinned_draws(
                &mut pass,
                &mut draw_params,
                &non_overlay_skinned,
                debug_blendshapes,
            );
            record_non_skinned_draws(&mut pass, &mut draw_params, &non_overlay_non_skinned);
        }

        Ok(())
    }
}

/// Overlay render pass: draws overlay meshes on top of the main scene.
///
/// Uses `LoadOp::Load` for color and depth; draws overlay batches (skinned and non-skinned)
/// with alpha blending to composite on top.
///
/// **Projection choice**:
/// - **Orthographic** (set [`RenderGraphContext::overlay_projection_override`]): Use for
///   screen-space UI (Canvas, HUD, fixed-size elements). Matches Unity Canvas render mode.
/// - **Perspective** (override `None`, default): Use for world-space overlays (3D UI in scene,
///   floating panels with depth). Overlay batches use the main view's projection.
pub struct OverlayRenderPass;

impl OverlayRenderPass {
    /// Creates a new overlay render pass.
    pub fn new() -> Self {
        Self
    }
}

impl Default for OverlayRenderPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for OverlayRenderPass {
    fn name(&self) -> &str {
        "overlay"
    }

    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        let (_, overlay_skinned, _, overlay_non_skinned) = collect_mesh_draws(ctx);

        if overlay_skinned.is_empty() && overlay_non_skinned.is_empty() {
            return Ok(());
        }

        let mut draw_params = MeshDrawParams {
            pipeline_manager: &mut ctx.pipeline_manager,
            device: &ctx.gpu.device,
            queue: &ctx.gpu.queue,
            config: &ctx.gpu.config,
            frame_index: ctx.frame_index,
            mesh_buffer_cache: &ctx.gpu.mesh_buffer_cache,
        };

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlay pass"),
            timestamp_writes: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.render_target.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: ctx.render_target.depth_view.map(|dv| {
                wgpu::RenderPassDepthStencilAttachment {
                    view: dv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                }
            }),
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let debug_blendshapes = ctx.session.render_config().debug_blendshapes;
        record_skinned_draws(&mut pass, &mut draw_params, &overlay_skinned, debug_blendshapes);
        record_non_skinned_draws(&mut pass, &mut draw_params, &overlay_non_skinned);

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

/// Reverse-Z orthographic projection matrix.
///
/// Used for UI and render tasks (e.g. CameraRenderTask with orthographic projection).
/// Maps z from [near, far] to NDC such that near → 1, far → -1 (reverse-Z depth).
///
/// * `half_width` - Half-width of the orthographic view volume.
/// * `half_height` - Half-height of the orthographic view volume.
/// * `near` - Near clip plane (closer to camera).
/// * `far` - Far clip plane (farther from camera).
pub fn orthographic_projection_reverse_z(
    half_width: f32,
    half_height: f32,
    near: f32,
    far: f32,
) -> Matrix4<f32> {
    let range = far - near;
    let z_scale = -2.0 / range;
    let z_offset = (far + near) / range;

    Matrix4::new(
        1.0 / half_width,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / half_height,
        0.0,
        0.0,
        0.0,
        0.0,
        z_scale,
        0.0,
        0.0,
        0.0,
        z_offset,
        1.0,
    )
}

/// Returns a projection matrix for the given camera parameters.
///
/// Chooses perspective or orthographic based on `params.projection`. Orthographic
/// is used for UI and render tasks (e.g. offscreen CameraRenderTask); the main
/// view continues to use the existing perspective path via [`reverse_z_projection`].
///
/// * `params` - Camera parameters from `CameraRenderTask` or similar.
/// * `aspect` - Aspect ratio (width / height).
pub fn projection_for_params(params: &CameraRenderParameters, aspect: f32) -> Matrix4<f32> {
    let near = params.near_clip.max(0.01);
    let far = params.far_clip;

    match params.projection {
        CameraProjection::orthographic => {
            let half_height = params.orthographic_size;
            let half_width = half_height * aspect;
            orthographic_projection_reverse_z(half_width, half_height, near, far)
        }
        CameraProjection::perspective | CameraProjection::panoramic => {
            reverse_z_projection(aspect, params.fov.to_radians(), near, far)
        }
    }
}
