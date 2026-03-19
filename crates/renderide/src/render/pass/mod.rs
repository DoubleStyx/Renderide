//! Render graph: RenderPass trait, contexts, and pass implementations.
//!
//! Extension point for shadows, post-processing, UI, probes.

mod clustered_light;
mod composite;
mod mesh_draw;
mod projection;
mod rtao_blur;
mod rtao_compute;

use nalgebra::Matrix4;

use super::SpaceDrawBatch;
use super::target::RenderTarget;
use super::view::ViewParams;
use crate::session::Session;
use mesh_draw::{
    CollectMeshDrawsContext, MeshDrawParams, collect_mesh_draws, record_non_skinned_draws,
    record_skinned_draws,
};

pub use clustered_light::ClusteredLightPass;
pub use composite::CompositePass;
pub use projection::{
    orthographic_projection_reverse_z, projection_for_params, reverse_z_projection,
};
pub use rtao_blur::RtaoBlurPass;
pub use rtao_compute::RtaoComputePass;

/// Pre-collected mesh draws and view parameters for the main view.
///
/// Produced by [`prepare_mesh_draws_for_view`] during the collect phase.
/// Passed to [`RenderLoop::render_frame`] to avoid CPU work in the render phase.
pub struct PreCollectedFrameData {
    /// Primary projection matrix for the main view.
    pub proj: Matrix4<f32>,
    /// Overlay projection override when overlays use orthographic.
    pub overlay_projection_override: Option<ViewParams>,
    /// Cached mesh draws: (non_overlay_skinned, overlay_skinned, non_overlay_non_skinned, overlay_non_skinned).
    pub(crate) cached_mesh_draws: (
        Vec<mesh_draw::SkinnedBatchedDraw>,
        Vec<mesh_draw::SkinnedBatchedDraw>,
        Vec<mesh_draw::BatchedDraw>,
        Vec<mesh_draw::BatchedDraw>,
    ),
}

/// Prepares mesh draws for the main view during the collect phase.
///
/// Runs [`ensure_mesh_buffers`] and [`collect_mesh_draws`] so this CPU work
/// is measured in the collect phase rather than the render phase.
pub fn prepare_mesh_draws_for_view(
    gpu: &mut crate::gpu::GpuState,
    session: &Session,
    draw_batches: &[SpaceDrawBatch],
    viewport: (u32, u32),
) -> PreCollectedFrameData {
    ensure_mesh_buffers(gpu, session, draw_batches);
    let (width, height) = viewport;
    let aspect = width as f32 / height.max(1) as f32;
    let view_params = ViewParams::perspective_from_session(session, aspect);
    let proj = view_params.to_projection_matrix();
    let overlay_projection_override =
        ViewParams::overlay_projection_for_frame(session, draw_batches, aspect);
    let collect_ctx = CollectMeshDrawsContext {
        session,
        draw_batches,
        gpu,
        proj,
        overlay_projection_override: overlay_projection_override.clone(),
    };
    let cached_mesh_draws = collect_mesh_draws(&collect_ctx);
    PreCollectedFrameData {
        proj,
        overlay_projection_override,
        cached_mesh_draws,
    }
}

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
    /// Color attachment view (output for this pass).
    pub color_view: &'a wgpu::TextureView,
    /// Optional depth attachment view.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, position G-buffer view for MRT mesh pass.
    pub mrt_position_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, normal G-buffer view for MRT mesh pass.
    pub mrt_normal_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, raw AO texture view (RTAO output, blur input).
    pub mrt_ao_raw_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, AO texture view for blur output and composite.
    pub mrt_ao_view: Option<&'a wgpu::TextureView>,
    /// When RTAO is enabled, mesh color input for composite pass (MRT color texture).
    pub mrt_color_input_view: Option<&'a wgpu::TextureView>,
}

/// Per-pass context passed to `RenderPass::execute`.
pub struct RenderPassContext<'a> {
    /// GPU state including device, queue, mesh cache, and depth texture.
    pub gpu: &'a mut crate::gpu::GpuState,
    /// Session for scene, assets, and view state.
    pub session: &'a Session,
    /// Draw batches for this frame.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager for mesh pipelines.
    pub pipeline_manager: &'a mut crate::gpu::PipelineManager,
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
    /// Cached mesh draws from a single collect per frame. Mesh and overlay passes use this.
    #[allow(clippy::type_complexity)]
    pub(crate) cached_mesh_draws: Option<(
        &'a [mesh_draw::SkinnedBatchedDraw],
        &'a [mesh_draw::SkinnedBatchedDraw],
        &'a [mesh_draw::BatchedDraw],
        &'a [mesh_draw::BatchedDraw],
    )>,
}

/// MRT (Multiple Render Target) views for RTAO pass.
///
/// When RTAO is enabled, the mesh pass renders to these instead of the surface.
pub struct MrtViews<'a> {
    /// Color attachment view (matches surface format for copy-back).
    pub color_view: &'a wgpu::TextureView,
    /// Color texture for copy to surface (same as color_view's texture).
    pub color_texture: &'a wgpu::Texture,
    /// Position G-buffer view (Rgba16Float).
    pub position_view: &'a wgpu::TextureView,
    /// Normal G-buffer view (Rgba16Float).
    pub normal_view: &'a wgpu::TextureView,
    /// Raw AO view (Rgba8Unorm). Written by RTAO compute, read by blur pass.
    pub ao_raw_view: &'a wgpu::TextureView,
    /// AO output view (Rgba8Unorm). Written by blur pass, read by composite.
    pub ao_view: &'a wgpu::TextureView,
}

/// Frame-level context created at the start of `render_frame`.
pub struct RenderGraphContext<'a> {
    /// GPU state.
    pub gpu: &'a mut crate::gpu::GpuState,
    /// Session.
    pub session: &'a Session,
    /// Draw batches.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager.
    pub pipeline_manager: &'a mut crate::gpu::PipelineManager,
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
    /// When RTAO is enabled and ray tracing is available, MRT views for mesh pass.
    pub mrt_views: Option<MrtViews<'a>>,
    /// Pre-collected mesh draws from the collect phase. When `Some`, skips collect in execute.
    #[allow(clippy::type_complexity)]
    pub(crate) pre_collected: Option<&'a (
        Vec<mesh_draw::SkinnedBatchedDraw>,
        Vec<mesh_draw::SkinnedBatchedDraw>,
        Vec<mesh_draw::BatchedDraw>,
        Vec<mesh_draw::BatchedDraw>,
    )>,
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
        Self { passes: Vec::new() }
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

        ctx.gpu.cluster_count_x = 0;
        ctx.gpu.cluster_count_y = 0;
        ctx.gpu.light_count = 0;

        let (
            color_view,
            mrt_position_view,
            mrt_normal_view,
            mrt_ao_raw_view,
            mrt_ao_view,
            mrt_color_input_view,
        ) = match &ctx.mrt_views {
            Some(mrt) => (
                mrt.color_view,
                Some(mrt.position_view),
                Some(mrt.normal_view),
                Some(mrt.ao_raw_view),
                Some(mrt.ao_view),
                Some(mrt.color_view),
            ),
            None => (ctx.target.color_view(), None, None, None, None, None),
        };
        let depth_view = ctx.target.depth_view().or(ctx.depth_view_override);
        let render_target = RenderTargetViews {
            color_view,
            depth_view,
            mrt_position_view,
            mrt_normal_view,
            mrt_ao_raw_view,
            mrt_ao_view,
            mrt_color_input_view,
        };

        if ctx.pre_collected.is_none() {
            ensure_mesh_buffers(ctx.gpu, ctx.session, ctx.draw_batches);
        }

        let computed;
        let cached_mesh_draws = match ctx.pre_collected {
            Some(pc) => Some((&pc.0[..], &pc.1[..], &pc.2[..], &pc.3[..])),
            None => {
                let collect_ctx = CollectMeshDrawsContext {
                    session: ctx.session,
                    draw_batches: ctx.draw_batches,
                    gpu: &*ctx.gpu,
                    proj: ctx.proj,
                    overlay_projection_override: ctx.overlay_projection_override.clone(),
                };
                computed = collect_mesh_draws(&collect_ctx);
                Some((
                    &computed.0[..],
                    &computed.1[..],
                    &computed.2[..],
                    &computed.3[..],
                ))
            }
        };

        if let Some(ref mut ray_tracing) = ctx.gpu.ray_tracing_state
            && let Some(ref accel) = ctx.gpu.accel_cache
        {
            ray_tracing.tlas = crate::gpu::build_tlas(
                &ctx.gpu.device,
                &mut encoder,
                accel,
                ctx.draw_batches,
                &mut ray_tracing.instance_scratch,
            );
        }

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
            cached_mesh_draws,
        };

        for pass in &mut self.passes {
            if pass.name() == "composite" || pass.name() == "overlay" {
                pass_ctx.render_target = RenderTargetViews {
                    color_view: ctx.target.color_view(),
                    depth_view: pass_ctx.render_target.depth_view,
                    mrt_position_view: pass_ctx.render_target.mrt_position_view,
                    mrt_normal_view: pass_ctx.render_target.mrt_normal_view,
                    mrt_ao_raw_view: pass_ctx.render_target.mrt_ao_raw_view,
                    mrt_ao_view: pass_ctx.render_target.mrt_ao_view,
                    mrt_color_input_view: pass_ctx.render_target.mrt_color_input_view,
                };
            }
            pass.execute(&mut pass_ctx)?;
        }

        if let (Some(query_set), Some(resolve_buffer), Some(staging_buffer)) = (
            ctx.timestamp_query_set,
            ctx.timestamp_resolve_buffer,
            ctx.timestamp_staging_buffer,
        ) {
            encoder.resolve_query_set(query_set, 0..2, resolve_buffer, 0);
            encoder.copy_buffer_to_buffer(
                resolve_buffer,
                0,
                staging_buffer,
                0,
                resolve_buffer.size(),
            );
        }

        if let Some(mrt) = &ctx.mrt_views {
            let (width, height) = ctx.viewport;
            let has_composite = self.passes.iter().any(|p| p.name() == "composite");
            if !has_composite {
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: mrt.color_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: ctx.target.texture(),
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
            }
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

/// Ensures all meshes referenced by draw batches are in the GPU mesh buffer cache.
fn ensure_mesh_buffers(
    gpu: &mut crate::gpu::GpuState,
    session: &crate::session::Session,
    draw_batches: &[SpaceDrawBatch],
) {
    let mesh_assets = session.asset_registry();
    for batch in draw_batches {
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
            if !gpu.mesh_buffer_cache.contains_key(&d.mesh_asset_id) {
                let stride = crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                let stride = if stride > 0 {
                    stride
                } else {
                    crate::gpu::compute_vertex_stride_from_mesh(mesh)
                };
                let ray_tracing = gpu.ray_tracing_available;
                if let Some(b) =
                    crate::gpu::create_mesh_buffers(&gpu.device, mesh, stride, ray_tracing)
                {
                    gpu.mesh_buffer_cache.insert(d.mesh_asset_id, b.clone());
                    if let Some(ref mut accel) = gpu.accel_cache
                        && let Some(blas) =
                            crate::gpu::build_blas_for_mesh(&gpu.device, &gpu.queue, mesh, &b)
                    {
                        accel.insert(d.mesh_asset_id, blas);
                    }
                }
            }
        }
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
        let (non_overlay_skinned, _, non_overlay_non_skinned, _) = ctx
            .cached_mesh_draws
            .as_ref()
            .expect("mesh pass requires cached_mesh_draws");

        let use_mrt = ctx.render_target.mrt_position_view.is_some()
            && ctx.render_target.mrt_normal_view.is_some();

        let light_buffer_version = ctx.gpu.light_buffer_cache.version;
        let cluster_buffer_version = ctx.gpu.cluster_buffer_cache.version;

        let pbr_scene = match (
            ctx.gpu.cluster_buffer_cache.get_buffers(ctx.viewport),
            ctx.gpu
                .light_buffer_cache
                .ensure_buffer(&ctx.gpu.device, ctx.gpu.light_count.max(1) as usize),
        ) {
            (Some(crefs), Some(lb))
                if ctx.gpu.cluster_count_x > 0 && ctx.gpu.cluster_count_y > 0 =>
            {
                let view_position = ctx
                    .draw_batches
                    .iter()
                    .find(|b| !b.is_overlay)
                    .map(|b| {
                        let m = crate::scene::render_transform_to_matrix(&b.view_transform);
                        [m.w_axis.x, m.w_axis.y, m.w_axis.z]
                    })
                    .unwrap_or([0.0, 0.0, 0.0]);
                Some(mesh_draw::PbrSceneParams {
                    view_position,
                    cluster_count_x: ctx.gpu.cluster_count_x,
                    cluster_count_y: ctx.gpu.cluster_count_y,
                    light_count: ctx.gpu.light_count,
                    light_buffer: lb,
                    cluster_light_counts: crefs.cluster_light_counts,
                    cluster_light_indices: crefs.cluster_light_indices,
                })
            }
            _ => None,
        };

        let mut draw_params = MeshDrawParams {
            pipeline_manager: ctx.pipeline_manager,
            device: &ctx.gpu.device,
            queue: &ctx.gpu.queue,
            config: &ctx.gpu.config,
            frame_index: ctx.frame_index,
            mesh_buffer_cache: &ctx.gpu.mesh_buffer_cache,
            skinned_bind_group_cache: &mut ctx.gpu.skinned_bind_group_cache,
            overlay_orthographic: false,
            use_mrt,
            use_pbr: ctx.session.render_config().use_pbr,
            pbr_scene,
            pbr_scene_bind_group_cache: &mut ctx.gpu.pbr_scene_bind_group_cache,
            last_pbr_scene_cache_light_version: &mut ctx.gpu.last_pbr_scene_cache_light_version,
            last_pbr_scene_cache_cluster_version: &mut ctx.gpu.last_pbr_scene_cache_cluster_version,
            light_buffer_version,
            cluster_buffer_version,
        };

        let timestamp_writes =
            ctx.timestamp_query_set
                .map(|query_set| wgpu::RenderPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });

        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = if use_mrt {
            let pos_view = ctx
                .render_target
                .mrt_position_view
                .expect("use_mrt is true only when both MRT views are Some");
            let norm_view = ctx
                .render_target
                .mrt_normal_view
                .expect("use_mrt is true only when both MRT views are Some");
            vec![
                Some(wgpu::RenderPassColorAttachment {
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
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: pos_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: norm_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ]
        } else {
            vec![Some(wgpu::RenderPassColorAttachment {
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
            })]
        };

        // Non-overlay pass: Clear framebuffer, draw all non-overlay batches.
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh pass (non-overlay)"),
                timestamp_writes: timestamp_writes.clone(),
                color_attachments: &color_attachments,
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
                non_overlay_skinned,
                debug_blendshapes,
            );
            record_non_skinned_draws(&mut pass, &mut draw_params, non_overlay_non_skinned);
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
///
/// ## GraphicsChunk stencil flow
///
/// Draws with `stencil_state.is_some()` use
/// [`crate::gpu::PipelineVariant::OverlayStencilContent`] or
/// [`crate::gpu::PipelineVariant::OverlayStencilSkinned`] pipelines. The depth-stencil attachment uses
/// `LoadOp::Load`/`StoreOp::Store` for stencil so MaskWrite → Content → MaskClear
/// phases can read/write stencil across draws. Per draw, the pass calls
/// `set_stencil_reference(stencil_state.reference)` before the draw. See
/// [`crate::stencil`] for GraphicsChunk RenderType (MaskWrite, Content, MaskClear).
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
        let (_, overlay_skinned, _, overlay_non_skinned) = ctx
            .cached_mesh_draws
            .as_ref()
            .expect("overlay pass requires cached_mesh_draws");

        if overlay_skinned.is_empty() && overlay_non_skinned.is_empty() {
            return Ok(());
        }

        let overlay_orthographic = ctx.overlay_projection_override.is_some();
        let light_buffer_version = ctx.gpu.light_buffer_cache.version;
        let cluster_buffer_version = ctx.gpu.cluster_buffer_cache.version;
        let mut draw_params = MeshDrawParams {
            pipeline_manager: ctx.pipeline_manager,
            device: &ctx.gpu.device,
            queue: &ctx.gpu.queue,
            config: &ctx.gpu.config,
            frame_index: ctx.frame_index,
            mesh_buffer_cache: &ctx.gpu.mesh_buffer_cache,
            skinned_bind_group_cache: &mut ctx.gpu.skinned_bind_group_cache,
            overlay_orthographic,
            use_mrt: false,
            use_pbr: false,
            pbr_scene: None,
            pbr_scene_bind_group_cache: &mut ctx.gpu.pbr_scene_bind_group_cache,
            last_pbr_scene_cache_light_version: &mut ctx.gpu.last_pbr_scene_cache_light_version,
            last_pbr_scene_cache_cluster_version: &mut ctx.gpu.last_pbr_scene_cache_cluster_version,
            light_buffer_version,
            cluster_buffer_version,
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
        // Stencil Load/Store preserves stencil across draws for GraphicsChunk
        // MaskWrite → Content → MaskClear flow.

        let debug_blendshapes = ctx.session.render_config().debug_blendshapes;
        record_skinned_draws(
            &mut pass,
            &mut draw_params,
            overlay_skinned,
            debug_blendshapes,
        );
        record_non_skinned_draws(&mut pass, &mut draw_params, overlay_non_skinned);

        Ok(())
    }
}
