//! Overlay render pass: draws overlay meshes on top of the main scene.
//!
//! Uses `LoadOp::Load` for color and depth; draws overlay batches (skinned and non-skinned)
//! with alpha blending to composite on top.
//!
//! ## Projection choice
//!
//! - **Orthographic** (set [`super::RenderGraphContext::overlay_projection_override`]): Use for
//!   screen-space UI (Canvas, HUD, fixed-size elements). Matches Unity Canvas render mode.
//! - **Perspective** (override `None`, default): Use for world-space overlays (3D UI in scene,
//!   floating panels with depth). Overlay batches use the main view's projection.
//!
//! ## GraphicsChunk stencil flow
//!
//! Draws with `stencil_state.is_some()` use
//! [`crate::gpu::PipelineVariant::OverlayStencilContent`] or
//! [`crate::gpu::PipelineVariant::OverlayStencilSkinned`] pipelines. The depth-stencil attachment uses
//! `LoadOp::Load`/`StoreOp::Store` for stencil so MaskWrite → Content → MaskClear
//! phases can read/write stencil across draws. Per draw, the pass calls
//! `set_stencil_reference(stencil_state.reference)` before the draw. See
//! [`crate::stencil`] for GraphicsChunk RenderType (MaskWrite, Content, MaskClear).

use super::mesh_draw::{MeshDrawParams, record_non_skinned_draws, record_skinned_draws};
use super::{PassResources, RenderPass, RenderPassContext, RenderPassError, ResourceSlot};

/// Overlay render pass: draws overlay meshes on top of the main scene.
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

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![ResourceSlot::Depth],
            writes: vec![ResourceSlot::Surface],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        let (_, overlay_skinned, _, overlay_non_skinned) = ctx
            .cached_mesh_draws
            .as_ref()
            .ok_or(RenderPassError::MissingCachedMeshDraws)?;

        if overlay_skinned.is_empty() && overlay_non_skinned.is_empty() {
            return Ok(());
        }

        let render_config = ctx.session.render_config();
        let mut native_ui_depth_bind = None;
        if render_config.use_native_ui_wgsl {
            let (vw, vh) = ctx.viewport;
            if ctx.gpu.depth_size == (vw, vh) && ctx.gpu.depth_texture.is_some() {
                ctx.gpu.ensure_ui_depth_copy_texture(vw, vh);
                if let (Some(src_tex), Some(dst_tex)) = (
                    ctx.gpu.depth_texture.as_ref(),
                    ctx.gpu.ui_depth_copy_texture.as_ref(),
                ) {
                    // Depth24PlusStencil8 copies must use `All` aspects on source (and destination);
                    // the bindable view for sampling is still depth-only (`ui_depth_copy_view`).
                    ctx.encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: src_tex,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: dst_tex,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: vw,
                            height: vh,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                ctx.gpu.ensure_native_ui_scene_depth_bind_group();
                native_ui_depth_bind = ctx.gpu.native_ui_scene_depth_bind_group.as_ref();
            }
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
            last_pbr_scene_cache_tlas_generation: &mut ctx.gpu.last_pbr_scene_cache_tlas_generation,
            light_buffer_version,
            cluster_buffer_version,
            pbr_tlas_generation: 0,
            pbr_tlas_ptr: None,
            mrt_gbuffer_origin_bind_group: None,
            rt_shadow_atlas_generation: ctx.gpu.rt_shadow_atlas_generation,
            last_pbr_scene_cache_rt_shadow_atlas_generation: &mut ctx
                .gpu
                .last_pbr_scene_cache_rt_shadow_atlas_generation,
            rt_shadow_bind: None,
            material_property_store: &ctx.session.asset_registry().material_property_store,
            render_config,
            native_ui_scene_depth_bind: native_ui_depth_bind,
            asset_registry: ctx.session.asset_registry(),
            texture2d_gpu: &mut ctx.gpu.texture2d_gpu,
            native_ui_material_bind_cache: &mut ctx.gpu.native_ui_material_bind_cache,
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
