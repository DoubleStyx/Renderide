//! [`RenderGraph::execute`] and scheduled unit traversal.

use crate::render::pass::error::RenderPassError;
use crate::render::pass::mesh_prep::{
    CachedMeshDrawsRef, ensure_mesh_buffers, run_collect_mesh_draws,
};

use super::context::{RenderGraphContext, RenderPassContext};
use super::resources::ResourceSlot;
use super::runtime::{ExecutionUnit, RenderGraph};
use super::views::{MrtViews, build_slot_map, render_target_views_for_pass};

impl RenderGraph {
    /// Creates an empty render graph.
    ///
    /// Schedulable graphs must be built with [`crate::render::pass::graph::GraphBuilder`] (or [`crate::render::pass::graph::build_main_render_graph`]) so
    /// edges and [`crate::render::pass::graph::GraphBuildError`] validation apply. An empty graph is only useful as a nested
    /// placeholder or before the builder moves content in; it has no composite/overlay [`crate::render::pass::graph::PassId`]
    /// metadata.
    pub fn new() -> Self {
        Self {
            rtao_mrt_cache: None,
            execution: Vec::new(),
            execution_order_pass_ids: Vec::new(),
            pass_resources: Vec::new(),
            composite_pass_id: None,
            overlay_pass_id: None,
        }
    }

    fn any_pass_writes_surface(&self) -> bool {
        self.execution.iter().any(|unit| match unit {
            ExecutionUnit::Pass { resources, .. } => {
                resources.writes.contains(&ResourceSlot::Surface)
            }
            ExecutionUnit::Subgraph(labeled) => labeled.graph.any_pass_writes_surface(),
        })
    }

    /// Runs this graph’s [`ExecutionUnit`] sequence on `encoder`. Used by [`execute`](Self::execute)
    /// and recursively for subgraphs.
    fn execute_scheduled_units(
        &mut self,
        ctx: &mut RenderGraphContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
        frame_index: u64,
        cached_mesh_draws: Option<CachedMeshDrawsRef<'_>>,
    ) -> Result<(), RenderPassError> {
        let (width, height) = ctx.viewport;
        let color_format = ctx.gpu.config.format;
        if ctx.enable_rtao_mrt {
            let with_shadow_atlas = ctx.session.render_config().ray_traced_shadows_use_compute
                && ctx.session.render_config().ray_traced_shadows_enabled
                && ctx.gpu.ray_tracing_available;
            let shadow_atlas_half_resolution =
                ctx.session.render_config().rt_shadow_atlas_half_resolution;
            let recreate = self.rtao_mrt_cache.as_ref().is_none_or(|c| {
                !c.matches_key(
                    width,
                    height,
                    color_format,
                    with_shadow_atlas,
                    shadow_atlas_half_resolution,
                )
            });
            if recreate {
                self.rtao_mrt_cache = Some(crate::gpu::rtao_textures::RtaoTextureCache::create(
                    &ctx.gpu.device,
                    width,
                    height,
                    color_format,
                    with_shadow_atlas,
                    shadow_atlas_half_resolution,
                ));
                ctx.gpu.rt_shadow_atlas_generation =
                    ctx.gpu.rt_shadow_atlas_generation.wrapping_add(1);
            }
            if let Some(ref c) = self.rtao_mrt_cache {
                if let Some(ref v) = c.shadow_atlas_view {
                    ctx.gpu.rt_shadow_atlas_main_view = Some(v.clone());
                    let (aw, ah) = if c.shadow_atlas_half_resolution {
                        (width.div_ceil(2).max(1), height.div_ceil(2).max(1))
                    } else {
                        (width.max(1), height.max(1))
                    };
                    ctx.gpu.rt_shadow_atlas_extent = Some((aw, ah));
                } else {
                    ctx.gpu.rt_shadow_atlas_main_view = None;
                    ctx.gpu.rt_shadow_atlas_extent = None;
                }
            }
        } else {
            let had_atlas = ctx.gpu.rt_shadow_atlas_main_view.is_some();
            self.rtao_mrt_cache = None;
            ctx.gpu.rt_shadow_atlas_main_view = None;
            ctx.gpu.rt_shadow_atlas_extent = None;
            if had_atlas {
                ctx.gpu.rt_shadow_atlas_generation =
                    ctx.gpu.rt_shadow_atlas_generation.wrapping_add(1);
            }
        }

        let mrt_views = self.rtao_mrt_cache.as_ref().map(|c| MrtViews {
            color_view: &c.color_view,
            color_texture: &c.color_texture,
            position_view: &c.position_view,
            position_texture: &c.position_texture,
            normal_view: &c.normal_view,
            normal_texture: &c.normal_texture,
            ao_raw_view: &c.ao_raw_view,
            ao_raw_texture: &c.ao_raw_texture,
            ao_view: &c.ao_view,
            ao_texture: &c.ao_texture,
        });

        let slot_map = build_slot_map(ctx.target, mrt_views.as_ref(), ctx.depth_view_override);

        for unit in &mut self.execution {
            match unit {
                ExecutionUnit::Pass { pass, resources } => {
                    let render_target = render_target_views_for_pass(&slot_map, Some(resources));

                    let mut pass_ctx = RenderPassContext {
                        gpu: ctx.gpu,
                        session: ctx.session,
                        draw_batches: ctx.draw_batches,
                        pipeline_manager: ctx.pipeline_manager,
                        frame_index,
                        viewport: ctx.viewport,
                        proj: ctx.proj,
                        overlay_projection_override: ctx.overlay_projection_override.as_ref(),
                        render_target,
                        encoder,
                        timestamp_query_set: ctx.timestamp_query_set,
                        cached_mesh_draws,
                    };
                    pass.execute(&mut pass_ctx)?;
                }
                ExecutionUnit::Subgraph(labeled) => {
                    labeled.graph.execute_scheduled_units(
                        ctx,
                        encoder,
                        frame_index,
                        cached_mesh_draws,
                    )?;
                }
            }
        }

        if let Some(mrt) = mrt_views.as_ref()
            && !self.any_pass_writes_surface()
        {
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

        Ok(())
    }

    /// Executes all passes in order, recording into a new command encoder.
    pub fn execute(&mut self, ctx: &mut RenderGraphContext) -> Result<(), RenderPassError> {
        let mut encoder = ctx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // Cluster counts and light_count are NOT reset here. ClusteredLightPass sets them when it
        // runs. If it skips, we keep the previous frame's values so the mesh pass can still use
        // cluster buffers from the last successful run (avoids "lights flash then vanish" when
        // clustered_light occasionally skips).

        let overlay_count = ctx.draw_batches.iter().filter(|b| b.is_overlay).count();
        let non_overlay_count = ctx.draw_batches.len().saturating_sub(overlay_count);
        logger::trace!(
            "render frame batches: {} overlay, {} non-overlay (clustered_light needs non-overlay)",
            overlay_count,
            non_overlay_count
        );

        if ctx.pre_collected.is_none() {
            ensure_mesh_buffers(ctx.gpu, ctx.session, ctx.draw_batches);
        }

        let computed;
        let cached_mesh_draws = match ctx.pre_collected {
            Some(pc) => Some((&pc.0[..], &pc.1[..], &pc.2[..], &pc.3[..])),
            None => {
                computed = run_collect_mesh_draws(
                    ctx.session,
                    ctx.draw_batches,
                    ctx.gpu,
                    ctx.proj,
                    ctx.overlay_projection_override.clone(),
                );
                let cached = &computed.0;
                Some((&cached.0[..], &cached.1[..], &cached.2[..], &cached.3[..]))
            }
        };

        let rc = ctx.session.render_config();
        if crate::gpu::needs_scene_ray_tracing_accel(
            ctx.gpu.ray_tracing_available,
            rc.rtao_enabled,
            rc.ray_traced_shadows_enabled,
        ) && let (Some(ref mut ray_tracing), Some(accel)) = (
            ctx.gpu.ray_tracing_state.as_mut(),
            ctx.gpu.accel_cache.as_ref(),
        ) {
            crate::gpu::update_tlas(
                &ctx.gpu.device,
                &mut encoder,
                ray_tracing,
                accel,
                ctx.draw_batches,
                &ctx.proj,
                ctx.overlay_projection_override.as_ref(),
                ctx.session.asset_registry(),
                rc.frustum_culling,
                &mut ctx.gpu.rigid_frustum_cull_cache,
            );
        }

        let frame_index = ctx.pipeline_manager.acquire_frame_index(&ctx.gpu.device);

        self.execute_scheduled_units(ctx, &mut encoder, frame_index, cached_mesh_draws)?;

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

        if let Some(hook) = ctx.before_submit.as_mut() {
            hook(&mut encoder);
        }

        let submission = ctx.gpu.queue.submit(std::iter::once(encoder.finish()));
        ctx.pipeline_manager
            .record_submission(submission, frame_index);
        Ok(())
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl RenderGraph {
    /// Returns pass names in execution order (depth-first; subgraph passes are prefixed with
    /// `label/`). For tests only.
    pub(super) fn pass_names(&self) -> Vec<String> {
        use super::pass_trait::RenderPass;

        let mut out = Vec::new();
        for unit in &self.execution {
            match unit {
                ExecutionUnit::Pass { pass, .. } => out.push(RenderPass::name(&**pass).to_string()),
                ExecutionUnit::Subgraph(labeled) => {
                    let prefix = labeled.label.as_str();
                    for n in labeled.graph.pass_names() {
                        out.push(format!("{prefix}/{n}"));
                    }
                }
            }
        }
        out
    }

    /// Returns composite and overlay PassIds for tests. For tests only.
    pub(super) fn special_pass_ids(
        &self,
    ) -> (Option<super::ids::PassId>, Option<super::ids::PassId>) {
        (self.composite_pass_id, self.overlay_pass_id)
    }

    /// Returns pass resource declarations in execution order. For tests only.
    pub(super) fn pass_resources(&self) -> &[super::resources::PassResources] {
        &self.pass_resources
    }
}
