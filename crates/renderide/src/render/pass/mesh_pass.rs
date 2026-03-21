//! Mesh render pass: draws non-overlay meshes from draw batches.
//!
//! Uses `LoadOp::Clear` for color and depth; draws all non-overlay batches (skinned and
//! non-skinned). Batches are pre-sorted (non-overlay first, then overlay) by the session.

use super::mesh_draw::{self, MeshDrawParams, record_non_skinned_draws, record_skinned_draws};
use super::{PassResources, RenderPass, RenderPassContext, RenderPassError, ResourceSlot};
use crate::gpu::pipeline::{RT_SHADOW_MODE_ATLAS, RT_SHADOW_MODE_TRACE};
use crate::render::batch::SpaceDrawBatch;
use crate::session::Session;

/// Non-overlay batch used for PBR and clustered lighting: same selection as [`super::ClusteredLightPass`].
pub(crate) fn pbr_primary_view_batch<'a>(
    draw_batches: &'a [SpaceDrawBatch],
    session: &'a Session,
) -> Option<&'a SpaceDrawBatch> {
    let space_id = session.primary_view_space_id().or_else(|| {
        draw_batches
            .iter()
            .find(|b| !b.is_overlay)
            .map(|b| b.space_id)
    });
    space_id
        .and_then(|sid| {
            draw_batches
                .iter()
                .find(|b| !b.is_overlay && b.space_id == sid)
        })
        .or_else(|| draw_batches.iter().find(|b| !b.is_overlay))
}

/// Third column of world-to-view (Z row for column vectors): `view_z = dot(xyz, world_pos) + w`.
pub(crate) fn pbr_view_space_z_coeffs_for_batch(batch: &SpaceDrawBatch) -> [f32; 4] {
    let view = crate::render::visibility::view_matrix_glam_for_batch(batch);
    [view.x_axis.z, view.y_axis.z, view.z_axis.z, view.w_axis.z]
}

/// World-space translation subtracted from fragment world position before writing the MRT position target.
///
/// The mesh pass uploads this to the MRT origin uniform (debug MRT) or it is implied by
/// `SceneUniforms::view_position` (PBR MRT). RTAO adds the same vector back when tracing.
pub(crate) fn mrt_gbuffer_world_origin(
    draw_batches: &[SpaceDrawBatch],
    session: &Session,
) -> [f32; 3] {
    let batch = pbr_primary_view_batch(draw_batches, session);
    batch
        .map(|b| {
            let m = crate::scene::render_transform_to_matrix(&b.view_transform);
            [m.w_axis.x, m.w_axis.y, m.w_axis.z]
        })
        .unwrap_or([0.0, 0.0, 0.0])
}

/// Mesh render pass: draws non-overlay meshes from draw batches.
///
/// [`Self::with_rtao_mrt_graph`] selects the **graph variant**: when true, resource declarations
/// include MRT g-buffer slots ([`ResourceSlot::Position`], [`ResourceSlot::Normal`],
/// [`ResourceSlot::AoRaw`]) used by the RTAO pipeline. When false, only [`ResourceSlot::Color`]
/// and [`ResourceSlot::Depth`] are declared so the mesh writes the main color target and depth;
/// [`ResourceSlot::Depth`] is required because [`super::OverlayRenderPass`] reads depth after mesh.
pub struct MeshRenderPass {
    rtao_mrt_graph: bool,
}

impl MeshRenderPass {
    /// Creates a mesh pass for the non-RTAO graph variant (color + depth to the surface).
    pub fn new() -> Self {
        Self::with_rtao_mrt_graph(false)
    }

    /// Creates a mesh pass aligned with the main render graph variant.
    ///
    /// Pass `true` when the graph includes RTAO compute, blur, and composite; `false` when mesh
    /// connects directly to overlay.
    pub fn with_rtao_mrt_graph(rtao_mrt_graph: bool) -> Self {
        Self { rtao_mrt_graph }
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

    fn resources(&self) -> PassResources {
        let writes = if self.rtao_mrt_graph {
            vec![
                ResourceSlot::Color,
                ResourceSlot::Position,
                ResourceSlot::Normal,
                ResourceSlot::AoRaw,
                ResourceSlot::Depth,
            ]
        } else {
            vec![ResourceSlot::Color, ResourceSlot::Depth]
        };
        PassResources {
            reads: vec![ResourceSlot::ClusterBuffers, ResourceSlot::LightBuffer],
            writes,
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        let (non_overlay_skinned, _, non_overlay_non_skinned, _) =
            ctx.cached_mesh_draws
                .as_ref()
                .ok_or(RenderPassError::MissingCachedMeshDraws)?;

        let use_mrt = ctx.render_target.mrt_position_view.is_some()
            && ctx.render_target.mrt_normal_view.is_some();

        let mrt_world_origin = mrt_gbuffer_world_origin(ctx.draw_batches, ctx.session);
        if use_mrt {
            ctx.gpu.ensure_mrt_gbuffer_origin_resources(
                ctx.pipeline_manager.mrt_gbuffer_origin_layout(),
            );
            ctx.gpu
                .write_mrt_gbuffer_origin(&ctx.gpu.queue, mrt_world_origin);
        }

        let use_ray_tracing_scene_early = {
            let rt = ctx.gpu.ray_tracing_state.as_ref();
            ctx.session.render_config().ray_traced_shadows_enabled
                && ctx.gpu.ray_tracing_available
                && rt.is_some_and(|r| r.tlas.is_some())
        };
        if use_ray_tracing_scene_early {
            ctx.gpu.ensure_rt_shadow_bind_resources();
        }

        let mrt_gbuffer_origin_bind_group = if use_mrt {
            ctx.gpu.mrt_gbuffer_origin_bind_group.as_ref()
        } else {
            None
        };

        let light_buffer_version = ctx.gpu.light_buffer_cache.version;
        let cluster_buffer_version = ctx.gpu.cluster_buffer_cache.version;

        let rt_state = ctx.gpu.ray_tracing_state.as_ref();

        let cluster_buffers = ctx
            .gpu
            .cluster_buffer_cache
            .get_buffers(ctx.viewport, ctx.gpu.cluster_count_z);
        let light_buffer = ctx
            .gpu
            .light_buffer_cache
            .ensure_buffer(&ctx.gpu.device, ctx.gpu.light_count.max(1) as usize);
        let (viewport_width, viewport_height) = ctx.viewport;
        let cluster_counts_ok = ctx.gpu.cluster_count_x > 0
            && ctx.gpu.cluster_count_y > 0
            && ctx.gpu.cluster_count_z > 0;
        let cluster_buffers_is_none = cluster_buffers.is_none();
        let light_buffer_is_none = light_buffer.is_none();

        let pbr_tlas_generation = rt_state.map(|r| r.tlas_generation).unwrap_or(0);
        let use_ray_tracing_scene = use_ray_tracing_scene_early;
        let pbr_tlas_ptr = if use_ray_tracing_scene {
            rt_state.and_then(|r| {
                r.tlas.as_ref().map(|t| {
                    std::ptr::NonNull::new(t as *const wgpu::Tlas as *mut wgpu::Tlas)
                        .expect("TLAS reference must be non-null")
                })
            })
        } else {
            None
        };

        let rc = ctx.session.render_config();
        let shadow_mode = if self.rtao_mrt_graph
            && rc.ray_traced_shadows_use_compute
            && rc.ray_traced_shadows_enabled
            && ctx.gpu.ray_tracing_available
            && ctx.gpu.rt_shadow_atlas_main_view.is_some()
        {
            RT_SHADOW_MODE_ATLAS
        } else {
            RT_SHADOW_MODE_TRACE
        };
        let rt_shadow_bind = if use_ray_tracing_scene {
            let uniform_buffer = ctx
                .gpu
                .rt_shadow_uniform_buffer
                .as_ref()
                .expect("ensure_rt_shadow_bind_resources creates rt shadow uniform buffer");
            let fallback_atlas = ctx
                .gpu
                .rt_shadow_fallback_atlas_view
                .as_ref()
                .expect("ensure_rt_shadow_bind_resources creates fallback atlas");
            let atlas_view = ctx
                .gpu
                .rt_shadow_atlas_main_view
                .as_ref()
                .unwrap_or(fallback_atlas);
            let sampler = ctx
                .gpu
                .rt_shadow_sampler
                .as_ref()
                .expect("ensure_rt_shadow_bind_resources creates rt shadow sampler");
            let (atlas_width, atlas_height) = ctx.gpu.rt_shadow_atlas_extent.unwrap_or((1, 1));
            Some(mesh_draw::RtShadowBindParams {
                uniform_buffer,
                atlas_view,
                sampler,
                soft_samples: rc.rt_soft_shadow_samples,
                cone_scale: rc.rt_soft_shadow_cone_scale,
                shadow_mode,
                full_viewport_width: viewport_width,
                full_viewport_height: viewport_height,
                atlas_width,
                atlas_height,
                gbuffer_origin: mrt_world_origin,
            })
        } else {
            None
        };

        let pbr_scene = match (cluster_buffers, light_buffer, cluster_counts_ok) {
            (Some(crefs), Some(lb), true) => {
                let batch = pbr_primary_view_batch(ctx.draw_batches, ctx.session);
                let view_space_z_coeffs = batch
                    .map(pbr_view_space_z_coeffs_for_batch)
                    .unwrap_or([0.0, 0.0, 0.0, 0.0]);
                Some(mesh_draw::PbrSceneParams {
                    view_position: mrt_world_origin,
                    view_space_z_coeffs,
                    cluster_count_x: ctx.gpu.cluster_count_x,
                    cluster_count_y: ctx.gpu.cluster_count_y,
                    cluster_count_z: ctx.gpu.cluster_count_z,
                    near_clip: ctx.session.near_clip().max(0.01),
                    far_clip: ctx.session.far_clip(),
                    light_count: ctx.gpu.light_count,
                    viewport_width,
                    viewport_height,
                    light_buffer: lb,
                    cluster_light_counts: crefs.cluster_light_counts,
                    cluster_light_indices: crefs.cluster_light_indices,
                    use_ray_tracing_scene,
                })
            }
            _ => {
                if ctx.session.render_config().use_pbr {
                    let reason = if !cluster_counts_ok {
                        format!(
                            "cluster_count_* zero (x={} y={} z={}) - clustered_light did not run this frame",
                            ctx.gpu.cluster_count_x,
                            ctx.gpu.cluster_count_y,
                            ctx.gpu.cluster_count_z
                        )
                    } else if cluster_buffers_is_none {
                        "cluster get_buffers returned None (viewport or cluster_count_z mismatch)"
                            .to_string()
                    } else if light_buffer_is_none {
                        "light buffer ensure_buffer returned None".to_string()
                    } else {
                        "unknown".to_string()
                    };
                    logger::trace!("PBR scene disabled (no clustered lighting): {}", reason);
                }
                None
            }
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
            last_pbr_scene_cache_tlas_generation: &mut ctx.gpu.last_pbr_scene_cache_tlas_generation,
            light_buffer_version,
            cluster_buffer_version,
            pbr_tlas_generation,
            pbr_tlas_ptr,
            mrt_gbuffer_origin_bind_group,
            rt_shadow_atlas_generation: ctx.gpu.rt_shadow_atlas_generation,
            last_pbr_scene_cache_rt_shadow_atlas_generation: &mut ctx
                .gpu
                .last_pbr_scene_cache_rt_shadow_atlas_generation,
            rt_shadow_bind,
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
                .ok_or(RenderPassError::MissingMrtViews)?;
            let norm_view = ctx
                .render_target
                .mrt_normal_view
                .ok_or(RenderPassError::MissingMrtViews)?;
            vec![
                Some(wgpu::RenderPassColorAttachment {
                    view: ctx.render_target.color_view,
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
                        g: 0.0,
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
