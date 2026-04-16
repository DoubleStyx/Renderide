//! Main forward pass: clear color + depth, debug normal shading for scene meshes.
//!
//! Draws are collected and **sorted by [`MaterialDrawBatchKey`](crate::render_graph::MaterialDrawBatchKey)**
//! so pipeline and batch key drive pipeline switches. **GPU instancing:** consecutive draws that share the
//! same mesh submesh and batch key (opaque, non-skinned) are merged into one indexed draw with
//! `instance_index` sampling [`crate::backend::PerDrawResources`] (`@group(2)` storage). Embedded `@group(1)`
//! skips redundant [`wgpu::RenderPass::set_bind_group`] when [`MaterialBindCacheKey`](crate::backend::MaterialBindCacheKey) matches
//! the previous draw (uniform updates still run each time via [`EmbeddedMaterialBindResources`](crate::backend::EmbeddedMaterialBindResources)).
//! Per-slot [`MaterialPropertyLookupIds`](crate::assets::material::MaterialPropertyLookupIds) are carried on each
//! [`WorldMeshDrawItem`](crate::render_graph::WorldMeshDrawItem) for `get_merged` when building `@group(1)` bind
//! groups for [`crate::materials::RasterPipelineKind::EmbeddedStem`] draws (see [`crate::backend::EmbeddedMaterialBindResources`]).
//!
//! Manifest raster binds use the composed WGSL **stem** from [`crate::materials::MaterialRouter::stem_for_shader_asset`]
//! (not a hard-coded Unlit path). Whether UV0 is bound is stored on [`MaterialDrawBatchKey::Embedded_needs_uv0`]
//! (same rule as the embedded raster pipeline and [`crate::materials::embedded_stem_needs_uv0_stream`], computed during draw collection).
//! Intersection tint subpasses use [`MaterialDrawBatchKey::Embedded_requires_intersection_pass`]
//! ([`crate::materials::embedded_stem_requires_intersection_pass`], WGSL reflection of `_IntersectColor`).
//!
//! ## VR stereo world draws
//!
//! OpenXR per-eye view–projection maps **stage** space to clip. For **non-overlay** draws with
//! `stereo_view_proj`, we use **identity** instead of the host `view_transform` world-to-camera so
//! `VP` is not `P·V_hmd·V_host`, which mixed stage with the host rig and caused playspace-relative
//! offsets. Overlays keep `view` for orthographic / UI alignment with the host camera rig.
//! Matrix composition lives in [`vp`].

mod encode;
mod vp;

use std::num::NonZeroU32;

use bytemuck::Zeroable;
use rayon::prelude::*;

use crate::assets::material::MaterialDictionary;
use crate::backend::mesh_deform::{
    write_per_draw_uniform_slab, PaddedPerDrawUniforms, PER_DRAW_UNIFORM_STRIDE,
};
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::materials::{
    MaterialPipelineDesc, MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;
use crate::pipelines::SHADER_PERM_MULTIVIEW_STEREO;
use crate::present::SWAPCHAIN_CLEAR_COLOR;
use crate::render_graph::camera::{
    effective_head_output_clip_planes, reverse_z_orthographic, reverse_z_perspective,
};
use crate::render_graph::cluster_frame::{cluster_frame_params, cluster_frame_params_stereo};
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};
use crate::render_graph::MAIN_FORWARD_DEPTH_CLEAR;
use crate::render_graph::{
    build_world_mesh_cull_proj_params, collect_and_sort_world_mesh_draws, WorldMeshCullInput,
    WorldMeshDrawItem,
};
use crate::render_graph::{
    world_mesh_draw_state_rows_from_sorted, world_mesh_draw_stats_from_sorted,
};

use encode::draw_subset;
use vp::compute_per_draw_vp_triple;

/// Minimum draws before parallelizing per-draw VP / model uniform packing (rayon overhead).
const PER_DRAW_VP_PARALLEL_MIN_DRAWS: usize = 256;

fn stencil_clear_ops(format: wgpu::TextureFormat) -> Option<wgpu::Operations<u32>> {
    format.has_stencil_aspect().then_some(wgpu::Operations {
        load: wgpu::LoadOp::Clear(0),
        store: wgpu::StoreOp::Store,
    })
}

fn stencil_load_ops(format: wgpu::TextureFormat) -> Option<wgpu::Operations<u32>> {
    format.has_stencil_aspect().then_some(wgpu::Operations {
        load: wgpu::LoadOp::Load,
        store: wgpu::StoreOp::Store,
    })
}

fn current_view_texture2d_asset_ids_from_draws(
    draws: &[WorldMeshDrawItem],
    backend: &crate::backend::RenderBackend,
) -> Vec<i32> {
    let Some(material_registry) = backend.materials.material_registry.as_ref() else {
        return Vec::new();
    };
    let Some(embedded_bind) = backend.materials.embedded_material_bind() else {
        return Vec::new();
    };
    let store = backend.material_property_store();
    let mut asset_ids = Vec::new();
    for item in draws {
        if !matches!(
            &item.batch_key.pipeline,
            RasterPipelineKind::EmbeddedStem(_)
        ) {
            continue;
        }
        let Some(stem) = material_registry.stem_for_shader_asset(item.batch_key.shader_asset_id)
        else {
            continue;
        };
        match embedded_bind.texture2d_asset_ids_for_stem(stem, store, item.lookup_ids) {
            Ok(ids) => asset_ids.extend(ids),
            Err(e) => logger::trace!("Texture HUD: failed to inspect embedded stem {stem}: {e}"),
        }
    }
    asset_ids.sort_unstable();
    asset_ids.dedup();
    asset_ids
}

/// Clears the backbuffer and depth, then draws meshes with material-batched raster pipelines.
#[derive(Debug, Default)]
pub struct WorldMeshForwardPass;

impl WorldMeshForwardPass {
    /// Creates a world mesh forward pass instance.
    pub fn new() -> Self {
        Self
    }
}

impl RenderPass for WorldMeshForwardPass {
    fn name(&self) -> &str {
        "WorldMeshForward"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![ResourceSlot::ClusterBuffers, ResourceSlot::LightBuffer],
            writes: vec![ResourceSlot::Backbuffer, ResourceSlot::Depth],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(bb) = ctx.backbuffer else {
            return Err(RenderPassError::MissingBackbuffer {
                pass: self.name().to_string(),
            });
        };
        let Some(depth) = ctx.depth_view else {
            return Err(RenderPassError::MissingDepth {
                pass: self.name().to_string(),
            });
        };
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

        // Merged instance batches use non-zero `first_instance` on `draw_indexed`. Downlevel
        // adapters without [`wgpu::DownlevelFlags::BASE_INSTANCE`] must use `false` (one draw per item).
        let supports_base_instance = ctx.gpu_limits.supports_base_instance;

        let hc = frame.host_camera;
        let use_multiview = frame.multiview_stereo
            && hc.vr_active
            && hc.stereo_view_proj.is_some()
            && ctx.gpu_limits.supports_multiview;

        let depth_format = frame.depth_texture.format();
        let pass_desc = MaterialPipelineDesc {
            surface_format: frame.surface_format,
            depth_stencil_format: Some(depth_format),
            sample_count: 1,
            multiview_mask: if use_multiview {
                NonZeroU32::new(3)
            } else {
                None
            },
        };
        let shader_perm = if use_multiview {
            SHADER_PERM_MULTIVIEW_STEREO
        } else {
            ShaderPermutation(0)
        };

        let render_context = frame.scene.active_main_render_context();
        let culling = if hc.suppress_occlusion_temporal {
            None
        } else {
            let cull_proj = build_world_mesh_cull_proj_params(frame.scene, frame.viewport_px, &hc);
            let depth_mode = frame.output_depth_mode();
            let view_id = frame.occlusion_view;
            let hi_z_temporal = frame.backend.occlusion.hi_z_temporal_snapshot(view_id);
            let hi_z = frame.backend.occlusion.hi_z_cull_data(depth_mode, view_id);
            Some(WorldMeshCullInput {
                proj: cull_proj,
                host_camera: &hc,
                hi_z,
                hi_z_temporal,
            })
        };

        let collection = if let Some(prefetched) = frame.prefetched_world_mesh_draws.take() {
            prefetched
        } else {
            let backend = &mut frame.backend;
            let fallback_router = MaterialRouter::new(RasterPipelineKind::DebugWorldNormals);
            let router_ref = backend
                .materials
                .material_registry
                .as_ref()
                .map(|r| &r.router)
                .unwrap_or(&fallback_router);
            let dict = MaterialDictionary::new(backend.material_property_store());
            let pipeline_property_ids =
                MaterialPipelinePropertyIds::new(backend.property_id_registry());
            collect_and_sort_world_mesh_draws(
                frame.scene,
                backend.mesh_pool(),
                &dict,
                router_ref,
                &pipeline_property_ids,
                shader_perm,
                render_context,
                hc.head_output_transform,
                culling.as_ref(),
                frame.transform_draw_filter.as_ref(),
            )
        };
        let track_current_view_textures = frame.offscreen_write_render_texture_asset_id.is_none();
        let backend = &mut frame.backend;
        if !hc.suppress_occlusion_temporal {
            if let Some(ref cull_in) = culling {
                let view_id = frame.occlusion_view;
                backend.occlusion.capture_hi_z_temporal_for_next_frame(
                    frame.scene,
                    cull_in.proj,
                    frame.viewport_px,
                    view_id,
                    hc.secondary_camera_world_to_view,
                );
            }
        }
        let draws = collection.items;
        if backend.debug_hud_textures_enabled() && track_current_view_textures {
            let asset_ids = current_view_texture2d_asset_ids_from_draws(&draws, backend);
            backend.note_debug_hud_current_view_texture_2d_asset_ids(asset_ids);
        }
        if backend.debug_hud_main_enabled() {
            let stats = world_mesh_draw_stats_from_sorted(
                &draws,
                Some((
                    collection.draws_pre_cull,
                    collection.draws_culled,
                    collection.draws_hi_z_culled,
                )),
                supports_base_instance,
            );
            let state_rows = world_mesh_draw_state_rows_from_sorted(&draws);
            backend.set_last_world_mesh_draw_stats(stats);
            backend.set_last_world_mesh_draw_state_rows(state_rows);
        }
        let (vw, vh) = frame.viewport_px;
        let aspect = vw as f32 / vh.max(1) as f32;
        let (near, far) = effective_head_output_clip_planes(
            hc.near_clip,
            hc.far_clip,
            hc.output_device,
            frame
                .scene
                .active_main_space()
                .map(|space| space.root_transform.scale),
        );
        let fov_rad =
            crate::render_graph::clamp_desktop_fov_degrees(hc.desktop_fov_degrees).to_radians();
        let world_proj = reverse_z_perspective(aspect, fov_rad, near, far);

        let has_overlay = !draws.is_empty() && draws.iter().any(|d| d.is_overlay);
        let overlay_proj = if has_overlay {
            Some(if let Some((half_h, on, of)) = hc.primary_ortho_task {
                reverse_z_orthographic(half_h * aspect, half_h, on, of)
            } else {
                reverse_z_orthographic(1.0 * aspect, 1.0, near, far)
            })
        } else {
            None
        };

        let scene = frame.scene;

        let mut slab_bytes = Vec::new();
        if !draws.is_empty() {
            {
                let Some(pd) = backend.frame_resources.per_draw.as_mut() else {
                    return Ok(());
                };
                pd.ensure_draw_slot_capacity(ctx.device, draws.len());
            }

            let slots: Vec<PaddedPerDrawUniforms> = if draws.len() >= PER_DRAW_VP_PARALLEL_MIN_DRAWS
            {
                draws
                    .par_iter()
                    .map(|item| {
                        let (vp_l, vp_r, model) = compute_per_draw_vp_triple(
                            scene,
                            item,
                            hc,
                            render_context,
                            world_proj,
                            overlay_proj,
                        );
                        if vp_l == vp_r {
                            PaddedPerDrawUniforms::new_single(vp_l, model)
                        } else {
                            PaddedPerDrawUniforms::new_stereo(vp_l, vp_r, model)
                        }
                    })
                    .collect()
            } else {
                draws
                    .iter()
                    .map(|item| {
                        let (vp_l, vp_r, model) = compute_per_draw_vp_triple(
                            scene,
                            item,
                            hc,
                            render_context,
                            world_proj,
                            overlay_proj,
                        );
                        if vp_l == vp_r {
                            PaddedPerDrawUniforms::new_single(vp_l, model)
                        } else {
                            PaddedPerDrawUniforms::new_stereo(vp_l, vp_r, model)
                        }
                    })
                    .collect()
            };

            slab_bytes = vec![0u8; draws.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE)];
            write_per_draw_uniform_slab(&slots, &mut slab_bytes);
        }

        let queue_guard = ctx
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let queue = &*queue_guard;

        if !draws.is_empty() {
            let Some(pd) = backend.frame_resources.per_draw.as_mut() else {
                return Ok(());
            };
            queue.write_buffer(&pd.per_draw_storage, 0, &slab_bytes);
        }
        let light_count_u = backend.frame_resources.frame_light_count_u32();
        let camera_world = hc
            .secondary_camera_world_position
            .unwrap_or_else(|| hc.head_output_transform.col(3).truncate());

        let stereo_cluster = use_multiview && hc.vr_active && hc.stereo_views.is_some();

        let uniforms = if stereo_cluster {
            if let Some((left, right)) = cluster_frame_params_stereo(&hc, scene, (vw, vh)) {
                left.frame_gpu_uniforms(camera_world, light_count_u, Some(&right))
            } else if let Some(mono) = cluster_frame_params(&hc, scene, (vw, vh)) {
                mono.frame_gpu_uniforms(camera_world, light_count_u, None)
            } else {
                FrameGpuUniforms::zeroed()
            }
        } else if let Some(mono) = cluster_frame_params(&hc, scene, (vw, vh)) {
            mono.frame_gpu_uniforms(camera_world, light_count_u, None)
        } else {
            FrameGpuUniforms::zeroed()
        };

        if let Some(fgpu) = backend.frame_resources.frame_gpu_mut() {
            fgpu.sync_cluster_viewport(ctx.device, (vw, vh), stereo_cluster);
        }
        backend
            .frame_resources
            .write_frame_uniform_and_lights_from_scratch(queue, &uniforms);

        if draws.is_empty() {
            // Still clear color + depth so offscreen render textures are defined (no draws → no geometry).
            {
                let _rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("world-mesh-forward-clear-only"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: bb,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(SWAPCHAIN_CLEAR_COLOR),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(MAIN_FORWARD_DEPTH_CLEAR),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: stencil_clear_ops(depth_format),
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: if use_multiview {
                        NonZeroU32::new(3)
                    } else {
                        None
                    },
                });
            }
            return Ok(());
        }

        let Some(per_draw_bg) = backend
            .frame_resources
            .per_draw
            .as_ref()
            .map(|d| d.bind_group.clone())
        else {
            return Ok(());
        };

        let mut regular_indices = Vec::with_capacity(draws.len());
        let mut intersect_indices = Vec::new();
        for (draw_idx, item) in draws.iter().enumerate() {
            if item.batch_key.embedded_requires_intersection_pass {
                intersect_indices.push(draw_idx);
            } else {
                regular_indices.push(draw_idx);
            }
        }

        let mut warned_missing_embedded_bind = false;
        let Some((frame_bg_arc, empty_bg_arc)) =
            backend.frame_resources.mesh_forward_frame_bind_groups()
        else {
            return Ok(());
        };

        let offscreen_write_rt = frame.offscreen_write_render_texture_asset_id;

        {
            let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world-mesh-forward-opaque"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: bb,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(SWAPCHAIN_CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(MAIN_FORWARD_DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: stencil_clear_ops(depth_format),
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: if use_multiview {
                    NonZeroU32::new(3)
                } else {
                    None
                },
            });
            draw_subset(
                &mut rpass,
                &regular_indices,
                &draws,
                backend,
                queue,
                frame_bg_arc.as_ref(),
                empty_bg_arc.as_ref(),
                per_draw_bg.as_ref(),
                &pass_desc,
                shader_perm,
                &mut warned_missing_embedded_bind,
                offscreen_write_rt,
                supports_base_instance,
            );
        }

        if !intersect_indices.is_empty() {
            if let Some(fgpu) = backend.frame_resources.frame_gpu_mut() {
                fgpu.copy_scene_depth_snapshot(
                    ctx.device,
                    ctx.encoder,
                    frame.depth_texture,
                    (vw, vh),
                    use_multiview,
                    stereo_cluster,
                );
            }
            let Some((frame_bg_arc, empty_bg_arc)) =
                backend.frame_resources.mesh_forward_frame_bind_groups()
            else {
                return Ok(());
            };
            let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world-mesh-forward-intersection"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: bb,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: stencil_load_ops(depth_format),
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: if use_multiview {
                    NonZeroU32::new(3)
                } else {
                    None
                },
            });
            draw_subset(
                &mut rpass,
                &intersect_indices,
                &draws,
                backend,
                queue,
                frame_bg_arc.as_ref(),
                empty_bg_arc.as_ref(),
                per_draw_bg.as_ref(),
                &pass_desc,
                shader_perm,
                &mut warned_missing_embedded_bind,
                offscreen_write_rt,
                supports_base_instance,
            );
        }

        Ok(())
    }
}
