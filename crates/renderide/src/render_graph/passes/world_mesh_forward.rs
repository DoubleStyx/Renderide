//! Main forward pass: clear color + depth, debug normal shading for scene meshes.
//!
//! Draws are collected and **sorted by [`MaterialDrawBatchKey`](crate::render_graph::MaterialDrawBatchKey)**
//! so pipeline and bind-group changes happen only when material / property-block / skinned path changes.
//! Per-slot [`MaterialPropertyLookupIds`](crate::assets::material::MaterialPropertyLookupIds) are carried on each
//! [`WorldMeshDrawItem`](crate::render_graph::WorldMeshDrawItem) for `get_merged` when building `@group(1)` bind
//! groups for [`crate::materials::RasterPipelineKind::EmbeddedStem`] draws (see [`crate::backend::EmbeddedMaterialBindResources`]).
//!
//! Manifest raster binds use the composed WGSL **stem** from [`crate::materials::MaterialRouter::stem_for_shader_asset`]
//! (not a hard-coded Unlit path). Whether UV0 is bound is stored on [`MaterialDrawBatchKey::embedded_needs_uv0`]
//! (same rule as the embedded raster pipeline and [`crate::materials::embedded_stem_needs_uv0_stream`], computed during draw collection).
//!
//! ## VR stereo world draws
//!
//! OpenXR per-eye view–projection maps **stage** space to clip. For **non-overlay** draws with
//! `stereo_view_proj`, we use **identity** instead of the host `view_transform` world-to-camera so
//! `VP` is not `P·V_hmd·V_host`, which mixed stage with the host rig and caused playspace-relative
//! offsets. Overlays keep `view` for orthographic / UI alignment.

use std::num::NonZeroU32;

use glam::Mat4;

use crate::assets::material::MaterialDictionary;
use crate::backend::{CLUSTER_COUNT_Z, TILE_SIZE};
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::{write_per_draw_uniform_slab, PaddedPerDrawUniforms, PER_DRAW_UNIFORM_STRIDE};
use crate::materials::{MaterialPipelineDesc, MaterialRouter, RasterPipelineKind};
use crate::pipelines::ShaderPermutation;
use crate::pipelines::SHADER_PERM_MULTIVIEW_STEREO;
use crate::present::SWAPCHAIN_CLEAR_COLOR;
use crate::render_graph::camera::{
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_orthographic,
    reverse_z_perspective, view_matrix_from_render_transform,
};
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};
#[cfg(feature = "debug-hud")]
use crate::render_graph::world_mesh_draw_stats_from_sorted;
use crate::render_graph::MAIN_FORWARD_DEPTH_CLEAR;
use crate::render_graph::{
    collect_and_sort_world_mesh_draws, resort_world_mesh_draws_for_camera, MaterialDrawBatchKey,
    WorldMeshDrawItem,
};
/// Clears the backbuffer and depth, then draws meshes with material-batched raster pipelines.
#[derive(Debug, Default)]
pub struct WorldMeshForwardPass;

impl WorldMeshForwardPass {
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

        let hc = frame.host_camera;
        let use_multiview = frame.multiview_stereo
            && hc.vr_active
            && hc.stereo_view_proj.is_some()
            && ctx.device.features().contains(wgpu::Features::MULTIVIEW);

        let pass_desc = MaterialPipelineDesc {
            surface_format: frame.surface_format,
            depth_stencil_format: Some(wgpu::TextureFormat::Depth32Float),
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

        let backend = &mut frame.backend;
        let render_context = frame.scene.active_main_render_context();
        let fallback_router = MaterialRouter::new(RasterPipelineKind::DebugWorldNormals);
        let router_ref = backend
            .material_registry
            .as_ref()
            .map(|r| &r.router)
            .unwrap_or(&fallback_router);
        let mut draws = {
            let dict = MaterialDictionary::new(backend.material_property_store());
            collect_and_sort_world_mesh_draws(
                frame.scene,
                &backend.mesh_pool,
                &dict,
                router_ref,
                shader_perm,
                render_context,
            )
        };
        let camera_world = hc.head_output_transform.col(3).truncate();
        resort_world_mesh_draws_for_camera(
            &mut draws,
            frame.scene,
            render_context,
            hc.head_output_transform,
            camera_world,
        );
        #[cfg(feature = "debug-hud")]
        {
            let stats = world_mesh_draw_stats_from_sorted(&draws);
            backend.set_last_world_mesh_draw_stats(stats);
        }
        if draws.is_empty() {
            return Ok(());
        }
        let lights_for_frame = backend.frame_lights().to_vec();
        {
            let Some(dbg) = backend.debug_draw.as_mut() else {
                return Ok(());
            };
            dbg.ensure_draw_slot_capacity(ctx.device, draws.len());
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
        let fov_rad = clamp_desktop_fov_degrees(hc.desktop_fov_degrees).to_radians();
        let world_proj = reverse_z_perspective(aspect, fov_rad, near, far);

        let has_overlay = draws.iter().any(|d| d.is_overlay);
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
        let mut slots: Vec<PaddedPerDrawUniforms> = Vec::with_capacity(draws.len());
        for item in &draws {
            let (vp_l, vp_r, model) = if let Some(space) = scene.space(item.space_id) {
                let view = view_matrix_from_render_transform(&space.view_transform);
                // OpenXR per-eye VP already includes the HMD pose; do not multiply host `view` again
                // for world-space meshes (avoids stage vs Resonite world double transform). Overlays
                // still use `view` for orthographic / UI alignment with the host camera rig.
                let vr_stereo_view = Mat4::IDENTITY;
                if let (true, Some((sl, sr))) = (hc.vr_active, hc.stereo_view_proj) {
                    if item.is_overlay {
                        let op = overlay_proj.unwrap_or(world_proj);
                        let base_vp = op * view;
                        if item.skinned {
                            (base_vp, base_vp, Mat4::IDENTITY)
                        } else {
                            let model = scene
                                .world_matrix_for_render_context(
                                    item.space_id,
                                    item.node_id as usize,
                                    render_context,
                                    hc.head_output_transform,
                                )
                                .unwrap_or(Mat4::IDENTITY);
                            (base_vp, base_vp, model)
                        }
                    } else if item.skinned {
                        (sl * vr_stereo_view, sr * vr_stereo_view, Mat4::IDENTITY)
                    } else {
                        let model = scene
                            .world_matrix_for_render_context(
                                item.space_id,
                                item.node_id as usize,
                                render_context,
                                hc.head_output_transform,
                            )
                            .unwrap_or(Mat4::IDENTITY);
                        (sl * vr_stereo_view, sr * vr_stereo_view, model)
                    }
                } else {
                    let proj = if item.is_overlay {
                        overlay_proj.unwrap_or(world_proj)
                    } else {
                        world_proj
                    };
                    let base_vp = proj * view;
                    if item.skinned {
                        (base_vp, base_vp, Mat4::IDENTITY)
                    } else {
                        let model = scene
                            .world_matrix_for_render_context(
                                item.space_id,
                                item.node_id as usize,
                                render_context,
                                hc.head_output_transform,
                            )
                            .unwrap_or(Mat4::IDENTITY);
                        (base_vp, base_vp, model)
                    }
                }
            } else {
                (Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY)
            };
            slots.push(if vp_l == vp_r {
                PaddedPerDrawUniforms::new_single(vp_l, model)
            } else {
                PaddedPerDrawUniforms::new_stereo(vp_l, vp_r, model)
            });
        }

        let mut slab_bytes = vec![0u8; draws.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE)];
        write_per_draw_uniform_slab(&slots, &mut slab_bytes);

        let queue_guard = ctx
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let queue = &*queue_guard;

        {
            let Some(dbg) = backend.debug_draw.as_mut() else {
                return Ok(());
            };
            queue.write_buffer(&dbg.per_draw_uniforms, 0, &slab_bytes);
        }
        let world_to_view = scene
            .active_main_space()
            .map(|s| view_matrix_from_render_transform(&s.view_transform))
            .unwrap_or(Mat4::IDENTITY);
        let z_coeffs = FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(world_to_view);
        let cluster_count_x = vw.div_ceil(TILE_SIZE);
        let cluster_count_y = vh.div_ceil(TILE_SIZE);
        let light_count_u = lights_for_frame.len().min(crate::backend::MAX_LIGHTS) as u32;
        let uniforms = FrameGpuUniforms::new_clustered(
            camera_world,
            z_coeffs,
            cluster_count_x,
            cluster_count_y,
            CLUSTER_COUNT_Z,
            near,
            far,
            light_count_u,
            vw.max(1),
            vh.max(1),
        );
        if let Some(fgpu) = backend.frame_gpu_mut() {
            fgpu.sync_cluster_viewport(ctx.device, (vw, vh));
            fgpu.write_frame_uniform_and_lights(queue, &uniforms, &lights_for_frame);
        }

        let Some(debug_bind_group) = backend.debug_draw.as_ref().map(|d| d.bind_group.clone())
        else {
            return Ok(());
        };

        let mut regular_indices = Vec::with_capacity(draws.len());
        let mut intersect_indices = Vec::new();
        for (draw_idx, item) in draws.iter().enumerate() {
            if is_pbs_intersection_draw(item) {
                intersect_indices.push(draw_idx);
            } else {
                regular_indices.push(draw_idx);
            }
        }

        let mut warned_missing_embedded_bind = false;
        let Some((frame_bg_arc, empty_bg_arc)) = backend.mesh_forward_frame_bind_groups() else {
            return Ok(());
        };

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
                    stencil_ops: None,
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
                debug_bind_group.as_ref(),
                &pass_desc,
                shader_perm,
                &mut warned_missing_embedded_bind,
            );
        }

        if !intersect_indices.is_empty() {
            if let Some(fgpu) = backend.frame_gpu_mut() {
                fgpu.copy_scene_depth_snapshot(
                    ctx.device,
                    ctx.encoder,
                    frame.depth_texture,
                    (vw, vh),
                    use_multiview,
                );
            }
            let Some((frame_bg_arc, empty_bg_arc)) = backend.mesh_forward_frame_bind_groups()
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
                    stencil_ops: None,
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
                debug_bind_group.as_ref(),
                &pass_desc,
                shader_perm,
                &mut warned_missing_embedded_bind,
            );
        }

        Ok(())
    }
}

fn is_pbs_intersection_draw(item: &WorldMeshDrawItem) -> bool {
    match &item.batch_key.pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            stem.starts_with("pbsintersectspecular")
                || stem.starts_with("custom_pbsintersectspecular")
        }
        RasterPipelineKind::DebugWorldNormals => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_subset(
    rpass: &mut wgpu::RenderPass<'_>,
    draw_indices: &[usize],
    draws: &[WorldMeshDrawItem],
    backend: &mut crate::backend::RenderBackend,
    queue: &wgpu::Queue,
    frame_bg: &wgpu::BindGroup,
    empty_bg: &wgpu::BindGroup,
    debug_bind_group: &wgpu::BindGroup,
    pass_desc: &MaterialPipelineDesc,
    shader_perm: ShaderPermutation,
    warned_missing_embedded_bind: &mut bool,
) {
    let mut last_batch_key: Option<MaterialDrawBatchKey> = None;
    let mut pipeline_ok = false;

    for draw_idx in draw_indices {
        let item = &draws[*draw_idx];
        if last_batch_key.as_ref() != Some(&item.batch_key) {
            last_batch_key = Some(item.batch_key.clone());
            let shader_asset_id = item.batch_key.shader_asset_id;
            pipeline_ok = match backend.material_registry.as_mut() {
                None => false,
                Some(reg) => {
                    match reg.pipeline_for_shader_asset(shader_asset_id, pass_desc, shader_perm) {
                        Some(pipeline) => {
                            rpass.set_pipeline(pipeline);
                            true
                        }
                        None => {
                            logger::trace!(
                            "WorldMeshForward: no pipeline for shader_asset_id {:?} pipeline {:?}, skipping draws until registered",
                            shader_asset_id,
                            item.batch_key.pipeline
                        );
                            false
                        }
                    }
                }
            };
        }

        if !pipeline_ok {
            continue;
        }

        let dynamic_offset = (*draw_idx * PER_DRAW_UNIFORM_STRIDE) as u32;
        rpass.set_bind_group(0, frame_bg, &[]);
        if matches!(
            &item.batch_key.pipeline,
            RasterPipelineKind::EmbeddedStem(_)
        ) {
            let stem = backend
                .material_registry
                .as_ref()
                .and_then(|r| r.stem_for_shader_asset(item.batch_key.shader_asset_id));
            if let (Some(mb), Some(stem)) = (backend.embedded_material_bind(), stem) {
                match mb.embedded_material_bind_group(
                    stem,
                    queue,
                    backend.material_property_store(),
                    backend.texture_pool(),
                    item.lookup_ids,
                ) {
                    Ok(bg) => rpass.set_bind_group(1, bg.as_ref(), &[]),
                    Err(_) => rpass.set_bind_group(1, empty_bg, &[]),
                }
            } else {
                if backend.embedded_material_bind().is_none() && !*warned_missing_embedded_bind {
                    logger::warn!(
                        "WorldMeshForward: embedded material bind resources unavailable; @group(1) uses empty bind group for embedded raster draws"
                    );
                    *warned_missing_embedded_bind = true;
                }
                rpass.set_bind_group(1, empty_bg, &[]);
            }
        } else {
            rpass.set_bind_group(1, empty_bg, &[]);
        }
        rpass.set_bind_group(2, debug_bind_group, &[dynamic_offset]);

        draw_mesh_submesh(
            rpass,
            item,
            &backend.mesh_pool,
            item.batch_key.embedded_needs_uv0,
            item.batch_key.embedded_needs_color,
        );
    }
}

fn draw_mesh_submesh(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    mesh_pool: &crate::resources::MeshPool,
    embedded_uv: bool,
    embedded_color: bool,
) {
    if item.mesh_asset_id < 0 || item.node_id < 0 || item.index_count == 0 {
        return;
    }
    let Some(mesh) = mesh_pool.get_mesh(item.mesh_asset_id) else {
        return;
    };
    if !mesh.debug_streams_ready() {
        return;
    }
    let Some(normals_bind) = mesh.normals_buffer.as_deref() else {
        return;
    };

    let use_deformed = item.skinned && mesh.has_skeleton;
    let use_blend_only = mesh.num_blendshapes > 0;

    let pos_buf = if use_deformed {
        mesh.deformed_positions_buffer.as_deref()
    } else if use_blend_only {
        mesh.deform_temp_buffer.as_deref()
    } else {
        mesh.positions_buffer.as_deref()
    };
    let Some(pos) = pos_buf else {
        return;
    };

    let normals_vb = if use_deformed {
        mesh.deformed_normals_buffer
            .as_deref()
            .unwrap_or(normals_bind)
    } else {
        normals_bind
    };

    rpass.set_vertex_buffer(0, pos.slice(..));
    rpass.set_vertex_buffer(1, normals_vb.slice(..));
    if embedded_uv || embedded_color {
        let Some(uv) = mesh.uv0_buffer.as_deref() else {
            return;
        };
        rpass.set_vertex_buffer(2, uv.slice(..));
    }
    if embedded_color {
        let Some(color) = mesh.color_buffer.as_deref() else {
            return;
        };
        rpass.set_vertex_buffer(3, color.slice(..));
    }
    rpass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);

    let first = item.first_index;
    let end = first.saturating_add(item.index_count);
    rpass.draw_indexed(first..end, 0, 0..1);
}
