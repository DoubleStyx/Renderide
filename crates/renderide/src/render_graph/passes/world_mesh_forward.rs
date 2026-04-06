//! Main forward pass: clear color + depth, debug normal shading for scene meshes.
//!
//! Draws are collected and **sorted by [`MaterialDrawBatchKey`](crate::render_graph::MaterialDrawBatchKey)**
//! so pipeline and bind-group changes happen only when material / property-block / skinned path changes.
//! Per-slot [`MaterialPropertyLookupIds`](crate::assets::material::MaterialPropertyLookupIds) are carried on each
//! [`WorldMeshDrawItem`](crate::render_graph::WorldMeshDrawItem) for upcoming per-material bind groups (`get_merged`).

use glam::Mat4;

use crate::assets::material::MaterialDictionary;
use crate::gpu::{write_per_draw_uniform_slab, PaddedPerDrawUniforms, PER_DRAW_UNIFORM_STRIDE};
use crate::materials::MaterialPipelineDesc;
use crate::pipelines::ShaderPermutation;
use crate::present::SWAPCHAIN_CLEAR_COLOR;
use crate::render_graph::camera::{
    clamp_desktop_fov_degrees, reverse_z_orthographic, reverse_z_perspective,
    view_matrix_from_render_transform,
};
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};
use crate::render_graph::MAIN_FORWARD_DEPTH_CLEAR;
use crate::render_graph::{
    collect_and_sort_world_mesh_draws, MaterialDrawBatchKey, WorldMeshDrawItem,
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
            reads: Vec::new(),
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

        let desc = MaterialPipelineDesc {
            surface_format: frame.surface_format,
            depth_stencil_format: Some(wgpu::TextureFormat::Depth32Float),
            sample_count: 1,
        };
        let backend = &mut frame.backend;
        let store_ref = backend.material_property_store();
        let dict = MaterialDictionary::new(store_ref);
        let mesh_pool = &backend.mesh_pool;
        let draws = collect_and_sort_world_mesh_draws(frame.scene, mesh_pool, &dict);
        if draws.is_empty() {
            return Ok(());
        }
        let Some(reg) = backend.material_registry.as_mut() else {
            return Ok(());
        };
        let Some(dbg) = backend.debug_draw.as_mut() else {
            return Ok(());
        };

        let (vw, vh) = frame.viewport_px;
        let aspect = vw as f32 / vh.max(1) as f32;
        let hc = frame.host_camera;
        let near = hc.near_clip.max(0.01);
        let far = hc.far_clip;
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

        dbg.ensure_draw_slot_capacity(ctx.device, draws.len());

        let scene = frame.scene;
        let mut slots: Vec<PaddedPerDrawUniforms> = Vec::with_capacity(draws.len());
        for item in &draws {
            let proj = if item.is_overlay {
                overlay_proj.unwrap_or(world_proj)
            } else {
                world_proj
            };
            let (vp, model) = if let Some(space) = scene.space(item.space_id) {
                let view = view_matrix_from_render_transform(&space.view_transform);
                let vp = proj * view;
                let node_u = item.node_id as usize;
                let model = scene
                    .world_matrix(item.space_id, node_u)
                    .unwrap_or(Mat4::IDENTITY);
                (vp, model)
            } else {
                (Mat4::IDENTITY, Mat4::IDENTITY)
            };
            slots.push(PaddedPerDrawUniforms::new(vp, model));
        }

        let mut slab_bytes = vec![0u8; draws.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE)];
        write_per_draw_uniform_slab(&slots, &mut slab_bytes);

        let queue = match ctx.queue.lock() {
            Ok(q) => q,
            Err(poisoned) => poisoned.into_inner(),
        };
        queue.write_buffer(&dbg.per_draw_uniforms, 0, &slab_bytes);

        let debug_bind_group = &dbg.bind_group;

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world-mesh-forward"),
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
            multiview_mask: None,
        });

        let mut last_batch_key: Option<MaterialDrawBatchKey> = None;
        let mut pipeline_ok = false;

        for (draw_idx, item) in draws.iter().enumerate() {
            if last_batch_key.as_ref() != Some(&item.batch_key) {
                last_batch_key = Some(item.batch_key);
                match reg.pipeline_for_family(item.batch_key.family_id, &desc, ShaderPermutation(0))
                {
                    Some(pipeline) => {
                        rpass.set_pipeline(pipeline);
                        pipeline_ok = true;
                    }
                    None => {
                        logger::trace!(
                            "WorldMeshForward: no pipeline for family {:?}, skipping draws until registered",
                            item.batch_key.family_id
                        );
                        pipeline_ok = false;
                    }
                }
            }

            if !pipeline_ok {
                continue;
            }

            // `lookup_ids` feeds future per-material bind groups (`MaterialDictionary::get_merged`).
            let _ = &item.lookup_ids;

            let dynamic_offset = (draw_idx * PER_DRAW_UNIFORM_STRIDE) as u32;
            rpass.set_bind_group(0, debug_bind_group, &[dynamic_offset]);

            draw_mesh_submesh(&mut rpass, item, mesh_pool);
        }

        Ok(())
    }
}

fn draw_mesh_submesh(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    mesh_pool: &crate::resources::MeshPool,
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
    let Some(normals) = mesh.normals_buffer.as_deref() else {
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

    rpass.set_vertex_buffer(0, pos.slice(..));
    rpass.set_vertex_buffer(1, normals.slice(..));
    rpass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);

    let first = item.first_index;
    let end = first.saturating_add(item.index_count);
    rpass.draw_indexed(first..end, 0, 0..1);
}
