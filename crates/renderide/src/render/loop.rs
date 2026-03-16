//! Render loop: executes one frame of mesh rendering.

use nalgebra::{Matrix4, Vector3};

use crate::gpu::{GpuMeshBuffers, GpuState, RenderPipeline, UniformData};
use crate::scene::render_transform_to_matrix;
use crate::session::Session;

/// Encapsulates the render frame logic.
pub struct RenderLoop;

impl RenderLoop {
    /// Renders one frame: clear, draw batches, present.
    pub fn render(
        gpu: &mut GpuState,
        session: &mut Session,
    ) -> Result<(), wgpu::SurfaceError> {
        let draw_batches = session.collect_draw_batches();

        let output = gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mesh_assets = session.asset_registry();

        let depth_view = gpu
            .depth_texture
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        let mut view_transform = session
            .primary_view_transform()
            .cloned()
            .unwrap_or_default();
        view_transform.scale = filter_scale(view_transform.scale);

        let aspect = gpu.config.width as f32 / gpu.config.height.max(1) as f32;
        let proj = reverse_z_projection(
            aspect,
            session.desktop_fov().to_radians(),
            session.near_clip().max(0.01),
            session.far_clip(),
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                depth_stencil_attachment: depth_view.as_ref().map(|dv| wgpu::RenderPassDepthStencilAttachment {
                    view: dv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let use_debug_uv = false;

            for batch in &draw_batches {
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
                    if !gpu.mesh_buffer_cache.contains_key(mesh_asset_id) {
                        let stride = crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                        let stride = if stride > 0 {
                            stride
                        } else {
                            crate::gpu::compute_vertex_stride_from_mesh(mesh)
                        };
                        if let Some(b) = crate::gpu::create_mesh_buffers(&gpu.device, mesh, stride) {
                            gpu.mesh_buffer_cache.insert(*mesh_asset_id, b);
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
            let scene_graph = session.scene_graph();

            for batch in &draw_batches {
                let mut batch_vt = batch.view_transform;
                batch_vt.scale = filter_scale(batch_vt.scale);
                let view_mat = render_transform_to_matrix(&batch_vt)
                    .try_inverse()
                    .unwrap_or_else(Matrix4::identity);
                let view_mat = apply_view_handedness_fix(view_mat);
                let view_proj = proj * view_mat;

                for (model, mesh_asset_id, is_skinned, material_id, bone_transform_ids) in &batch.draws {
                    let (buffers_ref, mesh) = if *mesh_asset_id >= 0 {
                        let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) else {
                            continue;
                        };
                        if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                            continue;
                        }
                        let Some(b) = gpu.mesh_buffer_cache.get(mesh_asset_id) else {
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
                        let Some(ids) = bone_transform_ids.as_ref() else {
                            continue;
                        };
                        let Some(_) = buffers_ref.vertex_buffer_skinned.as_ref() else {
                            continue;
                        };
                        let bone_matrices =
                            scene_graph.compute_bone_matrices(batch.space_id, ids, bind_poses);
                        gpu.pipeline_manager.skinned.upload_skinned(
                            &gpu.queue,
                            skinned_mvp,
                            &bone_matrices,
                        );
                        gpu.pipeline_manager.skinned.bind(&mut pass, None);
                        gpu.pipeline_manager.skinned.draw_skinned(
                            &mut pass,
                            buffers_ref,
                            &UniformData::Skinned {
                                mvp: skinned_mvp,
                                bone_matrices: &bone_matrices,
                            },
                        );
                        continue;
                    }

                    let use_uv_pipeline = use_debug_uv && buffers_ref.has_uvs;
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

            gpu.pipeline_manager.normal_debug.upload_batch(&gpu.queue, &mvp_models_normal);
            gpu.pipeline_manager.uv_debug.upload_batch(&gpu.queue, &mvp_models_uv);

            for (i, d) in normal_draws.iter().enumerate() {
                gpu.pipeline_manager.normal_debug.bind(&mut pass, Some(i as u32));
                gpu.pipeline_manager.normal_debug.draw_mesh(
                    &mut pass,
                    d.buffers,
                    &UniformData::Simple { mvp: d.mvp, model: d.model },
                );
            }
            for (i, d) in uv_draws.iter().enumerate() {
                gpu.pipeline_manager.uv_debug.bind(&mut pass, Some(i as u32));
                gpu.pipeline_manager.uv_debug.draw_mesh(
                    &mut pass,
                    d.buffers,
                    &UniformData::Simple { mvp: d.mvp, model: d.model },
                );
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn clamp_near_far(near: f32, far: f32) -> (f32, f32) {
    let near = near.max(0.001);
    let far = if far > near { far } else { near + 1.0 };
    (near, far)
}

fn reverse_z_projection(aspect: f32, vertical_fov: f32, near: f32, far: f32) -> Matrix4<f32> {
    let vertical_half = vertical_fov / 2.0;
    let tan_vertical_half = vertical_half.tan();
    let horizontal_fov = (tan_vertical_half * aspect)
        .atan()
        .clamp(0.1, std::f32::consts::FRAC_PI_2 - 0.1)
        * 2.0;
    let tan_horizontal_half = (horizontal_fov / 2.0).tan();
    let f_x = 1.0 / tan_horizontal_half;
    let f_y = 1.0 / tan_vertical_half;
    let proj = Matrix4::new(
        f_x, 0.0, 0.0, 0.0,
        0.0, f_y, 0.0, 0.0,
        0.0, 0.0, near / (far - near), (far * near) / (far - near),
        0.0, 0.0, -1.0, 0.0,
    );
    proj
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
