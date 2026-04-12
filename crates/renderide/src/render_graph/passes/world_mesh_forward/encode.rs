//! Encode indexed draws and material bind groups for [`super::WorldMeshForwardPass`].

use crate::materials::{MaterialPipelineDesc, RasterPipelineKind};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::MaterialDrawBatchKey;
use crate::render_graph::WorldMeshDrawItem;
use crate::resources::MeshPool;

use crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE;

pub(crate) fn is_pbs_intersection_draw(item: &WorldMeshDrawItem) -> bool {
    match &item.batch_key.pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            stem.starts_with("pbsintersectspecular")
                || stem.starts_with("custom_pbsintersectspecular")
        }
        RasterPipelineKind::DebugWorldNormals => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_subset(
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
            pipeline_ok = match backend.materials.material_registry.as_mut() {
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
                .materials
                .material_registry
                .as_ref()
                .and_then(|r| r.stem_for_shader_asset(item.batch_key.shader_asset_id));
            if let (Some(mb), Some(stem)) = (backend.materials.embedded_material_bind(), stem) {
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
                if backend.materials.embedded_material_bind().is_none()
                    && !*warned_missing_embedded_bind
                {
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
            backend.mesh_pool(),
            item.batch_key.embedded_needs_uv0,
            item.batch_key.embedded_needs_color,
        );
    }
}

pub(crate) fn draw_mesh_submesh(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    mesh_pool: &MeshPool,
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
