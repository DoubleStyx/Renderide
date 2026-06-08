//! Mesh-particle render-buffer expansion.

use glam::Mat4;
use hashbrown::HashMap;
use rayon::prelude::*;

use crate::particles::{ParticleDrawParams, PointParticle, PointRenderBufferAsset};
use crate::scene::{MeshRenderBufferEntry, MeshRendererInstanceId, RenderSpaceId};
use crate::shared::{LayerType, ShadowCastMode};

use super::super::FramePreparedDraw;
use super::context::ExpandCtx;
use super::render_buffers::ParticleRenderBufferPreparedKind;

/// Generated mesh-particle draws targeted per worker chunk.
const MESH_PARTICLE_EXPAND_PARALLEL_CHUNK_DRAWS: usize = 256;
/// Generated mesh-particle draw count required before expansion fans out.
const MESH_PARTICLE_EXPAND_PARALLEL_MIN_DRAWS: usize =
    MESH_PARTICLE_EXPAND_PARALLEL_CHUNK_DRAWS * 2;

/// Expands one mesh-particle renderer into one source-mesh draw per particle and submesh.
pub(super) fn try_expand_mesh_render_buffer_renderer(
    ctx: &mut ExpandCtx<'_>,
    point_render_buffers: &HashMap<i32, PointRenderBufferAsset>,
    renderable_index: usize,
    renderer: &MeshRenderBufferEntry,
) {
    if renderer.node_id < 0 || renderer.material_asset_id < 0 || renderer.mesh_asset_id < 0 {
        return;
    }
    let special_layer = ctx
        .scene
        .transform_special_layer(ctx.space_id, renderer.node_id as usize);
    let is_overlay = matches!(special_layer, Some(LayerType::Overlay));
    let is_hidden = matches!(special_layer, Some(LayerType::Hidden));
    let Some(point_buffer) = point_render_buffers.get(&renderer.point_render_buffer_asset_id)
    else {
        return;
    };
    let Some(mesh) = ctx.mesh_pool.get(renderer.mesh_asset_id) else {
        return;
    };
    if mesh.submeshes.is_empty() || point_buffer.points.is_empty() {
        return;
    }
    let active_submesh_count = mesh
        .submeshes
        .iter()
        .filter(|&&(_, index_count)| index_count > 0)
        .count();
    if active_submesh_count == 0 {
        return;
    }
    let Some(root_matrix) = ctx.scene.world_matrix_for_context(
        ctx.space_id,
        renderer.node_id as usize,
        ctx.render_context,
    ) else {
        return;
    };
    let generated_draw_count = point_buffer
        .points
        .len()
        .saturating_mul(active_submesh_count);
    if mesh_particle_parallel_is_worthwhile(generated_draw_count) {
        profiling::scope!("mesh::prepared_renderables::expand_mesh_particles_parallel");
        let active_submeshes = mesh
            .submeshes
            .iter()
            .enumerate()
            .filter_map(|(slot_index, &(first_index, index_count))| {
                (index_count > 0).then_some((slot_index, first_index, index_count))
            })
            .collect::<Vec<_>>();
        let point_chunk_size = MESH_PARTICLE_EXPAND_PARALLEL_CHUNK_DRAWS
            .saturating_div(active_submeshes.len())
            .max(1);
        let chunks = point_buffer
            .points
            .par_chunks(point_chunk_size)
            .with_min_len(1)
            .enumerate()
            .map(|(chunk_index, points)| {
                let base_point_index = chunk_index * point_chunk_size;
                build_mesh_particle_draw_chunk(MeshParticleDrawChunkInput {
                    points,
                    base_point_index,
                    active_submeshes: &active_submeshes,
                    root_matrix,
                    space_id: ctx.space_id,
                    renderable_index,
                    renderer,
                    is_overlay,
                    is_hidden,
                })
            })
            .collect::<Vec<_>>();
        for mut chunk in chunks {
            ctx.out.append(&mut chunk);
        }
        return;
    }

    append_mesh_particle_draws_serial(
        ctx.out,
        MeshParticleSerialDrawInput {
            points: &point_buffer.points,
            submeshes: &mesh.submeshes,
            root_matrix,
            space_id: ctx.space_id,
            renderable_index,
            renderer,
            is_overlay,
            is_hidden,
        },
    );
}

/// Inputs for serial mesh-particle expansion.
struct MeshParticleSerialDrawInput<'a> {
    /// Point particle rows to expand.
    points: &'a [PointParticle],
    /// Source mesh submesh ranges.
    submeshes: &'a [(u32, u32)],
    /// Scene node world matrix for the mesh-particle renderer.
    root_matrix: Mat4,
    /// Render space containing the renderer.
    space_id: RenderSpaceId,
    /// Renderer index within the mesh render-buffer table.
    renderable_index: usize,
    /// Source mesh-particle renderer row.
    renderer: &'a MeshRenderBufferEntry,
    /// Precomputed overlay-layer flag.
    is_overlay: bool,
    /// Precomputed hidden-layer flag.
    is_hidden: bool,
}

/// Inputs for expanding one chunk of mesh-particle points.
struct MeshParticleDrawChunkInput<'a> {
    /// Point particle rows assigned to this chunk.
    points: &'a [PointParticle],
    /// Point index of `points[0]` within the full render buffer.
    base_point_index: usize,
    /// Active source mesh submeshes emitted for every point.
    active_submeshes: &'a [(usize, u32, u32)],
    /// Scene node world matrix for the mesh-particle renderer.
    root_matrix: Mat4,
    /// Render space containing the renderer.
    space_id: RenderSpaceId,
    /// Renderer index within the mesh render-buffer table.
    renderable_index: usize,
    /// Source mesh-particle renderer row.
    renderer: &'a MeshRenderBufferEntry,
    /// Precomputed overlay-layer flag.
    is_overlay: bool,
    /// Precomputed hidden-layer flag.
    is_hidden: bool,
}

/// Returns whether mesh-particle draw expansion has at least two useful chunks.
fn mesh_particle_parallel_is_worthwhile(generated_draw_count: usize) -> bool {
    generated_draw_count >= MESH_PARTICLE_EXPAND_PARALLEL_MIN_DRAWS
        && rayon::current_num_threads() > 1
}

/// Appends prepared draws for mesh-particle points on the serial path.
fn append_mesh_particle_draws_serial(
    out: &mut Vec<FramePreparedDraw>,
    input: MeshParticleSerialDrawInput<'_>,
) {
    for (point_index, point) in input.points.iter().enumerate() {
        let model = input.root_matrix * point_transform_matrix(*point);
        for (slot_index, &(first_index, index_count)) in input.submeshes.iter().enumerate() {
            if index_count == 0 {
                continue;
            }
            out.push(FramePreparedDraw {
                space_id: input.space_id,
                renderable_index: input.renderable_index,
                instance_id: mesh_particle_renderer_instance_id(
                    input.renderable_index,
                    point_index,
                ),
                renderer_ordinal: 0,
                node_id: input.renderer.node_id,
                mesh_asset_id: input.renderer.mesh_asset_id,
                is_overlay: input.is_overlay,
                is_hidden: input.is_hidden,
                sorting_order: 0,
                shadow_cast_mode: ShadowCastMode::Off,
                skinned: false,
                world_space_deformed: false,
                blendshape_deformed: false,
                tangent_blendshape_deform_active: false,
                slot_index,
                material_stack_order: None,
                first_index,
                index_count,
                material_asset_id: input.renderer.material_asset_id,
                property_block_id: None,
                cull_geometry: None,
                rigid_world_matrix_override: Some(model),
                particle_draw: ParticleDrawParams::mesh(
                    input.renderer.alignment,
                    point.color,
                    point.frame_index,
                ),
            });
        }
    }
}

/// Builds prepared draws for one contiguous chunk of mesh-particle points.
fn build_mesh_particle_draw_chunk(input: MeshParticleDrawChunkInput<'_>) -> Vec<FramePreparedDraw> {
    let mut draws = Vec::with_capacity(
        input
            .points
            .len()
            .saturating_mul(input.active_submeshes.len()),
    );
    for (local_point_index, point) in input.points.iter().enumerate() {
        let point_index = input.base_point_index + local_point_index;
        let model = input.root_matrix * point_transform_matrix(*point);
        for &(slot_index, first_index, index_count) in input.active_submeshes {
            draws.push(FramePreparedDraw {
                space_id: input.space_id,
                renderable_index: input.renderable_index,
                instance_id: mesh_particle_renderer_instance_id(
                    input.renderable_index,
                    point_index,
                ),
                renderer_ordinal: 0,
                node_id: input.renderer.node_id,
                mesh_asset_id: input.renderer.mesh_asset_id,
                is_overlay: input.is_overlay,
                is_hidden: input.is_hidden,
                sorting_order: 0,
                shadow_cast_mode: ShadowCastMode::Off,
                skinned: false,
                world_space_deformed: false,
                blendshape_deformed: false,
                tangent_blendshape_deform_active: false,
                slot_index,
                material_stack_order: None,
                first_index,
                index_count,
                material_asset_id: input.renderer.material_asset_id,
                property_block_id: None,
                cull_geometry: None,
                rigid_world_matrix_override: Some(model),
                particle_draw: ParticleDrawParams::mesh(
                    input.renderer.alignment,
                    point.color,
                    point.frame_index,
                ),
            });
        }
    }
    draws
}

/// Builds a local mesh-particle transform from PhotonDust point data.
fn point_transform_matrix(point: PointParticle) -> Mat4 {
    Mat4::from_scale_rotation_translation(point.size, point.rotation, point.position)
}

/// Returns a stable renderer-local identity for one mesh-particle instance.
fn mesh_particle_renderer_instance_id(
    renderable_index: usize,
    point_index: usize,
) -> MeshRendererInstanceId {
    MeshRendererInstanceId(
        0x8000_0000_0000_0000
            | (ParticleRenderBufferPreparedKind::Mesh.tag() << 48)
            | ((renderable_index as u64) << 24)
            | point_index as u64,
    )
}
