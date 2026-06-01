//! PhotonDust render-buffer expansion.

use glam::Mat4;
use hashbrown::HashMap;

use crate::assets::mesh::GpuMesh;
use crate::gpu_pools::MeshPool;
use crate::particles::{ParticleDrawParams, PointRenderBufferAsset};
use crate::scene::{MeshRendererInstanceId, RenderSpaceId, SceneCoordinator};
use crate::shared::{LayerType, RenderingContext};
use crate::world_mesh::culling::{
    MeshCullGeometry, MeshCullTarget, mesh_world_geometry_for_cull_with_head,
};

use super::super::FramePreparedDraw;
use super::context::ExpandCtx;
use super::mesh_particles::try_expand_mesh_render_buffer_renderer;

/// Expands PhotonDust billboard and trail render-buffer renderers into prepared draw entries.
pub(in crate::world_mesh::draw_prep) fn expand_render_buffer_renderers_into(
    out: &mut Vec<FramePreparedDraw>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    point_render_buffers: &HashMap<i32, PointRenderBufferAsset>,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    profiling::scope!("mesh::prepared_renderables::expand_render_buffers");
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }
    let mut ctx = ExpandCtx {
        out,
        scene,
        mesh_pool,
        render_context,
        space_id,
        space_is_overlay: space.is_overlay(),
    };
    for (renderable_index, renderer) in space.billboard_render_buffers().iter().enumerate() {
        let Some(mesh_asset_id) = crate::particles::billboard_render_buffer_mesh_asset_id(
            renderer.point_render_buffer_asset_id,
        ) else {
            continue;
        };
        try_expand_render_buffer_renderer(
            &mut ctx,
            renderable_index,
            renderer.node_id,
            mesh_asset_id,
            renderer.material_asset_id,
            ParticleRenderBufferPreparedKind::Billboard,
            ParticleDrawParams::billboard(
                renderer.alignment,
                renderer.min_billboard_screen_size,
                renderer.max_billboard_screen_size,
                renderer.motion_vector_mode,
            ),
        );
    }
    for (renderable_index, renderer) in space.trail_render_buffers().iter().enumerate() {
        let Some(mesh_asset_id) = crate::particles::trail_render_buffer_mesh_asset_id(
            renderer.trails_render_buffer_asset_id,
            renderer.texture_mode,
        ) else {
            continue;
        };
        try_expand_render_buffer_renderer(
            &mut ctx,
            renderable_index,
            renderer.node_id,
            mesh_asset_id,
            renderer.material_asset_id,
            ParticleRenderBufferPreparedKind::Trail,
            ParticleDrawParams::trail(
                renderer.texture_mode,
                renderer.motion_vector_mode,
                renderer.generate_lighting_data,
            ),
        );
    }
    for (renderable_index, renderer) in space.mesh_render_buffers().iter().enumerate() {
        try_expand_mesh_render_buffer_renderer(
            &mut ctx,
            point_render_buffers,
            renderable_index,
            renderer,
        );
    }
}

/// Render-buffer family used to derive stable prepared renderer identities.
#[derive(Clone, Copy)]
pub(super) enum ParticleRenderBufferPreparedKind {
    /// Billboard point-buffer renderer.
    Billboard,
    /// Trail ribbon renderer.
    Trail,
    /// Mesh-particle renderer.
    Mesh,
}

impl ParticleRenderBufferPreparedKind {
    /// Stable high-bit tag for generated renderer-local ids.
    pub(super) fn tag(self) -> u64 {
        match self {
            Self::Billboard => 1,
            Self::Trail => 2,
            Self::Mesh => 3,
        }
    }
}

/// Expands one PhotonDust render-buffer renderer row into a single material draw.
fn try_expand_render_buffer_renderer(
    ctx: &mut ExpandCtx<'_>,
    renderable_index: usize,
    node_id: i32,
    mesh_asset_id: i32,
    material_asset_id: i32,
    kind: ParticleRenderBufferPreparedKind,
    particle_draw: ParticleDrawParams,
) {
    if node_id < 0 || material_asset_id < 0 {
        return;
    }
    if matches!(
        ctx.scene
            .transform_special_layer(ctx.space_id, node_id as usize),
        Some(LayerType::Hidden)
    ) {
        return;
    }
    let Some(mesh) = ctx.mesh_pool.get(mesh_asset_id) else {
        return;
    };
    let Some((first_index, index_count)) = mesh.submeshes.first().copied() else {
        return;
    };
    if index_count == 0 {
        return;
    }
    let cull_geometry = precompute_particle_cull_geometry(ctx, mesh, node_id);
    ctx.out.push(FramePreparedDraw {
        space_id: ctx.space_id,
        renderable_index,
        instance_id: particle_renderer_instance_id(kind, renderable_index),
        renderer_ordinal: 0,
        node_id,
        mesh_asset_id,
        is_overlay: ctx
            .scene
            .transform_is_in_overlay_layer(ctx.space_id, node_id as usize),
        sorting_order: 0,
        skinned: false,
        world_space_deformed: false,
        blendshape_deformed: false,
        tangent_blendshape_deform_active: false,
        slot_index: 0,
        first_index,
        index_count,
        material_asset_id,
        property_block_id: None,
        cull_geometry,
        rigid_world_matrix_override: None,
        particle_draw,
    });
}

/// Computes frame-invariant cull geometry for non-overlay render-buffer meshes.
fn precompute_particle_cull_geometry(
    ctx: &ExpandCtx<'_>,
    mesh: &GpuMesh,
    node_id: i32,
) -> Option<MeshCullGeometry> {
    if ctx.space_is_overlay {
        return None;
    }
    let target = MeshCullTarget {
        scene: ctx.scene,
        space_id: ctx.space_id,
        mesh,
        skinned: false,
        skinned_renderer: None,
        node_id,
    };
    Some(mesh_world_geometry_for_cull_with_head(
        &target,
        Mat4::IDENTITY,
        ctx.render_context,
    ))
}

/// Returns a renderer-local identity that cannot collide with host static/skinned ids.
fn particle_renderer_instance_id(
    kind: ParticleRenderBufferPreparedKind,
    renderable_index: usize,
) -> MeshRendererInstanceId {
    MeshRendererInstanceId(0x8000_0000_0000_0000 | (kind.tag() << 48) | renderable_index as u64)
}
