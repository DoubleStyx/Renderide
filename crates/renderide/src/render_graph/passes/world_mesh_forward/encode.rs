//! Encode indexed draws and material bind groups for [`super::WorldMeshForwardPass`].

use crate::backend::mesh_deform::GpuSkinCache;
use crate::backend::MaterialBindCacheKey;
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::materials::{MaterialPipelineDesc, RasterPipelineKind};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::world_mesh_draw_prep::build_instance_batches;
use crate::render_graph::MaterialDrawBatchKey;
use crate::render_graph::WorldMeshDrawItem;
use crate::resources::MeshPool;

/// Last `@group(1)` bind state for skipping redundant [`wgpu::RenderPass::set_bind_group`] when unchanged.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LastMaterialBindGroup1Key {
    Embedded(MaterialBindCacheKey),
    Empty,
}

/// Resolves and binds the raster pipeline for `item.batch_key`, or marks the batch as skippable.
fn set_world_mesh_pipeline_for_item(
    rpass: &mut wgpu::RenderPass<'_>,
    encode: &mut WorldMeshForwardEncodeRefs<'_>,
    item: &WorldMeshDrawItem,
    pass_desc: &MaterialPipelineDesc,
    shader_perm: ShaderPermutation,
    last_batch_key: &mut Option<MaterialDrawBatchKey>,
    pipeline_ok: &mut bool,
) {
    if last_batch_key.as_ref() == Some(&item.batch_key) {
        return;
    }
    *last_batch_key = Some(item.batch_key.clone());
    let shader_asset_id = item.batch_key.shader_asset_id;
    *pipeline_ok = match encode.materials.material_registry.as_mut() {
        None => false,
        Some(reg) => match reg.pipeline_for_shader_asset(shader_asset_id, pass_desc, shader_perm) {
            Some(pipeline) => {
                rpass.set_pipeline(pipeline.as_ref());
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
        },
    };
}

/// State for resolving and binding embedded `@group(1)` material data for one draw batch.
struct MaterialBindState<'a, 'b, 'c, 'd> {
    rpass: &'a mut wgpu::RenderPass<'b>,
    encode: &'a mut WorldMeshForwardEncodeRefs<'c>,
    queue: &'a wgpu::Queue,
    item: &'d WorldMeshDrawItem,
    empty_bg: &'a wgpu::BindGroup,
    last_material_bind_key: &'a mut Option<LastMaterialBindGroup1Key>,
    warned_missing_embedded_bind: &'a mut bool,
    offscreen_write_render_texture_asset_id: Option<i32>,
}

/// Binds `@group(1)` for embedded stems (texture/uniform pack) or the empty fallback.
fn set_world_mesh_material_bind_group(ctx: MaterialBindState<'_, '_, '_, '_>) {
    if matches!(
        &ctx.item.batch_key.pipeline,
        RasterPipelineKind::EmbeddedStem(_)
    ) {
        let stem = ctx
            .encode
            .materials
            .material_registry
            .as_ref()
            .and_then(|r| r.stem_for_shader_asset(ctx.item.batch_key.shader_asset_id));
        if let (Some(mb), Some(stem)) = (ctx.encode.materials.embedded_material_bind(), stem) {
            let pools = ctx.encode.embedded_texture_pools();
            match mb.embedded_material_bind_group_with_cache_key(
                stem,
                ctx.queue,
                ctx.encode.materials.material_property_store(),
                &pools,
                ctx.item.lookup_ids,
                ctx.offscreen_write_render_texture_asset_id,
            ) {
                Ok((cache_key, bg)) => {
                    if *ctx.last_material_bind_key
                        != Some(LastMaterialBindGroup1Key::Embedded(cache_key))
                    {
                        ctx.rpass.set_bind_group(1, bg.as_ref(), &[]);
                    }
                    *ctx.last_material_bind_key =
                        Some(LastMaterialBindGroup1Key::Embedded(cache_key));
                }
                Err(_) => {
                    if *ctx.last_material_bind_key != Some(LastMaterialBindGroup1Key::Empty) {
                        ctx.rpass.set_bind_group(1, ctx.empty_bg, &[]);
                    }
                    *ctx.last_material_bind_key = Some(LastMaterialBindGroup1Key::Empty);
                }
            }
        } else {
            if ctx.encode.materials.embedded_material_bind().is_none()
                && !*ctx.warned_missing_embedded_bind
            {
                logger::warn!(
                    "WorldMeshForward: embedded material bind resources unavailable; @group(1) uses empty bind group for embedded raster draws"
                );
                *ctx.warned_missing_embedded_bind = true;
            }
            if *ctx.last_material_bind_key != Some(LastMaterialBindGroup1Key::Empty) {
                ctx.rpass.set_bind_group(1, ctx.empty_bg, &[]);
            }
            *ctx.last_material_bind_key = Some(LastMaterialBindGroup1Key::Empty);
        }
    } else {
        if *ctx.last_material_bind_key != Some(LastMaterialBindGroup1Key::Empty) {
            ctx.rpass.set_bind_group(1, ctx.empty_bg, &[]);
        }
        *ctx.last_material_bind_key = Some(LastMaterialBindGroup1Key::Empty);
    }
}

/// Draw indices, bind groups, and pipeline state for one mesh-forward raster subpass.
pub(crate) struct ForwardDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Indices into `draws` for this subpass.
    pub draw_indices: &'c [usize],
    /// Sorted world mesh draws for the view.
    pub draws: &'c [WorldMeshDrawItem],
    /// Material registry, pools, and skin cache (disjoint borrows from [`crate::backend::RenderBackend`]).
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Queue for embedded material bind uploads.
    pub queue: &'a wgpu::Queue,
    /// Frame globals at `@group(0)`.
    pub frame_bg: &'a wgpu::BindGroup,
    /// Fallback material bind group when a stem has no resources.
    pub empty_bg: &'a wgpu::BindGroup,
    /// Per-draw storage slab at `@group(2)`.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Surface / depth / MSAA pipeline description.
    pub pass_desc: &'a MaterialPipelineDesc,
    /// Default vs multiview shader permutation.
    pub shader_perm: ShaderPermutation,
    /// Set true after logging missing embedded bind resources once.
    pub warned_missing_embedded_bind: &'a mut bool,
    /// Offscreen render-texture write target for embedded lookups.
    pub offscreen_write_render_texture_asset_id: Option<i32>,
    /// Whether `draw_indexed` may use non-zero `first_instance` / base instance.
    pub supports_base_instance: bool,
}

pub(crate) fn draw_subset(batch: ForwardDrawBatch<'_, '_, '_, '_>) {
    let mut last_batch_key: Option<MaterialDrawBatchKey> = None;
    let mut last_material_bind_key: Option<LastMaterialBindGroup1Key> = None;
    let mut pipeline_ok = false;

    batch.rpass.set_bind_group(0, batch.frame_bg, &[]);

    let batches = build_instance_batches(
        batch.draws,
        batch.draw_indices,
        batch.supports_base_instance,
    );

    for inst_batch in batches {
        let first_idx = inst_batch.first_draw_index;
        let item = &batch.draws[first_idx];

        set_world_mesh_pipeline_for_item(
            batch.rpass,
            batch.encode,
            item,
            batch.pass_desc,
            batch.shader_perm,
            &mut last_batch_key,
            &mut pipeline_ok,
        );

        if !pipeline_ok {
            continue;
        }

        set_world_mesh_material_bind_group(MaterialBindState {
            rpass: batch.rpass,
            encode: batch.encode,
            queue: batch.queue,
            item,
            empty_bg: batch.empty_bg,
            last_material_bind_key: &mut last_material_bind_key,
            warned_missing_embedded_bind: batch.warned_missing_embedded_bind,
            offscreen_write_render_texture_asset_id: batch.offscreen_write_render_texture_asset_id,
        });

        // Full-buffer bind group: no dynamic offset. Slot selection is `instance_index` from
        // `draw_indexed(..., first_idx..first_idx + count)` (single- and multi-instance batches).
        batch
            .rpass
            .set_bind_group(2, batch.per_draw_bind_group, &[]);

        let inst_start = first_idx as u32;
        let inst_range = inst_start..inst_start + inst_batch.instance_count;

        draw_mesh_submesh_instanced(
            batch.rpass,
            item,
            batch.encode.mesh_pool(),
            batch.encode.skin_cache,
            item.batch_key.embedded_needs_uv0,
            item.batch_key.embedded_needs_color,
            inst_range,
        );
    }
}

pub(crate) fn draw_mesh_submesh_instanced(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    mesh_pool: &MeshPool,
    skin_cache: Option<&GpuSkinCache>,
    embedded_uv: bool,
    embedded_color: bool,
    instances: std::ops::Range<u32>,
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
    let needs_cache_stream = use_deformed || use_blend_only;

    if needs_cache_stream {
        let Some(cache) = skin_cache else {
            return;
        };
        let key = (item.space_id, item.node_id);
        let Some(entry) = cache.lookup(&key) else {
            logger::trace!(
                "world mesh forward: skin cache miss for space {:?} node {}",
                item.space_id,
                item.node_id
            );
            return;
        };
        let pos_buf = cache.positions_arena();
        rpass.set_vertex_buffer(0, pos_buf.slice(entry.positions.byte_range()));
        if use_deformed {
            let Some(nrm_r) = entry.normals.as_ref() else {
                return;
            };
            let nrm_buf = cache.normals_arena();
            rpass.set_vertex_buffer(1, nrm_buf.slice(nrm_r.byte_range()));
        } else {
            rpass.set_vertex_buffer(1, normals_bind.slice(..));
        }
    } else {
        let Some(pos) = mesh.positions_buffer.as_deref() else {
            return;
        };
        rpass.set_vertex_buffer(0, pos.slice(..));
        rpass.set_vertex_buffer(1, normals_bind.slice(..));
    }
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
    rpass.draw_indexed(first..end, 0, instances);
}
