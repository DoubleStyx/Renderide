//! Encode indexed draws and material bind groups for [`super::WorldMeshForwardPass`].

use crate::backend::mesh_deform::GpuSkinCache;
use crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::backend::MaterialBindCacheKey;
use crate::materials::{MaterialPipelineDesc, MaterialPipelineSet, RasterPipelineKind};
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

/// State for resolving and binding embedded `@group(1)` material data for one draw batch.
struct MaterialBindState<'a, 'b, 'c> {
    rpass: &'a mut wgpu::RenderPass<'b>,
    backend: &'a mut crate::backend::RenderBackend,
    queue: &'a wgpu::Queue,
    item: &'c WorldMeshDrawItem,
    empty_bg: &'a wgpu::BindGroup,
    last_material_bind_key: &'a mut Option<LastMaterialBindGroup1Key>,
    warned_missing_embedded_bind: &'a mut bool,
    offscreen_write_render_texture_asset_id: Option<i32>,
}

/// Binds `@group(1)` for embedded stems (texture/uniform pack) or the empty fallback.
fn set_world_mesh_material_bind_group(ctx: MaterialBindState<'_, '_, '_>) {
    if matches!(
        &ctx.item.batch_key.pipeline,
        RasterPipelineKind::EmbeddedStem(_)
    ) {
        let stem = ctx
            .backend
            .materials
            .material_registry
            .as_ref()
            .and_then(|r| r.stem_for_shader_asset(ctx.item.batch_key.shader_asset_id));
        if let (Some(mb), Some(stem)) = (ctx.backend.materials.embedded_material_bind(), stem) {
            let pools = ctx.backend.embedded_texture_pools();
            match mb.embedded_material_bind_group_with_cache_key(
                stem,
                ctx.queue,
                ctx.backend.material_property_store(),
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
            if ctx.backend.materials.embedded_material_bind().is_none()
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
pub(crate) struct ForwardDrawBatch<'a, 'b, 'c> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Indices into `draws` for this subpass.
    pub draw_indices: &'c [usize],
    /// Sorted world mesh draws for the view.
    pub draws: &'c [WorldMeshDrawItem],
    /// Backend (material registry, mesh pool).
    pub backend: &'a mut crate::backend::RenderBackend,
    /// Queue for embedded material bind uploads.
    pub queue: &'a wgpu::Queue,
    /// GPU device for lazy mesh stream creation.
    pub device: &'a wgpu::Device,
    /// Frame globals at `@group(0)`.
    pub frame_bg: &'a wgpu::BindGroup,
    /// Fallback material bind group when a stem has no resources.
    pub empty_bg: &'a wgpu::BindGroup,
    /// Per-draw storage slab at `@group(2)`.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Full per-draw storage slab, used for one-row fallback binds when base instance is unavailable.
    pub per_draw_storage: &'a wgpu::Buffer,
    /// Bind layout for `@group(2)`.
    pub per_draw_bind_group_layout: &'a wgpu::BindGroupLayout,
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
    /// Per-instance deform vertex streams; pointer is valid for the frame encode scope only.
    ///
    /// # Safety
    ///
    /// Must point at [`crate::backend::FrameResourceManager`]'s cache, which outlives this draw.
    pub skin_cache: Option<*const GpuSkinCache>,
}

pub(crate) fn draw_subset(batch: ForwardDrawBatch<'_, '_, '_>) {
    let mut last_batch_key: Option<MaterialDrawBatchKey> = None;
    let mut last_material_bind_key: Option<LastMaterialBindGroup1Key> = None;
    let mut current_pipelines: Option<MaterialPipelineSet> = None;
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

        if last_batch_key.as_ref() != Some(&item.batch_key) {
            last_batch_key = Some(item.batch_key.clone());
            let shader_asset_id = item.batch_key.shader_asset_id;
            let material_blend_mode = item.batch_key.blend_mode;
            pipeline_ok = match batch.backend.materials.material_registry.as_mut() {
                None => {
                    current_pipelines = None;
                    false
                }
                Some(reg) => {
                    match reg.pipeline_for_shader_asset(
                        shader_asset_id,
                        batch.pass_desc,
                        batch.shader_perm,
                        material_blend_mode,
                        item.batch_key.render_state,
                    ) {
                        Some(pipelines) if !pipelines.is_empty() => {
                            current_pipelines = Some(pipelines);
                            true
                        }
                        Some(_) => {
                            current_pipelines = None;
                            logger::trace!(
                                "WorldMeshForward: empty pipeline set for shader_asset_id {:?} pipeline {:?}, skipping draws until registered",
                                shader_asset_id,
                                item.batch_key.pipeline
                            );
                            false
                        }
                        None => {
                            current_pipelines = None;
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

        set_world_mesh_material_bind_group(MaterialBindState {
            rpass: batch.rpass,
            backend: batch.backend,
            queue: batch.queue,
            item,
            empty_bg: batch.empty_bg,
            last_material_bind_key: &mut last_material_bind_key,
            warned_missing_embedded_bind: batch.warned_missing_embedded_bind,
            offscreen_write_render_texture_asset_id: batch.offscreen_write_render_texture_asset_id,
        });

        if batch.supports_base_instance {
            // Full-buffer bind group: slot selection is `instance_index` from
            // `draw_indexed(..., first_idx..first_idx + count)`.
            batch
                .rpass
                .set_bind_group(2, batch.per_draw_bind_group, &[]);
        } else {
            // Some downlevel stacks do not support non-zero `first_instance`. Bind the current
            // row as a one-element storage array and draw with instance index zero.
            debug_assert_eq!(inst_batch.instance_count, 1);
            let bg = per_draw_one_row_bind_group(
                batch.device,
                batch.per_draw_bind_group_layout,
                batch.per_draw_storage,
                first_idx,
            );
            batch.rpass.set_bind_group(2, &bg, &[]);
        }
        let inst_range = instance_range_for_batch(
            first_idx,
            inst_batch.instance_count,
            batch.supports_base_instance,
        );
        batch
            .rpass
            .set_stencil_reference(item.batch_key.render_state.stencil_reference());

        let Some(pipelines) = current_pipelines.as_ref() else {
            continue;
        };
        for pipeline in pipelines.iter() {
            batch.rpass.set_pipeline(pipeline);
            draw_mesh_submesh_instanced(
                batch.rpass,
                item,
                batch.backend.mesh_pool_mut(),
                batch.device,
                batch.skin_cache,
                item.batch_key.embedded_needs_uv0,
                item.batch_key.embedded_needs_color,
                item.batch_key.embedded_needs_extended_vertex_streams,
                inst_range.clone(),
            );
        }
    }
}

fn instance_range_for_batch(
    first_draw_index: usize,
    instance_count: u32,
    supports_base_instance: bool,
) -> std::ops::Range<u32> {
    if supports_base_instance {
        let start = first_draw_index as u32;
        start..start + instance_count
    } else {
        0..instance_count
    }
}

fn per_draw_one_row_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    storage: &wgpu::Buffer,
    draw_index: usize,
) -> wgpu::BindGroup {
    let offset = (draw_index * PER_DRAW_UNIFORM_STRIDE) as u64;
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh_forward_per_draw_one_row_bind_group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: storage,
                offset,
                size: std::num::NonZeroU64::new(PER_DRAW_UNIFORM_STRIDE as u64),
            }),
        }],
    })
}

pub(crate) fn draw_mesh_submesh_instanced(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    mesh_pool: &mut MeshPool,
    device: &wgpu::Device,
    skin_cache: Option<*const GpuSkinCache>,
    embedded_uv: bool,
    embedded_color: bool,
    embedded_extended_vertex_streams: bool,
    instances: std::ops::Range<u32>,
) {
    if item.mesh_asset_id < 0 || item.node_id < 0 || item.index_count == 0 {
        return;
    }
    if embedded_extended_vertex_streams
        && !mesh_pool.ensure_extended_vertex_streams(device, item.mesh_asset_id)
    {
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

    let use_deformed = item.world_space_deformed;
    let use_blend_only = mesh.num_blendshapes > 0;
    let needs_cache_stream = use_deformed || use_blend_only;

    if needs_cache_stream {
        let Some(cache_ptr) = skin_cache else {
            return;
        };
        // SAFETY: `skin_cache` is [`FrameResourceManager`]'s cache; lives for the full encode.
        let cache = unsafe { &*cache_ptr };
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
    if embedded_uv || embedded_color || embedded_extended_vertex_streams {
        let Some(uv) = mesh.uv0_buffer.as_deref() else {
            return;
        };
        rpass.set_vertex_buffer(2, uv.slice(..));
    }
    if embedded_color || embedded_extended_vertex_streams {
        let Some(color) = mesh.color_buffer.as_deref() else {
            return;
        };
        rpass.set_vertex_buffer(3, color.slice(..));
    }
    if embedded_extended_vertex_streams {
        let (Some(tangent), Some(uv1), Some(uv2), Some(uv3)) = (
            mesh.tangent_buffer.as_deref(),
            mesh.uv1_buffer.as_deref(),
            mesh.uv2_buffer.as_deref(),
            mesh.uv3_buffer.as_deref(),
        ) else {
            return;
        };
        rpass.set_vertex_buffer(4, tangent.slice(..));
        rpass.set_vertex_buffer(5, uv1.slice(..));
        rpass.set_vertex_buffer(6, uv2.slice(..));
        rpass.set_vertex_buffer(7, uv3.slice(..));
    }
    rpass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);

    let first = item.first_index;
    let end = first.saturating_add(item.index_count);
    rpass.draw_indexed(first..end, 0, instances);
}

#[cfg(test)]
mod tests {
    use super::instance_range_for_batch;

    #[test]
    fn no_base_instance_draws_from_zero() {
        assert_eq!(instance_range_for_batch(17, 1, false), 0..1);
    }

    #[test]
    fn base_instance_uses_sorted_draw_slot() {
        assert_eq!(instance_range_for_batch(17, 3, true), 17..20);
    }
}
