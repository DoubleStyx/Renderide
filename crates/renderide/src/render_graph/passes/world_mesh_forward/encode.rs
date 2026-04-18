//! Encode indexed draws and material bind groups for graph-managed world-mesh forward passes.

use crate::backend::mesh_deform::GpuSkinCache;
use crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::backend::MaterialBindCacheKey;
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::embedded_shaders;
use crate::materials::{
    embedded_composed_stem_for_permutation, MaterialPassDesc, MaterialPipelineDesc,
    MaterialPipelineSet, RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::world_mesh_draw_prep::for_each_instance_batch;
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
            .material_registry()
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
    /// Whether the packed frame light buffer contains any point/spot light.
    pub has_local_lights: bool,
}

fn declared_passes_for_pipeline(
    pipeline: &RasterPipelineKind,
    shader_perm: ShaderPermutation,
) -> &'static [MaterialPassDesc] {
    let RasterPipelineKind::EmbeddedStem(stem) = pipeline else {
        return &[];
    };
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), shader_perm);
    embedded_shaders::embedded_target_passes(&composed)
}

fn should_skip_pipeline_pass(
    declared_passes: &[MaterialPassDesc],
    pass_idx: usize,
    has_local_lights: bool,
) -> bool {
    !has_local_lights
        && declared_passes
            .get(pass_idx)
            .is_some_and(|pass| pass.name == "forward_delta")
}

pub(crate) fn draw_subset(batch: ForwardDrawBatch<'_, '_, '_, '_>) {
    let ForwardDrawBatch {
        rpass,
        draw_indices,
        draws,
        encode,
        queue,
        device,
        frame_bg,
        empty_bg,
        per_draw_bind_group,
        per_draw_storage,
        per_draw_bind_group_layout,
        pass_desc,
        shader_perm,
        warned_missing_embedded_bind,
        offscreen_write_render_texture_asset_id,
        supports_base_instance,
        has_local_lights,
    } = batch;

    let mut last_batch_key: Option<MaterialDrawBatchKey> = None;
    let mut last_material_bind_key: Option<LastMaterialBindGroup1Key> = None;
    let mut current_pipelines: Option<MaterialPipelineSet> = None;
    let mut current_declared_passes: &'static [MaterialPassDesc] = &[];
    let mut pipeline_ok = false;

    rpass.set_bind_group(0, frame_bg, &[]);

    for_each_instance_batch(draws, draw_indices, supports_base_instance, |inst_batch| {
        let first_idx = inst_batch.first_draw_index;
        let item = &draws[first_idx];

        let batch_key_changed = last_batch_key.as_ref() != Some(&item.batch_key);
        if batch_key_changed {
            last_batch_key = Some(item.batch_key.clone());
            current_declared_passes =
                declared_passes_for_pipeline(&item.batch_key.pipeline, shader_perm);
            let shader_asset_id = item.batch_key.shader_asset_id;
            let material_blend_mode = item.batch_key.blend_mode;
            pipeline_ok = match encode.materials.material_registry_mut() {
                None => {
                    current_pipelines = None;
                    false
                }
                Some(reg) => {
                    match reg.pipeline_for_shader_asset(
                        shader_asset_id,
                        pass_desc,
                        shader_perm,
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
            return;
        }

        // Material bind resolution (stem layout + texture signature hash + LRU lookups) only needs
        // to run when the batch key changes. Within a run of same-key batches, @group(1) and its
        // cached uniform buffer are invariant, so the previous `last_material_bind_key` still
        // reflects what is bound on the render pass.
        if batch_key_changed {
            set_world_mesh_material_bind_group(MaterialBindState {
                rpass: &mut *rpass,
                encode: &mut *encode,
                queue,
                item,
                empty_bg,
                last_material_bind_key: &mut last_material_bind_key,
                warned_missing_embedded_bind,
                offscreen_write_render_texture_asset_id,
            });
        }

        if supports_base_instance {
            // Full-buffer bind group: slot selection is `instance_index` from
            // `draw_indexed(..., first_idx..first_idx + count)`.
            rpass.set_bind_group(2, per_draw_bind_group, &[]);
        } else {
            // Some downlevel stacks do not support non-zero `first_instance`. Bind the current
            // row as a one-element storage array and draw with instance index zero.
            debug_assert_eq!(inst_batch.instance_count, 1);
            let bg = per_draw_one_row_bind_group(
                device,
                per_draw_bind_group_layout,
                per_draw_storage,
                first_idx,
            );
            rpass.set_bind_group(2, &bg, &[]);
        }
        let inst_range =
            instance_range_for_batch(first_idx, inst_batch.instance_count, supports_base_instance);
        rpass.set_stencil_reference(item.batch_key.render_state.stencil_reference());

        let Some(pipelines) = current_pipelines.as_ref() else {
            return;
        };
        let skin_cache = encode.skin_cache;
        for (pass_idx, pipeline) in pipelines.iter().enumerate() {
            if should_skip_pipeline_pass(current_declared_passes, pass_idx, has_local_lights) {
                continue;
            }
            rpass.set_pipeline(pipeline);
            draw_mesh_submesh_instanced(
                rpass,
                item,
                encode.mesh_pool_mut(),
                device,
                skin_cache,
                item.batch_key.embedded_needs_uv0,
                item.batch_key.embedded_needs_color,
                item.batch_key.embedded_needs_extended_vertex_streams,
                inst_range.clone(),
            );
        }
    });
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_mesh_submesh_instanced(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    mesh_pool: &mut MeshPool,
    device: &wgpu::Device,
    skin_cache: Option<&GpuSkinCache>,
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
    use crate::materials::default_pass;

    use super::{instance_range_for_batch, should_skip_pipeline_pass, MaterialPassDesc};

    #[test]
    fn no_base_instance_draws_from_zero() {
        assert_eq!(instance_range_for_batch(17, 1, false), 0..1);
    }

    #[test]
    fn base_instance_uses_sorted_draw_slot() {
        assert_eq!(instance_range_for_batch(17, 3, true), 17..20);
    }

    #[test]
    fn skips_forward_delta_only_when_no_local_lights() {
        let passes = [
            MaterialPassDesc {
                name: "forward",
                ..default_pass(false, true)
            },
            MaterialPassDesc {
                name: "forward_delta",
                ..default_pass(false, false)
            },
        ];

        assert!(!should_skip_pipeline_pass(&passes, 0, false));
        assert!(should_skip_pipeline_pass(&passes, 1, false));
        assert!(!should_skip_pipeline_pass(&passes, 1, true));
    }
}
