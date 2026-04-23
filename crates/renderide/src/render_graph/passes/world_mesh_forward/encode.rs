//! Encode indexed draws and material bind groups for graph-managed world-mesh forward passes.

use crate::backend::mesh_deform::GpuSkinCache;
use crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::backend::MaterialBindCacheKey;
use crate::backend::WorldMeshForwardEncodeRefs;
use crate::embedded_shaders;
use crate::gpu::GpuLimits;
use crate::materials::{
    embedded_composed_stem_for_permutation, MaterialPassDesc, MaterialPipelineDesc,
    MaterialPipelineSet, RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::world_mesh_draw_prep::for_each_instance_batch;
use crate::render_graph::MaterialDrawBatchKey;
use crate::render_graph::WorldMeshDrawItem;
use crate::resources::MeshPool;

/// Embedded material vertex stream requirements for one draw (matches pipeline reflection flags).
pub(crate) struct EmbeddedVertexStreamFlags {
    /// UV0 stream at `@location(2)`.
    pub embedded_uv: bool,
    /// Vertex color at `@location(3)`.
    pub embedded_color: bool,
    /// Extended streams (tangents, extra UVs) at `@location(4)` and above.
    pub embedded_extended_vertex_streams: bool,
}

/// GPU mesh pool and optional skin cache for [`draw_mesh_submesh_instanced`].
pub(crate) struct WorldMeshDrawGpuRefs<'a> {
    /// Resident meshes and vertex buffers.
    pub mesh_pool: &'a MeshPool,
    /// Skin/deform cache when the draw uses deformed or blendshape streams.
    pub skin_cache: Option<&'a GpuSkinCache>,
}

/// Last `@group(1)` bind state for skipping redundant [`wgpu::RenderPass::set_bind_group`] when unchanged.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LastMaterialBindGroup1Key {
    Embedded(MaterialBindCacheKey),
    Empty,
}

/// Compact identity for a [`wgpu::Buffer`] sub-range used to skip redundant vertex / index binds.
///
/// `byte_len == None` encodes a full-buffer `.slice(..)` bind; `Some(n)` is a ranged bind
/// of `byte_offset..byte_offset + n`. Two `BufferBindId`s are equal when they refer to the
/// same buffer object, offset, and length — a sufficient condition for the bind to be a no-op.
///
/// Buffer identity is a raw pointer cast to `usize`; the pointer is stable for the lifetime
/// of the mesh pool / skin cache (both outlive any single render pass).
#[derive(Clone, Copy, PartialEq, Eq)]
struct BufferBindId {
    ptr: usize,
    byte_offset: u64,
    byte_len: Option<u64>,
}

impl BufferBindId {
    /// Full-buffer bind (`buf.slice(..)`).
    fn full(buf: &wgpu::Buffer) -> Self {
        Self {
            ptr: buf as *const wgpu::Buffer as usize,
            byte_offset: 0,
            byte_len: None,
        }
    }

    /// Ranged bind (`buf.slice(byte_start..byte_end)`).
    fn ranged(buf: &wgpu::Buffer, byte_start: u64, byte_end: u64) -> Self {
        Self {
            ptr: buf as *const wgpu::Buffer as usize,
            byte_offset: byte_start,
            byte_len: Some(byte_end - byte_start),
        }
    }
}

/// Per-render-pass last-bound vertex and index buffer state for bind deduplication.
///
/// Tracks the last-submitted buffer identity for each of the 8 vertex slots and the index
/// buffer. Reset at every new render pass (i.e. at the start of [`draw_subset`]).
pub(crate) struct LastMeshBindState {
    /// Last bound buffer identity per vertex slot 0–7; `None` = never bound this pass.
    vertex: [Option<BufferBindId>; 8],
    /// Last bound index buffer (pointer-as-usize identity) and format; `None` = never bound.
    index: Option<(usize, wgpu::IndexFormat)>,
}

impl LastMeshBindState {
    fn new() -> Self {
        Self {
            vertex: [None; 8],
            index: None,
        }
    }
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
    /// Material registry, pools, and skin cache (see [`crate::render_graph::FrameRenderParams::world_mesh_forward_encode_refs`]).
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Queue for embedded material bind uploads.
    pub queue: &'a wgpu::Queue,
    /// Device limits snapshot (dynamic storage offset alignment for `@group(2)`).
    pub gpu_limits: &'a GpuLimits,
    /// Frame globals at `@group(0)`.
    pub frame_bg: &'a wgpu::BindGroup,
    /// Fallback material bind group when a stem has no resources.
    pub empty_bg: &'a wgpu::BindGroup,
    /// Per-draw storage slab at `@group(2)` (dynamic storage offset; see [`Self::supports_base_instance`]).
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

/// Resolves raster pipeline set for `item`’s batch key, logging when the registry has no match.
fn resolve_pipelines_for_batch_item(
    encode: &mut WorldMeshForwardEncodeRefs<'_>,
    pass_desc: &MaterialPipelineDesc,
    shader_perm: ShaderPermutation,
    item: &WorldMeshDrawItem,
) -> (bool, Option<MaterialPipelineSet>) {
    let shader_asset_id = item.batch_key.shader_asset_id;
    let material_blend_mode = item.batch_key.blend_mode;
    match encode.materials.material_registry() {
        None => (false, None),
        Some(reg) => {
            match reg.pipeline_for_shader_asset(
                shader_asset_id,
                pass_desc,
                shader_perm,
                material_blend_mode,
                item.batch_key.render_state,
            ) {
                Some(pipelines) if !pipelines.is_empty() => (true, Some(pipelines)),
                Some(_) => {
                    logger::trace!(
                        "WorldMeshForward: empty pipeline set for shader_asset_id {:?} pipeline {:?}, skipping draws until registered",
                        shader_asset_id,
                        item.batch_key.pipeline
                    );
                    (false, None)
                }
                None => {
                    logger::trace!(
                        "WorldMeshForward: no pipeline for shader_asset_id {:?} pipeline {:?}, skipping draws until registered",
                        shader_asset_id,
                        item.batch_key.pipeline
                    );
                    (false, None)
                }
            }
        }
    }
}

pub(crate) fn draw_subset(batch: ForwardDrawBatch<'_, '_, '_, '_>) {
    let ForwardDrawBatch {
        rpass,
        draw_indices,
        draws,
        encode,
        queue,
        gpu_limits,
        frame_bg,
        empty_bg,
        per_draw_bind_group,
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
    // Dedup state for binds that are stable across consecutive draws in the same pass.
    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_stencil_ref: Option<u32> = None;

    rpass.set_bind_group(0, frame_bg, &[]);

    for_each_instance_batch(draws, draw_indices, supports_base_instance, |inst_batch| {
        let first_idx = inst_batch.first_draw_index;
        let item = &draws[first_idx];

        let batch_key_changed = last_batch_key.as_ref() != Some(&item.batch_key);
        if batch_key_changed {
            last_batch_key = Some(item.batch_key.clone());
            let (ok, pipes) =
                resolve_pipelines_for_batch_item(encode, pass_desc, shader_perm, item);
            pipeline_ok = ok;
            current_pipelines = pipes;
            current_declared_passes =
                declared_passes_for_pipeline(&item.batch_key.pipeline, shader_perm);
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

        let storage_align = gpu_limits.min_storage_buffer_offset_alignment();
        let per_draw_dyn_offset = if supports_base_instance {
            // Base-instance path: all rows accessed via `first_instance`; dynamic offset is
            // always zero for the entire pass so the bind is skipped after the first draw.
            0u32
        } else {
            // Downlevel: `first_instance` is always zero; select the draw row via dynamic offset.
            debug_assert_eq!(inst_batch.instance_count, 1);
            let raw = (first_idx * PER_DRAW_UNIFORM_STRIDE) as u32;
            debug_assert_eq!(
                raw % storage_align,
                0,
                "per-draw offset must match min_storage_buffer_offset_alignment"
            );
            raw
        };
        if last_per_draw_dyn_offset != Some(per_draw_dyn_offset) {
            rpass.set_bind_group(2, per_draw_bind_group, &[per_draw_dyn_offset]);
            last_per_draw_dyn_offset = Some(per_draw_dyn_offset);
        }

        let inst_range =
            instance_range_for_batch(first_idx, inst_batch.instance_count, supports_base_instance);

        let stencil_ref = item.batch_key.render_state.stencil_reference();
        if last_stencil_ref != Some(stencil_ref) {
            rpass.set_stencil_reference(stencil_ref);
            last_stencil_ref = Some(stencil_ref);
        }

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
                WorldMeshDrawGpuRefs {
                    mesh_pool: encode.mesh_pool(),
                    skin_cache,
                },
                EmbeddedVertexStreamFlags {
                    embedded_uv: item.batch_key.embedded_needs_uv0,
                    embedded_color: item.batch_key.embedded_needs_color,
                    embedded_extended_vertex_streams: item
                        .batch_key
                        .embedded_needs_extended_vertex_streams,
                },
                inst_range.clone(),
                &mut last_mesh,
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

/// Binds one vertex slot only when the buffer identity or range has changed since the last bind.
///
/// Using `global_id()` rather than pointer equality is safe because wgpu `Buffer`s are
/// refcounted and their IDs are stable for the lifetime of the object.
macro_rules! bind_vertex_if_changed {
    ($rpass:expr, $slot:expr, $buf:expr, $id:expr, $last:expr) => {{
        if $last[$slot as usize] != Some($id) {
            $rpass.set_vertex_buffer($slot, $buf);
            $last[$slot as usize] = Some($id);
        }
    }};
}

#[expect(
    clippy::too_many_lines,
    reason = "hot draw path keeps bind/set decisions inline for register reuse"
)]
pub(crate) fn draw_mesh_submesh_instanced(
    rpass: &mut wgpu::RenderPass<'_>,
    item: &WorldMeshDrawItem,
    gpu: WorldMeshDrawGpuRefs<'_>,
    streams: EmbeddedVertexStreamFlags,
    instances: std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
) {
    if item.mesh_asset_id < 0 || item.node_id < 0 || item.index_count == 0 {
        return;
    }
    let EmbeddedVertexStreamFlags {
        embedded_uv,
        embedded_color,
        embedded_extended_vertex_streams,
    } = streams;
    if embedded_extended_vertex_streams
        && !gpu
            .mesh_pool
            .get_mesh(item.mesh_asset_id)
            .is_some_and(|mesh| mesh.extended_vertex_streams_ready())
    {
        logger::trace!(
            "WorldMeshForward: extended vertex streams missing for mesh_asset_id {}; draw skipped until pre-warm catches up",
            item.mesh_asset_id
        );
        return;
    }
    let Some(mesh) = gpu.mesh_pool.get_mesh(item.mesh_asset_id) else {
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
        let Some(cache) = gpu.skin_cache else {
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
        let pos_range = entry.positions.byte_range();
        bind_vertex_if_changed!(
            rpass,
            0,
            pos_buf.slice(pos_range.start..pos_range.end),
            BufferBindId::ranged(pos_buf, pos_range.start, pos_range.end),
            last_mesh.vertex
        );
        if use_deformed {
            let Some(nrm_r) = entry.normals.as_ref() else {
                return;
            };
            let nrm_buf = cache.normals_arena();
            let nrm_range = nrm_r.byte_range();
            bind_vertex_if_changed!(
                rpass,
                1,
                nrm_buf.slice(nrm_range.start..nrm_range.end),
                BufferBindId::ranged(nrm_buf, nrm_range.start, nrm_range.end),
                last_mesh.vertex
            );
        } else {
            bind_vertex_if_changed!(
                rpass,
                1,
                normals_bind.slice(..),
                BufferBindId::full(normals_bind),
                last_mesh.vertex
            );
        }
    } else {
        let Some(pos) = mesh.positions_buffer.as_deref() else {
            return;
        };
        bind_vertex_if_changed!(
            rpass,
            0,
            pos.slice(..),
            BufferBindId::full(pos),
            last_mesh.vertex
        );
        bind_vertex_if_changed!(
            rpass,
            1,
            normals_bind.slice(..),
            BufferBindId::full(normals_bind),
            last_mesh.vertex
        );
    }
    if embedded_uv || embedded_color || embedded_extended_vertex_streams {
        let Some(uv) = mesh.uv0_buffer.as_deref() else {
            return;
        };
        bind_vertex_if_changed!(
            rpass,
            2,
            uv.slice(..),
            BufferBindId::full(uv),
            last_mesh.vertex
        );
    }
    if embedded_color || embedded_extended_vertex_streams {
        let Some(color) = mesh.color_buffer.as_deref() else {
            return;
        };
        bind_vertex_if_changed!(
            rpass,
            3,
            color.slice(..),
            BufferBindId::full(color),
            last_mesh.vertex
        );
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
        bind_vertex_if_changed!(
            rpass,
            4,
            tangent.slice(..),
            BufferBindId::full(tangent),
            last_mesh.vertex
        );
        bind_vertex_if_changed!(
            rpass,
            5,
            uv1.slice(..),
            BufferBindId::full(uv1),
            last_mesh.vertex
        );
        bind_vertex_if_changed!(
            rpass,
            6,
            uv2.slice(..),
            BufferBindId::full(uv2),
            last_mesh.vertex
        );
        bind_vertex_if_changed!(
            rpass,
            7,
            uv3.slice(..),
            BufferBindId::full(uv3),
            last_mesh.vertex
        );
    }

    let index_key = (
        mesh.index_buffer.as_ref() as *const wgpu::Buffer as usize,
        mesh.index_format,
    );
    if last_mesh.index != Some(index_key) {
        rpass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);
        last_mesh.index = Some(index_key);
    }

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
