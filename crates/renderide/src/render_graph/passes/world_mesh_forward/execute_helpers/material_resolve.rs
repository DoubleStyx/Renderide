//! Material pipeline and embedded bind-group precomputation for world-mesh forward draws.

use rayon::prelude::*;

use crate::backend::WorldMeshForwardEncodeRefs;
use crate::embedded_shaders;
use crate::materials::{
    embedded_composed_stem_for_permutation, MaterialPassDesc, MaterialPipelineDesc,
    RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::frame_params::PrecomputedMaterialBind;
use crate::render_graph::world_mesh_draw_prep::WorldMeshDrawItem;

/// Resolves per-batch pipeline sets and `@group(1)` bind groups for the sorted draw list.
///
/// Works in two phases:
///
/// 1. **Boundary detection (serial)** — single O(N) scan to find where
///    [`crate::render_graph::MaterialDrawBatchKey`] changes.
///
/// 2. **Resolution (parallel via rayon)** — for each unique batch, resolves the pipeline set
///    from the material registry and the embedded `@group(1)` bind group from the LRU cache.
///    Rayon workers share borrowed refs to the material system and asset pools (all `Sync`);
///    cache access uses the existing `Mutex<LruCache>` internals so concurrent hits are cheap
///    (~50 ns lock + Arc clone) and concurrent misses produce a correct result.
///
/// Both raster sub-passes (opaque and intersect) drive `set_pipeline` / `set_bind_group` from
/// `PreparedWorldMeshForwardFrame::precomputed_batches` — no LRU lookups during `RenderPass`.
pub(super) fn precompute_material_resolve_batches(
    encode: &WorldMeshForwardEncodeRefs<'_>,
    queue: &wgpu::Queue,
    draws: &[WorldMeshDrawItem],
    shader_perm: ShaderPermutation,
    pass_desc: &MaterialPipelineDesc,
    offscreen_write_render_texture_asset_id: Option<i32>,
) -> Vec<PrecomputedMaterialBind> {
    profiling::scope!("world_mesh::precompute_material_binds");
    if draws.is_empty() {
        return Vec::new();
    }

    let boundaries = collect_material_batch_boundaries(draws);

    // Borrow the pieces that rayon workers will share (`&` = Sync).
    let registry = encode.materials.material_registry();
    let embedded_bind = encode.materials.embedded_material_bind();
    let store = encode.materials.material_property_store();
    let pools = encode.embedded_texture_pools();

    boundaries
        .into_par_iter()
        .map(|(first, last)| {
            resolve_one_material_batch(
                draws,
                first,
                last,
                registry,
                embedded_bind,
                store,
                &pools,
                queue,
                shader_perm,
                pass_desc,
                offscreen_write_render_texture_asset_id,
            )
        })
        .collect()
}

/// Walks `draws` once and emits `(first_idx, last_idx)` runs of identical [`MaterialDrawBatchKey`].
///
/// `draws` is assumed pre-sorted by batch key (the world-mesh draw collector guarantees this), so
/// each adjacent-equal run is one material batch. Returns at least one boundary when `draws` is
/// non-empty; callers handle the empty case before calling.
fn collect_material_batch_boundaries(draws: &[WorldMeshDrawItem]) -> Vec<(usize, usize)> {
    let mut boundaries: Vec<(usize, usize)> = Vec::new();
    let mut current_start = 0usize;
    let mut last_key = &draws[0].batch_key;
    for (idx, item) in draws.iter().enumerate().skip(1) {
        if &item.batch_key != last_key {
            boundaries.push((current_start, idx - 1));
            current_start = idx;
            last_key = &item.batch_key;
        }
    }
    boundaries.push((current_start, draws.len() - 1));
    boundaries
}

/// Resolves the pipeline set, declared passes, and `@group(1)` bind group for one material batch.
///
/// Called from a rayon worker once per `(first_idx, last_idx)` boundary returned by
/// [`collect_material_batch_boundaries`]. All borrowed parameters are `Sync`; the cache locks
/// inside `embedded_material_bind_group_with_cache_key` keep concurrent hits cheap.
#[expect(
    clippy::too_many_arguments,
    reason = "all args are owned by the parallel closure body extracted from precompute_material_resolve_batches"
)]
fn resolve_one_material_batch<'a>(
    draws: &[WorldMeshDrawItem],
    first: usize,
    last: usize,
    registry: Option<&crate::materials::MaterialRegistry>,
    embedded_bind: Option<&crate::backend::EmbeddedMaterialBindResources>,
    store: &crate::assets::material::MaterialPropertyStore,
    pools: &crate::backend::EmbeddedTexturePools<'a>,
    queue: &wgpu::Queue,
    shader_perm: ShaderPermutation,
    pass_desc: &MaterialPipelineDesc,
    offscreen_write_render_texture_asset_id: Option<i32>,
) -> PrecomputedMaterialBind {
    let item = &draws[first];
    let batch_key = &item.batch_key;
    let grab_pass_desc;
    let pass_desc = if batch_key.embedded_requires_grab_pass && pass_desc.sample_count > 1 {
        grab_pass_desc = MaterialPipelineDesc {
            sample_count: 1,
            ..*pass_desc
        };
        &grab_pass_desc
    } else {
        pass_desc
    };

    let (pipelines, declared_passes) = if let Some(reg) = registry {
        let pipes = reg.pipeline_for_shader_asset(
            batch_key.shader_asset_id,
            pass_desc,
            shader_perm,
            batch_key.blend_mode,
            batch_key.render_state,
            batch_key.front_face,
        );
        let passes = declared_passes_for_pipeline_kind(&batch_key.pipeline, shader_perm);
        match pipes {
            Some(p) if !p.is_empty() => (Some(p), passes),
            Some(_) => {
                logger::trace!(
                    "WorldMeshForward: empty pipeline for shader {:?}, skipping batch",
                    batch_key.shader_asset_id
                );
                (None, passes)
            }
            None => {
                logger::trace!(
                    "WorldMeshForward: no pipeline for shader {:?}, skipping batch",
                    batch_key.shader_asset_id
                );
                (None, passes)
            }
        }
    } else {
        (None, &[] as &'static [MaterialPassDesc])
    };

    let bind_group = if matches!(&batch_key.pipeline, RasterPipelineKind::EmbeddedStem(_)) {
        if let (Some(mb), Some(reg)) = (embedded_bind, registry) {
            match reg.stem_for_shader_asset(batch_key.shader_asset_id) {
                Some(stem) => mb
                    .embedded_material_bind_group_with_cache_key(
                        stem,
                        queue,
                        store,
                        pools,
                        item.lookup_ids,
                        offscreen_write_render_texture_asset_id,
                    )
                    .ok()
                    .map(|(_, bg)| bg),
                None => None,
            }
        } else {
            if embedded_bind.is_none() {
                logger::warn!(
                    "WorldMeshForward: embedded material bind resources unavailable; \
                         @group(1) uses empty bind group for embedded raster draws"
                );
            }
            None
        }
    } else {
        None
    };

    PrecomputedMaterialBind {
        first_draw_idx: first,
        last_draw_idx: last,
        front_face: batch_key.front_face,
        bind_group,
        pipelines,
        declared_passes,
    }
}

/// Returns the declared pass descriptors for `pipeline` at `shader_perm` (zero-alloc `&'static`).
fn declared_passes_for_pipeline_kind(
    pipeline: &RasterPipelineKind,
    shader_perm: ShaderPermutation,
) -> &'static [MaterialPassDesc] {
    let RasterPipelineKind::EmbeddedStem(stem) = pipeline else {
        return &[];
    };
    let composed = embedded_composed_stem_for_permutation(stem.as_ref(), shader_perm);
    embedded_shaders::embedded_target_passes(&composed)
}
