//! Material batch packet resolution for world-mesh forward draws.
//!
//! The resolver is the single boundary between sorted CPU draw runs and concrete raster state.
//! Backend frame planning builds [`PipelineVariantKey`] once per batch so raster recording cannot
//! drift on MSAA, front-face, blend, render-state, or shader permutations.

use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use hashbrown::HashMap;
use rayon::prelude::*;

use crate::cpu_parallelism::{
    RELEVANCE_PACKET_MIN_ITEMS, admit_relevance_items, current_reference_worker_count,
    record_parallel_admission,
};
use crate::frame_upload_batch::GraphUploadSink;
use crate::graph_inputs::OffscreenWriteTarget;
use crate::log_throttle::LogThrottle;
use crate::materials::ShaderPermutation;
use crate::materials::embedded::{EmbeddedMaterialBindError, MaterialBindCacheKey};
use crate::materials::{
    EmbeddedMaterialBindResources, EmbeddedMaterialBindShader, EmbeddedTexturePools,
};
use crate::materials::{
    MaterialBlendMode, MaterialPassRouting, MaterialPipelineDesc, MaterialPipelineResolution,
    MaterialPipelineSet, MaterialPipelineVariantSpec, MaterialRegistry, MaterialRenderState,
    MaterialShaderSpecializationKey, RasterFrontFace, RasterPipelineKind, RasterPrimitiveTopology,
    ensure_render_buffer_billboard_variant_bits, remap_variant_bits_for_billboard,
};
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::world_mesh::MaterialDrawBatchKey;
use crate::world_mesh::draw_prep::WorldMeshDrawItem;

/// Material boundary runs assigned to one packet-resolution worker.
const MATERIAL_BATCH_PARALLEL_CHUNK_RUNS: usize = RELEVANCE_PACKET_MIN_ITEMS;
/// Material boundary run count required before packet resolution uses Rayon.
const MATERIAL_BATCH_PARALLEL_MIN_RUNS: usize = MATERIAL_BATCH_PARALLEL_CHUNK_RUNS * 2;

/// Throttles repeated embedded-bind failures so a single bad material cannot flood logs.
static EMBEDDED_MATERIAL_BIND_FAILURE_LOG: LogThrottle = LogThrottle::new();

/// Inclusive `(first_draw_idx, last_draw_idx)` span over the sorted world-mesh draw list
/// identifying one contiguous material batch run.
pub(crate) type MaterialBatchBoundary = (usize, usize);

/// Draw-local identity that determines material pipeline and group-1 resolution.
///
/// Transparent sorting can split one material into hundreds of non-contiguous runs. Hashing the
/// cached batch-key hash keeps this deduplication cheap while `PartialEq` still compares the full
/// key, so a hash collision cannot alias distinct material state.
#[derive(Clone, Copy, Eq, PartialEq)]
struct MaterialBatchResolutionIdentity<'a> {
    batch_key_hash: u64,
    batch_key: &'a MaterialDrawBatchKey,
    material_asset_id: i32,
    property_block_slot0: Option<i32>,
    renderer_property_block_id: Option<i32>,
}

impl Hash for MaterialBatchResolutionIdentity<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.batch_key_hash.hash(state);
        self.material_asset_id.hash(state);
        self.property_block_slot0.hash(state);
        self.renderer_property_block_id.hash(state);
    }
}

/// Kind-only summary of the group-1 binding carried by a material packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaterialGroup1BindingKind {
    /// Empty material bind group used by the Null fallback pipeline.
    Empty,
    /// Reflected embedded material bind group used by embedded raster pipelines.
    Embedded,
}

/// Explicit group-1 binding state for one world-mesh material packet.
#[derive(Clone)]
pub(crate) enum MaterialGroup1Binding {
    /// Bind the shared empty material bind group.
    Empty,
    /// Bind an embedded material group and optional uniform dynamic offset.
    Embedded {
        /// Cache key that describes the bind group's resolved texture/uniform identity.
        bind_key: MaterialBindCacheKey,
        /// Reflected bind group matching the selected embedded pipeline layout.
        bind_group: Arc<wgpu::BindGroup>,
        /// Dynamic offset into the material uniform arena, when the material block is dynamic.
        uniform_dynamic_offset: Option<u32>,
    },
}

impl MaterialGroup1Binding {
    /// Returns the kind-only binding identity for validation and tests.
    fn kind(&self) -> MaterialGroup1BindingKind {
        match self {
            Self::Empty => MaterialGroup1BindingKind::Empty,
            Self::Embedded { .. } => MaterialGroup1BindingKind::Embedded,
        }
    }
}

/// One resolved per-batch draw packet covering a contiguous range of sorted draws with the same
/// [`crate::world_mesh::MaterialDrawBatchKey`].
///
/// Populated by backend frame planning so the recording loop can drive pipeline and bind-group state
/// entirely from this table, without material-cache lookups inside `RenderPass`.
#[derive(Clone)]
pub(crate) struct MaterialBatchPacket {
    /// First draw index (into the sorted draw list) covered by this entry.
    pub first_draw_idx: usize,
    /// Last draw index (inclusive) covered by this entry.
    pub last_draw_idx: usize,
    /// Exact pipeline variant requested for this batch.
    pub(crate) pipeline_key: PipelineVariantKey,
    /// Actual pipeline kind selected for this packet, or [`None`] when the batch is skipped.
    pub(crate) resolved_pipeline_kind: Option<RasterPipelineKind>,
    /// Explicit `@group(1)` binding that matches [`Self::resolved_pipeline_kind`].
    pub(crate) group1_binding: MaterialGroup1Binding,
    /// Resolved pipeline set for this batch, or `None` when the pipeline is unavailable (skip draws).
    pub pipelines: Option<MaterialPipelineSet>,
}

/// Inputs needed to build a [`PipelineVariantKey`] for one material draw run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PipelineVariantKeyInput {
    /// Base pass descriptor for the owning view.
    pub pass_desc: MaterialPipelineDesc,
    /// Shader permutation selected for the owning view.
    pub shader_perm: ShaderPermutation,
    /// Renderer-local shader specialization constants for material keyword branches.
    pub shader_specialization: MaterialShaderSpecializationKey,
    /// Host shader asset id for diagnostics and material registry lookup.
    pub shader_asset_id: i32,
    /// Resolved material blend state.
    pub blend_mode: MaterialBlendMode,
    /// Resolved material render state.
    pub render_state: MaterialRenderState,
    /// Runtime material routing decisions for per-pass pipeline state.
    pub pass_routing: MaterialPassRouting,
    /// Front-face winding selected from the draw transform.
    pub front_face: RasterFrontFace,
    /// Primitive topology selected from the mesh's per-submesh topology.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// Exact material pipeline variant used by backend frame planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PipelineVariantKey {
    /// Host shader asset id for diagnostics and material registry lookup.
    pub shader_asset_id: i32,
    /// Color attachment format.
    pub surface_format: wgpu::TextureFormat,
    /// Optional depth/stencil format.
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    /// Effective sample count for the active render pass.
    pub sample_count: u32,
    /// Optional multiview mask.
    pub multiview_mask: Option<std::num::NonZeroU32>,
    /// Shader permutation selected for the view.
    pub shader_perm: ShaderPermutation,
    /// Renderer-local shader specialization constants for material keyword branches.
    pub shader_specialization: MaterialShaderSpecializationKey,
    /// Resolved material blend state.
    pub blend_mode: MaterialBlendMode,
    /// Resolved material render state.
    pub render_state: MaterialRenderState,
    /// Runtime material routing decisions for per-pass pipeline state.
    pub pass_routing: MaterialPassRouting,
    /// Front-face winding selected from the draw transform.
    pub front_face: RasterFrontFace,
    /// Primitive topology selected from the mesh's per-submesh topology.
    pub primitive_topology: RasterPrimitiveTopology,
}

impl PipelineVariantKey {
    /// Builds the key used for material packet resolution.
    pub(crate) fn new(input: PipelineVariantKeyInput) -> Self {
        let PipelineVariantKeyInput {
            pass_desc,
            shader_perm,
            shader_specialization,
            shader_asset_id,
            blend_mode,
            render_state,
            pass_routing,
            front_face,
            primitive_topology,
        } = input;
        Self {
            shader_asset_id,
            surface_format: pass_desc.surface_format,
            depth_stencil_format: pass_desc.depth_stencil_format,
            sample_count: pass_desc.sample_count,
            multiview_mask: pass_desc.multiview_mask,
            shader_perm,
            shader_specialization,
            blend_mode,
            render_state,
            pass_routing,
            front_face,
            primitive_topology,
        }
    }

    /// Rehydrates the material pipeline descriptor used by [`MaterialRegistry`].
    pub(crate) fn pass_desc(self) -> MaterialPipelineDesc {
        MaterialPipelineDesc {
            surface_format: self.surface_format,
            depth_stencil_format: self.depth_stencil_format,
            sample_count: self.sample_count,
            multiview_mask: self.multiview_mask,
        }
    }

    /// Rehydrates the material pipeline variant selectors used by [`MaterialRegistry`].
    pub(crate) fn variant_spec(self) -> MaterialPipelineVariantSpec {
        MaterialPipelineVariantSpec {
            permutation: self.shader_perm,
            shader_specialization: self.shader_specialization,
            blend_mode: self.blend_mode,
            render_state: self.render_state,
            pass_routing: self.pass_routing,
            front_face: self.front_face,
            primitive_topology: self.primitive_topology,
        }
    }

    /// Builds a key directly from a sorted draw item and view-level pipeline state.
    pub(crate) fn for_draw_item(
        item: &WorldMeshDrawItem,
        pass_desc: MaterialPipelineDesc,
        shader_perm: ShaderPermutation,
    ) -> Self {
        let batch_key = &item.batch_key;
        Self::new(PipelineVariantKeyInput {
            pass_desc,
            shader_perm,
            shader_specialization: batch_key.shader_specialization,
            shader_asset_id: batch_key.shader_asset_id,
            blend_mode: batch_key.blend_mode,
            render_state: batch_key.render_state,
            pass_routing: batch_key.pass_routing(),
            front_face: batch_key.front_face,
            primitive_topology: batch_key.primitive_topology,
        })
    }
}

/// Exact registry request shared by material packets that use the same raster pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineResolutionRequest<'a> {
    requested_kind: &'a RasterPipelineKind,
    pipeline_key: PipelineVariantKey,
}

/// Material pipeline and embedded-bind resolver for one world-mesh forward view plan.
pub(crate) struct MaterialDrawResolver<'a> {
    /// Material registry used for pipeline lookup.
    registry: Option<&'a MaterialRegistry>,
    /// Embedded material bind resources used for `@group(1)` lookup.
    embedded_bind: Option<&'a EmbeddedMaterialBindResources>,
    /// Material property store used by embedded bind resolution.
    store: &'a crate::materials::host_data::MaterialPropertyStore,
    /// Texture pools used by embedded bind resolution.
    pools: EmbeddedTexturePools<'a>,
    /// Upload sink used by embedded uniform updates.
    uploads: GraphUploadSink<'a>,
    /// View-level material pipeline descriptor before per-material overrides.
    pass_desc: MaterialPipelineDesc,
    /// Shader permutation for this view.
    shader_perm: ShaderPermutation,
    /// Offscreen target being written by this view, if any.
    offscreen_write_target: OffscreenWriteTarget,
    /// Whether this view flips front-face winding before draw-local transform parity is applied.
    front_face_flip: bool,
}

impl<'a> MaterialDrawResolver<'a> {
    /// Builds a resolver from the forward encode references for this view.
    pub(crate) fn new(
        encode: &'a WorldMeshForwardEncodeRefs<'_>,
        uploads: GraphUploadSink<'a>,
        pass_desc: MaterialPipelineDesc,
        shader_perm: ShaderPermutation,
        offscreen_write_target: OffscreenWriteTarget,
        front_face_flip: bool,
    ) -> Self {
        Self {
            registry: encode.materials.material_registry(),
            embedded_bind: encode.materials.embedded_material_bind(),
            store: encode.materials.material_property_store(),
            pools: encode.embedded_texture_pools(),
            uploads,
            pass_desc,
            shader_perm,
            offscreen_write_target,
            front_face_flip,
        }
    }

    /// Resolves every contiguous material run in `draws` into record-ready packets.
    ///
    /// `boundaries_scratch` is cleared and refilled with the material-batch boundary spans; the
    /// caller owns the buffer so its capacity survives across frames and reallocates only on
    /// growth past the previous high-water mark.
    pub(crate) fn resolve_batches(
        &self,
        draws: &[WorldMeshDrawItem],
        boundaries_scratch: &mut Vec<MaterialBatchBoundary>,
    ) -> Vec<MaterialBatchPacket> {
        profiling::scope!("world_mesh_forward::resolve_material_packets");
        boundaries_scratch.clear();
        if draws.is_empty() {
            return Vec::new();
        }

        collect_material_batch_boundaries_into(draws, boundaries_scratch);
        let (unique_boundaries, boundary_template_indices) = {
            profiling::scope!("world_mesh_forward::deduplicate_material_packets");
            collect_unique_material_batch_boundaries(draws, boundaries_scratch)
        };
        let (pipeline_requests, template_pipeline_request_indices) = {
            profiling::scope!("world_mesh_forward::deduplicate_material_pipelines");
            collect_unique_pipeline_resolution_requests(
                draws,
                &unique_boundaries,
                self.pass_desc,
                self.shader_perm,
                self.front_face_flip,
            )
        };
        let pipeline_resolutions: Vec<OnceLock<Option<MaterialPipelineResolution>>> =
            std::iter::repeat_with(OnceLock::new)
                .take(pipeline_requests.len())
                .collect();
        let admission =
            admit_relevance_items(unique_boundaries.len(), current_reference_worker_count());
        record_parallel_admission(
            "material_batch_resolve",
            boundaries_scratch.len(),
            unique_boundaries.len(),
            admission,
        );
        let templates = if unique_boundaries.len() < MATERIAL_BATCH_PARALLEL_MIN_RUNS
            || !admission.is_parallel()
        {
            let mut packets = Vec::with_capacity(unique_boundaries.len());
            for (template_idx, &(first, last)) in unique_boundaries.iter().enumerate() {
                let request_idx = template_pipeline_request_indices[template_idx];
                packets.push(self.resolve_one_batch(
                    draws,
                    first,
                    last,
                    pipeline_requests[request_idx],
                    &pipeline_resolutions[request_idx],
                ));
            }
            packets
        } else {
            let chunk_size = admission
                .chunk_size()
                .unwrap_or(MATERIAL_BATCH_PARALLEL_CHUNK_RUNS);
            unique_boundaries
                .par_iter()
                .with_min_len(chunk_size)
                .copied()
                .enumerate()
                .map(|(template_idx, (first, last))| {
                    let request_idx = template_pipeline_request_indices[template_idx];
                    self.resolve_one_batch(
                        draws,
                        first,
                        last,
                        pipeline_requests[request_idx],
                        &pipeline_resolutions[request_idx],
                    )
                })
                .collect()
        };

        if unique_boundaries.len() == boundaries_scratch.len() {
            return templates;
        }

        profiling::scope!("world_mesh_forward::expand_material_packet_runs");
        boundaries_scratch
            .iter()
            .zip(boundary_template_indices)
            .map(|(&(first_draw_idx, last_draw_idx), template_idx)| {
                let mut packet = templates[template_idx].clone();
                packet.first_draw_idx = first_draw_idx;
                packet.last_draw_idx = last_draw_idx;
                packet
            })
            .collect()
    }

    /// Resolves one material run into a record-ready packet.
    fn resolve_one_batch(
        &self,
        draws: &[WorldMeshDrawItem],
        first: usize,
        last: usize,
        pipeline_request: PipelineResolutionRequest<'_>,
        pipeline_resolution: &OnceLock<Option<MaterialPipelineResolution>>,
    ) -> MaterialBatchPacket {
        let item = &draws[first];
        debug_assert_eq!(pipeline_request.requested_kind, &item.batch_key.pipeline);
        let pipeline_key = pipeline_request.pipeline_key;
        let pipeline_resolution = resolve_pipeline_once(pipeline_resolution, || {
            self.resolve_pipeline_resolution(pipeline_request.requested_kind, pipeline_key)
        });
        let resolved = self.resolve_pipeline_and_group1(item, pipeline_key, pipeline_resolution);

        if let Some((resolution, group1_binding)) = resolved {
            debug_assert!(material_group1_binding_matches_pipeline(
                group1_binding.kind(),
                &resolution.kind
            ));
            return MaterialBatchPacket {
                first_draw_idx: first,
                last_draw_idx: last,
                pipeline_key,
                resolved_pipeline_kind: Some(resolution.kind),
                group1_binding,
                pipelines: Some(resolution.pipelines),
            };
        }

        MaterialBatchPacket {
            first_draw_idx: first,
            last_draw_idx: last,
            pipeline_key,
            resolved_pipeline_kind: None,
            group1_binding: MaterialGroup1Binding::Empty,
            pipelines: None,
        }
    }

    /// Resolves the material pipeline and matching group-1 binding for one batch.
    fn resolve_pipeline_and_group1(
        &self,
        item: &WorldMeshDrawItem,
        pipeline_key: PipelineVariantKey,
        resolution: Option<&MaterialPipelineResolution>,
    ) -> Option<(MaterialPipelineResolution, MaterialGroup1Binding)> {
        let resolution = resolution.cloned()?;
        match &resolution.kind {
            RasterPipelineKind::Null => Some((resolution, MaterialGroup1Binding::Empty)),
            RasterPipelineKind::EmbeddedStem(stem) => {
                match self.resolve_embedded_group1_binding(item, stem.as_ref()) {
                    Ok(group1_binding) => Some((resolution, group1_binding)),
                    Err(error) => {
                        let fallback = self.resolve_null_fallback_pipeline(pipeline_key);
                        self.log_embedded_bind_failure(
                            item,
                            stem.as_ref(),
                            &error,
                            fallback.is_some(),
                        );
                        fallback.map(|fallback_resolution| {
                            (fallback_resolution, MaterialGroup1Binding::Empty)
                        })
                    }
                }
            }
        }
    }

    /// Resolves the material pipeline set and concrete raster kind for one batch.
    fn resolve_pipeline_resolution(
        &self,
        pipeline_kind: &RasterPipelineKind,
        pipeline_key: PipelineVariantKey,
    ) -> Option<MaterialPipelineResolution> {
        let registry = self.registry?;

        let pass_desc = pipeline_key.pass_desc();
        let resolution = registry.pipeline_for_resolved_kind(
            pipeline_key.shader_asset_id,
            pipeline_kind,
            &pass_desc,
            pipeline_key.variant_spec(),
        );

        match resolution {
            Some(resolution) if !resolution.pipelines.is_empty() => Some(resolution),
            Some(resolution) => {
                logger::trace!(
                    "WorldMeshForward: empty pipeline for shader {:?} requested_kind {:?} resolved_kind {:?}, skipping batch",
                    pipeline_key.shader_asset_id,
                    pipeline_kind,
                    resolution.kind
                );
                None
            }
            None => {
                logger::trace!(
                    "WorldMeshForward: no pipeline for shader {:?} kind {:?}, skipping batch",
                    pipeline_key.shader_asset_id,
                    pipeline_kind
                );
                None
            }
        }
    }

    /// Resolves a ready Null fallback pipeline for a batch.
    fn resolve_null_fallback_pipeline(
        &self,
        pipeline_key: PipelineVariantKey,
    ) -> Option<MaterialPipelineResolution> {
        let registry = self.registry?;
        let pass_desc = pipeline_key.pass_desc();
        let resolution =
            registry.null_pipeline_for_variant(&pass_desc, pipeline_key.variant_spec());
        match resolution {
            Some(resolution) if !resolution.pipelines.is_empty() => Some(resolution),
            Some(_) => {
                logger::trace!(
                    "WorldMeshForward: empty Null fallback pipeline for shader {:?}, skipping batch",
                    pipeline_key.shader_asset_id
                );
                None
            }
            None => {
                logger::trace!(
                    "WorldMeshForward: Null fallback pipeline unavailable for shader {:?}, skipping batch",
                    pipeline_key.shader_asset_id
                );
                None
            }
        }
    }

    /// Resolves the embedded material bind group for an embedded pipeline stem.
    fn resolve_embedded_group1_binding(
        &self,
        item: &WorldMeshDrawItem,
        stem: &str,
    ) -> Result<MaterialGroup1Binding, EmbeddedMaterialBindError> {
        let Some(bind) = self.embedded_bind else {
            return Err(EmbeddedMaterialBindError::from(
                "embedded material bind resources unavailable",
            ));
        };

        let shader_variant_bits = self.resolve_embedded_shader_variant_bits(item, stem);
        let (bind_key, bind_group) = bind.embedded_material_bind_group_with_cache_key(
            EmbeddedMaterialBindShader {
                stem,
                shader_variant_bits,
            },
            self.uploads,
            self.store,
            &self.pools,
            item.lookup_ids,
            self.offscreen_write_target,
        )?;
        Ok(MaterialGroup1Binding::Embedded {
            bind_key,
            bind_group: bind_group.bind_group,
            uniform_dynamic_offset: bind_group.uniform_dynamic_offset,
        })
    }

    /// Resolves source shader variant bits for a possibly rerouted embedded draw.
    fn resolve_embedded_shader_variant_bits(
        &self,
        item: &WorldMeshDrawItem,
        stem: &str,
    ) -> Option<u32> {
        let batch_key = &item.batch_key;
        let source_bits = self
            .registry
            .and_then(|registry| registry.variant_bits_for_shader_asset(batch_key.shader_asset_id));
        if !stem.starts_with("billboardunlit") {
            return source_bits;
        }
        let mut bits = source_bits.unwrap_or(0);
        if let Some(source_stem) = self
            .registry
            .and_then(|registry| registry.stem_for_shader_asset(batch_key.shader_asset_id))
        {
            bits = remap_variant_bits_for_billboard(source_stem, bits);
        }
        if batch_key.uses_render_buffer_billboard {
            Some(ensure_render_buffer_billboard_variant_bits(bits))
        } else {
            source_bits.map(|_| bits)
        }
    }

    /// Emits a throttled diagnostic for embedded bind failures and the selected fallback action.
    fn log_embedded_bind_failure(
        &self,
        item: &WorldMeshDrawItem,
        stem: &str,
        error: &EmbeddedMaterialBindError,
        fallback_ready: bool,
    ) {
        let Some(occurrence) = EMBEDDED_MATERIAL_BIND_FAILURE_LOG.should_log(8, 128) else {
            return;
        };
        let action = if fallback_ready {
            "using Null fallback"
        } else {
            "skipping batch until fallback is ready"
        };
        logger::warn!(
            "WorldMeshForward: embedded material bind group failed \
             (shader_asset_id={}, material_asset_id={}, slot_property_block={:?}, \
             renderer_property_block={:?}, stem={}, occurrence={}); {}: {}",
            item.batch_key.shader_asset_id,
            item.lookup_ids.material_asset_id,
            item.lookup_ids.mesh_property_block_slot0,
            item.lookup_ids.mesh_renderer_property_block_id,
            stem,
            occurrence,
            action,
            error
        );
    }
}

/// Returns the group-1 binding kind required by a concrete raster pipeline kind.
fn required_group1_binding_kind(kind: &RasterPipelineKind) -> MaterialGroup1BindingKind {
    match kind {
        RasterPipelineKind::Null => MaterialGroup1BindingKind::Empty,
        RasterPipelineKind::EmbeddedStem(_) => MaterialGroup1BindingKind::Embedded,
    }
}

/// Returns whether a group-1 binding kind is layout-compatible with a raster pipeline kind.
fn material_group1_binding_matches_pipeline(
    binding_kind: MaterialGroup1BindingKind,
    pipeline_kind: &RasterPipelineKind,
) -> bool {
    binding_kind == required_group1_binding_kind(pipeline_kind)
}

/// Walks `draws` once and writes `(first_idx, last_idx)` runs of identical material batch keys
/// into the caller-supplied `out` buffer. `out` is cleared before filling.
fn collect_material_batch_boundaries_into(
    draws: &[WorldMeshDrawItem],
    out: &mut Vec<MaterialBatchBoundary>,
) {
    out.clear();
    let mut current_start = 0usize;
    let mut last_key = &draws[0].batch_key;
    let mut last_key_hash = draws[0].batch_key_hash;
    let mut last_renderer_property_block_id = draws[0].lookup_ids.mesh_renderer_property_block_id;
    for (idx, item) in draws.iter().enumerate().skip(1) {
        let renderer_property_block_id = item.lookup_ids.mesh_renderer_property_block_id;
        if item.batch_key_hash != last_key_hash
            || &item.batch_key != last_key
            || renderer_property_block_id != last_renderer_property_block_id
        {
            out.push((current_start, idx - 1));
            current_start = idx;
            last_key = &item.batch_key;
            last_key_hash = item.batch_key_hash;
            last_renderer_property_block_id = renderer_property_block_id;
        }
    }
    out.push((current_start, draws.len() - 1));
}

/// Collapses non-contiguous runs that resolve to the same material packet template.
///
/// The returned index list stays aligned with `boundaries`; callers clone the selected template
/// and restore each run's draw range after the expensive pipeline/bind resolution happens once.
fn collect_unique_material_batch_boundaries(
    draws: &[WorldMeshDrawItem],
    boundaries: &[MaterialBatchBoundary],
) -> (Vec<MaterialBatchBoundary>, Vec<usize>) {
    let mut unique_boundaries = Vec::with_capacity(boundaries.len());
    let mut boundary_template_indices = Vec::with_capacity(boundaries.len());
    let mut template_by_identity = HashMap::with_capacity(boundaries.len());

    for &(first, last) in boundaries {
        let item = &draws[first];
        let identity = MaterialBatchResolutionIdentity {
            batch_key_hash: item.batch_key_hash,
            batch_key: &item.batch_key,
            material_asset_id: item.lookup_ids.material_asset_id,
            property_block_slot0: item.lookup_ids.mesh_property_block_slot0,
            renderer_property_block_id: item.lookup_ids.mesh_renderer_property_block_id,
        };
        let next_template_idx = unique_boundaries.len();
        let template_idx = *template_by_identity.entry(identity).or_insert_with(|| {
            unique_boundaries.push((first, last));
            next_template_idx
        });
        boundary_template_indices.push(template_idx);
    }

    (unique_boundaries, boundary_template_indices)
}

/// Deduplicates exact pipeline requests while retaining one request index per material template.
fn collect_unique_pipeline_resolution_requests<'a>(
    draws: &'a [WorldMeshDrawItem],
    material_boundaries: &[MaterialBatchBoundary],
    pass_desc: MaterialPipelineDesc,
    shader_perm: ShaderPermutation,
    front_face_flip: bool,
) -> (Vec<PipelineResolutionRequest<'a>>, Vec<usize>) {
    let mut unique_requests = Vec::with_capacity(material_boundaries.len());
    let mut material_request_indices = Vec::with_capacity(material_boundaries.len());
    let mut request_indices = HashMap::with_capacity(material_boundaries.len());

    for &(first, _) in material_boundaries {
        let item = &draws[first];
        let mut pipeline_key = PipelineVariantKey::for_draw_item(item, pass_desc, shader_perm);
        if front_face_flip {
            pipeline_key.front_face = pipeline_key.front_face.flipped();
        }
        let request = PipelineResolutionRequest {
            requested_kind: &item.batch_key.pipeline,
            pipeline_key,
        };
        let next_request_idx = unique_requests.len();
        let request_idx = *request_indices.entry(request).or_insert_with(|| {
            unique_requests.push(request);
            next_request_idx
        });
        material_request_indices.push(request_idx);
    }

    (unique_requests, material_request_indices)
}

/// Resolves one local cache slot, including unavailable pipeline results.
fn resolve_pipeline_once(
    slot: &OnceLock<Option<MaterialPipelineResolution>>,
    resolve: impl FnOnce() -> Option<MaterialPipelineResolution>,
) -> Option<&MaterialPipelineResolution> {
    slot.get_or_init(resolve).as_ref()
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::sync::Arc;

    use super::*;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn base_desc() -> MaterialPipelineDesc {
        MaterialPipelineDesc {
            surface_format: wgpu::TextureFormat::Rgba16Float,
            depth_stencil_format: Some(wgpu::TextureFormat::Depth24PlusStencil8),
            sample_count: 4,
            multiview_mask: NonZeroU32::new(3),
        }
    }

    fn key_for() -> PipelineVariantKey {
        PipelineVariantKey::new(PipelineVariantKeyInput {
            pass_desc: base_desc(),
            shader_perm: ShaderPermutation(1),
            shader_specialization: MaterialShaderSpecializationKey::disabled(),
            shader_asset_id: 42,
            blend_mode: MaterialBlendMode::Opaque,
            render_state: MaterialRenderState::default(),
            pass_routing: MaterialPassRouting::default(),
            front_face: RasterFrontFace::CounterClockwise,
            primitive_topology: RasterPrimitiveTopology::TriangleList,
        })
    }

    /// Builds an embedded pipeline kind for packet-selection tests.
    fn embedded_pipeline(stem: &'static str) -> RasterPipelineKind {
        RasterPipelineKind::EmbeddedStem(Arc::from(stem))
    }

    #[test]
    fn pipeline_key_preserves_regular_sample_count() {
        let key = key_for();
        assert_eq!(key.sample_count, 4);
        assert_eq!(key.pass_desc().sample_count, 4);
    }

    #[test]
    fn pipeline_key_preserves_grab_pass_sample_count() {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 42,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 7,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.shader_asset_id = 42;
        item.batch_key.blend_mode = MaterialBlendMode::Opaque;
        item.batch_key.front_face = RasterFrontFace::CounterClockwise;
        item.batch_key.embedded_uses_scene_color_snapshot = true;

        let key = PipelineVariantKey::for_draw_item(&item, base_desc(), ShaderPermutation(1));
        assert_eq!(key.sample_count, 4);
        assert_eq!(key.pass_desc().sample_count, 4);
        assert_eq!(key.surface_format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(
            key.depth_stencil_format,
            Some(wgpu::TextureFormat::Depth24PlusStencil8)
        );
        assert_eq!(key.multiview_mask, NonZeroU32::new(3));
    }

    #[test]
    fn pipeline_key_changes_when_front_face_changes() {
        let mut a = key_for();
        let mut b = key_for();
        a.front_face = RasterFrontFace::Clockwise;
        b.front_face = RasterFrontFace::CounterClockwise;
        assert_ne!(a, b);
    }

    /// Null pipelines require the shared empty material bind group.
    #[test]
    fn null_pipeline_requires_empty_group1_binding() {
        assert_eq!(
            required_group1_binding_kind(&RasterPipelineKind::Null),
            MaterialGroup1BindingKind::Empty
        );
        assert!(material_group1_binding_matches_pipeline(
            MaterialGroup1BindingKind::Empty,
            &RasterPipelineKind::Null
        ));
    }

    /// Embedded pipelines require the reflected embedded material bind group.
    #[test]
    fn embedded_pipeline_requires_embedded_group1_binding() {
        let kind = embedded_pipeline("xstoon2.0_default");
        assert_eq!(
            required_group1_binding_kind(&kind),
            MaterialGroup1BindingKind::Embedded
        );
        assert!(material_group1_binding_matches_pipeline(
            MaterialGroup1BindingKind::Embedded,
            &kind
        ));
    }

    /// The empty material bind group is never layout-compatible with embedded pipelines.
    #[test]
    fn empty_group1_binding_does_not_match_embedded_pipeline() {
        let kind = embedded_pipeline("xstoon2.0_default");
        assert!(!material_group1_binding_matches_pipeline(
            MaterialGroup1BindingKind::Empty,
            &kind
        ));
    }

    #[test]
    fn render_buffer_billboard_draws_split_material_batch_boundaries() {
        let mut ordinary = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 42,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 7,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        ordinary.batch_key.pipeline = embedded_pipeline("billboardunlit_default");
        ordinary.batch_key.shader_asset_id = 42;

        let mut render_buffer = ordinary.clone();
        render_buffer.mesh_asset_id = crate::particles::billboard_render_buffer_mesh_asset_id(3)
            .expect("valid render-buffer billboard id");
        render_buffer.batch_key.uses_render_buffer_billboard = true;

        let draws = vec![ordinary, render_buffer];
        let mut boundaries = Vec::new();

        collect_material_batch_boundaries_into(&draws, &mut boundaries);

        assert_eq!(boundaries, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn non_contiguous_identical_material_runs_share_one_resolution_template() {
        let first = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 42,
            property_block: Some(420),
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 7,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        let middle = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 99,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 8,
            node_id: 2,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: true,
        });
        let mut last = first.clone();
        last.node_id = 3;
        last.collect_order = 2;
        let draws = vec![first, middle, last];
        let mut boundaries = Vec::new();
        collect_material_batch_boundaries_into(&draws, &mut boundaries);

        let (unique, template_indices) =
            collect_unique_material_batch_boundaries(&draws, &boundaries);

        assert_eq!(boundaries, vec![(0, 0), (1, 1), (2, 2)]);
        assert_eq!(unique, vec![(0, 0), (1, 1)]);
        assert_eq!(template_indices, vec![0, 1, 0]);
    }

    #[test]
    fn material_resolution_dedup_preserves_renderer_property_block_identity() {
        let first = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 42,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 7,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        let middle = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 99,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 8,
            node_id: 2,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: true,
        });
        let mut last = first.clone();
        last.node_id = 3;
        last.collect_order = 2;
        last.lookup_ids.mesh_renderer_property_block_id = Some(700);
        let draws = vec![first, middle, last];
        let mut boundaries = Vec::new();
        collect_material_batch_boundaries_into(&draws, &mut boundaries);

        let (unique, template_indices) =
            collect_unique_material_batch_boundaries(&draws, &boundaries);

        assert_eq!(unique, boundaries);
        assert_eq!(template_indices, vec![0, 1, 2]);
    }

    #[test]
    fn adjacent_renderer_property_blocks_split_material_packets() {
        let first = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 42,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 7,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        let mut second = first.clone();
        second.node_id = 2;
        second.collect_order = 1;
        second.lookup_ids.mesh_renderer_property_block_id = Some(700);
        let draws = vec![first, second];
        let mut boundaries = Vec::new();

        collect_material_batch_boundaries_into(&draws, &mut boundaries);
        let (unique, template_indices) =
            collect_unique_material_batch_boundaries(&draws, &boundaries);

        assert_eq!(boundaries, vec![(0, 0), (1, 1)]);
        assert_eq!(unique, boundaries);
        assert_eq!(template_indices, vec![0, 1]);
    }

    #[test]
    fn distinct_material_templates_share_one_pipeline_resolution_request() {
        let first = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 42,
            property_block: Some(420),
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 7,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        let second = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 99,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 8,
            node_id: 2,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: true,
        });
        let draws = vec![first, second];
        let mut boundaries = Vec::new();
        collect_material_batch_boundaries_into(&draws, &mut boundaries);
        let (material_templates, _) = collect_unique_material_batch_boundaries(&draws, &boundaries);

        let (requests, request_indices) = collect_unique_pipeline_resolution_requests(
            &draws,
            &material_templates,
            base_desc(),
            ShaderPermutation(1),
            true,
        );

        assert_eq!(material_templates.len(), 2);
        assert_eq!(requests.len(), 1);
        assert_eq!(request_indices, vec![0, 0]);
        assert_eq!(
            requests[0].pipeline_key.front_face,
            RasterFrontFace::CounterClockwise
        );
    }

    #[test]
    fn pipeline_resolution_request_keeps_kind_and_shader_id_exact() {
        let first = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        let mut different_kind = first.clone();
        different_kind.batch_key.pipeline = embedded_pipeline("unlit_default");
        different_kind.lookup_ids.material_asset_id = 2;
        let mut different_shader_id = first.clone();
        different_shader_id.batch_key.shader_asset_id = 77;
        different_shader_id.lookup_ids.material_asset_id = 3;
        let draws = vec![first, different_kind, different_shader_id];
        let boundaries = vec![(0, 0), (1, 1), (2, 2)];

        let (requests, request_indices) = collect_unique_pipeline_resolution_requests(
            &draws,
            &boundaries,
            base_desc(),
            ShaderPermutation(1),
            false,
        );

        assert_eq!(requests.len(), 3);
        assert_eq!(request_indices, vec![0, 1, 2]);
    }

    #[test]
    fn unavailable_pipeline_resolution_is_probed_once_per_local_slot() {
        let slot = OnceLock::new();
        let probe_count = std::cell::Cell::new(0usize);

        for _ in 0..3 {
            let resolution = resolve_pipeline_once(&slot, || {
                probe_count.set(probe_count.get() + 1);
                None
            });
            assert!(resolution.is_none());
        }

        assert_eq!(probe_count.get(), 1);
    }

    /// A draw batch snapshot that stayed Null still requires empty group 1 even if routing changes.
    #[test]
    fn stale_draw_batch_pipeline_requires_empty_group1_even_if_current_route_is_embedded() {
        let current_router_route = embedded_pipeline("xstoon2.0_default");
        let stale_draw_batch_pipeline = RasterPipelineKind::Null;

        assert_eq!(
            required_group1_binding_kind(&stale_draw_batch_pipeline),
            MaterialGroup1BindingKind::Empty
        );
        assert_ne!(stale_draw_batch_pipeline, current_router_route);
    }
}
