//! Main forward pass: prefill depth, clear color, draw scene meshes, MSAA resolve.
//!
//! ## Pass graph structure
//!
//! World-mesh forward rendering is split across these graph passes:
//!
//! 1. Backend frame planning prepares sorted draws, packs per-draw VP/model uniforms
//!    (rayon-parallel above the existing threshold), uploads the per-draw slab and frame
//!    uniforms via the graph upload sink, and stores the prepared state in
//!    [`WorldMeshForwardPlanSlot`] before any graph pass records.
//! 2. [`WorldMeshForwardDepthPrepass`] -- **[`RasterPass`]** that clears and fills depth for
//!    conservative opaque draws.
//! 3. [`WorldMeshForwardOpaquePass`] -- **[`RasterPass`]** that clears HDR color, loads the
//!    prepass depth attachment, records pre-skybox opaque and alpha-test phases, then records the
//!    skybox/background.
//! 4. [`WorldMeshForwardGtaoDepthResolvePass`] -- optional **[`EncoderPass`]** that resolves
//!    current MSAA depth before GTAO samples the single-sample frame depth.
//! 5. [`WorldMeshForwardNormalPass`] -- optional **[`RasterPass`]** that writes smooth
//!    view-space vertex normals for GTAO, using the opaque depth buffer as an equality mask.
//! 6. [`WorldMeshDepthSnapshotPass`] -- **[`EncoderPass`]** that resolves MSAA depth (when active)
//!    and copies single-sample depth into the scene-depth snapshot for depth-snapshot materials.
//! 7. [`WorldMeshForwardIntersectPass`] -- **[`RasterPass`]** that draws nontransparent
//!    intersection groups.
//! 8. [`WorldMeshForwardTransparentSequencePass`] -- **[`EncoderPass`]** that draws the sorted
//!    transparent phase, including transparent intersection materials, resolving/copying a fresh
//!    scene-color snapshot immediately before each grab-pass phase group.
//! 9. [`WorldMeshForwardDepthResolvePass`] -- **[`EncoderPass`]** that resolves the final MSAA
//!    depth into the single-sample frame depth used by Hi-Z.
//!
//! ## VR stereo world draws
//!
//! OpenXR per-eye view-projection maps **stage** space to clip. For non-overlay draws with
//! [`crate::camera::StereoViewMatrices`], identity is used instead of the host
//! `view_transform` world-to-camera to avoid mixing stage with the host rig. Overlays keep
//! `view` for orthographic / UI alignment. Matrix composition lives in [`vp`].

mod attachments;
mod camera;
mod color_resolve;
mod color_snapshot;
mod current_view_textures;
mod depth_prepass;
mod depth_resolve;
mod depth_snapshot;
mod encode;
mod frame_uniforms;
mod material_batch;
mod material_resolve;
mod normal_pass;
mod prepare;
mod raster_recording;
mod skybox;
mod slab;
mod state;
mod transparent_sequence;
mod vp;

pub use depth_prepass::{WorldMeshForwardDepthPrepass, WorldMeshForwardDepthPrepassGraphResources};
pub(crate) use material_batch::{MaterialBatchBoundary, MaterialBatchPacket, MaterialDrawResolver};
pub(crate) use normal_pass::GTAO_VIEW_NORMAL_FORMAT;
pub use normal_pass::{WorldMeshForwardNormalGraphResources, WorldMeshForwardNormalPass};
pub(crate) use prepare::{WorldMeshForwardPrepareContext, prepare_world_mesh_forward_frame};
pub(crate) use skybox::SkyboxRenderer as WorldMeshForwardSkyboxRenderer;
pub(crate) use state::{
    PreparedWorldMeshForwardFrame, WorldMeshForwardPipelineState, WorldMeshForwardPlanSlot,
};
pub use transparent_sequence::WorldMeshForwardTransparentSequencePass;

use std::num::NonZeroU32;

use crate::graph_inputs::{MsaaViewsSlot, PerViewFramePlanSlot};
use crate::render_graph::context::{EncoderPassCtx, RasterPassCtx};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::stereo_mask_or_template;
use crate::render_graph::pass::{DepthAttachmentTemplate, RenderPassTemplate};
use crate::render_graph::pass::{EncoderPass, PassBuilder, RasterPass};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess, TextureAccess,
    TextureHandle,
};
use crate::world_mesh::{InstancePlan, WorldMeshPhase};

use attachments::declare_forward_color_depth_attachments;
use depth_resolve::encode_msaa_depth_resolve_after_clear_only;
use depth_snapshot::{
    EncodeCtx as DepthSnapshotEncodeCtx, encode_world_mesh_forward_depth_snapshot,
};
use raster_recording::{
    record_world_mesh_forward_intersection_graph_raster,
    record_world_mesh_forward_opaque_graph_raster, stencil_load_ops,
};
use skybox::record_prepared_skybox;

use crate::gpu_pools::{
    CubemapPool, MeshPool, RenderTexturePool, Texture3dPool, TexturePool, VideoTexturePool,
};
use crate::materials::MaterialSystem;
use crate::materials::embedded::EmbeddedTexturePools;
use crate::mesh_deform::GpuSkinCache;

/// Graph-managed opaque/clear subpass for world-mesh forward rendering.
#[derive(Debug)]
pub struct WorldMeshForwardOpaquePass {
    resources: WorldMeshForwardGraphResources,
}

/// Copies the resolved forward depth into the scene-depth snapshot for depth-snapshot materials.
#[derive(Debug)]
pub struct WorldMeshDepthSnapshotPass {
    resources: WorldMeshForwardGraphResources,
}

/// Draws nontransparent intersection material groups after the scene-depth snapshot is available.
#[derive(Debug)]
pub struct WorldMeshForwardIntersectPass {
    resources: WorldMeshForwardGraphResources,
}

/// Resolves the final MSAA forward depth into the single-sample frame depth target.
#[derive(Debug)]
pub struct WorldMeshForwardDepthResolvePass {
    resources: WorldMeshForwardGraphResources,
}

/// Resolves MSAA forward depth before GTAO samples the single-sample frame depth target.
#[derive(Debug)]
pub struct WorldMeshForwardGtaoDepthResolvePass {
    resources: WorldMeshForwardGraphResources,
}

/// MSAA-only transient graph resources shared by world-mesh forward passes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ForwardMsaaResources {
    /// Multisampled HDR scene-color transient.
    pub scene_color_hdr: TextureHandle,
    /// Graph-owned forward depth target.
    pub depth: TextureHandle,
    /// Graph-owned R32Float intermediate for resolving MSAA depth.
    pub depth_r32: TextureHandle,
}

/// Graph resources shared by world-mesh forward prepare/opaque/intersect/resolve passes.
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshForwardGraphResources {
    /// Single-sample HDR scene-color transient (forward resolve target).
    pub scene_color_hdr: TextureHandle,
    /// Imported frame depth target.
    pub depth: ImportedTextureHandle,
    /// Multisampled forward transients when MSAA is active.
    pub msaa: Option<ForwardMsaaResources>,
    /// Imported cluster light-count storage buffer.
    pub cluster_light_counts: ImportedBufferHandle,
    /// Imported cluster light-index storage buffer.
    pub cluster_light_indices: ImportedBufferHandle,
    /// Imported light storage buffer.
    pub lights: ImportedBufferHandle,
    /// Imported per-draw storage slab.
    pub per_draw_slab: ImportedBufferHandle,
    /// Imported frame uniform buffer.
    pub frame_uniforms: ImportedBufferHandle,
}

impl WorldMeshForwardGraphResources {
    /// Returns whether this graph variant records multisampled forward attachments.
    pub fn msaa_enabled(self) -> bool {
        self.msaa.is_some()
    }
}

/// Disjoint borrows required by world-mesh forward encoding.
pub(crate) struct WorldMeshForwardEncodeRefs<'a> {
    /// Material registry, embedded binds, and property store.
    pub(crate) materials: &'a MaterialSystem,
    /// Resident mesh pool.
    pub(crate) mesh_pool: &'a MeshPool,
    /// Resident 2D texture pool.
    pub(crate) texture_pool: &'a TexturePool,
    /// Resident 3D texture pool.
    pub(crate) texture3d_pool: &'a Texture3dPool,
    /// Resident cubemap pool.
    pub(crate) cubemap_pool: &'a CubemapPool,
    /// Host render texture pool.
    pub(crate) render_texture_pool: &'a RenderTexturePool,
    /// Resident video texture pool.
    pub(crate) video_texture_pool: &'a VideoTexturePool,
    /// Arena-backed deformed positions and normals keyed by renderable.
    pub(crate) skin_cache: Option<&'a GpuSkinCache>,
}

impl<'a> WorldMeshForwardEncodeRefs<'a> {
    /// Builds encode refs from a graph pass frame's disjoint system slices.
    pub(crate) fn from_frame(frame: &crate::graph_inputs::GraphPassFrame<'a>) -> Self {
        Self {
            materials: frame.shared.materials,
            mesh_pool: frame.shared.asset_resources.mesh_pool(),
            texture_pool: frame.shared.asset_resources.texture_pool(),
            texture3d_pool: frame.shared.asset_resources.texture3d_pool(),
            cubemap_pool: frame.shared.asset_resources.cubemap_pool(),
            render_texture_pool: frame.shared.asset_resources.render_texture_pool(),
            video_texture_pool: frame.shared.asset_resources.video_texture_pool(),
            skin_cache: frame.shared.skin_cache,
        }
    }

    /// Mesh pool for draw recording after any required stream uploads were pre-warmed.
    pub(crate) fn mesh_pool(&self) -> &MeshPool {
        self.mesh_pool
    }

    /// Pool views for embedded `@group(1)` texture resolution.
    pub(crate) fn embedded_texture_pools(&self) -> EmbeddedTexturePools<'_> {
        EmbeddedTexturePools {
            texture: self.texture_pool,
            texture3d: self.texture3d_pool,
            cubemap: self.cubemap_pool,
            render_texture: self.render_texture_pool,
            video_texture: self.video_texture_pool,
        }
    }
}

/// Returns whether the intersection raster tail has view-local work to record.
fn forward_intersection_raster_needed(opaque_recorded: bool, plan: &InstancePlan) -> bool {
    opaque_recorded && !plan.phase_is_empty(WorldMeshPhase::Intersection)
}

impl WorldMeshForwardOpaquePass {
    /// Creates a graph-managed opaque world mesh forward pass instance.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
    }
}

impl WorldMeshDepthSnapshotPass {
    /// Creates a world mesh depth snapshot pass instance.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
    }
}

impl WorldMeshForwardIntersectPass {
    /// Creates a world mesh intersection raster pass instance.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
    }
}

impl WorldMeshForwardDepthResolvePass {
    /// Creates a world mesh final depth-resolve pass instance.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
    }
}

impl WorldMeshForwardGtaoDepthResolvePass {
    /// Creates a pre-GTAO depth-resolve pass instance.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
    }
}

/// Declares the resources touched by the manual MSAA depth resolve encoder path.
fn declare_msaa_depth_resolve_accesses(
    b: &mut PassBuilder<'_>,
    resources: WorldMeshForwardGraphResources,
) {
    b.read_optional_blackboard::<MsaaViewsSlot>();
    let Some(msaa) = resources.msaa else {
        return;
    };
    b.read_texture(
        msaa.depth,
        TextureAccess::Sampled {
            stages: wgpu::ShaderStages::COMPUTE,
        },
    );
    b.write_texture(
        msaa.depth_r32,
        TextureAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE,
            access: StorageAccess::WriteOnly,
        },
    );
    b.read_texture(
        msaa.depth_r32,
        TextureAccess::Sampled {
            stages: wgpu::ShaderStages::FRAGMENT,
        },
    );
    b.import_texture(
        resources.depth,
        TextureAccess::DepthAttachment {
            depth: wgpu::Operations {
                load: wgpu::LoadOp::Clear(crate::gpu::MAIN_FORWARD_DEPTH_CLEAR),
                store: wgpu::StoreOp::Store,
            },
            stencil: None,
        },
    );
}

/// Encodes the shared MSAA depth resolve and records command stats for the current view.
fn record_msaa_depth_resolve(ctx: &mut EncoderPassCtx<'_, '_, '_>) -> bool {
    let frame = &mut *ctx.pass_frame;
    let msaa_views = ctx.blackboard.get::<MsaaViewsSlot>();
    let msaa_depth_resolve = frame.view.msaa_depth_resolve.clone();
    let resolved = encode_msaa_depth_resolve_after_clear_only(
        ctx.device,
        ctx.encoder,
        frame,
        msaa_views,
        msaa_depth_resolve.as_deref(),
        ctx.profiler,
    );
    if let Some(stats) = ctx
        .blackboard
        .get_mut::<crate::render_graph::blackboard::GraphCommandStatsSlot>()
    {
        stats.record_resolve_result(resolved);
    }
    resolved
}

/// Marks world-mesh depth fresh when a resolve pass updated the imported depth target.
fn mark_depth_resolved(
    blackboard: &mut crate::render_graph::blackboard::Blackboard,
    resolved: bool,
) {
    if resolved && let Some(prepared) = blackboard.get_mut::<WorldMeshForwardPlanSlot>() {
        prepared.depth_freshness.mark_resolved();
    }
}

pub(in crate::passes::world_mesh_forward) fn declare_forward_draw_reads(
    b: &mut PassBuilder<'_>,
    resources: WorldMeshForwardGraphResources,
) {
    b.import_buffer(
        resources.cluster_light_counts,
        BufferAccess::Storage {
            stages: wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    );
    b.import_buffer(
        resources.cluster_light_indices,
        BufferAccess::Storage {
            stages: wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    );
    b.import_buffer(
        resources.lights,
        BufferAccess::Storage {
            stages: wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    );
    b.import_buffer(
        resources.per_draw_slab,
        BufferAccess::Storage {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    );
    b.import_buffer(
        resources.frame_uniforms,
        BufferAccess::Uniform {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            dynamic_offset: false,
        },
    );
}

impl RasterPass for WorldMeshForwardOpaquePass {
    fn name(&self) -> &str {
        "WorldMeshForwardOpaque"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_blackboard::<PerViewFramePlanSlot>();
        b.read_optional_blackboard::<WorldMeshForwardPlanSlot>();
        b.write_blackboard::<WorldMeshForwardPlanSlot>();
        {
            let mut r = b.raster();
            // No `resolve_target` here: when MSAA is active, the multisampled buffer is preserved
            // and resolved by the transparent sequence using the Karis HDR-aware bracket.
            let color_ops = wgpu::Operations {
                load: wgpu::LoadOp::Clear(crate::present::SWAPCHAIN_CLEAR_COLOR),
                store: wgpu::StoreOp::Store,
            };
            let depth_ops = wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            };
            declare_forward_color_depth_attachments(&mut r, self.resources, color_ops, depth_ops);
        };
        declare_forward_draw_reads(b, self.resources);
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        let use_multiview = ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .is_some_and(|prepared| prepared.pipeline.use_multiview);
        stereo_mask_or_template(use_multiview, template.multiview_mask)
    }

    fn stencil_ops_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        depth: &DepthAttachmentTemplate,
    ) -> Option<wgpu::Operations<u32>> {
        let Some(format) = ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .and_then(|prepared| prepared.pipeline.pass_desc.depth_stencil_format)
        else {
            return depth.stencil;
        };
        format.has_stencil_aspect().then_some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(0),
            store: wgpu::StoreOp::Store,
        })
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::opaque_record");
        let frame = &mut *ctx.pass_frame;

        let Some(mut prepared) = ctx.blackboard.take::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        let pre_skybox_recorded = record_world_mesh_forward_opaque_graph_raster(
            rpass,
            ctx.device,
            frame,
            ctx.blackboard,
            &prepared,
        );
        let skybox_recorded = prepared
            .skybox
            .as_ref()
            .is_none_or(|skybox| record_prepared_skybox(rpass, frame, ctx.blackboard, skybox));
        prepared.opaque_recorded = pre_skybox_recorded && skybox_recorded;
        ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }
}

impl EncoderPass for WorldMeshDepthSnapshotPass {
    fn name(&self) -> &str {
        "WorldMeshDepthSnapshot"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.encoder();
        b.read_optional_blackboard::<WorldMeshForwardPlanSlot>();
        b.write_blackboard::<WorldMeshForwardPlanSlot>();
        declare_msaa_depth_resolve_accesses(b, self.resources);
        b.import_texture(self.resources.depth, TextureAccess::CopySrc);
        Ok(())
    }

    fn should_record(&self, ctx: &EncoderPassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
        Ok(ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .is_some_and(|prepared| prepared.helper_needs.depth_snapshot))
    }

    fn record(&self, ctx: &mut EncoderPassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::depth_snapshot_record");
        let frame = &mut *ctx.pass_frame;
        let Some(mut prepared) = ctx.blackboard.take::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        let msaa_views = ctx.blackboard.get::<MsaaViewsSlot>();
        let msaa_depth_resolve = frame.view.msaa_depth_resolve.clone();
        let resolve_msaa_depth =
            frame.view.sample_count > 1 && !prepared.depth_freshness.is_current();
        let result = encode_world_mesh_forward_depth_snapshot(DepthSnapshotEncodeCtx {
            device: ctx.device,
            encoder: ctx.encoder,
            frame,
            prepared: &prepared,
            msaa_views,
            msaa_depth_resolve: msaa_depth_resolve.as_deref(),
            profiler: ctx.profiler,
            resolve_msaa_depth,
        });
        if result.resolved_depth {
            prepared.depth_freshness.mark_resolved();
        }
        if result.copied {
            prepared.depth_snapshot_recorded = true;
        }
        if let Some(stats) = ctx
            .blackboard
            .get_mut::<crate::render_graph::blackboard::GraphCommandStatsSlot>()
        {
            if frame.view.sample_count > 1 {
                stats.record_resolve_result(result.resolved_depth);
            }
            stats.record_copy_result(result.copied);
        }
        ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }
}

impl RasterPass for WorldMeshForwardIntersectPass {
    fn name(&self) -> &str {
        "WorldMeshForwardIntersect"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_blackboard::<PerViewFramePlanSlot>();
        b.read_optional_blackboard::<WorldMeshForwardPlanSlot>();
        b.write_blackboard::<WorldMeshForwardPlanSlot>();
        {
            let mut r = b.raster();
            let color_ops = wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            };
            let depth_ops = wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            };
            declare_forward_color_depth_attachments(&mut r, self.resources, color_ops, depth_ops);
        };
        declare_forward_draw_reads(b, self.resources);
        Ok(())
    }

    fn should_record(&self, ctx: &RasterPassCtx<'_, '_>) -> Result<bool, RenderPassError> {
        Ok(ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .is_some_and(|prepared| {
                forward_intersection_raster_needed(prepared.opaque_recorded, &prepared.plan)
            }))
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        let use_multiview = ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .is_some_and(|prepared| prepared.pipeline.use_multiview);
        stereo_mask_or_template(use_multiview, template.multiview_mask)
    }

    fn stencil_ops_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        depth: &DepthAttachmentTemplate,
    ) -> Option<wgpu::Operations<u32>> {
        let Some(format) = ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .and_then(|prepared| prepared.pipeline.pass_desc.depth_stencil_format)
        else {
            return depth.stencil;
        };
        stencil_load_ops(Some(format))
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::intersect_record");
        let frame = &mut *ctx.pass_frame;

        let Some(mut prepared) = ctx.blackboard.take::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        let recorded = if prepared.opaque_recorded {
            record_world_mesh_forward_intersection_graph_raster(
                rpass,
                ctx.device,
                frame,
                ctx.blackboard,
                &prepared,
            )
        } else {
            false
        };
        if recorded {
            prepared.tail_raster_recorded = true;
            if frame.view.sample_count > 1 {
                prepared.depth_freshness.mark_dirty();
            }
        }
        ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }
}

impl EncoderPass for WorldMeshForwardGtaoDepthResolvePass {
    fn name(&self) -> &str {
        "WorldMeshForwardGtaoDepthResolve"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.encoder();
        b.read_optional_blackboard::<WorldMeshForwardPlanSlot>();
        b.write_blackboard::<WorldMeshForwardPlanSlot>();
        declare_msaa_depth_resolve_accesses(b, self.resources);
        Ok(())
    }

    fn should_record(&self, ctx: &EncoderPassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
        Ok(ctx.pass_frame.view.sample_count > 1
            && ctx
                .blackboard
                .get::<WorldMeshForwardPlanSlot>()
                .is_some_and(|prepared| {
                    prepared.opaque_recorded && !prepared.depth_freshness.is_current()
                }))
    }

    fn record(&self, ctx: &mut EncoderPassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::gtao_depth_resolve_record");
        let resolved = record_msaa_depth_resolve(ctx);
        mark_depth_resolved(ctx.blackboard, resolved);
        Ok(())
    }
}

impl EncoderPass for WorldMeshForwardDepthResolvePass {
    fn name(&self) -> &str {
        "WorldMeshForwardDepthResolve"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.encoder();
        b.read_optional_blackboard::<WorldMeshForwardPlanSlot>();
        b.write_blackboard::<WorldMeshForwardPlanSlot>();
        declare_msaa_depth_resolve_accesses(b, self.resources);
        Ok(())
    }

    fn should_record(&self, ctx: &EncoderPassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
        Ok(ctx.pass_frame.view.sample_count > 1
            && ctx
                .blackboard
                .get::<WorldMeshForwardPlanSlot>()
                .is_some_and(|prepared| {
                    prepared.opaque_recorded && !prepared.depth_freshness.is_current()
                }))
    }

    fn record(&self, ctx: &mut EncoderPassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::depth_resolve_record");
        let resolved = record_msaa_depth_resolve(ctx);
        mark_depth_resolved(ctx.blackboard, resolved);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        InstancePlan, forward_intersection_raster_needed,
        transparent_sequence::forward_transparent_sequence_needed,
    };
    use crate::world_mesh::{DrawGroup, WorldMeshPhase};

    /// Empty helper plans skip their raster pass even after opaque work records successfully.
    #[test]
    fn helper_raster_needed_requires_matching_groups() {
        let empty = InstancePlan::default();
        assert!(!forward_intersection_raster_needed(true, &empty));
        assert!(!forward_transparent_sequence_needed(true, &empty));

        let mut intersect = InstancePlan::default();
        intersect
            .phase_mut(WorldMeshPhase::Intersection)
            .push(DrawGroup {
                representative_draw_idx: 0,
                instance_range: 0..1,
                material_packet_idx: 0,
            });
        assert!(forward_intersection_raster_needed(true, &intersect));
        assert!(!forward_intersection_raster_needed(false, &intersect));

        let mut transparent = InstancePlan::default();
        transparent
            .phase_mut(WorldMeshPhase::Transparent)
            .push(DrawGroup {
                representative_draw_idx: 0,
                instance_range: 0..1,
                material_packet_idx: 0,
            });
        assert!(forward_transparent_sequence_needed(true, &transparent));
        assert!(!forward_transparent_sequence_needed(false, &transparent));
    }
}
