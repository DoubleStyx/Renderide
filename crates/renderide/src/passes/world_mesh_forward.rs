//! Main forward pass: clear color + depth, draw scene meshes, MSAA resolve.
//!
//! ## Pass graph structure
//!
//! World-mesh forward rendering is split across these graph passes:
//!
//! 1. [`WorldMeshForwardPreparePass`] -- **[`CallbackPass`]** that collects + sorts draws, packs
//!    per-draw VP/model uniforms (rayon-parallel above the existing threshold), and uploads the
//!    per-draw slab and frame uniforms via `Queue::write_buffer`. Stores the prepared state in
//!    [`WorldMeshForwardPlanSlot`] for downstream passes.
//! 2. [`WorldMeshForwardOpaquePass`] -- **[`RasterPass`]** that opens the HDR color + depth
//!    attachments with `LoadOp::Clear` and records opaque draws.
//! 3. [`WorldMeshDepthSnapshotPass`] -- **[`ComputePass`]** that resolves MSAA depth (when active)
//!    and copies single-sample depth into the scene-depth snapshot for intersection materials.
//! 4. [`WorldMeshForwardIntersectPass`] -- **[`RasterPass`]** that draws intersection materials.
//! 5. [`WorldMeshForwardColorResolvePass`] -- optional **[`RasterPass`]** that resolves MSAA
//!    scene color before the grab-pass snapshot.
//! 6. [`WorldMeshColorSnapshotPass`] -- **[`ComputePass`]** that copies resolved HDR color into
//!    the grab-pass scene-color snapshot.
//! 7. [`WorldMeshForwardTransparentPass`] -- **[`RasterPass`]** that draws grab-pass transparent
//!    materials into the active frame-sampled HDR target.
//! 8. [`WorldMeshForwardColorResolvePass`] -- optional final **[`RasterPass`]** that resolves
//!    MSAA scene color after grab-pass transparent draws.
//! 9. [`WorldMeshForwardDepthResolvePass`] -- **[`ComputePass`]** that resolves the final MSAA
//!    depth into the single-sample frame depth used by Hi-Z.
//!
//! ## VR stereo world draws
//!
//! OpenXR per-eye view-projection maps **stage** space to clip. For non-overlay draws with
//! [`crate::camera::StereoViewMatrices`], identity is used instead of the host
//! `view_transform` world-to-camera to avoid mixing stage with the host rig. Overlays keep
//! `view` for orthographic / UI alignment. Matrix composition lives in [`vp`].

mod color_resolve;
mod current_view_textures;
mod encode;
mod execute_helpers;
mod material_batch;
mod skybox;
mod state;
mod vp;

pub use color_resolve::{
    WorldMeshForwardColorResolveGraphResources, WorldMeshForwardColorResolvePass,
};
pub(crate) use material_batch::{MaterialBatchPacket, MaterialDrawResolver, PipelineVariantKey};
pub(crate) use state::{
    PrefetchedWorldMeshDrawsSlot, PreparedWorldMeshForwardFrame, WorldMeshForwardPipelineState,
    WorldMeshForwardPlanSlot,
};

use std::num::NonZeroU32;

use crate::render_graph::compiled::{DepthAttachmentTemplate, RenderPassTemplate};
use crate::render_graph::context::{CallbackCtx, ComputePassCtx, RasterPassCtx};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::MsaaViewsSlot;
use crate::render_graph::gpu_cache::stereo_mask_or_template;
use crate::render_graph::pass::{CallbackPass, ComputePass, PassBuilder, RasterPass};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess, TextureAccess,
    TextureHandle,
};
use crate::world_mesh::InstancePlan;

use execute_helpers::{
    encode_msaa_depth_resolve_after_clear_only, encode_world_mesh_forward_color_snapshot,
    encode_world_mesh_forward_depth_snapshot, prepare_world_mesh_forward_frame,
    record_world_mesh_forward_intersection_graph_raster,
    record_world_mesh_forward_opaque_graph_raster,
    record_world_mesh_forward_transparent_graph_raster, stencil_load_ops,
};
use skybox::{SkyboxRenderer, record_prepared_skybox};

/// Prepares sorted world-mesh forward draw state for subsequent graph nodes.
///
/// The pass is a [`CallbackPass`] (no encoder); it records deferred uploads and stores one
/// [`PreparedWorldMeshForwardFrame`] in the per-view blackboard via
/// [`WorldMeshForwardPlanSlot`].
#[derive(Debug, Default)]
pub struct WorldMeshForwardPreparePass {
    skybox: SkyboxRenderer,
}

/// Graph-managed opaque/clear subpass for world-mesh forward rendering.
#[derive(Debug)]
pub struct WorldMeshForwardOpaquePass {
    resources: WorldMeshForwardGraphResources,
}

/// Copies the resolved forward depth into the scene-depth snapshot for intersection materials.
#[derive(Debug)]
pub struct WorldMeshDepthSnapshotPass {
    resources: WorldMeshForwardGraphResources,
}

/// Draws intersection materials and resolves forward color when MSAA is active.
#[derive(Debug)]
pub struct WorldMeshForwardIntersectPass {
    resources: WorldMeshForwardGraphResources,
}

/// Copies the resolved forward color into the scene-color snapshot for grab-pass materials.
#[derive(Debug)]
pub struct WorldMeshColorSnapshotPass {
    resources: WorldMeshForwardGraphResources,
}

/// Draws grab-pass transparent materials after the scene-color snapshot is available.
#[derive(Debug)]
pub struct WorldMeshForwardTransparentPass {
    resources: WorldMeshForwardGraphResources,
}

/// Resolves the final MSAA forward depth into the single-sample frame depth target.
#[derive(Debug)]
pub struct WorldMeshForwardDepthResolvePass {
    resources: WorldMeshForwardGraphResources,
}

/// Graph resources shared by world-mesh forward prepare/opaque/intersect/resolve passes.
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshForwardGraphResources {
    /// Single-sample HDR scene-color transient (forward resolve target).
    pub scene_color_hdr: TextureHandle,
    /// Multisampled HDR scene-color transient when MSAA is active.
    pub scene_color_hdr_msaa: TextureHandle,
    /// Imported frame depth target.
    pub depth: ImportedTextureHandle,
    /// Graph-owned forward depth target used when MSAA is active.
    pub msaa_depth: TextureHandle,
    /// Graph-owned R32Float intermediate for resolving MSAA depth.
    pub msaa_depth_r32: TextureHandle,
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

/// Returns whether the intersection raster tail has view-local work to record.
fn forward_intersection_raster_needed(opaque_recorded: bool, plan: &InstancePlan) -> bool {
    opaque_recorded && !plan.intersect_groups.is_empty()
}

/// Returns whether the grab-pass transparent raster tail has view-local work to record.
fn forward_transparent_raster_needed(opaque_recorded: bool, plan: &InstancePlan) -> bool {
    opaque_recorded && !plan.transparent_groups.is_empty()
}

impl WorldMeshForwardPreparePass {
    /// Creates a world mesh forward prepare pass instance.
    ///
    /// The `_resources` parameter is accepted for API symmetry with the other forward passes
    /// but is not stored: the prepare pass operates on per-view blackboard slots rather than
    /// graph resource handles.
    pub fn new(_resources: WorldMeshForwardGraphResources) -> Self {
        Self {
            skybox: SkyboxRenderer::default(),
        }
    }
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

impl WorldMeshColorSnapshotPass {
    /// Creates a world mesh color snapshot pass instance.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
    }
}

impl WorldMeshForwardTransparentPass {
    /// Creates a world mesh grab-pass transparent raster pass instance.
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

fn declare_forward_draw_reads(b: &mut PassBuilder<'_>, resources: WorldMeshForwardGraphResources) {
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

impl CallbackPass for WorldMeshForwardPreparePass {
    fn name(&self) -> &str {
        "WorldMeshForwardPrepare"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.callback();
        b.cull_exempt();
        Ok(())
    }

    fn run(&self, ctx: &mut CallbackCtx<'_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::prepare");
        let frame = &mut *ctx.pass_frame;

        let prepared = prepare_world_mesh_forward_frame(
            ctx.device,
            ctx.queue.as_ref(),
            ctx.upload_batch,
            ctx.gpu_limits,
            frame,
            ctx.blackboard,
            &self.skybox,
        );
        if let Some(prepared) = prepared {
            ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        }
        Ok(())
    }

    fn release_view_resources(&mut self, retired_views: &[crate::camera::ViewId]) {
        self.skybox.release_view_resources(retired_views);
    }
}

impl RasterPass for WorldMeshForwardOpaquePass {
    fn name(&self) -> &str {
        "WorldMeshForwardOpaque"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            r.frame_sampled_color(
                self.resources.scene_color_hdr,
                self.resources.scene_color_hdr_msaa,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::present::SWAPCHAIN_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
                Option::<ImportedTextureHandle>::None,
            );
            r.frame_sampled_depth(
                self.resources.depth,
                self.resources.msaa_depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::gpu::MAIN_FORWARD_DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
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
        let skybox_recorded = prepared
            .skybox
            .as_ref()
            .is_none_or(|skybox| record_prepared_skybox(rpass, frame, skybox));
        let mesh_recorded = record_world_mesh_forward_opaque_graph_raster(
            rpass,
            ctx.device,
            ctx.queue.as_ref(),
            frame,
            &prepared,
        );
        prepared.opaque_recorded = skybox_recorded && mesh_recorded;
        ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }
}

impl ComputePass for WorldMeshDepthSnapshotPass {
    fn name(&self) -> &str {
        "WorldMeshDepthSnapshot"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        // Declare only what is actually used: msaa_depth as sampled input,
        // msaa_depth_r32 as storage write output, and depth as CopySrc.
        // Note: msaa_depth_r32 lifetime is extended by WorldMeshForwardDepthResolvePass
        // which also writes it, covering the intersection pass that reads it implicitly
        // via the frame bind group.
        b.read_texture(
            self.resources.msaa_depth,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::COMPUTE,
            },
        );
        b.write_texture(
            self.resources.msaa_depth_r32,
            TextureAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::WriteOnly,
            },
        );
        b.import_texture(self.resources.depth, TextureAccess::CopySrc);
        Ok(())
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::depth_snapshot_record");
        let frame = &mut *ctx.pass_frame;
        let Some(mut prepared) = ctx.blackboard.take::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        let msaa_views = ctx.blackboard.get::<MsaaViewsSlot>();
        let msaa_depth_resolve = frame.view.msaa_depth_resolve.clone();
        let recorded = encode_world_mesh_forward_depth_snapshot(
            ctx.device,
            ctx.encoder,
            frame,
            &prepared,
            msaa_views,
            msaa_depth_resolve.as_deref(),
            ctx.profiler,
        );
        if recorded {
            prepared.depth_snapshot_recorded = true;
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
        {
            let mut r = b.raster();
            // No `resolve_target` here: when MSAA is active, the multisampled buffer is preserved
            // and resolved by [`WorldMeshForwardColorResolvePass`] using the Karis HDR-aware
            // bracket. wgpu's automatic linear average underestimates very bright samples at
            // contrast edges, producing visible aliasing where bright/dark samples meet (e.g.
            // specular sparks against dark surfaces) -- the custom resolve fixes that.
            r.frame_sampled_color(
                self.resources.scene_color_hdr,
                self.resources.scene_color_hdr_msaa,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                Option::<ImportedTextureHandle>::None,
            );
            r.frame_sampled_depth(
                self.resources.depth,
                self.resources.msaa_depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
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
                ctx.queue.as_ref(),
                frame,
                &prepared,
            )
        } else {
            false
        };
        if recorded {
            prepared.tail_raster_recorded = true;
        }
        ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }
}

impl ComputePass for WorldMeshColorSnapshotPass {
    fn name(&self) -> &str {
        "WorldMeshColorSnapshot"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.read_texture(self.resources.scene_color_hdr, TextureAccess::CopySrc);
        Ok(())
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::color_snapshot_record");
        let frame = &mut *ctx.pass_frame;
        let Some(prepared) = ctx.blackboard.get::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        encode_world_mesh_forward_color_snapshot(
            ctx.graph_resources,
            ctx.encoder,
            frame,
            prepared,
            self.resources,
        );
        Ok(())
    }
}

impl RasterPass for WorldMeshForwardTransparentPass {
    fn name(&self) -> &str {
        "WorldMeshForwardTransparent"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            // Keep grab-pass transparent draws on the same active color/depth sample tier as the
            // opaque/intersection passes. In MSAA mode this preserves the current MSAA stencil
            // attachment for UI masks; a final color resolve moves the updated HDR color back to
            // the single-sample scene target for post-processing.
            r.frame_sampled_color(
                self.resources.scene_color_hdr,
                self.resources.scene_color_hdr_msaa,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                Option::<ImportedTextureHandle>::None,
            );
            r.frame_sampled_depth(
                self.resources.depth,
                self.resources.msaa_depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
        };
        declare_forward_draw_reads(b, self.resources);
        Ok(())
    }

    fn should_record(&self, ctx: &RasterPassCtx<'_, '_>) -> Result<bool, RenderPassError> {
        Ok(ctx
            .blackboard
            .get::<WorldMeshForwardPlanSlot>()
            .is_some_and(|prepared| {
                forward_transparent_raster_needed(prepared.opaque_recorded, &prepared.plan)
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
        profiling::scope!("world_mesh_forward::transparent_record");
        let frame = &mut *ctx.pass_frame;

        let Some(mut prepared) = ctx.blackboard.take::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        let recorded = if prepared.opaque_recorded {
            record_world_mesh_forward_transparent_graph_raster(
                rpass,
                ctx.device,
                ctx.queue.as_ref(),
                frame,
                &prepared,
            )
        } else {
            false
        };
        if recorded {
            prepared.tail_raster_recorded = true;
        }
        ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        Ok(())
    }
}

impl ComputePass for WorldMeshForwardDepthResolvePass {
    fn name(&self) -> &str {
        "WorldMeshForwardDepthResolve"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.read_texture(
            self.resources.msaa_depth,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::COMPUTE,
            },
        );
        b.write_texture(
            self.resources.msaa_depth_r32,
            TextureAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::WriteOnly,
            },
        );
        b.read_texture(
            self.resources.msaa_depth_r32,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        Ok(())
    }

    fn record(&self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::depth_resolve_record");
        let frame = &mut *ctx.pass_frame;
        let msaa_views = ctx.blackboard.get::<MsaaViewsSlot>();
        let msaa_depth_resolve = frame.view.msaa_depth_resolve.clone();
        encode_msaa_depth_resolve_after_clear_only(
            ctx.device,
            ctx.encoder,
            frame,
            msaa_views,
            msaa_depth_resolve.as_deref(),
            ctx.profiler,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        InstancePlan, forward_intersection_raster_needed, forward_transparent_raster_needed,
    };
    use crate::world_mesh::DrawGroup;

    /// Empty helper plans skip their raster pass even after opaque work records successfully.
    #[test]
    fn helper_raster_needed_requires_matching_groups() {
        let empty = InstancePlan::default();
        assert!(!forward_intersection_raster_needed(true, &empty));
        assert!(!forward_transparent_raster_needed(true, &empty));

        let mut intersect = InstancePlan::default();
        intersect.intersect_groups.push(DrawGroup {
            representative_draw_idx: 0,
            instance_range: 0..1,
            material_packet_idx: 0,
        });
        assert!(forward_intersection_raster_needed(true, &intersect));
        assert!(!forward_intersection_raster_needed(false, &intersect));

        let mut transparent = InstancePlan::default();
        transparent.transparent_groups.push(DrawGroup {
            representative_draw_idx: 0,
            instance_range: 0..1,
            material_packet_idx: 0,
        });
        assert!(forward_transparent_raster_needed(true, &transparent));
        assert!(!forward_transparent_raster_needed(false, &transparent));
    }
}
