//! Main forward pass: clear color + depth, draw scene meshes, MSAA resolve.
//!
//! ## Pass graph structure
//!
//! World-mesh forward rendering is split across five passes:
//!
//! 1. [`WorldMeshForwardPreparePass`] — **[`CallbackPass`]** that collects + sorts draws, packs
//!    per-draw VP/model uniforms (rayon-parallel above the existing threshold), and uploads the
//!    per-draw slab and frame uniforms via `Queue::write_buffer`. Stores the prepared state in
//!    [`crate::render_graph::frame_params::FrameRenderParams::prepared_world_mesh_forward`] for
//!    downstream passes. (Phase 2 will move this to the typed blackboard.)
//! 2. [`WorldMeshForwardOpaquePass`] — **[`RasterPass`]** that opens the HDR color + depth
//!    attachments with `LoadOp::Clear` and records opaque draws.
//! 3. [`WorldMeshDepthSnapshotPass`] — **[`ComputePass`]** that resolves MSAA depth (when active)
//!    and copies single-sample depth into the scene-depth snapshot for intersection materials.
//! 4. [`WorldMeshForwardIntersectPass`] — **[`RasterPass`]** that draws intersection materials
//!    and resolves MSAA color when active.
//! 5. [`WorldMeshForwardDepthResolvePass`] — **[`ComputePass`]** that resolves the final MSAA
//!    depth into the single-sample frame depth used by Hi-Z.
//!
//! ## VR stereo world draws
//!
//! OpenXR per-eye view–projection maps **stage** space to clip. For non-overlay draws with
//! [`crate::render_graph::StereoViewMatrices`], identity is used instead of the host
//! `view_transform` world-to-camera to avoid mixing stage with the host rig. Overlays keep
//! `view` for orthographic / UI alignment. Matrix composition lives in [`vp`].

mod current_view_textures;
mod encode;
mod execute_helpers;
mod vp;

use std::num::NonZeroU32;

use crate::render_graph::compiled::{DepthAttachmentTemplate, RenderPassTemplate};
use crate::render_graph::context::{CallbackCtx, ComputePassCtx, RasterPassCtx};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::{PrefetchedWorldMeshDrawsSlot, WorldMeshForwardPlanSlot};
use crate::render_graph::pass::{CallbackPass, ComputePass, PassBuilder, RasterPass};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess, TextureAccess,
    TextureHandle,
};

use execute_helpers::{
    encode_msaa_depth_resolve_after_clear_only, encode_world_mesh_forward_depth_snapshot,
    prepare_world_mesh_forward_frame, record_world_mesh_forward_intersection_graph_raster,
    record_world_mesh_forward_opaque_graph_raster, resolve_forward_msaa_views, stencil_load_ops,
};

/// Prepares sorted world-mesh forward draw state for subsequent graph nodes.
///
/// The pass is a [`CallbackPass`] (no encoder); it issues `Queue::write_buffer` uploads and
/// stores results in the per-view blackboard via [`WorldMeshForwardPlanSlot`] and
/// [`crate::render_graph::PrecomputedMaterialBindsSlot`].
#[derive(Debug, Default)]
pub struct WorldMeshForwardPreparePass;

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

impl WorldMeshForwardPreparePass {
    /// Creates a world mesh forward prepare pass instance.
    ///
    /// The `_resources` parameter is accepted for API symmetry with the other forward passes
    /// but is not stored: the prepare pass operates on per-view blackboard slots rather than
    /// graph resource handles.
    pub fn new(_resources: WorldMeshForwardGraphResources) -> Self {
        Self
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
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

        // Transfer prefetched draws from blackboard to frame params so the prepare helper
        // can use them (it calls `take_or_collect_world_mesh_draws` which checks frame params).
        // Phase 5+ will move this entirely into the blackboard via `PrefetchedWorldMeshDrawsSlot`.
        if let Some(prefetched) = ctx.blackboard.take::<PrefetchedWorldMeshDrawsSlot>() {
            // Temporarily stash in the blackboard slot so execute_helpers.rs can pick it up.
            // Since execute_helpers currently reads from frame (which we moved away from),
            // pass it directly to prepare_world_mesh_forward_frame via a new blackboard-aware path.
            // For now: use the `prefetched_world_mesh_draws` slot that prepare_frame reads.
            ctx.blackboard
                .insert::<PrefetchedWorldMeshDrawsSlot>(prefetched);
        }

        let prepared = prepare_world_mesh_forward_frame(
            ctx.device,
            ctx.queue.as_ref(),
            ctx.upload_batch,
            ctx.gpu_limits,
            frame,
            ctx.blackboard,
        );
        if let Some(prepared) = prepared {
            ctx.blackboard.insert::<WorldMeshForwardPlanSlot>(prepared);
        }
        Ok(())
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
                    load: wgpu::LoadOp::Clear(crate::render_graph::MAIN_FORWARD_DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
        }
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
        if use_multiview {
            NonZeroU32::new(3)
        } else {
            template.multiview_mask
        }
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
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

        let Some(mut prepared) = ctx.blackboard.take::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        let recorded = record_world_mesh_forward_opaque_graph_raster(
            rpass,
            ctx.device,
            ctx.queue.as_ref(),
            frame,
            &prepared,
        );
        prepared.opaque_recorded = recorded;
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
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let msaa_views = resolve_forward_msaa_views(
            ctx.graph_resources,
            self.resources,
            frame.view.sample_count,
            frame.view.multiview_stereo,
        );

        let Some(mut prepared) = ctx.blackboard.take::<WorldMeshForwardPlanSlot>() else {
            return Ok(());
        };
        let msaa_depth_resolve = frame.view.msaa_depth_resolve.clone();
        let recorded = encode_world_mesh_forward_depth_snapshot(
            ctx.device,
            ctx.encoder,
            frame,
            &prepared,
            msaa_views.as_ref(),
            msaa_depth_resolve.as_deref(),
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
            r.frame_sampled_color(
                self.resources.scene_color_hdr,
                self.resources.scene_color_hdr_msaa,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                Some(self.resources.scene_color_hdr),
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
        }
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
        if use_multiview {
            NonZeroU32::new(3)
        } else {
            template.multiview_mask
        }
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
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

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
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let msaa_views = resolve_forward_msaa_views(
            ctx.graph_resources,
            self.resources,
            frame.view.sample_count,
            frame.view.multiview_stereo,
        );
        let msaa_depth_resolve = frame.view.msaa_depth_resolve.clone();
        encode_msaa_depth_resolve_after_clear_only(
            ctx.device,
            ctx.encoder,
            frame,
            msaa_views.as_ref(),
            msaa_depth_resolve.as_deref(),
        );
        // No blackboard interaction needed: depth resolve is purely encoder-driven.
        Ok(())
    }
}
