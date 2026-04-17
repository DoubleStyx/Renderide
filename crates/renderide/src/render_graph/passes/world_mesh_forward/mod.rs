//! Main forward pass: clear color + depth, debug normal shading for scene meshes.
//!
//! Draws are collected and **sorted by [`MaterialDrawBatchKey`](crate::render_graph::MaterialDrawBatchKey)**
//! so pipeline and batch key drive pipeline switches. **GPU instancing:** consecutive draws that share the
//! same mesh submesh and batch key (opaque, non-skinned) are merged into one indexed draw with
//! `instance_index` sampling [`crate::backend::PerDrawResources`] (`@group(2)` storage). Embedded `@group(1)`
//! skips redundant [`wgpu::RenderPass::set_bind_group`] when [`MaterialBindCacheKey`](crate::backend::MaterialBindCacheKey) matches
//! the previous draw (uniform updates still run each time via [`EmbeddedMaterialBindResources`](crate::backend::EmbeddedMaterialBindResources)).
//! Per-slot [`MaterialPropertyLookupIds`](crate::assets::material::MaterialPropertyLookupIds) are carried on each
//! [`WorldMeshDrawItem`](crate::render_graph::WorldMeshDrawItem) for `get_merged` when building `@group(1)` bind
//! groups for [`crate::materials::RasterPipelineKind::EmbeddedStem`] draws (see [`crate::backend::EmbeddedMaterialBindResources`]).
//!
//! Manifest raster binds use the composed WGSL **stem** from [`crate::materials::MaterialRouter::stem_for_shader_asset`]
//! (not a hard-coded Unlit path). Whether UV0 is bound is stored on [`MaterialDrawBatchKey::Embedded_needs_uv0`]
//! (same rule as the embedded raster pipeline and [`crate::materials::embedded_stem_needs_uv0_stream`], computed during draw collection).
//! Intersection tint subpasses use [`MaterialDrawBatchKey::Embedded_requires_intersection_pass`]
//! ([`crate::materials::embedded_stem_requires_intersection_pass`], WGSL reflection of `_IntersectColor`).
//!
//! ## VR stereo world draws
//!
//! OpenXR per-eye view–projection maps **stage** space to clip. For **non-overlay** draws with
//! `stereo_view_proj`, we use **identity** instead of the host `view_transform` world-to-camera so
//! `VP` is not `P·V_hmd·V_host`, which mixed stage with the host rig and caused playspace-relative
//! offsets. Overlays keep `view` for orthographic / UI alignment with the host camera rig.
//! Matrix composition lives in [`vp`].

mod encode;
mod execute_helpers;
mod vp;

use std::num::NonZeroU32;

use crate::render_graph::context::{
    GraphRasterPassContext, GraphResolvedResources, RenderPassContext, ResolvedGraphTexture,
};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::FrameRenderParams;
use crate::render_graph::pass::{PassBuilder, RenderPass};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, StorageAccess, TextureAccess,
    TextureHandle,
};

use execute_helpers::{
    encode_clear_only_pass, encode_msaa_color_resolve_after_opaque,
    encode_msaa_depth_resolve_after_clear_only, encode_world_mesh_forward_depth_snapshot,
    encode_world_mesh_forward_draw_passes, prepare_world_mesh_forward_frame,
    record_world_mesh_forward_intersection_graph_raster,
    record_world_mesh_forward_opaque_graph_raster, stencil_load_ops, ForwardPassEncodeFrame,
    ForwardPassEncodeViews,
};

/// Clears the backbuffer and depth, then draws meshes with material-batched raster pipelines.
#[derive(Debug)]
pub struct WorldMeshForwardPass {
    resources: WorldMeshForwardGraphResources,
}

/// Prepares sorted world-mesh forward draw state for subsequent graph nodes.
#[derive(Debug)]
pub struct WorldMeshForwardPreparePass {
    resources: WorldMeshForwardGraphResources,
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

/// Resolves the final MSAA forward depth into the single-sample frame depth target.
#[derive(Debug)]
pub struct WorldMeshForwardDepthResolvePass {
    resources: WorldMeshForwardGraphResources,
}

/// Graph resources used by [`WorldMeshForwardPass`].
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshForwardGraphResources {
    /// Imported frame color target.
    pub color: ImportedTextureHandle,
    /// Imported frame depth target.
    pub depth: ImportedTextureHandle,
    /// Graph-owned forward color target used when MSAA is active.
    pub msaa_color: TextureHandle,
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

impl WorldMeshForwardPass {
    /// Creates a world mesh forward pass instance.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
    }
}

impl WorldMeshForwardPreparePass {
    /// Creates a world mesh forward prepare pass instance.
    pub fn new(resources: WorldMeshForwardGraphResources) -> Self {
        Self { resources }
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

impl RenderPass for WorldMeshForwardPreparePass {
    fn name(&self) -> &str {
        "WorldMeshForwardPrepare"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.copy();
        b.import_buffer(self.resources.per_draw_slab, BufferAccess::CopyDst);
        b.import_buffer(self.resources.frame_uniforms, BufferAccess::CopyDst);
        b.import_buffer(self.resources.lights, BufferAccess::CopyDst);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

        let queue_guard = ctx
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let queue = &*queue_guard;
        frame.prepared_world_mesh_forward =
            prepare_world_mesh_forward_frame(ctx.device, queue, ctx.gpu_limits, frame);
        Ok(())
    }
}

impl RenderPass for WorldMeshForwardOpaquePass {
    fn name(&self) -> &str {
        "WorldMeshForwardOpaque"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            r.frame_sampled_color(
                self.resources.color,
                self.resources.msaa_color,
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

    fn execute(&mut self, _ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        Ok(())
    }

    fn graph_managed_raster(&self) -> bool {
        true
    }

    fn graph_raster_multiview_mask(
        &self,
        ctx: &GraphRasterPassContext<'_, '_>,
        template: &crate::render_graph::RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        let use_multiview = ctx
            .frame
            .as_ref()
            .and_then(|frame| frame.prepared_world_mesh_forward.as_ref())
            .is_some_and(|prepared| prepared.pipeline.use_multiview);
        if use_multiview {
            NonZeroU32::new(3)
        } else {
            template.multiview_mask
        }
    }

    fn graph_raster_stencil_ops(
        &self,
        ctx: &GraphRasterPassContext<'_, '_>,
        depth: &crate::render_graph::DepthAttachmentTemplate,
    ) -> Option<wgpu::Operations<u32>> {
        let Some(format) = ctx
            .frame
            .as_ref()
            .and_then(|frame| frame.prepared_world_mesh_forward.as_ref())
            .and_then(|prepared| prepared.pipeline.pass_desc.depth_stencil_format)
        else {
            return depth.stencil;
        };
        format.has_stencil_aspect().then_some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(0),
            store: wgpu::StoreOp::Store,
        })
    }

    fn execute_graph_raster(
        &mut self,
        ctx: &mut GraphRasterPassContext<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        apply_graph_forward_msaa_views(frame, ctx.graph_resources, self.resources);

        let Some(mut prepared) = frame.prepared_world_mesh_forward.take() else {
            return Ok(());
        };
        let queue_guard = ctx
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let queue = &*queue_guard;
        let recorded = record_world_mesh_forward_opaque_graph_raster(
            rpass, ctx.device, queue, frame, &prepared,
        );
        prepared.opaque_recorded = recorded;
        frame.prepared_world_mesh_forward = Some(prepared);
        Ok(())
    }
}

impl RenderPass for WorldMeshDepthSnapshotPass {
    fn name(&self) -> &str {
        "WorldMeshDepthSnapshot"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            r.depth(
                self.resources.depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
        }
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
        b.import_texture(self.resources.depth, TextureAccess::CopySrc);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        apply_graph_forward_msaa_views(frame, ctx.graph_resources, self.resources);

        let Some(mut prepared) = frame.prepared_world_mesh_forward.take() else {
            return Ok(());
        };
        let msaa_depth_resolve = frame.backend.msaa_depth_resolve();
        let recorded = encode_world_mesh_forward_depth_snapshot(
            ctx.device,
            ctx.encoder,
            frame,
            &prepared,
            msaa_depth_resolve.as_deref(),
        );
        if recorded {
            prepared.depth_snapshot_recorded = true;
        }
        frame.prepared_world_mesh_forward = Some(prepared);
        Ok(())
    }
}

impl RenderPass for WorldMeshForwardIntersectPass {
    fn name(&self) -> &str {
        "WorldMeshForwardIntersect"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            r.frame_sampled_color(
                self.resources.color,
                self.resources.msaa_color,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                Some(self.resources.color),
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

    fn execute(&mut self, _ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        Ok(())
    }

    fn graph_managed_raster(&self) -> bool {
        true
    }

    fn graph_raster_multiview_mask(
        &self,
        ctx: &GraphRasterPassContext<'_, '_>,
        template: &crate::render_graph::RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        let use_multiview = ctx
            .frame
            .as_ref()
            .and_then(|frame| frame.prepared_world_mesh_forward.as_ref())
            .is_some_and(|prepared| prepared.pipeline.use_multiview);
        if use_multiview {
            NonZeroU32::new(3)
        } else {
            template.multiview_mask
        }
    }

    fn graph_raster_stencil_ops(
        &self,
        ctx: &GraphRasterPassContext<'_, '_>,
        depth: &crate::render_graph::DepthAttachmentTemplate,
    ) -> Option<wgpu::Operations<u32>> {
        let Some(format) = ctx
            .frame
            .as_ref()
            .and_then(|frame| frame.prepared_world_mesh_forward.as_ref())
            .and_then(|prepared| prepared.pipeline.pass_desc.depth_stencil_format)
        else {
            return depth.stencil;
        };
        stencil_load_ops(Some(format))
    }

    fn execute_graph_raster(
        &mut self,
        ctx: &mut GraphRasterPassContext<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        apply_graph_forward_msaa_views(frame, ctx.graph_resources, self.resources);

        let Some(mut prepared) = frame.prepared_world_mesh_forward.take() else {
            return Ok(());
        };
        let recorded = if prepared.opaque_recorded {
            let queue_guard = ctx
                .queue
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let queue = &*queue_guard;
            record_world_mesh_forward_intersection_graph_raster(
                rpass, ctx.device, queue, frame, &prepared,
            )
        } else {
            false
        };
        if recorded {
            prepared.tail_raster_recorded = true;
        }
        frame.prepared_world_mesh_forward = Some(prepared);
        Ok(())
    }
}

impl RenderPass for WorldMeshForwardDepthResolvePass {
    fn name(&self) -> &str {
        "WorldMeshForwardDepthResolve"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            r.depth(
                self.resources.depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
        }
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

    fn execute(&mut self, ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        apply_graph_forward_msaa_views(frame, ctx.graph_resources, self.resources);
        let msaa_depth_resolve = frame.backend.msaa_depth_resolve();
        encode_msaa_depth_resolve_after_clear_only(
            ctx.device,
            ctx.encoder,
            frame,
            msaa_depth_resolve.as_deref(),
        );
        Ok(())
    }
}

impl RenderPass for WorldMeshForwardPass {
    fn name(&self) -> &str {
        "WorldMeshForward"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        {
            let mut r = b.raster();
            r.color(
                self.resources.color,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::present::SWAPCHAIN_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
                Option::<ImportedTextureHandle>::None,
            );
            r.depth(
                self.resources.depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::render_graph::MAIN_FORWARD_DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
            r.color(
                self.resources.msaa_color,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::present::SWAPCHAIN_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
                Some(self.resources.color),
            );
            r.depth(
                self.resources.msaa_depth,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::render_graph::MAIN_FORWARD_DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                },
                None,
            );
        }
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
        b.import_buffer(
            self.resources.cluster_light_counts,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::FRAGMENT,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.cluster_light_indices,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::FRAGMENT,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.lights,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::FRAGMENT,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.per_draw_slab,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.frame_uniforms,
            BufferAccess::Uniform {
                stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(bb) = ctx.backbuffer else {
            return Err(RenderPassError::MissingBackbuffer {
                pass: self.name().to_string(),
            });
        };
        let Some(depth) = ctx.depth_view else {
            return Err(RenderPassError::MissingDepth {
                pass: self.name().to_string(),
            });
        };
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

        let queue_guard = ctx
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let queue = &*queue_guard;

        let prepared = if let Some(prepared) = frame.prepared_world_mesh_forward.take() {
            prepared
        } else {
            let Some(prepared) =
                prepare_world_mesh_forward_frame(ctx.device, queue, ctx.gpu_limits, frame)
            else {
                return Ok(());
            };
            prepared
        };

        apply_graph_forward_msaa_views(frame, ctx.graph_resources, self.resources);

        let msaa_color = frame.msaa_color_view.clone();
        let msaa_depth = frame.msaa_depth_view.clone();
        let color_view = msaa_color.as_ref().unwrap_or(bb);
        let depth_raster = msaa_depth.as_ref().unwrap_or(depth);
        let resolve_swapchain = if frame.sample_count > 1 {
            Some(bb)
        } else {
            None
        };

        let msaa_depth_resolve = frame.backend.msaa_depth_resolve();

        if prepared.draws.is_empty() {
            if prepared.opaque_recorded {
                if frame.sample_count > 1 {
                    encode_msaa_color_resolve_after_opaque(
                        ctx.encoder,
                        color_view,
                        resolve_swapchain,
                        prepared.pipeline.use_multiview,
                    );
                    encode_msaa_depth_resolve_after_clear_only(
                        ctx.device,
                        ctx.encoder,
                        frame,
                        msaa_depth_resolve.as_deref(),
                    );
                }
            } else {
                encode_clear_only_pass(
                    ctx.encoder,
                    color_view,
                    depth_raster,
                    prepared.pipeline.pass_desc.depth_stencil_format,
                    resolve_swapchain,
                    prepared.pipeline.use_multiview,
                );
                encode_msaa_depth_resolve_after_clear_only(
                    ctx.device,
                    ctx.encoder,
                    frame,
                    msaa_depth_resolve.as_deref(),
                );
            }
            return Ok(());
        }

        if !encode_world_mesh_forward_draw_passes(
            ForwardPassEncodeFrame {
                encoder: ctx.encoder,
                device: ctx.device,
                frame,
                queue,
            },
            &prepared,
            ForwardPassEncodeViews {
                color_view,
                depth_raster_view: depth_raster,
                resolve_swapchain,
                msaa_depth_resolve: msaa_depth_resolve.as_deref(),
            },
        ) {
            return Ok(());
        }

        Ok(())
    }
}

fn apply_graph_forward_msaa_views(
    frame: &mut FrameRenderParams<'_>,
    graph_resources: Option<&GraphResolvedResources>,
    resources: WorldMeshForwardGraphResources,
) {
    if frame.sample_count <= 1 {
        return;
    }
    let Some(graph_resources) = graph_resources else {
        return;
    };
    let (Some(color), Some(depth), Some(r32)) = (
        graph_resources.transient_texture(resources.msaa_color),
        graph_resources.transient_texture(resources.msaa_depth),
        graph_resources.transient_texture(resources.msaa_depth_r32),
    ) else {
        return;
    };

    if frame.multiview_stereo {
        let (Some(depth_layers), Some(r32_layers)) =
            (first_two_layer_views(depth), first_two_layer_views(r32))
        else {
            return;
        };
        frame.msaa_color_view = Some(color.view.clone());
        frame.msaa_depth_view = Some(depth.view.clone());
        frame.msaa_depth_resolve_r32_view = Some(r32.view.clone());
        frame.msaa_depth_is_array = true;
        frame.msaa_stereo_depth_layer_views = Some(depth_layers);
        frame.msaa_stereo_r32_layer_views = Some(r32_layers);
    } else {
        frame.msaa_color_view = Some(color.view.clone());
        frame.msaa_depth_view = Some(depth.view.clone());
        frame.msaa_depth_resolve_r32_view = Some(r32.view.clone());
        frame.msaa_depth_is_array = false;
        frame.msaa_stereo_depth_layer_views = None;
        frame.msaa_stereo_r32_layer_views = None;
    }
}

fn first_two_layer_views(texture: &ResolvedGraphTexture) -> Option<[wgpu::TextureView; 2]> {
    Some([
        texture.layer_views.first()?.clone(),
        texture.layer_views.get(1)?.clone(),
    ])
}
