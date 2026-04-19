//! [`CompiledRenderGraph`] execution: multi-view scheduling, resource resolution, and submits.

use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

use crate::backend::RenderBackend;
use crate::gpu::GpuContext;
use crate::scene::SceneCoordinator;

use super::super::context::{
    GraphRasterPassContext, GraphResolvedResources, PostSubmitContext, RenderPassContext,
    ResolvedGraphBuffer, ResolvedGraphTexture, ResolvedImportedBuffer, ResolvedImportedTexture,
};
use super::super::error::GraphExecuteError;
use super::super::frame_params::{HostCameraFrame, OcclusionViewId};
use super::super::resources::{
    BackendFrameBufferKind, BufferImportSource, FrameTargetRole, ImportSource,
    ImportedBufferHandle, ImportedTextureHandle, TextureHandle,
};
use super::super::transient_pool::{BufferKey, TextureKey, TransientPool};
use super::helpers;
use super::{
    CompiledRenderGraph, ExternalFrameTargets, FrameView, FrameViewTarget,
    MultiViewExecutionContext, OffscreenSingleViewExecuteSpec, ResolvedView,
};

/// Key for reusing transient pool allocations across [`FrameView`]s with identical surface layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GraphResolveKey {
    viewport_px: (u32, u32),
    surface_format: wgpu::TextureFormat,
    depth_stencil_format: wgpu::TextureFormat,
    sample_count: u32,
    multiview_stereo: bool,
}

impl GraphResolveKey {
    fn from_resolved(resolved: &ResolvedView<'_>) -> Self {
        Self {
            viewport_px: resolved.viewport_px,
            surface_format: resolved.surface_format,
            depth_stencil_format: resolved.depth_texture.format(),
            sample_count: resolved.sample_count,
            multiview_stereo: resolved.multiview_stereo,
        }
    }
}

/// View surface properties used when resolving transient [`TextureKey`] values for a graph view.
pub(crate) struct TransientTextureResolveSurfaceParams {
    /// Viewport extent in pixels.
    pub viewport_px: (u32, u32),
    /// Swapchain or offscreen color format for format resolution.
    pub surface_format: wgpu::TextureFormat,
    /// Depth attachment format for format resolution.
    pub depth_stencil_format: wgpu::TextureFormat,
    /// MSAA sample count for the view.
    pub sample_count: u32,
    /// Stereo multiview (two layers) vs single-view.
    pub multiview_stereo: bool,
}

impl CompiledRenderGraph {
    /// Ordered pass count.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Whether this graph targets the swapchain this frame.
    pub fn needs_surface_acquire(&self) -> bool {
        self.needs_surface_acquire
    }

    /// Desktop single-view entry: delegates to [`Self::execute_multi_view`] (one swapchain view).
    ///
    /// Submit count follows the multi-view rules (optional frame-global encoder + submit, then
    /// per-view encoder + submit). Matches [`crate::present::present_clear_frame`] recovery behavior
    /// for surface acquire (timeout/occluded skip, validation reconfigure).
    pub fn execute(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
    ) -> Result<(), GraphExecuteError> {
        let mut single = [FrameView {
            host_camera,
            target: FrameViewTarget::Swapchain,
            draw_filter: None,
            prefetched_world_mesh_draws: None,
        }];
        self.execute_multi_view(gpu, scene, backend, &mut single)
    }

    /// Records passes against pre-built multiview array targets (OpenXR swapchain path).
    pub fn execute_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        let mut single = [FrameView {
            host_camera,
            target: FrameViewTarget::ExternalMultiview(external),
            draw_filter: None,
            prefetched_world_mesh_draws: None,
        }];
        self.execute_multi_view(gpu, scene, backend, &mut single)
    }

    /// Renders the graph to a single-view offscreen color/depth target (secondary camera → render texture).
    pub fn execute_offscreen_single_view(
        &mut self,
        gpu: &mut GpuContext,
        backend: &mut RenderBackend,
        spec: OffscreenSingleViewExecuteSpec<'_>,
    ) -> Result<(), GraphExecuteError> {
        let scene = spec.scene;
        let host_camera = spec.host_camera;
        let external = spec.external;
        let transform_filter = spec.transform_filter;
        let prefetched_world_mesh_draws = spec.prefetched_world_mesh_draws;
        let mut single = [FrameView {
            host_camera,
            target: FrameViewTarget::OffscreenRt(external),
            draw_filter: transform_filter,
            prefetched_world_mesh_draws,
        }];
        self.execute_multi_view(gpu, scene, backend, &mut single)
    }

    /// Records all views: one encoder + submit for frame-global work, then one encoder + submit per view.
    ///
    /// Per-view passes use [`wgpu::Queue::write_buffer`] for camera uniforms, per-draw slabs, and
    /// cluster params. Those writes are ordered **before** the next `queue.submit`; a single submit
    /// for all views would leave only the last view’s uploads visible to every view’s GPU commands,
    /// so each view is isolated in its own submit.
    ///
    /// Frame-global passes ([`PassPhase::FrameGlobal`]) run once in the first encoder; per-view
    /// passes run for each [`FrameView`] in order.
    ///
    /// `views` is borrowed for the duration of execution (callers can pass a stack-allocated
    /// one-element array or reuse a [`Vec`] through `as_mut_slice()` to avoid allocating a fresh
    /// [`Vec`] each frame for single-view paths).
    ///
    /// Swapchain acquisition routes through [`GpuContext::acquire_with_recovery`], which uses the
    /// window stored inside `gpu` for size queries. Headless contexts have no window and no
    /// swapchain views (the headless driver substitutes `Swapchain` for `OffscreenRt` upstream),
    /// so this path never reaches the swapchain branch in headless mode.
    pub fn execute_multi_view<'a>(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        views: &mut [FrameView<'a>],
    ) -> Result<(), GraphExecuteError> {
        if views.is_empty() {
            return Ok(());
        }

        let needs_swapchain = views
            .iter()
            .any(|v| matches!(v.target, FrameViewTarget::Swapchain));

        let (frame, backbuffer_view_holder): (
            Option<wgpu::SurfaceTexture>,
            Option<wgpu::TextureView>,
        ) = match helpers::acquire_swapchain_for_multi_view_if_needed(
            needs_swapchain,
            self.needs_surface_acquire,
            gpu,
        )? {
            helpers::MultiViewSwapchainAcquire::NotNeeded => (None, None),
            helpers::MultiViewSwapchainAcquire::SkipPresent => return Ok(()),
            helpers::MultiViewSwapchainAcquire::Acquired {
                frame,
                backbuffer_view,
            } => (Some(frame), Some(backbuffer_view)),
        };

        // `resolve_view_from_target` and submits need `&mut GpuContext` while passes need `&Device`,
        // `&GpuLimits`, and `&Arc<Queue>`. Those cannot be borrowed directly from `gpu` alongside
        // the mutable context borrow, so we hold refcounted clones for the frame (cheap atomics).
        let device_arc = gpu.device().clone();
        let queue_arc = gpu.queue().clone();
        let limits_arc = gpu.limits().clone();
        let device = device_arc.as_ref();
        let gpu_limits = limits_arc.as_ref();

        backend.transient_pool_mut().begin_generation();

        let mut mv_ctx = MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc: &queue_arc,
            backbuffer_view_holder: &backbuffer_view_holder,
        };

        let mut transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources> = HashMap::new();

        self.execute_multi_view_frame_global_passes(&mut mv_ctx, views, &mut transient_by_key)?;

        // Per-view: separate encoder + submit so queue writes before each submit apply only to this view.
        for view in views.iter_mut() {
            self.execute_multi_view_submit_for_one_view(&mut mv_ctx, view, &mut transient_by_key)?;
        }

        {
            let pool = mv_ctx.backend.transient_pool_mut();
            for (_, resources) in transient_by_key {
                resources.release_to_pool(pool);
            }
            pool.gc_tick(120);
        }

        if let Some(f) = frame {
            f.present();
        }
        Ok(())
    }

    /// One per-view encoder, per-view [`PassPhase::PerView`] passes, submit, and Hi-Z bookkeeping.
    fn execute_multi_view_submit_for_one_view(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        view: &mut FrameView<'_>,
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
    ) -> Result<(), GraphExecuteError> {
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc,
            backbuffer_view_holder,
        } = mv_ctx;

        let prefetched = view.prefetched_world_mesh_draws.take();
        let draw_filter = view.draw_filter.clone();
        let host_camera = view.host_camera;
        let target_is_swapchain = matches!(view.target, FrameViewTarget::Swapchain);
        let resolved = Self::resolve_view_from_target(&view.target, gpu, backbuffer_view_holder)?;
        let key = GraphResolveKey::from_resolved(&resolved);
        let resolved_resources = match transient_by_key.entry(key) {
            Entry::Vacant(v) => {
                let mut resources = GraphResolvedResources::with_capacity(
                    self.transient_textures.len(),
                    self.transient_buffers.len(),
                    self.imported_textures.len(),
                    self.imported_buffers.len(),
                );
                let alloc_viewport = helpers::clamp_viewport_for_transient_alloc(
                    resolved.viewport_px,
                    gpu_limits.max_texture_dimension_2d(),
                );
                self.resolve_transient_textures(
                    device,
                    backend.transient_pool_mut(),
                    TransientTextureResolveSurfaceParams {
                        viewport_px: alloc_viewport,
                        surface_format: resolved.surface_format,
                        depth_stencil_format: resolved.depth_texture.format(),
                        sample_count: resolved.sample_count,
                        multiview_stereo: resolved.multiview_stereo,
                    },
                    &mut resources,
                )?;
                self.resolve_transient_buffers(
                    device,
                    backend.transient_pool_mut(),
                    alloc_viewport,
                    &mut resources,
                )?;
                v.insert(resources)
            }
            Entry::Occupied(o) => o.into_mut(),
        };
        self.resolve_imported_textures(&resolved, resolved_resources);
        self.resolve_imported_buffers(&backend.frame_resources, &resolved, resolved_resources);
        let graph_resources: &GraphResolvedResources = &*resolved_resources;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-per-view"),
        });
        {
            let mut frame_params = helpers::frame_render_params_from_resolved(
                scene,
                backend,
                &resolved,
                host_camera,
                draw_filter,
                prefetched,
            );
            helpers::populate_forward_msaa_from_graph_resources(
                &mut frame_params,
                Some(graph_resources),
                self.main_graph_msaa_transient_handles,
            );
            for &pass_idx in &self.per_view_pass_indices {
                let pass = &mut self.passes[pass_idx];
                if pass.graph_managed_raster() {
                    let template = helpers::pass_info_raster_template(&self.pass_info, pass_idx)?;
                    let mut ctx = GraphRasterPassContext {
                        device,
                        gpu_limits,
                        queue: queue_arc,
                        backbuffer: resolved.backbuffer,
                        depth_view: Some(resolved.depth_view),
                        frame: Some(&mut frame_params),
                        graph_resources: Some(graph_resources),
                    };
                    helpers::execute_graph_managed_raster_pass(
                        pass.as_mut(),
                        &template,
                        graph_resources,
                        &mut encoder,
                        &mut ctx,
                    )?;
                } else {
                    let mut ctx = RenderPassContext {
                        device,
                        gpu_limits,
                        queue: queue_arc,
                        encoder: &mut encoder,
                        backbuffer: resolved.backbuffer,
                        depth_view: Some(resolved.depth_view),
                        frame: Some(&mut frame_params),
                        graph_resources: Some(graph_resources),
                    };
                    pass.execute(&mut ctx)?;
                }
            }
        }

        if target_is_swapchain {
            let Some(bb) = backbuffer_view_holder.as_ref() else {
                return Err(GraphExecuteError::MissingSwapchainView);
            };
            let viewport_px = gpu.surface_extent_px();
            let queue_ref: &wgpu::Queue = queue_arc.as_ref();
            if let Err(e) =
                backend.encode_debug_hud_overlay(device, queue_ref, &mut encoder, bb, viewport_px)
            {
                logger::warn!("debug HUD overlay: {e}");
            }
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands_with_queue(queue_ref, cmd);
        } else {
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands(cmd);
        }

        let occlusion_view = view.occlusion_view_id();
        let mut post_ctx = PostSubmitContext {
            device,
            occlusion: &mut backend.occlusion,
            occlusion_view,
            host_camera: view.host_camera,
        };
        for &pass_idx in &self.per_view_pass_indices {
            self.passes[pass_idx]
                .post_submit(&mut post_ctx)
                .map_err(GraphExecuteError::Pass)?;
        }
        Ok(())
    }

    /// Runs [`PassPhase::FrameGlobal`] passes once per tick using the first view for host/scene context.
    fn execute_multi_view_frame_global_passes(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
    ) -> Result<(), GraphExecuteError> {
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc,
            backbuffer_view_holder,
        } = mv_ctx;

        if self.frame_global_pass_indices.is_empty() {
            return Ok(());
        }
        let first = views.first().ok_or(GraphExecuteError::NoViewsInBatch)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-frame-global"),
        });
        // Frame-global phase (e.g. mesh deform): use first view for host camera / scene context.
        {
            let resolved =
                Self::resolve_view_from_target(&first.target, gpu, backbuffer_view_holder)?;
            let key = GraphResolveKey::from_resolved(&resolved);
            let resolved_resources = match transient_by_key.entry(key) {
                Entry::Vacant(v) => {
                    let mut resources = GraphResolvedResources::with_capacity(
                        self.transient_textures.len(),
                        self.transient_buffers.len(),
                        self.imported_textures.len(),
                        self.imported_buffers.len(),
                    );
                    let alloc_viewport = helpers::clamp_viewport_for_transient_alloc(
                        resolved.viewport_px,
                        gpu_limits.max_texture_dimension_2d(),
                    );
                    self.resolve_transient_textures(
                        device,
                        backend.transient_pool_mut(),
                        TransientTextureResolveSurfaceParams {
                            viewport_px: alloc_viewport,
                            surface_format: resolved.surface_format,
                            depth_stencil_format: resolved.depth_texture.format(),
                            sample_count: resolved.sample_count,
                            multiview_stereo: resolved.multiview_stereo,
                        },
                        &mut resources,
                    )?;
                    self.resolve_transient_buffers(
                        device,
                        backend.transient_pool_mut(),
                        alloc_viewport,
                        &mut resources,
                    )?;
                    v.insert(resources)
                }
                Entry::Occupied(o) => o.into_mut(),
            };
            self.resolve_imported_textures(&resolved, resolved_resources);
            self.resolve_imported_buffers(&backend.frame_resources, &resolved, resolved_resources);
            let graph_resources: &GraphResolvedResources = &*resolved_resources;
            {
                let mut frame_params = helpers::frame_render_params_from_resolved(
                    scene,
                    backend,
                    &resolved,
                    first.host_camera,
                    first.draw_filter.clone(),
                    None,
                );
                helpers::populate_forward_msaa_from_graph_resources(
                    &mut frame_params,
                    Some(graph_resources),
                    self.main_graph_msaa_transient_handles,
                );
                for &pass_idx in &self.frame_global_pass_indices {
                    let pass = &mut self.passes[pass_idx];
                    if pass.graph_managed_raster() {
                        let template =
                            helpers::pass_info_raster_template(&self.pass_info, pass_idx)?;
                        let mut ctx = GraphRasterPassContext {
                            device,
                            gpu_limits,
                            queue: queue_arc,
                            backbuffer: None,
                            depth_view: None,
                            frame: Some(&mut frame_params),
                            graph_resources: Some(graph_resources),
                        };
                        helpers::execute_graph_managed_raster_pass(
                            pass.as_mut(),
                            &template,
                            graph_resources,
                            &mut encoder,
                            &mut ctx,
                        )?;
                    } else {
                        let mut ctx = RenderPassContext {
                            device,
                            gpu_limits,
                            queue: queue_arc,
                            encoder: &mut encoder,
                            backbuffer: None,
                            depth_view: None,
                            frame: Some(&mut frame_params),
                            graph_resources: Some(graph_resources),
                        };
                        pass.execute(&mut ctx)?;
                    }
                }
            }
        }
        let cmd = encoder.finish();
        gpu.submit_tracked_frame_commands(cmd);

        let occlusion_view = first.occlusion_view_id();
        let host_camera = first.host_camera;
        let mut post_ctx = PostSubmitContext {
            device,
            occlusion: &mut backend.occlusion,
            occlusion_view,
            host_camera,
        };
        for &pass_idx in &self.frame_global_pass_indices {
            self.passes[pass_idx]
                .post_submit(&mut post_ctx)
                .map_err(GraphExecuteError::Pass)?;
        }
        Ok(())
    }

    #[allow(clippy::map_entry)] // insert-on-miss pattern is clearer than Entry API here
    fn resolve_transient_textures(
        &self,
        device: &wgpu::Device,
        pool: &mut TransientPool,
        surface: TransientTextureResolveSurfaceParams,
        resources: &mut GraphResolvedResources,
    ) -> Result<(), GraphExecuteError> {
        let mut physical_slots: HashMap<usize, ResolvedGraphTexture> = HashMap::new();
        for (idx, compiled) in self.transient_textures.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            if !physical_slots.contains_key(&compiled.physical_slot) {
                let array_layers = compiled.desc.array_layers.resolve(surface.multiview_stereo);
                let key = TextureKey {
                    format: compiled
                        .desc
                        .format
                        .resolve(surface.surface_format, surface.depth_stencil_format),
                    extent: helpers::resolve_transient_extent(
                        compiled.desc.extent,
                        surface.viewport_px,
                        array_layers,
                    ),
                    mip_levels: compiled.desc.mip_levels,
                    sample_count: compiled.desc.sample_count.resolve(surface.sample_count),
                    dimension: compiled.desc.dimension,
                    array_layers,
                    usage_bits: compiled.usage.bits() as u64,
                };
                let lease = pool.acquire_texture_resource(
                    device,
                    key,
                    compiled.desc.label,
                    compiled.usage,
                )?;
                let layer_views = helpers::create_transient_layer_views(&lease.texture, key);
                physical_slots.insert(
                    compiled.physical_slot,
                    ResolvedGraphTexture {
                        pool_id: lease.pool_id,
                        physical_slot: compiled.physical_slot,
                        texture: lease.texture,
                        view: lease.view,
                        layer_views,
                    },
                );
            }
            let resolved = physical_slots[&compiled.physical_slot].clone();
            resources.set_transient_texture(TextureHandle(idx as u32), resolved);
        }
        Ok(())
    }

    #[allow(clippy::map_entry)]
    fn resolve_transient_buffers(
        &self,
        device: &wgpu::Device,
        pool: &mut TransientPool,
        viewport_px: (u32, u32),
        resources: &mut GraphResolvedResources,
    ) -> Result<(), GraphExecuteError> {
        let mut physical_slots: HashMap<usize, ResolvedGraphBuffer> = HashMap::new();
        for (idx, compiled) in self.transient_buffers.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            if !physical_slots.contains_key(&compiled.physical_slot) {
                let key = BufferKey {
                    size_policy: compiled.desc.size_policy,
                    usage_bits: compiled.usage.bits() as u64,
                };
                let size = helpers::resolve_buffer_size(compiled.desc.size_policy, viewport_px);
                let lease = pool.acquire_buffer_resource(
                    device,
                    key,
                    compiled.desc.label,
                    compiled.usage,
                    size,
                )?;
                physical_slots.insert(
                    compiled.physical_slot,
                    ResolvedGraphBuffer {
                        pool_id: lease.pool_id,
                        physical_slot: compiled.physical_slot,
                        buffer: lease.buffer,
                        size: lease.size,
                    },
                );
            }
            let resolved = physical_slots[&compiled.physical_slot].clone();
            resources
                .set_transient_buffer(super::super::resources::BufferHandle(idx as u32), resolved);
        }
        Ok(())
    }

    fn resolve_imported_textures(
        &self,
        resolved: &ResolvedView<'_>,
        resources: &mut GraphResolvedResources,
    ) {
        for (idx, import) in self.imported_textures.iter().enumerate() {
            let view = match &import.source {
                ImportSource::FrameTarget(FrameTargetRole::ColorAttachment) => {
                    resolved.backbuffer.cloned()
                }
                ImportSource::FrameTarget(FrameTargetRole::DepthAttachment) => {
                    Some(resolved.depth_view.clone())
                }
                ImportSource::External | ImportSource::PingPong(_) => None,
            };
            if let Some(view) = view {
                resources.set_imported_texture(
                    ImportedTextureHandle(idx as u32),
                    ResolvedImportedTexture { view },
                );
            }
        }
    }

    fn resolve_imported_buffers(
        &self,
        frame_resources: &crate::backend::FrameResourceManager,
        resolved: &ResolvedView<'_>,
        resources: &mut GraphResolvedResources,
    ) {
        let frame_gpu = frame_resources.frame_gpu();
        let cluster_refs = frame_gpu.and_then(|fgpu| {
            fgpu.cluster_cache.get_buffers(
                resolved.viewport_px,
                crate::backend::CLUSTER_COUNT_Z,
                resolved.multiview_stereo,
            )
        });
        for (idx, import) in self.imported_buffers.iter().enumerate() {
            let buffer = match &import.source {
                BufferImportSource::BackendFrameResource(BackendFrameBufferKind::Lights) => {
                    frame_gpu.map(|fgpu| fgpu.lights_buffer.clone())
                }
                BufferImportSource::BackendFrameResource(BackendFrameBufferKind::FrameUniforms) => {
                    frame_gpu.map(|fgpu| fgpu.frame_uniform.clone())
                }
                BufferImportSource::BackendFrameResource(
                    BackendFrameBufferKind::ClusterLightCounts,
                ) => cluster_refs
                    .as_ref()
                    .map(|refs| refs.cluster_light_counts.clone()),
                BufferImportSource::BackendFrameResource(
                    BackendFrameBufferKind::ClusterLightIndices,
                ) => cluster_refs
                    .as_ref()
                    .map(|refs| refs.cluster_light_indices.clone()),
                BufferImportSource::BackendFrameResource(BackendFrameBufferKind::PerDrawSlab) => {
                    frame_resources
                        .per_draw()
                        .map(|per_draw| per_draw.per_draw_storage.clone())
                }
                BufferImportSource::External | BufferImportSource::PingPong(_) => None,
            };
            if let Some(buffer) = buffer {
                resources.set_imported_buffer(
                    ImportedBufferHandle(idx as u32),
                    ResolvedImportedBuffer { buffer },
                );
            }
        }
    }

    fn resolve_view_from_target<'a>(
        target: &'a FrameViewTarget<'a>,
        gpu: &'a mut GpuContext,
        backbuffer_view_holder: &'a Option<wgpu::TextureView>,
    ) -> Result<ResolvedView<'a>, GraphExecuteError> {
        match target {
            FrameViewTarget::Swapchain => {
                let surface_format = gpu.config_format();
                let viewport_px = gpu.surface_extent_px();
                let bb = backbuffer_view_holder
                    .as_ref()
                    .map(|v| v as &wgpu::TextureView);
                let Some(bb_ref) = bb else {
                    return Err(GraphExecuteError::MissingSwapchainView);
                };
                let sample_count = gpu.swapchain_msaa_effective().max(1);
                let (depth_tex, depth_view) = gpu
                    .ensure_depth_target()
                    .map_err(|_| GraphExecuteError::DepthTarget)?;

                Ok(ResolvedView {
                    depth_texture: depth_tex,
                    depth_view,
                    backbuffer: Some(bb_ref),
                    surface_format,
                    viewport_px,
                    multiview_stereo: false,
                    offscreen_write_render_texture_asset_id: None,
                    occlusion_view: OcclusionViewId::Main,
                    sample_count,
                    msaa_color_view: None,
                    msaa_depth_view: None,
                    msaa_depth_resolve_r32_view: None,
                    msaa_depth_is_array: false,
                    msaa_stereo_depth_layer_views: None,
                    msaa_stereo_r32_layer_views: None,
                })
            }
            FrameViewTarget::ExternalMultiview(ext) => {
                let sample_count = gpu.swapchain_msaa_effective_stereo().max(1);
                Ok(ResolvedView {
                    depth_texture: ext.depth_texture,
                    depth_view: ext.depth_view,
                    backbuffer: Some(ext.color_view),
                    surface_format: ext.surface_format,
                    viewport_px: ext.extent_px,
                    multiview_stereo: true,
                    offscreen_write_render_texture_asset_id: None,
                    occlusion_view: OcclusionViewId::Main,
                    sample_count,
                    msaa_color_view: None,
                    msaa_depth_view: None,
                    msaa_depth_resolve_r32_view: None,
                    msaa_depth_is_array: false,
                    msaa_stereo_depth_layer_views: None,
                    msaa_stereo_r32_layer_views: None,
                })
            }
            FrameViewTarget::OffscreenRt(ext) => Ok(ResolvedView {
                depth_texture: ext.depth_texture,
                depth_view: ext.depth_view,
                backbuffer: Some(ext.color_view),
                surface_format: ext.color_format,
                viewport_px: ext.extent_px,
                multiview_stereo: false,
                offscreen_write_render_texture_asset_id: Some(ext.render_texture_asset_id),
                occlusion_view: OcclusionViewId::OffscreenRenderTexture(
                    ext.render_texture_asset_id,
                ),
                sample_count: 1,
                msaa_color_view: None,
                msaa_depth_view: None,
                msaa_depth_resolve_r32_view: None,
                msaa_depth_is_array: false,
                msaa_stereo_depth_layer_views: None,
                msaa_stereo_r32_layer_views: None,
            }),
        }
    }
}
