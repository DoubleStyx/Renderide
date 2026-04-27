//! Resource resolution helpers: transient pool leases and imported-resource lookups plus
//! per-view target resolution used before every pass encode.

use hashbrown::HashMap;

use crate::backend::{HistoryRegistry, HistoryResourceScope};
use crate::gpu::GpuContext;

use super::super::super::context::{
    GraphResolvedResources, ResolvedGraphBuffer, ResolvedGraphTexture, ResolvedImportedBuffer,
    ResolvedImportedHistoryTexture, ResolvedImportedTexture,
};
use super::super::super::error::GraphExecuteError;
use super::super::super::frame_params::OcclusionViewId;
use super::super::super::resources::{
    BackendFrameBufferKind, BufferImportSource, FrameTargetRole, HistorySlotId, ImportSource,
    ImportedBufferDecl, ImportedBufferHandle, ImportedTextureDecl, ImportedTextureHandle,
    SubresourceHandle, TextureHandle,
};
use super::super::super::transient_pool::{BufferKey, TextureKey, TransientPool};
use super::super::helpers;
use super::super::{CompiledRenderGraph, FrameViewTarget, ResolvedView};
use super::{OwnedResolvedView, TransientTextureResolveSurfaceParams};

impl CompiledRenderGraph {
    /// Acquires transient texture leases for this view and inserts them into `resources`.
    pub(super) fn resolve_transient_textures(
        &self,
        device: &wgpu::Device,
        limits: &crate::gpu::GpuLimits,
        pool: &mut TransientPool,
        surface: TransientTextureResolveSurfaceParams,
        resources: &mut GraphResolvedResources,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::resolve_transient_textures");
        let mut physical_slots: HashMap<usize, ResolvedGraphTexture> = HashMap::new();
        for (idx, compiled) in self.transient_textures.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            let resolved = if let Some(existing) = physical_slots.get(&compiled.physical_slot) {
                existing.clone()
            } else {
                let array_layers = compiled.desc.array_layers.resolve(surface.multiview_stereo);
                let key = TextureKey {
                    format: compiled.desc.format.resolve(
                        surface.surface_format,
                        surface.depth_stencil_format,
                        surface.scene_color_format,
                    ),
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
                    limits,
                    key,
                    compiled.desc.label,
                    compiled.usage,
                )?;
                let layer_views = helpers::create_transient_layer_views(&lease.texture, key);
                let inserted = ResolvedGraphTexture {
                    pool_id: lease.pool_id,
                    physical_slot: compiled.physical_slot,
                    texture: lease.texture,
                    view: lease.view,
                    layer_views,
                };
                let cloned = inserted.clone();
                physical_slots.insert(compiled.physical_slot, inserted);
                cloned
            };
            resources.set_transient_texture(TextureHandle(idx as u32), resolved);
        }
        Ok(())
    }

    /// Acquires transient buffer leases for this view and inserts them into `resources`.
    pub(super) fn resolve_transient_buffers(
        &self,
        device: &wgpu::Device,
        limits: &crate::gpu::GpuLimits,
        pool: &mut TransientPool,
        viewport_px: (u32, u32),
        resources: &mut GraphResolvedResources,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::resolve_transient_buffers");
        let mut physical_slots: HashMap<usize, ResolvedGraphBuffer> = HashMap::new();
        for (idx, compiled) in self.transient_buffers.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            let resolved = if let Some(existing) = physical_slots.get(&compiled.physical_slot) {
                existing.clone()
            } else {
                let key = BufferKey {
                    size_policy: compiled.desc.size_policy,
                    usage_bits: compiled.usage.bits() as u64,
                };
                let size = helpers::resolve_buffer_size(compiled.desc.size_policy, viewport_px);
                let lease = pool.acquire_buffer_resource(
                    device,
                    limits,
                    key,
                    compiled.desc.label,
                    compiled.usage,
                    size,
                )?;
                let inserted = ResolvedGraphBuffer {
                    pool_id: lease.pool_id,
                    physical_slot: compiled.physical_slot,
                    buffer: lease.buffer,
                    size: lease.size,
                };
                let cloned = inserted.clone();
                physical_slots.insert(compiled.physical_slot, inserted);
                cloned
            };
            resources.set_transient_buffer(
                super::super::super::resources::BufferHandle(idx as u32),
                resolved,
            );
        }
        Ok(())
    }

    /// Binds imported textures (frame color / depth attachments) into `resources`.
    pub(super) fn resolve_imported_textures(
        &self,
        resolved: &ResolvedView<'_>,
        history: &HistoryRegistry,
        resources: &mut GraphResolvedResources,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::resolve_imported_textures");
        for (idx, import) in self.imported_textures.iter().enumerate() {
            let resolved_import = match &import.source {
                ImportSource::FrameTarget(FrameTargetRole::ColorAttachment) => resolved
                    .backbuffer
                    .cloned()
                    .map(|view| ResolvedImportedTexture {
                        view,
                        history: None,
                    }),
                ImportSource::FrameTarget(FrameTargetRole::DepthAttachment) => {
                    Some(ResolvedImportedTexture {
                        view: resolved.depth_view.clone(),
                        history: None,
                    })
                }
                ImportSource::External => None,
                ImportSource::PingPong(slot) => {
                    let scope = history_scope_for_texture(*slot, resolved);
                    let texture_slot =
                        history.texture_slot_scoped(*slot, scope).ok_or_else(|| {
                            GraphExecuteError::missing_history_texture(*slot, import.label)
                        })?;
                    let guard = texture_slot.lock();
                    let (half_idx, half_name) = texture_history_half(import, history);
                    let texture = guard.half(half_idx).ok_or_else(|| {
                        GraphExecuteError::unallocated_history_texture(*slot, half_name)
                    })?;
                    Some(ResolvedImportedTexture {
                        view: texture.view.clone(),
                        history: Some(ResolvedImportedHistoryTexture {
                            texture: texture.texture.clone(),
                            mip_views: texture.mip_views.clone(),
                        }),
                    })
                }
            };
            if let Some(resolved_import) = resolved_import {
                resources.set_imported_texture(ImportedTextureHandle(idx as u32), resolved_import);
            }
        }
        Ok(())
    }

    /// Resolves subresource views declared on [`super::super::CompiledRenderGraph::subresources`]
    /// against their parent transient texture.
    ///
    /// Run after [`Self::resolve_transient_textures`] so the parent `wgpu::Texture` handles
    /// already exist. Subresources whose parent is not resolved (because the parent's transient
    /// index is culled or its lifetime is `None`) are left as `None` — callers that look them up
    /// get a harmless `None` instead of an encoder-time panic.
    pub(super) fn resolve_subresource_views(&self, resources: &mut GraphResolvedResources) {
        if self.subresources.is_empty() {
            return;
        }
        profiling::scope!("render::resolve_subresource_views");
        for (idx, desc) in self.subresources.iter().enumerate() {
            let Some(parent) = resources.transient_texture(desc.parent) else {
                continue;
            };
            let view = parent.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(desc.label),
                base_mip_level: desc.base_mip_level,
                mip_level_count: Some(desc.mip_level_count.max(1)),
                base_array_layer: desc.base_array_layer,
                array_layer_count: Some(desc.array_layer_count.max(1)),
                ..Default::default()
            });
            resources.set_subresource_view(SubresourceHandle(idx as u32), view);
        }
    }

    /// Binds imported backend buffers (lights, cluster tables, per-draw slab) into `resources`.
    pub(super) fn resolve_imported_buffers(
        &self,
        frame_resources: &crate::backend::FrameResourceManager,
        history: &HistoryRegistry,
        resolved: &ResolvedView<'_>,
        resources: &mut GraphResolvedResources,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::resolve_imported_buffers");
        let frame_gpu = frame_resources.frame_gpu();
        // All views share one cluster buffer; safe under single-submit because each view's
        // compute-then-raster sequence completes before the next view's compute overwrites.
        let cluster_refs = frame_resources.shared_cluster_buffer_refs();
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
                        .per_view_per_draw(resolved.occlusion_view)
                        .map(|per_draw| per_draw.lock().per_draw_storage.clone())
                }
                BufferImportSource::External => None,
                BufferImportSource::PingPong(slot) => {
                    let scope = history_scope_for_buffer(*slot, resolved);
                    let buffer_slot =
                        history.buffer_slot_scoped(*slot, scope).ok_or_else(|| {
                            GraphExecuteError::missing_history_buffer(*slot, import.label)
                        })?;
                    let guard = buffer_slot.lock();
                    let (half_idx, half_name) = buffer_history_half(import, history);
                    let buffer = guard.half(half_idx).ok_or_else(|| {
                        GraphExecuteError::unallocated_history_buffer(*slot, half_name)
                    })?;
                    Some(buffer.clone())
                }
            };
            if let Some(buffer) = buffer {
                resources.set_imported_buffer(
                    ImportedBufferHandle(idx as u32),
                    ResolvedImportedBuffer { buffer },
                );
            }
        }
        Ok(())
    }

    /// Resolves a [`FrameViewTarget`] into a [`ResolvedView`] with color/depth attachments.
    pub(super) fn resolve_view_from_target<'a>(
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
                    .map_err(GraphExecuteError::DepthTarget)?;

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
            }),
        }
    }

    /// Same as [`Self::resolve_view_from_target`] but owns its color/depth handles.
    pub(super) fn resolve_owned_view_from_target(
        target: &FrameViewTarget<'_>,
        gpu: &mut GpuContext,
        backbuffer_view_holder: &Option<wgpu::TextureView>,
    ) -> Result<OwnedResolvedView, GraphExecuteError> {
        let resolved = Self::resolve_view_from_target(target, gpu, backbuffer_view_holder)?;
        Ok(OwnedResolvedView {
            depth_texture: resolved.depth_texture.clone(),
            depth_view: resolved.depth_view.clone(),
            backbuffer: resolved.backbuffer.cloned(),
            surface_format: resolved.surface_format,
            viewport_px: resolved.viewport_px,
            multiview_stereo: resolved.multiview_stereo,
            offscreen_write_render_texture_asset_id: resolved
                .offscreen_write_render_texture_asset_id,
            occlusion_view: resolved.occlusion_view,
            sample_count: resolved.sample_count,
        })
    }
}

/// Resolves a texture history slot's scope for the current view.
fn history_scope_for_texture(
    slot: HistorySlotId,
    resolved: &ResolvedView<'_>,
) -> HistoryResourceScope {
    if slot == HistorySlotId::HI_Z {
        HistoryResourceScope::View(resolved.occlusion_view)
    } else {
        HistoryResourceScope::Global
    }
}

/// Resolves a buffer history slot's scope for the current view.
fn history_scope_for_buffer(
    slot: HistorySlotId,
    resolved: &ResolvedView<'_>,
) -> HistoryResourceScope {
    if slot == HistorySlotId::HI_Z {
        HistoryResourceScope::View(resolved.occlusion_view)
    } else {
        HistoryResourceScope::Global
    }
}

/// Selects the ping-pong half for a texture import based on declared access intent.
fn texture_history_half(
    import: &ImportedTextureDecl,
    history: &HistoryRegistry,
) -> (usize, &'static str) {
    if import.initial_access.writes() || import.final_access.writes() {
        (history.current_index(), "current")
    } else {
        (history.previous_index(), "previous")
    }
}

/// Selects the ping-pong half for a buffer import based on declared access intent.
fn buffer_history_half(
    import: &ImportedBufferDecl,
    history: &HistoryRegistry,
) -> (usize, &'static str) {
    if import.initial_access.writes() || import.final_access.writes() {
        (history.current_index(), "current")
    } else {
        (history.previous_index(), "previous")
    }
}
