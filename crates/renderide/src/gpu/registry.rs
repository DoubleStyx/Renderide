//! Pipeline registry: maps (shader_id, [`PipelineVariant`]) to [`RenderPipeline`] instances.
//!
//! Built-ins are indexed by [`PipelineKey`] and mirrored in [`super::pipeline_descriptor_cache::PipelineDescriptorCache`]
//! for stable descriptor hashing. Host-unlit programs share one [`super::pipeline::HostUnlitPipeline`] per
//! shader asset id (see [`PipelineVariant::Material`]).

use std::collections::HashMap;
use std::sync::Arc;

use crate::assets::{
    MaterialPropertyStore, resolve_native_ui_surface_blend_text,
    resolve_native_ui_surface_blend_unlit,
};
use crate::config::RenderConfig;

use super::pipeline::mrt::create_mrt_gbuffer_origin_bind_group_layout;
use super::pipeline::{
    HostUnlitPipeline, NormalDebugMRTPipeline, NormalDebugPipeline,
    OverlayStencilMaskClearPipeline, OverlayStencilMaskClearSkinnedPipeline,
    OverlayStencilMaskWritePipeline, OverlayStencilMaskWriteSkinnedPipeline,
    OverlayStencilPipeline, OverlayStencilSkinnedPipeline, PbrHostAlbedoPipeline, PbrMRTPipeline,
    PbrMrtRayQueryPipeline, PbrPipeline, PbrRayQueryPipeline, RenderPipeline, SkinnedMRTPipeline,
    SkinnedPbrMRTPipeline, SkinnedPbrMrtRayQueryPipeline, SkinnedPbrPipeline,
    SkinnedPbrRayQueryPipeline, SkinnedPipeline, UiTextUnlitNativePipeline, UiUnlitNativePipeline,
    UvDebugMRTPipeline, UvDebugPipeline,
};
use super::pipeline_descriptor_cache::PipelineDescriptorCache;

/// Key for pipeline lookup: shader_id (None = builtin) and variant.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct PipelineKey(pub Option<i32>, pub PipelineVariant);

/// Variant of render pipeline (debug, skinned, material, PBR).
///
/// Ord is used for draw batching: MaskWrite < Content < MaskClear for GraphicsChunk flow.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum PipelineVariant {
    /// Normal debug: colors surfaces by smooth normal.
    NormalDebug,
    /// UV debug: colors surfaces by UV coordinates.
    UvDebug,
    /// Skinned mesh: transforms vertices by weighted bone matrices.
    Skinned,
    /// Normal debug MRT: color, position, normal for RTAO.
    NormalDebugMRT,
    /// UV debug MRT: color, position, normal for RTAO.
    UvDebugMRT,
    /// Skinned MRT: color, position, normal for RTAO.
    SkinnedMRT,
    /// Overlay stencil MaskWrite: compare=Always, pass_op=Replace, write_mask=0xFF.
    OverlayStencilMaskWrite,
    /// Overlay stencil Content: compare=Equal, pass_op=Keep, write_mask=0.
    OverlayStencilContent,
    /// Overlay stencil MaskClear: compare=Always, pass_op=Zero, write_mask=0xFF.
    OverlayStencilMaskClear,
    /// Skinned overlay stencil MaskWrite.
    OverlayStencilMaskWriteSkinned,
    /// Skinned overlay stencil Content.
    OverlayStencilSkinned,
    /// Skinned overlay stencil MaskClear.
    OverlayStencilMaskClearSkinned,
    /// Normal debug with depth test disabled for orthographic screen-space overlay.
    OverlayNoDepthNormalDebug,
    /// UV debug with depth test disabled for orthographic screen-space overlay.
    OverlayNoDepthUvDebug,
    /// Skinned with depth test disabled for orthographic screen-space overlay.
    OverlayNoDepthSkinned,
    /// Host-resolved material: uses [`HostUnlitPipeline`] when the property store lists a shader
    /// for `material_id` (material / property block id from the draw).
    Material { material_id: i32 },
    /// Native WGSL Resonite `UI/Unlit` (orthographic overlay, canvas vertices).
    NativeUiUnlit { material_id: i32 },
    /// Native WGSL `UI/Text/Unlit`.
    NativeUiTextUnlit { material_id: i32 },
    /// Native `UI_Unlit` with GraphicsChunk stencil test in the overlay pass.
    NativeUiUnlitStencil { material_id: i32 },
    /// Native `UI_TextUnlit` with GraphicsChunk stencil test in the overlay pass.
    NativeUiTextUnlitStencil { material_id: i32 },
    /// PBR pipeline.
    Pbr,
    /// PBR MRT: PBR with G-buffer output for RTAO.
    PbrMRT,
    /// Skinned PBR: bone skinning with PBS lighting.
    SkinnedPbr,
    /// Skinned PBR MRT: PBR with G-buffer output for RTAO.
    SkinnedPbrMRT,
    /// PBR with fragment ray queries and TLAS shadow rays (requires ray tracing).
    PbrRayQuery,
    /// Forward PBR with host `_MainTex` albedo multiply (requires mesh UV0).
    PbrHostAlbedo,
    /// PBR MRT with fragment ray queries.
    PbrMRTRayQuery,
    /// Skinned PBR with fragment ray queries.
    SkinnedPbrRayQuery,
    /// Skinned PBR MRT with fragment ray queries.
    SkinnedPbrMRTRayQuery,
}

/// Maps pipeline keys to render pipelines. Supports builtin registration and lazy creation.
pub struct PipelineRegistry {
    pipelines: HashMap<PipelineKey, Arc<dyn RenderPipeline>>,
    descriptor_cache: PipelineDescriptorCache,
}

impl PipelineRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            descriptor_cache: PipelineDescriptorCache::default(),
        }
    }

    fn put_builtin(
        &mut self,
        variant: PipelineVariant,
        config: &wgpu::SurfaceConfiguration,
        pipeline: Arc<dyn RenderPipeline>,
    ) {
        self.pipelines
            .insert(PipelineKey(None, variant), Arc::clone(&pipeline));
        self.descriptor_cache.insert(
            PipelineDescriptorCache::builtin_key(variant, config.format),
            pipeline,
        );
    }

    fn put_lazy(
        &mut self,
        key: PipelineKey,
        variant: PipelineVariant,
        config: &wgpu::SurfaceConfiguration,
        pipeline: Arc<dyn RenderPipeline>,
    ) {
        self.pipelines.insert(key, Arc::clone(&pipeline));
        self.descriptor_cache.insert(
            PipelineDescriptorCache::builtin_key(variant, config.format),
            pipeline,
        );
    }

    /// Registers builtin pipelines for the given device and surface configuration.
    ///
    /// `mrt_gbuffer_origin_layout` must match [`PipelineManager::mrt_gbuffer_origin_layout`].
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn register_builtin(
        &mut self,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        mrt_gbuffer_origin_layout: &wgpu::BindGroupLayout,
    ) {
        self.put_builtin(
            PipelineVariant::NormalDebug,
            config,
            Arc::new(NormalDebugPipeline::new(device, config, false)),
        );
        self.put_builtin(
            PipelineVariant::UvDebug,
            config,
            Arc::new(UvDebugPipeline::new(device, config, false)),
        );
        self.put_builtin(
            PipelineVariant::Skinned,
            config,
            Arc::new(SkinnedPipeline::new(device, config, None, false)),
        );
        self.put_builtin(
            PipelineVariant::NormalDebugMRT,
            config,
            Arc::new(NormalDebugMRTPipeline::new(
                device,
                config,
                mrt_gbuffer_origin_layout,
            )),
        );
        self.put_builtin(
            PipelineVariant::UvDebugMRT,
            config,
            Arc::new(UvDebugMRTPipeline::new(
                device,
                config,
                mrt_gbuffer_origin_layout,
            )),
        );
        self.put_builtin(
            PipelineVariant::SkinnedMRT,
            config,
            Arc::new(SkinnedMRTPipeline::new(
                device,
                config,
                mrt_gbuffer_origin_layout,
            )),
        );
        self.put_builtin(
            PipelineVariant::OverlayStencilMaskWrite,
            config,
            Arc::new(OverlayStencilMaskWritePipeline::new(device, config)),
        );
        self.put_builtin(
            PipelineVariant::OverlayStencilContent,
            config,
            Arc::new(OverlayStencilPipeline::new(device, config)),
        );
        self.put_builtin(
            PipelineVariant::OverlayStencilMaskClear,
            config,
            Arc::new(OverlayStencilMaskClearPipeline::new(device, config)),
        );
        self.put_builtin(
            PipelineVariant::OverlayStencilMaskWriteSkinned,
            config,
            Arc::new(OverlayStencilMaskWriteSkinnedPipeline::new(device, config)),
        );
        self.put_builtin(
            PipelineVariant::OverlayStencilSkinned,
            config,
            Arc::new(OverlayStencilSkinnedPipeline::new(device, config)),
        );
        self.put_builtin(
            PipelineVariant::OverlayStencilMaskClearSkinned,
            config,
            Arc::new(OverlayStencilMaskClearSkinnedPipeline::new(device, config)),
        );
        self.put_builtin(
            PipelineVariant::OverlayNoDepthNormalDebug,
            config,
            Arc::new(NormalDebugPipeline::new(device, config, true)),
        );
        self.put_builtin(
            PipelineVariant::OverlayNoDepthUvDebug,
            config,
            Arc::new(UvDebugPipeline::new(device, config, true)),
        );
        self.put_builtin(
            PipelineVariant::OverlayNoDepthSkinned,
            config,
            Arc::new(SkinnedPipeline::new(device, config, None, true)),
        );
        self.put_builtin(
            PipelineVariant::PbrMRT,
            config,
            Arc::new(PbrMRTPipeline::new(device, config)),
        );
        self.put_builtin(
            PipelineVariant::SkinnedPbr,
            config,
            Arc::new(SkinnedPbrPipeline::new(device, config)),
        );
        self.put_builtin(
            PipelineVariant::SkinnedPbrMRT,
            config,
            Arc::new(SkinnedPbrMRTPipeline::new(device, config)),
        );
    }

    /// Returns the pipeline for the key, or lazily creates it for Material/Pbr.
    /// Builtins must be registered via [`Self::register_builtin`] before use.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn get_or_create(
        &mut self,
        key: PipelineKey,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        material_store: Option<&MaterialPropertyStore>,
        render_config: &RenderConfig,
    ) -> Option<Arc<dyn RenderPipeline>> {
        if let Some(p) = self.pipelines.get(&key) {
            return Some(Arc::clone(p));
        }
        match &key.1 {
            PipelineVariant::Material { material_id } => {
                let store = material_store?;
                let shader_id = store.shader_asset_for_block(*material_id)?;
                let dk = PipelineDescriptorCache::host_unlit_key(shader_id, config.format);
                let pipeline: Arc<dyn RenderPipeline> =
                    if let Some(p) = self.descriptor_cache.get(dk) {
                        p
                    } else {
                        let p: Arc<dyn RenderPipeline> =
                            Arc::new(HostUnlitPipeline::new(device, config));
                        self.descriptor_cache.insert(dk, Arc::clone(&p));
                        p
                    };
                self.pipelines.insert(key, Arc::clone(&pipeline));
                Some(pipeline)
            }
            PipelineVariant::NativeUiUnlit { material_id } => {
                let store = material_store?;
                let shader_id = store.shader_asset_for_block(*material_id)?;
                let blend = resolve_native_ui_surface_blend_unlit(
                    store,
                    *material_id,
                    &render_config.ui_unlit_property_ids,
                    render_config.native_ui_default_surface_blend,
                );
                let dk =
                    PipelineDescriptorCache::native_ui_unlit_key(shader_id, config.format, blend);
                let pipeline: Arc<dyn RenderPipeline> =
                    if let Some(p) = self.descriptor_cache.get(dk) {
                        p
                    } else {
                        let p: Arc<dyn RenderPipeline> =
                            Arc::new(UiUnlitNativePipeline::new(device, config, blend));
                        self.descriptor_cache.insert(dk, Arc::clone(&p));
                        p
                    };
                self.pipelines.insert(key, Arc::clone(&pipeline));
                Some(pipeline)
            }
            PipelineVariant::NativeUiTextUnlit { material_id } => {
                let store = material_store?;
                let shader_id = store.shader_asset_for_block(*material_id)?;
                let blend = resolve_native_ui_surface_blend_text(
                    store,
                    *material_id,
                    &render_config.ui_text_unlit_property_ids,
                    render_config.native_ui_default_surface_blend,
                );
                let dk =
                    PipelineDescriptorCache::native_ui_text_key(shader_id, config.format, blend);
                let pipeline: Arc<dyn RenderPipeline> =
                    if let Some(p) = self.descriptor_cache.get(dk) {
                        p
                    } else {
                        let p: Arc<dyn RenderPipeline> =
                            Arc::new(UiTextUnlitNativePipeline::new(device, config, blend));
                        self.descriptor_cache.insert(dk, Arc::clone(&p));
                        p
                    };
                self.pipelines.insert(key, Arc::clone(&pipeline));
                Some(pipeline)
            }
            PipelineVariant::NativeUiUnlitStencil { material_id } => {
                let store = material_store?;
                let shader_id = store.shader_asset_for_block(*material_id)?;
                let blend = resolve_native_ui_surface_blend_unlit(
                    store,
                    *material_id,
                    &render_config.ui_unlit_property_ids,
                    render_config.native_ui_default_surface_blend,
                );
                let dk = PipelineDescriptorCache::native_ui_unlit_stencil_key(
                    shader_id,
                    config.format,
                    blend,
                );
                let pipeline: Arc<dyn RenderPipeline> =
                    if let Some(p) = self.descriptor_cache.get(dk) {
                        p
                    } else {
                        let p: Arc<dyn RenderPipeline> = Arc::new(
                            UiUnlitNativePipeline::new_with_stencil(device, config, blend),
                        );
                        self.descriptor_cache.insert(dk, Arc::clone(&p));
                        p
                    };
                self.pipelines.insert(key, Arc::clone(&pipeline));
                Some(pipeline)
            }
            PipelineVariant::NativeUiTextUnlitStencil { material_id } => {
                let store = material_store?;
                let shader_id = store.shader_asset_for_block(*material_id)?;
                let blend = resolve_native_ui_surface_blend_text(
                    store,
                    *material_id,
                    &render_config.ui_text_unlit_property_ids,
                    render_config.native_ui_default_surface_blend,
                );
                let dk = PipelineDescriptorCache::native_ui_text_stencil_key(
                    shader_id,
                    config.format,
                    blend,
                );
                let pipeline: Arc<dyn RenderPipeline> =
                    if let Some(p) = self.descriptor_cache.get(dk) {
                        p
                    } else {
                        let p: Arc<dyn RenderPipeline> = Arc::new(
                            UiTextUnlitNativePipeline::new_with_stencil(device, config, blend),
                        );
                        self.descriptor_cache.insert(dk, Arc::clone(&p));
                        p
                    };
                self.pipelines.insert(key, Arc::clone(&pipeline));
                Some(pipeline)
            }
            PipelineVariant::Pbr => {
                let pipeline: Arc<dyn RenderPipeline> = Arc::new(PbrPipeline::new(device, config));
                self.put_lazy(key, PipelineVariant::Pbr, config, Arc::clone(&pipeline));
                Some(pipeline)
            }
            PipelineVariant::PbrHostAlbedo => {
                let pipeline: Arc<dyn RenderPipeline> =
                    Arc::new(PbrHostAlbedoPipeline::new(device, config));
                self.put_lazy(
                    key,
                    PipelineVariant::PbrHostAlbedo,
                    config,
                    Arc::clone(&pipeline),
                );
                Some(pipeline)
            }
            PipelineVariant::PbrRayQuery => {
                if !device
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                {
                    return None;
                }
                let pipeline: Arc<dyn RenderPipeline> =
                    Arc::new(PbrRayQueryPipeline::new(device, config));
                self.put_lazy(
                    key,
                    PipelineVariant::PbrRayQuery,
                    config,
                    Arc::clone(&pipeline),
                );
                Some(pipeline)
            }
            PipelineVariant::PbrMRTRayQuery => {
                if !device
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                {
                    return None;
                }
                let pipeline: Arc<dyn RenderPipeline> =
                    Arc::new(PbrMrtRayQueryPipeline::new(device, config));
                self.put_lazy(
                    key,
                    PipelineVariant::PbrMRTRayQuery,
                    config,
                    Arc::clone(&pipeline),
                );
                Some(pipeline)
            }
            PipelineVariant::SkinnedPbrRayQuery => {
                if !device
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                {
                    return None;
                }
                let pipeline: Arc<dyn RenderPipeline> =
                    Arc::new(SkinnedPbrRayQueryPipeline::new(device, config));
                self.put_lazy(
                    key,
                    PipelineVariant::SkinnedPbrRayQuery,
                    config,
                    Arc::clone(&pipeline),
                );
                Some(pipeline)
            }
            PipelineVariant::SkinnedPbrMRTRayQuery => {
                if !device
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                {
                    return None;
                }
                let pipeline: Arc<dyn RenderPipeline> =
                    Arc::new(SkinnedPbrMrtRayQueryPipeline::new(device, config));
                self.put_lazy(
                    key,
                    PipelineVariant::SkinnedPbrMRTRayQuery,
                    config,
                    Arc::clone(&pipeline),
                );
                Some(pipeline)
            }
            _ => None,
        }
    }

    /// Removes [`PipelineKey`] rows for the given material id. Does not remove shared host-unlit
    /// descriptor-cache entries (other materials may share the same shader asset).
    pub fn evict_material(&mut self, material_id: i32) {
        self.pipelines.retain(|k, _| match &k.1 {
            PipelineVariant::Material { material_id: m } if *m == material_id => false,
            PipelineVariant::NativeUiUnlit { material_id: m } if *m == material_id => false,
            PipelineVariant::NativeUiTextUnlit { material_id: m } if *m == material_id => false,
            PipelineVariant::NativeUiUnlitStencil { material_id: m } if *m == material_id => false,
            PipelineVariant::NativeUiTextUnlitStencil { material_id: m } if *m == material_id => {
                false
            }
            _ => true,
        });
    }

    /// Drops the descriptor-cache slot for a host-unlit program so the next use rebuilds it.
    ///
    /// Existing [`PipelineKey`] rows may still hold a strong [`Arc`] to the old pipeline until
    /// [`Self::evict_material`] removes those keys or the registry is recreated.
    pub fn evict_host_unlit_shader(&mut self, shader_asset_id: i32, format: wgpu::TextureFormat) {
        self.descriptor_cache
            .remove_host_unlit(shader_asset_id, format);
        self.descriptor_cache
            .remove_native_ui(shader_asset_id, format);
    }
}

impl Default for PipelineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages render pipelines and [`super::GpuFrameScheduler`] state for batched uniform ring buffers.
pub struct PipelineManager {
    registry: PipelineRegistry,
    frame_scheduler: super::frame_scheduler::GpuFrameScheduler,
    /// Shared layout for MRT debug pipelines' group 1 (g-buffer world origin). Also used by [`super::GpuState::ensure_mrt_gbuffer_origin_resources`].
    mrt_gbuffer_origin_bgl: wgpu::BindGroupLayout,
}

impl PipelineManager {
    /// Creates the pipeline manager and registers builtin pipelines.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let mrt_gbuffer_origin_bgl = create_mrt_gbuffer_origin_bind_group_layout(device);
        let mut registry = PipelineRegistry::new();
        registry.register_builtin(device, config, &mrt_gbuffer_origin_bgl);
        Self {
            registry,
            frame_scheduler: super::frame_scheduler::GpuFrameScheduler::new(),
            mrt_gbuffer_origin_bgl,
        }
    }

    /// Bind group layout for [`super::pipeline::mrt::MrtGbufferOriginUniform`] (MRT debug fragment group 1).
    pub fn mrt_gbuffer_origin_layout(&self) -> &wgpu::BindGroupLayout {
        &self.mrt_gbuffer_origin_bgl
    }

    /// Acquires the next ring-buffer frame index, waiting if too many submits are still in flight.
    pub fn acquire_frame_index(&mut self, device: &wgpu::Device) -> u64 {
        self.frame_scheduler.acquire_frame_index(device)
    }

    /// Records a queue submission that used `frame_index` from [`Self::acquire_frame_index`].
    pub fn record_submission(&mut self, submission: wgpu::SubmissionIndex, frame_index: u64) {
        self.frame_scheduler
            .record_submission(submission, frame_index);
    }

    /// Returns the pipeline for the key, creating it lazily for Material/Pbr if needed.
    pub fn get_pipeline(
        &mut self,
        key: PipelineKey,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        material_store: Option<&MaterialPropertyStore>,
        render_config: &RenderConfig,
    ) -> Option<Arc<dyn RenderPipeline>> {
        self.registry
            .get_or_create(key, device, config, material_store, render_config)
    }

    /// Evicts pipelines for the given material. Call when a material is unloaded.
    pub fn evict_material(&mut self, material_id: i32) {
        self.registry.evict_material(material_id);
    }

    /// Evicts the host-unlit GPU pipeline cached for `shader_asset_id`.
    pub fn evict_host_unlit_shader(&mut self, shader_asset_id: i32, format: wgpu::TextureFormat) {
        self.registry
            .evict_host_unlit_shader(shader_asset_id, format);
    }
}
