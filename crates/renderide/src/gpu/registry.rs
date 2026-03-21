//! Pipeline registry: maps (shader_id, PipelineVariant) to RenderPipeline instances.
//!
//! Enables arbitrary shaders and prepares for host-uploaded shaders.

use std::collections::HashMap;
use std::sync::Arc;

use super::pipeline::mrt::create_mrt_gbuffer_origin_bind_group_layout;
use super::pipeline::{
    MaterialPipeline, NormalDebugMRTPipeline, NormalDebugPipeline, OverlayStencilMaskClearPipeline,
    OverlayStencilMaskClearSkinnedPipeline, OverlayStencilMaskWritePipeline,
    OverlayStencilMaskWriteSkinnedPipeline, OverlayStencilPipeline, OverlayStencilSkinnedPipeline,
    PbrMRTPipeline, PbrMrtRayQueryPipeline, PbrPipeline, PbrRayQueryPipeline, RenderPipeline,
    SkinnedMRTPipeline, SkinnedPbrMRTPipeline, SkinnedPbrMrtRayQueryPipeline, SkinnedPbrPipeline,
    SkinnedPbrRayQueryPipeline, SkinnedPipeline, UvDebugMRTPipeline, UvDebugPipeline,
};

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
    /// Material-based pipeline for a specific material.
    Material { material_id: i32 },
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
}

impl PipelineRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
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
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::NormalDebug),
            Arc::new(NormalDebugPipeline::new(device, config, false)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::UvDebug),
            Arc::new(UvDebugPipeline::new(device, config, false)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::Skinned),
            Arc::new(SkinnedPipeline::new(device, config, None, false)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::NormalDebugMRT),
            Arc::new(NormalDebugMRTPipeline::new(
                device,
                config,
                mrt_gbuffer_origin_layout,
            )),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::UvDebugMRT),
            Arc::new(UvDebugMRTPipeline::new(
                device,
                config,
                mrt_gbuffer_origin_layout,
            )),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::SkinnedMRT),
            Arc::new(SkinnedMRTPipeline::new(
                device,
                config,
                mrt_gbuffer_origin_layout,
            )),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayStencilMaskWrite),
            Arc::new(OverlayStencilMaskWritePipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayStencilContent),
            Arc::new(OverlayStencilPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayStencilMaskClear),
            Arc::new(OverlayStencilMaskClearPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayStencilMaskWriteSkinned),
            Arc::new(OverlayStencilMaskWriteSkinnedPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayStencilSkinned),
            Arc::new(OverlayStencilSkinnedPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayStencilMaskClearSkinned),
            Arc::new(OverlayStencilMaskClearSkinnedPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayNoDepthNormalDebug),
            Arc::new(NormalDebugPipeline::new(device, config, true)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayNoDepthUvDebug),
            Arc::new(UvDebugPipeline::new(device, config, true)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayNoDepthSkinned),
            Arc::new(SkinnedPipeline::new(device, config, None, true)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::PbrMRT),
            Arc::new(PbrMRTPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::SkinnedPbr),
            Arc::new(SkinnedPbrPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::SkinnedPbrMRT),
            Arc::new(SkinnedPbrMRTPipeline::new(device, config)),
        );
    }

    /// Returns the pipeline for the key, or lazily creates it for Material/Pbr.
    /// Builtins must be registered via `register_builtin` before use.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn get_or_create(
        &mut self,
        key: PipelineKey,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> Option<Arc<dyn RenderPipeline>> {
        if let Some(p) = self.pipelines.get(&key) {
            return Some(Arc::clone(p));
        }
        let pipeline: Arc<dyn RenderPipeline> = match &key.1 {
            PipelineVariant::Material { .. } => Arc::new(MaterialPipeline::new(device, config)),
            PipelineVariant::Pbr => Arc::new(PbrPipeline::new(device, config)),
            PipelineVariant::PbrRayQuery => {
                if !device
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                {
                    return None;
                }
                Arc::new(PbrRayQueryPipeline::new(device, config))
            }
            PipelineVariant::PbrMRTRayQuery => {
                if !device
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                {
                    return None;
                }
                Arc::new(PbrMrtRayQueryPipeline::new(device, config))
            }
            PipelineVariant::SkinnedPbrRayQuery => {
                if !device
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                {
                    return None;
                }
                Arc::new(SkinnedPbrRayQueryPipeline::new(device, config))
            }
            PipelineVariant::SkinnedPbrMRTRayQuery => {
                if !device
                    .features()
                    .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
                {
                    return None;
                }
                Arc::new(SkinnedPbrMrtRayQueryPipeline::new(device, config))
            }
            _ => return None,
        };
        self.pipelines.insert(key, Arc::clone(&pipeline));
        Some(pipeline)
    }

    /// Removes pipelines for the given material ID. Call when a material is unloaded to avoid unbounded growth.
    pub fn evict_material(&mut self, material_id: i32) {
        self.pipelines
            .retain(|k, _| !matches!(&k.1, PipelineVariant::Material { material_id: mid } if *mid == material_id));
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
    ) -> Option<Arc<dyn RenderPipeline>> {
        self.registry.get_or_create(key, device, config)
    }

    /// Evicts pipelines for the given material. Call when a material is unloaded.
    pub fn evict_material(&mut self, material_id: i32) {
        self.registry.evict_material(material_id);
    }
}
