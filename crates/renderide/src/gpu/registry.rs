//! Pipeline registry: maps (shader_id, PipelineVariant) to RenderPipeline instances.
//!
//! Enables arbitrary shaders and prepares for host-uploaded shaders.

use std::collections::HashMap;
use std::sync::Arc;

use super::pipeline::{
    MaterialPipeline, NormalDebugMRTPipeline, NormalDebugPipeline, OverlayStencilMaskClearPipeline,
    OverlayStencilMaskClearSkinnedPipeline, OverlayStencilMaskWritePipeline,
    OverlayStencilMaskWriteSkinnedPipeline, OverlayStencilPipeline, OverlayStencilSkinnedPipeline,
    PbrMRTPipeline, PbrPipeline, RenderPipeline, SkinnedMRTPipeline, SkinnedPbrMRTPipeline,
    SkinnedPbrPipeline, SkinnedPipeline, UvDebugMRTPipeline, UvDebugPipeline,
};

/// Key for pipeline lookup: shader_id (None = builtin) and variant.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct PipelineKey(pub Option<i32>, pub PipelineVariant);

/// Variant of render pipeline (debug, skinned, material, PBR).
///
/// Ord is used for draw batching: MaskWrite < Content < MaskClear for GraphicsChunk flow.
#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
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
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn register_builtin(&mut self, device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) {
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
            Arc::new(NormalDebugMRTPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::UvDebugMRT),
            Arc::new(UvDebugMRTPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::SkinnedMRT),
            Arc::new(SkinnedMRTPipeline::new(device, config)),
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
            _ => return None,
        };
        self.pipelines.insert(key.clone(), Arc::clone(&pipeline));
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

/// Manages render pipelines via a registry and frame advancement for ring buffers.
pub struct PipelineManager {
    registry: PipelineRegistry,
    frame_index: u64,
}

impl PipelineManager {
    /// Creates the pipeline manager and registers builtin pipelines.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let mut registry = PipelineRegistry::new();
        registry.register_builtin(device, config);
        Self {
            registry,
            frame_index: 0,
        }
    }

    /// Advances frame index for ring buffer and returns the value to use this frame.
    pub fn advance_frame(&mut self) -> u64 {
        let idx = self.frame_index;
        self.frame_index = self.frame_index.wrapping_add(1);
        idx
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
