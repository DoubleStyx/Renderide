//! Pipeline registry: maps (shader_id, PipelineVariant) to RenderPipeline instances.
//!
//! Enables arbitrary shaders and prepares for host-uploaded shaders.

use std::collections::HashMap;
use std::sync::Arc;

use super::pipeline::{
    MaterialPipeline, NormalDebugPipeline, OverlayStencilPipeline, OverlayStencilSkinnedPipeline,
    PbrPipeline, RenderPipeline, SkinnedPipeline, UvDebugPipeline,
};

/// Key for pipeline lookup: shader_id (None = builtin) and variant.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct PipelineKey(pub Option<i32>, pub PipelineVariant);

/// Variant of render pipeline (debug, skinned, material, PBR).
#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum PipelineVariant {
    /// Normal debug: colors surfaces by smooth normal.
    NormalDebug,
    /// UV debug: colors surfaces by UV coordinates.
    UvDebug,
    /// Skinned mesh: transforms vertices by weighted bone matrices.
    Skinned,
    /// Overlay with stencil for GraphicsChunk masking (Content: Equal compare, Keep op).
    OverlayStencilContent,
    /// Skinned overlay with stencil for GraphicsChunk masking.
    OverlayStencilSkinned,
    /// Material-based pipeline for a specific material.
    Material { material_id: i32 },
    /// PBR pipeline.
    Pbr,
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
    pub fn register_builtin(
        &mut self,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) {
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::NormalDebug),
            Arc::new(NormalDebugPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::UvDebug),
            Arc::new(UvDebugPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::Skinned),
            Arc::new(SkinnedPipeline::new(device, config, false)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayStencilContent),
            Arc::new(OverlayStencilPipeline::new(device, config)),
        );
        self.pipelines.insert(
            PipelineKey(None, PipelineVariant::OverlayStencilSkinned),
            Arc::new(OverlayStencilSkinnedPipeline::new(device, config)),
        );
    }

    /// Returns the pipeline for the key, or lazily creates it for Material/Pbr.
    /// Builtins must be registered via `register_builtin` before use.
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
}
