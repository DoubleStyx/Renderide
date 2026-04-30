//! Material-driven raster state resolved from Unity-style properties (`_Stencil`, `_ZWrite`, `_Cull`, …).
//!
//! Used by the mesh-forward draw prep path and reflective raster pipeline builders to key
//! [`wgpu::RenderPipeline`] instances consistently with host material overrides.

mod from_maps;
mod types;
mod unity_mapping;

pub use from_maps::{material_render_state_for_lookup, material_render_state_from_maps};
pub use types::{
    MaterialCullOverride, MaterialDepthOffsetState, MaterialRenderState, MaterialStencilState,
    RasterFrontFace,
};
