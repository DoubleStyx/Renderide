//! Material-driven raster state resolved from host shader properties (`_Stencil`, `_ZWrite`, `_Cull`, ...).
//!
//! Used by the mesh-forward draw prep path and reflective raster pipeline builders to key
//! [`wgpu::RenderPipeline`] instances consistently with host material overrides.

mod from_maps;
mod types;
mod unity_mapping;

#[cfg(test)]
pub use from_maps::material_render_state_for_lookup;
pub use from_maps::material_render_state_from_maps;
#[cfg(test)]
pub(crate) use types::{MaterialCullOverride, MaterialDepthOffsetState, MaterialStencilState};
pub use types::{
    MaterialDepthCompareDomain, MaterialDepthCompareOverride, MaterialRenderState, RasterFrontFace,
    RasterPrimitiveTopology,
};
