//! Render-context override state mirrored from host `RenderTransformOverride*` / `RenderMaterialOverride*`.

mod apply;
mod space_impl;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use apply::{
    ExtractedRenderMaterialOverridesUpdate, ExtractedRenderTransformOverridesUpdate,
    apply_render_material_overrides_update_extracted,
    apply_render_transform_overrides_update_extracted, extract_render_material_overrides_update,
    extract_render_transform_overrides_update,
};
pub(in crate::scene) use types::decode_packed_mesh_renderer_target;
pub use types::{
    MeshRendererOverrideTarget, RenderMaterialOverrideEntry, RenderTransformOverrideEntry,
};
