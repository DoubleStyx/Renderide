//! Render-context override state mirrored from host `RenderTransformOverride*` / `RenderMaterialOverride*`.

mod apply;
mod space_impl;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use apply::{
    apply_render_material_overrides_update_extracted,
    apply_render_transform_overrides_update_extracted, extract_render_material_overrides_update,
    extract_render_transform_overrides_update, ExtractedRenderMaterialOverridesUpdate,
    ExtractedRenderTransformOverridesUpdate,
};
pub use types::{
    MeshRendererOverrideTarget, RenderMaterialOverrideEntry, RenderTransformOverrideEntry,
};
