//! Maps host material property ids to `@group(1)` WGSL bindings (textures, uniform blocks).
//!
//! [`crate::assets::material::MaterialPropertyStore`] and [`crate::assets::material::PropertyIdRegistry`]
//! capture Unity [`Material`](https://docs.unity3d.com/ScriptReference/Material.html) /
//! [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html) data on
//! CPU. GPU uploads use [`super::reflect_raster_material_wgsl`] (or per-shader manifest entries) to pack
//! [`crate::assets::material::MaterialPropertyValue`] into bind group **1** (group **0** is always frame globals).

use crate::assets::material::PropertyIdRegistry;

/// Stable property-name interning for Resonite `Shader "Unlit"` (`_Tex`, `_Color`, `_Tex_ST`, `_Cutoff`).
#[derive(Clone, Copy, Debug)]
pub struct WorldUnlitPropertyIds {
    /// `_Tex`
    pub tex: i32,
    /// `_Color`
    pub color: i32,
    /// `_Tex_ST`
    pub tex_st: i32,
    /// `_Cutoff`
    pub cutoff: i32,
}

impl WorldUnlitPropertyIds {
    /// Interns Unity-style property names in a fixed order for stable ids.
    pub fn new(registry: &PropertyIdRegistry) -> Self {
        Self {
            tex: registry.intern("_Tex"),
            color: registry.intern("_Color"),
            tex_st: registry.intern("_Tex_ST"),
            cutoff: registry.intern("_Cutoff"),
        }
    }
}

/// Per-logical-shader layout describing which property ids feed which `@group(1)` bindings (reserved).
#[derive(Debug, Default)]
pub struct MaterialPropertyGpuLayout {
    /// Reserved until manifests list `property_name` → `binding` pairs alongside reflection.
    pub _pending: (),
}

impl MaterialPropertyGpuLayout {
    /// Builds an empty layout (no GPU bindings yet).
    pub fn empty() -> Self {
        Self::default()
    }
}
