//! Maps host material property ids to `@group(1)` WGSL bindings (textures, uniform blocks).
//!
//! [`crate::materials::host_data::MaterialPropertyStore`] and [`crate::materials::host_data::PropertyIdRegistry`]
//! capture Unity [`Material`](https://docs.unity3d.com/ScriptReference/Material.html) /
//! [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html) data on
//! CPU. GPU uploads use [`super::reflect_raster_material_wgsl`] and WGSL identifiers that match host
//! property names to pack [`crate::materials::host_data::MaterialPropertyValue`] into bind group **1**
//! (group **0** is always frame globals).

/// Per-logical-shader layout describing which property ids feed which `@group(1)` bindings (reserved).
#[derive(Debug, Default)]
pub struct MaterialPropertyGpuLayout {
    /// Reserved until the shipped shader metadata lists `property_name` -> `binding` pairs alongside reflection.
    pub _pending: (),
}

impl MaterialPropertyGpuLayout {
    /// Builds an empty layout (no GPU bindings yet).
    pub fn empty() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_layout_matches_default_layout() {
        assert_eq!(
            format!("{:?}", MaterialPropertyGpuLayout::empty()),
            format!("{:?}", MaterialPropertyGpuLayout::default())
        );
    }
}
