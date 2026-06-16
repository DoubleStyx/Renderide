//! Unity shader asset names mapped to composed WGSL stems through the runtime shader package.
//!
//! Resolution uses [`crate::assets::util::normalize_unity_shader_lookup_key`] and probes
//! the package route manifest generated from material source stems.

use crate::assets::util::normalize_unity_shader_lookup_key;

#[cfg(test)]
mod tests;

/// Returns the default package material stem for a Unity shader asset name.
pub fn embedded_default_stem_for_shader_asset_name(name: &str) -> Option<String> {
    let key = normalize_unity_shader_lookup_key(name);
    crate::materials::shader_package::default_material_stem_for_asset_key(&key)
        .map(|stem| stem.to_string())
}
