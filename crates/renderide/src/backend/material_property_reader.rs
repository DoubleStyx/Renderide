//! Name-based material property readers shared by backend environment systems.

use crate::assets::texture::{HostTextureAssetKind, unpack_host_texture_packed};
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};

/// Reads a packed texture property by host name.
pub(crate) fn texture_property(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
) -> Option<(i32, HostTextureAssetKind)> {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Texture(packed)) => unpack_host_texture_packed(*packed),
        _ => None,
    }
}

/// Reads a scalar float material property by host name.
pub(crate) fn float_property(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
    fallback: f32,
) -> f32 {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Float(v)) => *v,
        _ => fallback,
    }
}

/// Reads a float4 material property by host name.
pub(crate) fn float4_property(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
    fallback: [f32; 4],
) -> [f32; 4] {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Float4(v)) => *v,
        _ => fallback,
    }
}

/// Reads up to sixteen float4 rows from a material property.
pub(crate) fn float4_array16_property(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
) -> [[f32; 4]; 16] {
    let pid = registry.intern(name);
    let mut out = [[0.0; 4]; 16];
    if let Some(MaterialPropertyValue::Float4Array(values)) = store.get_merged(lookup, pid) {
        for (dst, src) in out.iter_mut().zip(values.iter()) {
            *dst = *src;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float4_property_returns_fallback_for_missing_property() {
        let store = MaterialPropertyStore::new();
        let registry = PropertyIdRegistry::new();
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 1,
            mesh_property_block_slot0: None,
        };

        assert_eq!(
            float4_property(&store, &registry, lookup, "_Missing", [1.0, 2.0, 3.0, 4.0]),
            [1.0, 2.0, 3.0, 4.0]
        );
    }
}
