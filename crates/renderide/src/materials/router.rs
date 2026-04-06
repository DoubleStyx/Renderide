//! Maps host shader asset ids from `set_shader` to renderer [`super::MaterialFamilyId`].
//!
//! Populated from [`crate::assets::shader::resolve_shader_upload`] when the host sends
//! [`crate::shared::ShaderUpload`]. Unknown ids use [`MaterialRouter::fallback`].

use std::collections::HashMap;

use super::MaterialFamilyId;

/// Shader asset id → material family; unknown ids use [`Self::fallback`].
#[derive(Debug)]
pub struct MaterialRouter {
    shader_to_family: HashMap<i32, MaterialFamilyId>,
    /// Default when `shader_to_family` has no entry.
    pub fallback: MaterialFamilyId,
}

impl MaterialRouter {
    /// Builds a router with only a fallback family.
    pub fn new(fallback: MaterialFamilyId) -> Self {
        Self {
            shader_to_family: HashMap::new(),
            fallback,
        }
    }

    /// Inserts or replaces a host shader → family mapping.
    pub fn set_shader_family(&mut self, shader_asset_id: i32, family: MaterialFamilyId) {
        self.shader_to_family.insert(shader_asset_id, family);
    }

    /// Resolves the family for a host shader asset id.
    pub fn family_for_shader_asset(&self, shader_asset_id: i32) -> MaterialFamilyId {
        self.shader_to_family
            .get(&shader_asset_id)
            .copied()
            .unwrap_or(self.fallback)
    }

    /// Drops a host shader id mapping after [`crate::shared::ShaderUnload`].
    pub fn remove_shader_family(&mut self, shader_asset_id: i32) {
        self.shader_to_family.remove(&shader_asset_id);
    }

    /// Returns the mapped family when the host id was registered via [`Self::set_shader_family`].
    pub fn get_shader_family(&self, shader_asset_id: i32) -> Option<MaterialFamilyId> {
        self.shader_to_family.get(&shader_asset_id).copied()
    }

    /// Host shader asset ids and families, sorted by id (for debug HUD).
    pub fn routes_sorted_for_hud(&self) -> Vec<(i32, MaterialFamilyId)> {
        let mut v: Vec<_> = self
            .shader_to_family
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        v.sort_by_key(|(k, _)| *k);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::MaterialRouter;
    use crate::materials::{MaterialFamilyId, SOLID_COLOR_FAMILY_ID};

    #[test]
    fn remove_shader_family_clears_entry() {
        let mut r = MaterialRouter::new(MaterialFamilyId(99));
        r.set_shader_family(7, SOLID_COLOR_FAMILY_ID);
        assert_eq!(r.get_shader_family(7), Some(SOLID_COLOR_FAMILY_ID));
        r.remove_shader_family(7);
        assert_eq!(r.get_shader_family(7), None);
        assert_eq!(r.family_for_shader_asset(7), MaterialFamilyId(99));
    }
}
