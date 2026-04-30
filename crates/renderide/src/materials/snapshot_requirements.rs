//! Unified flags describing whether a material consumes scene-color, scene-depth, or
//! intersection-pass GPU resources.
//!
//! Reflection of a material's WGSL surfaces three independent boolean requirements: whether the
//! shader samples a scene-color snapshot, whether it samples a scene-depth snapshot, and whether
//! it needs an intersection pre-pass. They were carried as three separate fields on
//! [`crate::materials::ReflectedRasterLayout`] and exposed through six near-identical free
//! functions (three for raw WGSL, three for embedded stems). [`SnapshotRequirements`] folds them
//! into one struct so callers can ask for the whole set in a single lookup.

/// Scene-snapshot resources required by a reflected material.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SnapshotRequirements {
    /// True when the shader samples a scene-color snapshot texture.
    pub uses_scene_color: bool,
    /// True when the shader samples a scene-depth snapshot texture.
    pub uses_scene_depth: bool,
    /// True when the material requires the renderer to schedule an intersection pre-pass.
    pub requires_intersection_pass: bool,
}

impl SnapshotRequirements {
    /// Returns true when any snapshot flag is set.
    pub fn any(self) -> bool {
        self.uses_scene_color || self.uses_scene_depth || self.requires_intersection_pass
    }
}
