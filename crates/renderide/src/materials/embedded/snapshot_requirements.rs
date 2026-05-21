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

/// How a material expects the scene-color snapshot to be refreshed.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SceneColorSnapshotMode {
    /// The material does not sample the scene-color snapshot.
    #[default]
    None,
    /// Unity unnamed grab-pass behavior: copy immediately before each draw group.
    PerObjectGrab,
    /// Unity named grab-pass behavior: copy once at the first matching draw and reuse it.
    NamedBackgroundGrab,
}

impl SceneColorSnapshotMode {
    /// Returns true when the mode samples a scene-color snapshot.
    pub fn uses_scene_color(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl SnapshotRequirements {
    /// Returns true when any snapshot flag is set.
    #[cfg(test)]
    pub fn any(self) -> bool {
        self.uses_scene_color || self.uses_scene_depth || self.requires_intersection_pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_is_false_for_default_requirements() {
        assert!(!SnapshotRequirements::default().any());
    }

    #[test]
    fn any_is_true_for_each_individual_requirement() {
        assert!(
            SnapshotRequirements {
                uses_scene_color: true,
                ..Default::default()
            }
            .any()
        );
        assert!(
            SnapshotRequirements {
                uses_scene_depth: true,
                ..Default::default()
            }
            .any()
        );
        assert!(
            SnapshotRequirements {
                requires_intersection_pass: true,
                ..Default::default()
            }
            .any()
        );
    }

    #[test]
    fn scene_color_snapshot_mode_reports_usage() {
        assert!(!SceneColorSnapshotMode::None.uses_scene_color());
        assert!(SceneColorSnapshotMode::PerObjectGrab.uses_scene_color());
        assert!(SceneColorSnapshotMode::NamedBackgroundGrab.uses_scene_color());
    }
}
