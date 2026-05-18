//! Per-view blackboard slots that propagate live post-processing settings into the chain.

use super::blackboard::BlackboardSlot;

/// Blackboard slot for the live [`crate::config::GtaoSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so
/// the GTAO chain ([`crate::passes::post_processing::GtaoEffect`]) reads the current slider
/// values without rebuilding the compiled render graph. Non-topology slider changes don't
/// flip [`crate::render_graph::post_process_chain::chain::PostProcessChainSignature`] -- this
/// slot is the path that propagates those edits into the per-stage UBO writes.
pub struct GtaoSettingsSlot;
impl BlackboardSlot for GtaoSettingsSlot {
    type Value = GtaoSettingsValue;
}

/// Live [`crate::config::GtaoSettings`] carried on the per-view blackboard.
///
/// Wraps `GtaoSettings` by value; the blackboard slot trait needs a concrete type living in this
/// module and the inner settings type lives in `crate::config`.
#[derive(Clone, Copy, Debug)]
pub struct GtaoSettingsValue(pub crate::config::GtaoSettings);

/// Blackboard slot for the live [`crate::config::BloomSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so the
/// bloom passes read the current slider values without rebuilding the compiled render graph.
/// Non-topology edits (intensity, low-frequency boost, threshold, composite mode, ...) flow in via
/// this slot; only the effective `max_mip_dimension` changes force a rebuild because it resizes
/// the mip-chain transient textures -- the chain signature tracks that value explicitly.
pub struct BloomSettingsSlot;
impl BlackboardSlot for BloomSettingsSlot {
    type Value = BloomSettingsValue;
}

/// Live [`crate::config::BloomSettings`] carried on the per-view blackboard.
#[derive(Clone, Copy, Debug)]
pub struct BloomSettingsValue(pub crate::config::BloomSettings);

/// Blackboard slot for the live [`crate::config::MotionBlurSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so
/// motion blur sample counts, shutter scale, and pixel clamp updates take effect without
/// rebuilding the compiled render graph. Topology fields still rebuild the graph when they
/// add or remove the velocity and resolve passes.
pub struct MotionBlurSettingsSlot;
impl BlackboardSlot for MotionBlurSettingsSlot {
    type Value = MotionBlurSettingsValue;
}

/// Live [`crate::config::MotionBlurSettings`] carried on the per-view blackboard.
#[derive(Clone, Copy, Debug)]
pub struct MotionBlurSettingsValue(pub crate::config::MotionBlurSettings);

/// Blackboard slot for the live [`crate::config::AutoExposureSettings`] snapshot.
///
/// Seeded each frame from [`crate::config::RendererSettings`] before per-view recording so the
/// auto-exposure histogram pass can update its GPU settings buffer without rebuilding the graph.
/// The frame delta is carried alongside the settings because exposure adaptation is temporal.
pub struct AutoExposureSettingsSlot;
impl BlackboardSlot for AutoExposureSettingsSlot {
    type Value = AutoExposureSettingsValue;
}

/// Live auto-exposure settings, frame delta, and adaptation policy carried on the per-view blackboard.
#[derive(Clone, Copy, Debug)]
pub struct AutoExposureSettingsValue {
    /// Current renderer-config auto-exposure settings.
    pub settings: crate::config::AutoExposureSettings,
    /// Wall-clock delta for temporal adaptation, in seconds.
    pub delta_seconds: f32,
    /// Whether exposure should snap to the current metered target for this view.
    pub instant_adaptation: bool,
}

impl AutoExposureSettingsValue {
    /// Builds a per-view auto-exposure settings value from the live renderer settings.
    pub(crate) fn for_view(
        settings: crate::config::AutoExposureSettings,
        delta_seconds: f32,
        view_id: crate::camera::ViewId,
    ) -> Self {
        Self {
            settings,
            delta_seconds,
            instant_adaptation: matches!(
                view_id,
                crate::camera::ViewId::CameraRenderTask(_)
                    | crate::camera::ViewId::Camera360RenderTaskFace(_)
            ),
        }
    }
}

impl Default for AutoExposureSettingsValue {
    fn default() -> Self {
        Self {
            settings: crate::config::AutoExposureSettings::default(),
            delta_seconds: 1.0 / 60.0,
            instant_adaptation: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::camera::ViewId;
    use crate::config::AutoExposureSettings;
    use crate::scene::RenderSpaceId;

    use super::AutoExposureSettingsValue;

    #[test]
    fn camera_render_tasks_use_instant_auto_exposure_adaptation() {
        let value = AutoExposureSettingsValue::for_view(
            AutoExposureSettings::default(),
            1.0 / 60.0,
            ViewId::camera_render_task(RenderSpaceId(7), 0),
        );

        assert!(value.instant_adaptation);
    }

    #[test]
    fn camera360_face_tasks_use_instant_auto_exposure_adaptation() {
        let value = AutoExposureSettingsValue::for_view(
            AutoExposureSettings::default(),
            1.0 / 60.0,
            ViewId::camera360_render_task_face(RenderSpaceId(7), 0, 3),
        );

        assert!(value.instant_adaptation);
    }

    #[test]
    fn persistent_views_keep_temporal_auto_exposure_adaptation() {
        let main = AutoExposureSettingsValue::for_view(
            AutoExposureSettings::default(),
            1.0 / 60.0,
            ViewId::Main,
        );
        let secondary = AutoExposureSettingsValue::for_view(
            AutoExposureSettings::default(),
            1.0 / 60.0,
            ViewId::secondary_camera(RenderSpaceId(7), 3),
        );

        assert!(!main.instant_adaptation);
        assert!(!secondary.instant_adaptation);
    }
}
