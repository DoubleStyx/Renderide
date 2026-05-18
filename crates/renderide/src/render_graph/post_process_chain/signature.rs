//! Topology fingerprint that gates render-graph rebuilds when the chain shape changes.

use crate::config::{PostProcessingSettings, TonemapMode};

/// Topology fingerprint for the post-processing chain at graph compile time.
///
/// Changes to any field force a render-graph rebuild. Non-topology parameters (intensity,
/// threshold, composite mode, etc.) flow to the passes via per-view blackboard slots
/// ([`crate::render_graph::post_process_settings::BloomSettingsSlot`],
/// [`crate::render_graph::post_process_settings::GtaoSettingsSlot`],
/// [`crate::render_graph::post_process_settings::AutoExposureSettingsSlot`], and
/// [`crate::render_graph::post_process_settings::MotionBlurSettingsSlot`]) and therefore do
/// **not** need to be tracked here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PostProcessChainSignature {
    /// Ground-Truth Ambient Occlusion pass active.
    pub gtao: bool,
    /// Number of GTAO depth-aware denoise iterations baked into the graph (`0..=3`).
    /// Topology field: `>= 2` adds an intermediate denoise pass and a second AO ping-pong
    /// transient; `3` adds another ping-pong iteration, so a change must rebuild. `0` when
    /// GTAO is inactive.
    pub gtao_denoise_passes: u32,
    /// Dual-filter bloom pass active.
    pub bloom: bool,
    /// Screen-space motion blur pass active.
    pub motion_blur: bool,
    /// Histogram-based auto-exposure pass active.
    pub auto_exposure: bool,
    /// Stephen Hill ACES Fitted tonemap pass active.
    pub aces_tonemap: bool,
    /// Analytic AgX tonemap pass active.
    pub agx_tonemap: bool,
    /// Effective bloom mip 0 target height (px). Baked into the mip-chain transient texture
    /// extents at graph-build time via
    /// [`crate::render_graph::resources::TransientExtent::BackbufferScaledMip`], so a change here
    /// must rebuild. `0` when bloom is inactive.
    pub bloom_max_mip_dimension: u32,
}

impl PostProcessChainSignature {
    /// Derives the signature from live [`PostProcessingSettings`].
    pub fn from_settings(settings: &PostProcessingSettings) -> Self {
        let master = settings.enabled;
        let gtao = master && settings.gtao.enabled;
        let bloom = master && settings.bloom.enabled && settings.bloom.intensity > 0.0;
        let motion_blur = master && settings.motion_blur.is_effectively_enabled();
        let auto_exposure = master && settings.auto_exposure.enabled;
        Self {
            gtao,
            gtao_denoise_passes: if gtao {
                settings.gtao.denoise_passes.min(3)
            } else {
                0
            },
            bloom,
            motion_blur,
            auto_exposure,
            aces_tonemap: master && matches!(settings.tonemap.mode, TonemapMode::AcesFitted),
            agx_tonemap: master && matches!(settings.tonemap.mode, TonemapMode::AgX),
            bloom_max_mip_dimension: if bloom {
                settings.bloom.effective_max_mip_dimension()
            } else {
                0
            },
        }
    }

    /// Returns `true` when no effects are active and the chain should be skipped entirely.
    #[cfg(test)]
    pub fn is_empty(self) -> bool {
        !self.gtao
            && !self.bloom
            && !self.motion_blur
            && !self.auto_exposure
            && !self.aces_tonemap
            && !self.agx_tonemap
    }

    /// Number of active effects.
    pub fn active_count(self) -> usize {
        usize::from(self.gtao)
            + usize::from(self.bloom)
            + usize::from(self.motion_blur)
            + usize::from(self.auto_exposure)
            + usize::from(self.aces_tonemap)
            + usize::from(self.agx_tonemap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TonemapSettings;

    #[test]
    fn signature_from_settings_matches_master_toggle() {
        let mut s = PostProcessingSettings {
            enabled: false,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
            ..Default::default()
        };
        s.auto_exposure.enabled = true;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());

        s.enabled = true;
        let sig = PostProcessChainSignature::from_settings(&s);
        assert!(sig.aces_tonemap);
        assert!(!sig.agx_tonemap);
        assert!(sig.gtao);
        assert!(sig.bloom);
        assert!(sig.motion_blur);
        assert!(sig.auto_exposure);
        assert_eq!(sig.active_count(), 5);

        s.tonemap.mode = TonemapMode::AgX;
        let sig = PostProcessChainSignature::from_settings(&s);
        assert!(!sig.aces_tonemap);
        assert!(sig.agx_tonemap);
        assert_eq!(sig.active_count(), 5);

        s.tonemap.mode = TonemapMode::None;
        assert!(PostProcessChainSignature::from_settings(&s).gtao);
        assert!(PostProcessChainSignature::from_settings(&s).bloom);
        assert!(PostProcessChainSignature::from_settings(&s).auto_exposure);
        s.gtao.enabled = false;
        s.bloom.enabled = false;
        s.motion_blur.enabled = false;
        s.auto_exposure.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());
    }

    #[test]
    fn signature_tracks_gtao_toggle_independently_of_tonemap() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.bloom.enabled = false;
        s.motion_blur.enabled = false;
        s.auto_exposure.enabled = false;
        let sig = PostProcessChainSignature::from_settings(&s);
        assert!(sig.gtao);
        assert!(!sig.aces_tonemap);
        assert!(!sig.agx_tonemap);
        assert!(!sig.bloom);
        assert_eq!(sig.active_count(), 1);

        s.gtao.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());

        s.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());
    }

    #[test]
    fn signature_tracks_agx_tonemap_independently_of_aces() {
        let s = PostProcessingSettings {
            enabled: true,
            gtao: crate::config::GtaoSettings {
                enabled: false,
                ..Default::default()
            },
            bloom: crate::config::BloomSettings {
                enabled: false,
                ..Default::default()
            },
            motion_blur: crate::config::MotionBlurSettings {
                enabled: false,
                ..Default::default()
            },
            auto_exposure: crate::config::AutoExposureSettings {
                enabled: false,
                ..Default::default()
            },
            tonemap: TonemapSettings {
                mode: TonemapMode::AgX,
            },
        };

        let sig = PostProcessChainSignature::from_settings(&s);

        assert!(sig.agx_tonemap);
        assert!(!sig.aces_tonemap);
        assert_eq!(sig.active_count(), 1);
    }

    #[test]
    fn signature_clamps_gtao_denoise_topology_to_soft_preset() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.bloom.enabled = false;
        s.motion_blur.enabled = false;
        s.gtao.denoise_passes = 99;

        let sig = PostProcessChainSignature::from_settings(&s);

        assert!(sig.gtao);
        assert_eq!(sig.gtao_denoise_passes, 3);
    }

    #[test]
    fn signature_tracks_bloom_toggle_and_intensity_gate() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.gtao.enabled = false;
        s.auto_exposure.enabled = false;
        s.motion_blur.enabled = false;
        s.bloom.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());

        s.bloom.enabled = true;
        s.bloom.intensity = 0.15;
        let sig = PostProcessChainSignature::from_settings(&s);
        assert!(sig.bloom);
        assert_eq!(sig.active_count(), 1);

        s.bloom.intensity = 0.0;
        assert!(
            !PostProcessChainSignature::from_settings(&s).bloom,
            "intensity=0 must gate bloom off even when enabled"
        );
    }

    #[test]
    fn signature_tracks_effective_bloom_max_mip_dimension() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.gtao.enabled = false;
        s.auto_exposure.enabled = false;
        s.motion_blur.enabled = false;
        s.bloom.max_mip_dimension = 511;

        let sig = PostProcessChainSignature::from_settings(&s);

        assert!(sig.bloom);
        assert_eq!(sig.bloom_max_mip_dimension, 256);
    }

    #[test]
    fn signature_tracks_auto_exposure_toggle_independently_of_tonemap() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.gtao.enabled = false;
        s.bloom.enabled = false;
        s.motion_blur.enabled = false;
        s.auto_exposure.enabled = true;

        let sig = PostProcessChainSignature::from_settings(&s);

        assert!(sig.auto_exposure);
        assert_eq!(sig.active_count(), 1);

        s.auto_exposure.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());
    }

    #[test]
    fn signature_tracks_motion_blur_effective_gate() {
        let mut s = PostProcessingSettings {
            enabled: true,
            gtao: crate::config::GtaoSettings {
                enabled: false,
                ..Default::default()
            },
            bloom: crate::config::BloomSettings {
                enabled: false,
                ..Default::default()
            },
            auto_exposure: crate::config::AutoExposureSettings {
                enabled: false,
                ..Default::default()
            },
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };

        let sig = PostProcessChainSignature::from_settings(&s);
        assert!(sig.motion_blur);
        assert_eq!(sig.active_count(), 1);

        s.motion_blur.sample_count = 0;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());

        s.motion_blur.sample_count = 8;
        s.motion_blur.shutter_angle = 0.0;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());
    }
}
