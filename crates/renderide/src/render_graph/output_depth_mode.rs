//! Classification of main depth attachment layout for Hi-Z and occlusion policy wiring.

use crate::xr::XR_VIEW_COUNT;

/// How the main forward depth buffer is laid out for GPU sampling and CPU readback.
///
/// Derived from [`super::frame_params::FrameRenderParams::multiview_stereo`] and the same signals
/// used for multiview world draws: stereo uses a two-layer `D2Array` depth target; desktop uses a
/// single-layer depth texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputDepthMode {
    /// Single `D2` depth texture (window / mirror path).
    DesktopSingle,
    /// `D2Array` depth with `layer_count` eyes (HMD multiview path).
    StereoArray {
        /// Number of array layers (expected [`XR_VIEW_COUNT`] for OpenXR stereo).
        layer_count: u32,
    },
}

impl OutputDepthMode {
    /// Stereo when `multiview_stereo` is set (external OpenXR targets); desktop otherwise.
    ///
    /// Mirror windows typically use [`Self::DesktopSingle`] even when VR is active elsewhere.
    pub fn from_multiview_stereo(multiview_stereo: bool) -> Self {
        if multiview_stereo {
            Self::StereoArray {
                layer_count: XR_VIEW_COUNT,
            }
        } else {
            Self::DesktopSingle
        }
    }

    /// `true` when occlusion should maintain per-eye Hi-Z data ([`Self::StereoArray`]).
    pub fn is_stereo_array(self) -> bool {
        matches!(self, Self::StereoArray { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_vs_multiview_stereo() {
        assert_eq!(
            OutputDepthMode::from_multiview_stereo(false),
            OutputDepthMode::DesktopSingle
        );
        match OutputDepthMode::from_multiview_stereo(true) {
            OutputDepthMode::StereoArray { layer_count } => {
                assert_eq!(layer_count, XR_VIEW_COUNT);
            }
            OutputDepthMode::DesktopSingle => panic!("expected stereo array"),
        }
    }
}
