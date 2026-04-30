//! Classification of main depth attachment layout for Hi-Z and occlusion policy wiring.

use thiserror::Error;

/// Number of stereo view layers in the multiview depth array (left + right eye).
///
/// Inlined here to keep `render_graph/` independent of `xr/`. Stays in sync with
/// `crate::xr::STEREO_LAYER_COUNT`.
const STEREO_LAYER_COUNT: u32 = 2;

/// Errors when code expects a stereo depth array but the mode is desktop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum OutputDepthModeError {
    /// [`OutputDepthMode::DesktopSingle`] was found where stereo array layout was required.
    #[error("expected stereo depth array, got desktop single")]
    ExpectedStereoArray,
}

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
        /// Number of array layers (expected [`STEREO_LAYER_COUNT`] for OpenXR stereo).
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
                layer_count: STEREO_LAYER_COUNT,
            }
        } else {
            Self::DesktopSingle
        }
    }

    /// `true` when occlusion should maintain per-eye Hi-Z data ([`Self::StereoArray`]).
    pub fn is_stereo_array(self) -> bool {
        matches!(self, Self::StereoArray { .. })
    }

    /// Layer count when this mode is [`Self::StereoArray`]; otherwise [`OutputDepthModeError::ExpectedStereoArray`].
    pub fn try_stereo_layer_count(self) -> Result<u32, OutputDepthModeError> {
        match self {
            Self::StereoArray { layer_count } => Ok(layer_count),
            Self::DesktopSingle => Err(OutputDepthModeError::ExpectedStereoArray),
        }
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
        assert_eq!(
            OutputDepthMode::from_multiview_stereo(true).try_stereo_layer_count(),
            Ok(STEREO_LAYER_COUNT)
        );
    }
}
