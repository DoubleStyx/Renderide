//! Graph-independent frame-view contracts shared by materials, passes, and graph execution.

use crate::camera::{
    camera_state_motion_blur, camera_state_post_processing, camera_state_screen_space_reflections,
};
use crate::color_space::DEFAULT_SKYBOX_CLEAR_COLOR;
use crate::shared::{CameraClearMode, CameraRenderParameters, CameraState};

/// Offscreen target currently being written by a view.
///
/// The renderer uses this for two separate decisions: any offscreen target needs the offscreen
/// projection convention, while only host render textures need material self-sampling suppression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OffscreenWriteTarget {
    /// The view writes directly to the desktop swapchain or an external multiview target.
    #[default]
    None,
    /// The view writes to an offscreen target that is not a host render-texture asset.
    Untracked,
    /// The view writes to a host render texture with the supplied asset id and sampling policy.
    HostRenderTexture {
        /// Host render-texture asset id.
        asset_id: i32,
        /// Material sampling policy for this render texture while it is being written.
        self_sampling: RenderTextureSelfSampling,
    },
}

/// Material sampling policy for a render texture while a camera writes that same texture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RenderTextureSelfSampling {
    /// Hide the render texture from materials while the view writes it.
    #[default]
    Suppress,
    /// Allow materials to sample the render texture contents completed before this view.
    AllowPreviousContents,
}

impl OffscreenWriteTarget {
    /// Builds a host render-texture target using the default same-target sampling suppression.
    #[inline]
    pub const fn host_render_texture(asset_id: i32) -> Self {
        Self::host_render_texture_with_self_sampling(asset_id, RenderTextureSelfSampling::Suppress)
    }

    /// Builds a host render-texture target with an explicit same-target sampling policy.
    #[inline]
    pub const fn host_render_texture_with_self_sampling(
        asset_id: i32,
        self_sampling: RenderTextureSelfSampling,
    ) -> Self {
        Self::HostRenderTexture {
            asset_id,
            self_sampling,
        }
    }

    /// Returns `true` when the view writes to any offscreen target.
    #[inline]
    pub const fn is_offscreen(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Applies the render-target projection convention for this write target.
    ///
    /// Offscreen color attachments are written in the host texture orientation, so their
    /// clip-space projection gets a Y flip. Screen-space consumers built from the view projection,
    /// including clustered-light froxels and frame unprojection constants, must use the same
    /// adjusted projection as the forward draw path.
    #[inline]
    pub(crate) fn render_projection(self, projection: glam::Mat4) -> glam::Mat4 {
        if self.is_offscreen() {
            offscreen_projection_y_flip() * projection
        } else {
            projection
        }
    }

    /// Returns the host render-texture asset id for this write target.
    #[inline]
    pub const fn host_render_texture_asset_id(self) -> Option<i32> {
        match self {
            Self::HostRenderTexture { asset_id, .. } => Some(asset_id),
            Self::None | Self::Untracked => None,
        }
    }

    /// Returns the same-target material sampling policy for this write target.
    #[inline]
    pub const fn render_texture_self_sampling(self) -> Option<RenderTextureSelfSampling> {
        match self {
            Self::HostRenderTexture { self_sampling, .. } => Some(self_sampling),
            Self::None | Self::Untracked => None,
        }
    }

    /// Returns `true` when material bindings should mask this render texture while rendering.
    #[inline]
    pub fn suppresses_render_texture_sampling(self, sampled_asset_id: i32) -> bool {
        self.host_render_texture_asset_id() == Some(sampled_asset_id)
            && self.render_texture_self_sampling() == Some(RenderTextureSelfSampling::Suppress)
    }
}

/// Per-view winding policy before draw-local transform parity is applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ViewWinding {
    /// Whether the camera view matrix mirrors handedness, as planar reflections do.
    mirror_reflection: bool,
}

impl ViewWinding {
    /// View policy for ordinary non-reflection cameras.
    #[inline]
    pub const fn normal() -> Self {
        Self {
            mirror_reflection: false,
        }
    }

    /// View policy for planar mirror reflection cameras.
    #[inline]
    pub const fn mirror_reflection() -> Self {
        Self {
            mirror_reflection: true,
        }
    }

    /// Returns whether final raster front-face winding must be flipped for this view.
    #[inline]
    pub const fn flips_front_face_for(self, write_target: OffscreenWriteTarget) -> bool {
        write_target.is_offscreen() ^ self.mirror_reflection
    }
}

#[inline]
fn offscreen_projection_y_flip() -> glam::Mat4 {
    glam::Mat4::from_diagonal(glam::Vec4::new(1.0, -1.0, 1.0, 1.0))
}

/// Per-view background clear contract propagated from host camera state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameViewClear {
    /// Host camera clear mode for this view.
    pub mode: CameraClearMode,
    /// Host background color used when [`CameraClearMode::Color`] is selected.
    pub color: glam::Vec4,
}

impl FrameViewClear {
    /// Main-view clear mode: render the active render-space skybox.
    #[inline]
    pub fn skybox() -> Self {
        Self {
            mode: CameraClearMode::Skybox,
            color: DEFAULT_SKYBOX_CLEAR_COLOR,
        }
    }

    /// Color clear mode with the supplied linear RGBA background.
    #[inline]
    pub fn color(color: glam::Vec4) -> Self {
        Self {
            mode: CameraClearMode::Color,
            color,
        }
    }

    /// Converts host camera state into a frame-view clear descriptor.
    #[inline]
    pub fn from_camera_state(state: &CameraState) -> Self {
        Self {
            mode: state.clear_mode,
            color: state.background_color,
        }
    }

    /// Converts host camera readback parameters into a frame-view clear descriptor.
    #[inline]
    pub fn from_camera_render_parameters(parameters: &CameraRenderParameters) -> Self {
        Self {
            mode: parameters.clear_mode,
            color: parameters.clear_color,
        }
    }
}

impl Default for FrameViewClear {
    #[inline]
    fn default() -> Self {
        Self::skybox()
    }
}

/// Post-processing permissions requested by a single view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewPostProcessing {
    /// `true` when this view should run the post-processing stack.
    pub enabled: bool,
    /// `true` when this view allows screen-space reflections to record.
    pub screen_space_reflections: bool,
    /// `true` when this view allows motion blur to record.
    pub motion_blur: bool,
}

impl ViewPostProcessing {
    /// Builds a view post-processing policy from decoded host camera settings.
    pub const fn new(enabled: bool, screen_space_reflections: bool, motion_blur: bool) -> Self {
        Self {
            enabled,
            screen_space_reflections: enabled && screen_space_reflections,
            motion_blur: enabled && motion_blur,
        }
    }

    /// Primary/HMD view policy: allow the renderer-global post-processing stack to run.
    pub const fn primary_view() -> Self {
        Self::new(true, true, true)
    }

    /// Reflection-probe and other raw-capture policy: bypass all post-processing effects.
    pub const fn disabled() -> Self {
        Self::new(false, false, false)
    }

    /// Converts host camera readback parameters into a view post-processing policy.
    ///
    /// Camera render tasks explicitly disable motion blur to match the host camera-capture path.
    pub fn from_camera_render_parameters(parameters: &CameraRenderParameters) -> Self {
        Self::new(
            parameters.post_processing,
            parameters.screen_space_reflections,
            false,
        )
    }

    /// Converts secondary render-texture camera state flags into a view post-processing policy.
    pub fn from_camera_state(state: &CameraState) -> Self {
        Self::new(
            camera_state_post_processing(state.flags),
            camera_state_screen_space_reflections(state.flags),
            camera_state_motion_blur(state.flags),
        )
    }

    /// Returns `true` when this view should run the post-processing stack.
    pub const fn is_enabled(self) -> bool {
        self.enabled
    }
}

impl Default for ViewPostProcessing {
    fn default() -> Self {
        Self::primary_view()
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameViewClear, OffscreenWriteTarget, RenderTextureSelfSampling, ViewWinding};

    #[test]
    fn offscreen_target_helpers_distinguish_write_targets() {
        assert!(!OffscreenWriteTarget::None.is_offscreen());
        assert_eq!(
            OffscreenWriteTarget::None.host_render_texture_asset_id(),
            None
        );
        assert!(OffscreenWriteTarget::Untracked.is_offscreen());
        assert_eq!(
            OffscreenWriteTarget::Untracked.host_render_texture_asset_id(),
            None
        );
        let host_target = OffscreenWriteTarget::host_render_texture(77);
        assert!(host_target.is_offscreen());
        assert_eq!(host_target.host_render_texture_asset_id(), Some(77));
        assert_eq!(
            host_target.render_texture_self_sampling(),
            Some(RenderTextureSelfSampling::Suppress)
        );
        assert!(host_target.suppresses_render_texture_sampling(77));
        assert!(!host_target.suppresses_render_texture_sampling(78));
    }

    #[test]
    fn host_render_texture_can_allow_previous_contents() {
        let host_target = OffscreenWriteTarget::host_render_texture_with_self_sampling(
            77,
            RenderTextureSelfSampling::AllowPreviousContents,
        );

        assert_eq!(
            host_target.render_texture_self_sampling(),
            Some(RenderTextureSelfSampling::AllowPreviousContents)
        );
        assert!(!host_target.suppresses_render_texture_sampling(77));
    }

    #[test]
    fn offscreen_projection_flips_y() {
        let projection = glam::Mat4::IDENTITY;

        assert_eq!(
            OffscreenWriteTarget::host_render_texture(77).render_projection(projection),
            glam::Mat4::from_diagonal(glam::Vec4::new(1.0, -1.0, 1.0, 1.0))
        );
        assert_eq!(
            OffscreenWriteTarget::None.render_projection(projection),
            projection
        );
    }

    #[test]
    fn winding_flips_for_offscreen_and_reflections() {
        assert!(!ViewWinding::normal().flips_front_face_for(OffscreenWriteTarget::None));
        assert!(
            ViewWinding::normal()
                .flips_front_face_for(OffscreenWriteTarget::host_render_texture(77))
        );
        assert!(ViewWinding::mirror_reflection().flips_front_face_for(OffscreenWriteTarget::None));
        assert!(
            !ViewWinding::mirror_reflection()
                .flips_front_face_for(OffscreenWriteTarget::host_render_texture(77))
        );
    }

    #[test]
    fn default_clear_is_skybox() {
        let clear = FrameViewClear::default();

        assert_eq!(clear.mode, crate::shared::CameraClearMode::Skybox);
        assert_eq!(clear.color, crate::color_space::DEFAULT_SKYBOX_CLEAR_COLOR);
    }

    #[test]
    fn clear_from_camera_state_preserves_mode_and_color() {
        let state = crate::shared::CameraState {
            clear_mode: crate::shared::CameraClearMode::Color,
            background_color: glam::Vec4::new(0.2, 0.3, 0.4, 1.0),
            ..Default::default()
        };

        let clear = FrameViewClear::from_camera_state(&state);

        assert_eq!(clear.mode, crate::shared::CameraClearMode::Color);
        assert_eq!(clear.color, state.background_color);
    }
}
