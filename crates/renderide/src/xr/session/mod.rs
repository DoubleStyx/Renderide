//! OpenXR session: view/projection math ([`view_math`]) and frame loop ([`state::XrSessionState`]).
//!
//! OpenXR [`xr::Posef`] transforms **from the view-local frame to the reference (stage) frame**
//! (right-handed, Y-up, −Z forward). Scene content and render-space rigs from the host use a
//! Unity-style left-handed world basis for parity with FrooxEngine; [`openxr_pose_to_engine`] and
//! [`openxr_pose_to_host_tracking`] apply the same RH→LH mapping used for IPC so HMD views and
//! scene transforms share one world basis before [`crate::render_graph::apply_view_handedness_fix`]
//! applies the clip-space Z handling for the mesh path.
//!
//! ## Stereo convention (runtime `views` order)
//!
//! For the primary stereo view configuration (`PRIMARY_STEREO`), `views[0]` is the left eye and
//! `views[1]` the right eye. [`headset_center_pose_from_stereo_views`] averages both for the
//! center-eye pose sent over IPC via [`openxr_pose_to_host_tracking`].

mod state;
mod view_math;

#[cfg(test)]
mod tests;

pub use state::XrSessionState;
pub(in crate::xr) use state::XrSessionStateDescriptor;
pub use view_math::{
    center_view_projection_from_stereo_views_aligned, headset_center_pose_from_stereo_views,
    headset_pose_from_xr_view, openxr_pose_to_engine, openxr_pose_to_host_tracking,
    tracking_space_to_world_matrix, view_from_xr_view_aligned, view_projection_from_xr_view,
    view_projection_from_xr_view_aligned,
};
