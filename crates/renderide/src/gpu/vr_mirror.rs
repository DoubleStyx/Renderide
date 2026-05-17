//! VR desktop mirror: copy renderer-owned HMD color into OpenXR and the desktop mirror surface.
//!
//! The surface blit uses **cover** (fill) mapping: the window is filled with a uniform scale of the
//! staging texture; aspect mismatch is resolved by cropping the center (no letterboxing).
//!
//! Used instead of a second full world render when OpenXR multiview has already drawn the scene.
//!
//! When stereo MSAA is active ([`crate::gpu::GpuContext::swapchain_msaa_effective_stereo`] > 1) the
//! forward pass resolves into a single-sample renderer-owned HMD color texture. The final copy pass
//! then writes that owned color into the acquired OpenXR swapchain and the left-eye mirror staging
//! texture, so the desktop mirror never samples an OpenXR-owned image.

mod cover;
mod eye_blit;
mod pipelines;
mod resources;
mod surface_blit;

/// OpenXR `PRIMARY_STEREO` layer index used for the desktop mirror (left eye).
pub const VR_MIRROR_EYE_LAYER: u32 = 0;

/// HMD color format the final-copy and staging blits read and write.
///
/// Matches the OpenXR swapchain format used by the XR layer. The staging texture matches it so the
/// desktop mirror can display an HMD eye without importing XR modules.
pub(crate) const HMD_MIRROR_SOURCE_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba8UnormSrgb;

pub use resources::VrMirrorBlitResources;
