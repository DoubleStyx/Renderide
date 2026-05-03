//! VR desktop mirror: copy one HMD eye into a staging texture, then blit to the window surface.
//!
//! The surface blit uses **cover** (fill) mapping: the window is filled with a uniform scale of the
//! staging texture; aspect mismatch is resolved by cropping the center (no letterboxing).
//!
//! Used instead of a second full world render when OpenXR multiview has already drawn the scene.
//!
//! When stereo MSAA is active ([`crate::gpu::GpuContext::swapchain_msaa_effective_stereo`] > 1) the
//! forward pass resolves into the single-sample OpenXR swapchain image, so this mirror always samples
//! already-resolved color and does not need to be aware of the sample count.

mod cover;
mod eye_blit;
mod pipelines;
mod resources;
mod surface_blit;

/// OpenXR `PRIMARY_STEREO` layer index used for the desktop mirror (left eye).
pub const VR_MIRROR_EYE_LAYER: u32 = 0;

/// HMD swapchain color format the eye-to-staging blit reads and writes.
///
/// Mirrors [`crate::xr::XR_COLOR_FORMAT`] -- the OpenXR runtime allocates its swapchain in this
/// format and the staging texture matches it byte-for-byte. Defined locally so `gpu/` does not
/// import from `xr/` (the dependency direction is `xr -> gpu`).
pub(crate) const HMD_MIRROR_SOURCE_FORMAT: wgpu::TextureFormat =
    wgpu::TextureFormat::Rgba8UnormSrgb;

pub use resources::VrMirrorBlitResources;
