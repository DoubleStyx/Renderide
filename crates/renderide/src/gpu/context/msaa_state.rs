//! Per-frame MSAA state on [`GpuContext`].
//!
//! Stores the device's supported MSAA tier lists (desktop and stereo) discovered at
//! construction by [`crate::gpu::adapter::msaa_support::MsaaSupport`], plus the effective
//! sample counts the runtime selects each tick by clamping the user request through
//! [`crate::gpu::adapter::msaa_support::clamp_msaa_request_to_supported`].

use crate::gpu::adapter::msaa_support::clamp_msaa_request_to_supported;

use super::GpuContext;

impl GpuContext {
    /// Adapter-reported maximum MSAA sample count for the swapchain color format and depth.
    pub fn msaa_max_sample_count(&self) -> u32 {
        self.msaa_supported_sample_counts
            .last()
            .copied()
            .unwrap_or(1)
    }

    /// Adapter-reported maximum MSAA sample count for **2D array** color + depth (stereo / OpenXR path).
    ///
    /// Returns `1` when the device lacks [`wgpu::Features::MULTISAMPLE_ARRAY`] or
    /// [`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`], in which case the stereo forward
    /// path silently falls back to no MSAA.
    pub fn msaa_max_sample_count_stereo(&self) -> u32 {
        self.msaa_supported_sample_counts_stereo
            .last()
            .copied()
            .unwrap_or(1)
    }

    /// Effective MSAA sample count for the main window this frame (after [`Self::set_swapchain_msaa_requested`]).
    pub fn swapchain_msaa_effective(&self) -> u32 {
        self.swapchain_msaa_effective
    }

    /// Effective stereo MSAA sample count for the OpenXR path this frame (after
    /// [`Self::set_swapchain_msaa_requested_stereo`]). `1` = off.
    pub fn swapchain_msaa_effective_stereo(&self) -> u32 {
        self.swapchain_msaa_effective_stereo
    }

    /// Sets requested MSAA for the desktop swapchain path; values are rounded to a **format-valid**
    /// tier ([`Self::msaa_max_sample_count`]), not merely capped by the maximum tier.
    ///
    /// Call each frame before graph execution (from [`crate::config::RenderingSettings::msaa`]).
    pub fn set_swapchain_msaa_requested(&mut self, requested: u32) {
        self.swapchain_msaa_effective =
            clamp_msaa_request_to_supported(requested, &self.msaa_supported_sample_counts);
    }

    /// Sets requested MSAA for the OpenXR stereo path; clamps to a format-valid tier against the
    /// stereo supported list. When `MULTISAMPLE_ARRAY` is unavailable the stereo list is empty and
    /// the effective count silently becomes `1`.
    ///
    /// Call each XR frame before graph execution (from [`crate::config::RenderingSettings::msaa`]).
    pub fn set_swapchain_msaa_requested_stereo(&mut self, requested: u32) {
        let requested = requested.max(1);
        let effective =
            clamp_msaa_request_to_supported(requested, &self.msaa_supported_sample_counts_stereo);
        if self.swapchain_msaa_requested_stereo != requested
            || self.swapchain_msaa_effective_stereo != effective
        {
            if requested > 1 && effective != requested {
                logger::info!(
                    "VR MSAA clamped: requested {}x -> effective {}x (supported={:?})",
                    requested,
                    effective,
                    self.msaa_supported_sample_counts_stereo
                );
            }
            self.swapchain_msaa_requested_stereo = requested;
            self.swapchain_msaa_effective_stereo = effective;
        }
    }
}
