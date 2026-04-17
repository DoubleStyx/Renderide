//! [`GpuContext`]: instance, surface, device, and swapchain state.
//!
//! Submodules hold MSAA tier math, bootstrap, depth and MSAA texture helpers, surface acquire, and
//! tracked submit/timing glue; this module keeps the public type and wires them together.

mod bootstrap;
mod depth_attachment;
mod frame_queue;
mod msaa_resources;
mod msaa_tiers;
mod surface_acquire;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::frame_cpu_gpu_timing::{FrameCpuGpuTiming, FrameCpuGpuTimingHandle};
use super::limits::{GpuLimits, GpuLimitsError};
use thiserror::Error;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use bootstrap::GpuBootstrapParts;
use msaa_tiers::clamp_msaa_request_to_supported;

/// Multisampled color + depth targets for the main window forward path ([`GpuContext::ensure_msaa_targets`]).
pub struct MsaaTargets {
    /// Multisampled color texture (`sample_count` &gt; 1).
    pub color_texture: wgpu::Texture,
    /// Default [`wgpu::TextureView`] for [`Self::color_texture`].
    pub color_view: wgpu::TextureView,
    /// Multisampled depth texture ([`wgpu::TextureFormat::Depth32Float`]).
    pub depth_texture: wgpu::Texture,
    /// Default [`wgpu::TextureView`] for [`Self::depth_texture`].
    pub depth_view: wgpu::TextureView,
    /// Effective sample count (2, 4, or 8).
    pub sample_count: u32,
    /// Pixel extent `(width, height)`.
    pub extent: (u32, u32),
    /// Swapchain color format used for [`Self::color_texture`].
    pub color_format: wgpu::TextureFormat,
}

/// Multisampled 2-layer `D2Array` color + depth targets for the OpenXR single-pass stereo forward path
/// ([`GpuContext::ensure_msaa_stereo_targets`]).
///
/// Color resolves into the single-sample OpenXR swapchain image; depth resolves into the stereo
/// [`wgpu::TextureFormat::Depth32Float`] array via compute + multiview blit.
pub struct MsaaStereoTargets {
    /// Multisampled `D2` array texture (`depth_or_array_layers = 2`, `sample_count > 1`).
    pub color_texture: wgpu::Texture,
    /// `D2Array` color view for the multiview render-pass attachment.
    pub color_view: wgpu::TextureView,
    /// Multisampled `D2` array depth texture (2 layers, `sample_count > 1`).
    pub depth_texture: wgpu::Texture,
    /// `D2Array` depth view for the multiview render-pass attachment.
    pub depth_view: wgpu::TextureView,
    /// Per-eye (`D2`, single-layer) depth views used by the compute depth resolve shader,
    /// which binds as `texture_depth_multisampled_2d` (WGSL has no array variant yet).
    pub depth_layer_views: [wgpu::TextureView; 2],
    /// Effective sample count (2, 4, or 8).
    pub sample_count: u32,
    /// Pixel extent per eye `(width, height)`.
    pub extent: (u32, u32),
    /// OpenXR swapchain color format used for [`Self::color_texture`].
    pub color_format: wgpu::TextureFormat,
}

/// Single-sample `R32Float` 2-layer array temp used when resolving the stereo MSAA depth.
///
/// The compute pass writes per eye via `layer_views`; the fullscreen multiview blit samples the
/// whole `D2Array` via `array_view` and writes into the stereo `Depth32Float` target.
pub(crate) struct MsaaStereoDepthResolveR32 {
    /// Owning 2-layer `R32Float` texture. Kept to document ownership and anchor lifetime of the
    /// derived views; wgpu refcounts the underlying object so the field is intentionally not read.
    #[allow(dead_code)]
    pub(crate) texture: wgpu::Texture,
    /// `D2Array` sampled view for the multiview blit source.
    pub(crate) array_view: wgpu::TextureView,
    /// Per-eye (`D2`, single-layer) storage views for the compute pass.
    pub(crate) layer_views: [wgpu::TextureView; 2],
    /// Pixel extent per eye `(width, height)`.
    pub(crate) extent: (u32, u32),
}

/// GPU stack for presentation and future render passes.
pub struct GpuContext {
    /// Adapter metadata from construction (for diagnostics).
    adapter_info: wgpu::AdapterInfo,
    /// MSAA tiers supported for the configured surface color format and [`wgpu::TextureFormat::Depth32Float`]
    /// (sorted ascending: 2, 4, …). Empty means MSAA is unavailable.
    msaa_supported_sample_counts: Vec<u32>,
    /// MSAA tiers supported for **2D array** color + [`wgpu::TextureFormat::Depth32Float`] on the OpenXR
    /// path (sorted ascending). Empty when the adapter lacks
    /// [`wgpu::Features::MULTISAMPLE_ARRAY`] / [`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`],
    /// which silently clamps the stereo request to `1` (MSAA off).
    msaa_supported_sample_counts_stereo: Vec<u32>,
    /// Effective swapchain MSAA sample count this frame (1 = off), set via [`Self::set_swapchain_msaa_requested`].
    swapchain_msaa_effective: u32,
    /// Requested stereo MSAA (from settings) before clamping; set each XR frame by the runtime.
    swapchain_msaa_requested_stereo: u32,
    /// Effective stereo MSAA sample count (1 = off), set via [`Self::set_swapchain_msaa_requested_stereo`].
    swapchain_msaa_effective_stereo: u32,
    /// Effective limits and derived caps for this device (shared across backend and uploads).
    limits: Arc<GpuLimits>,
    device: Arc<wgpu::Device>,
    queue: Arc<Mutex<wgpu::Queue>>,
    /// Kept as `'static` so the context can move independently of the window borrow; the window
    /// must outlive this value (owned alongside it in the app handler).
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// Depth target matching [`Self::config`] extent; recreated after resize.
    depth_attachment: Option<(wgpu::Texture, wgpu::TextureView)>,
    depth_extent_px: (u32, u32),
    /// Multisampled targets for desktop MSAA; [`None`] when off or extent/sample count unchanged.
    msaa_targets: Option<MsaaTargets>,
    /// Multisampled 2-layer targets for stereo / OpenXR MSAA; [`None`] when off or stale.
    msaa_stereo_targets: Option<MsaaStereoTargets>,
    /// Single-sample R32Float resolve temp for MSAA depth → depth blit ([`crate::gpu::MsaaDepthResolveResources`]).
    msaa_depth_resolve_r32: Option<(wgpu::Texture, wgpu::TextureView)>,
    msaa_depth_resolve_r32_extent: (u32, u32),
    /// Stereo R32Float resolve temp (2 layers) for MSAA depth → stereo depth blit.
    msaa_stereo_depth_resolve_r32: Option<MsaaStereoDepthResolveR32>,
    /// Debug HUD: wall-clock CPU (tick start → last submit) and GPU (last submit → idle) timing.
    frame_timing: FrameCpuGpuTimingHandle,
}

/// GPU initialization or resize failure.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No suitable adapter was found.
    #[error("request_adapter failed: {0}")]
    Adapter(String),
    /// Device creation failed.
    #[error("request_device failed: {0}")]
    Device(String),
    /// Surface could not be created from the window.
    #[error("create_surface failed: {0}")]
    Surface(String),
    /// No default surface configuration for this adapter.
    #[error("surface unsupported")]
    SurfaceUnsupported,
    /// Device reports limits below Renderide minimums.
    #[error("GPU limits: {0}")]
    Limits(#[from] GpuLimitsError),
}

impl GpuContext {
    /// Asynchronously builds GPU state for `window`.
    ///
    /// `gpu_validation_layers` selects whether to request backend validation before `WGPU_*` env
    /// overrides; see [`crate::gpu::instance_flags_for_gpu_init`].
    pub async fn new(
        window: Arc<Window>,
        vsync: bool,
        gpu_validation_layers: bool,
    ) -> Result<Self, GpuError> {
        let GpuBootstrapParts {
            adapter_info,
            msaa_supported_sample_counts,
            msaa_supported_sample_counts_stereo,
            limits,
            device,
            queue,
            surface,
            config,
        } = bootstrap::bootstrap_desktop(window, vsync, gpu_validation_layers).await?;

        Ok(Self {
            adapter_info,
            msaa_supported_sample_counts,
            msaa_supported_sample_counts_stereo,
            swapchain_msaa_effective: 1,
            swapchain_msaa_requested_stereo: 1,
            swapchain_msaa_effective_stereo: 1,
            limits,
            device,
            queue,
            surface,
            config,
            depth_attachment: None,
            depth_extent_px: (0, 0),
            msaa_targets: None,
            msaa_stereo_targets: None,
            msaa_depth_resolve_r32: None,
            msaa_depth_resolve_r32_extent: (0, 0),
            msaa_stereo_depth_resolve_r32: None,
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
        })
    }

    /// Builds GPU state using an existing wgpu instance/device from OpenXR bootstrap (mirror window).
    pub fn new_from_openxr_bootstrap(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<Mutex<wgpu::Queue>>,
        window: Arc<Window>,
        vsync: bool,
    ) -> Result<Self, GpuError> {
        let GpuBootstrapParts {
            adapter_info,
            msaa_supported_sample_counts,
            msaa_supported_sample_counts_stereo,
            limits,
            device,
            queue,
            surface,
            config,
        } = bootstrap::bootstrap_openxr_mirror(instance, adapter, device, queue, window, vsync)?;

        Ok(Self {
            adapter_info,
            msaa_supported_sample_counts,
            msaa_supported_sample_counts_stereo,
            swapchain_msaa_effective: 1,
            swapchain_msaa_requested_stereo: 1,
            swapchain_msaa_effective_stereo: 1,
            limits,
            device,
            queue,
            surface,
            config,
            depth_attachment: None,
            depth_extent_px: (0, 0),
            msaa_targets: None,
            msaa_stereo_targets: None,
            msaa_depth_resolve_r32: None,
            msaa_depth_resolve_r32_extent: (0, 0),
            msaa_stereo_depth_resolve_r32: None,
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
        })
    }

    /// Updates vertical sync / present mode and reconfigures the surface (hot-reload from settings).
    pub fn set_vsync(&mut self, vsync: bool) {
        let mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        if self.config.present_mode == mode {
            return;
        }
        self.config.present_mode = mode;
        self.surface.configure(&self.device, &self.config);
        logger::info!(
            "Present mode set to {:?} (vsync={})",
            self.config.present_mode,
            vsync
        );
    }

    /// Current swapchain configuration extent.
    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.config.width, self.config.height)
    }

    /// Swapchain pixel size `(width, height)`.
    pub fn surface_extent_px(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigures the swapchain after resize or after [`wgpu::CurrentSurfaceTexture::Lost`] /
    /// [`wgpu::CurrentSurfaceTexture::Outdated`].
    pub fn reconfigure(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.depth_attachment = None;
        self.depth_extent_px = (0, 0);
        self.msaa_targets = None;
        self.msaa_depth_resolve_r32 = None;
        self.msaa_depth_resolve_r32_extent = (0, 0);
    }

    /// Frees the stereo MSAA color + depth targets and R32F resolve temp.
    ///
    /// Call when the OpenXR swapchain is recreated (resolution change, loss) so the next frame
    /// reallocates at the correct extent.
    pub fn reset_msaa_stereo_targets(&mut self) {
        self.msaa_stereo_targets = None;
        self.msaa_stereo_depth_resolve_r32 = None;
    }

    /// Borrows the configured surface for acquire/submit.
    pub fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }

    /// Centralized device limits and derived caps ([`GpuLimits`]).
    pub fn limits(&self) -> &Arc<GpuLimits> {
        &self.limits
    }

    /// WGPU device for buffer/texture/pipeline creation.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Shared handle also passed to [`crate::runtime::RendererRuntime`] for uploads.
    pub fn queue(&self) -> &Arc<Mutex<wgpu::Queue>> {
        &self.queue
    }

    /// Submits render work for this frame; records last submit and GPU idle for the debug HUD timing HUD.
    pub fn submit_tracked_frame_commands(&self, cmd: wgpu::CommandBuffer) {
        frame_queue::submit_tracked_frame_commands(&self.frame_timing, &self.queue, cmd);
    }

    /// Same as [`Self::submit_tracked_frame_commands`] but uses an already-locked queue (e.g. debug HUD overlay encode).
    pub fn submit_tracked_frame_commands_with_queue(
        &self,
        queue: &mut wgpu::Queue,
        cmd: wgpu::CommandBuffer,
    ) {
        frame_queue::submit_tracked_frame_commands_with_queue(&self.frame_timing, queue, cmd);
    }

    /// Call at the start of each winit frame tick (same instant as [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn begin_frame_timing(&self, frame_start: Instant) {
        frame_queue::begin_frame_timing(&self.frame_timing, frame_start);
    }

    /// Call after all tracked queue submits for this tick (before reading HUD metrics).
    ///
    /// Finalizes CPU-until-submit for this tick. GPU idle time for the HUD comes from
    /// [`super::frame_cpu_gpu_timing::FrameCpuGpuTiming::last_completed_gpu_idle_ms`], which is
    /// updated asynchronously when [`wgpu::Queue::on_submitted_work_done`] runs—no blocking poll here.
    pub fn end_frame_timing(&self) {
        frame_queue::end_frame_timing(&self.frame_timing);
    }

    /// CPU time for this tick and the **latest completed** GPU submit→idle ms (may lag; see
    /// [`super::frame_cpu_gpu_timing::FrameCpuGpuTiming::last_completed_gpu_idle_ms`]).
    pub fn frame_cpu_gpu_ms_for_hud(&self) -> (Option<f64>, Option<f64>) {
        frame_queue::frame_cpu_gpu_ms_for_hud(&self.frame_timing)
    }

    /// Swapchain color format from the active surface configuration.
    pub fn config_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// WGPU adapter description captured at init ([`Self::new`]).
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Process-local GPU memory from wgpu’s allocator when the active backend supports
    /// [`wgpu::Device::generate_allocator_report`].
    ///
    /// Returns `(allocated_bytes, reserved_bytes)`, or `(None, None)` when the backend does not report.
    /// The **Stats** debug HUD tab uses these totals every capture; the **GPU memory** tab uses a
    /// throttled full [`wgpu::AllocatorReport`] via [`crate::runtime::RendererRuntime`].
    pub fn gpu_allocator_bytes(&self) -> (Option<u64>, Option<u64>) {
        self.device
            .generate_allocator_report()
            .map(|r| (Some(r.total_allocated_bytes), Some(r.total_reserved_bytes)))
            .unwrap_or((None, None))
    }

    /// Swapchain present mode (vsync policy).
    pub fn present_mode(&self) -> wgpu::PresentMode {
        self.config.present_mode
    }

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
    /// tier ([`Self::msaa_supported_sample_counts`]), not merely capped by the maximum tier.
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
                    "VR MSAA clamped: requested {}× → effective {}× (supported={:?})",
                    requested,
                    effective,
                    self.msaa_supported_sample_counts_stereo
                );
            }
            self.swapchain_msaa_requested_stereo = requested;
            self.swapchain_msaa_effective_stereo = effective;
        }
    }

    /// Ensures a single-sample [`wgpu::TextureFormat::R32Float`] texture for MSAA depth resolve + blit.
    pub fn ensure_msaa_depth_resolve_r32_view(
        &mut self,
    ) -> Result<&wgpu::TextureView, &'static str> {
        let extent = (self.config.width.max(1), self.config.height.max(1));
        msaa_resources::ensure_msaa_depth_resolve_r32_view(
            &self.device,
            extent,
            &mut self.msaa_depth_resolve_r32,
            &mut self.msaa_depth_resolve_r32_extent,
        )
    }

    /// Ensures multisampled color/depth targets for the main surface; returns [`None`] when `requested_samples` ≤ 1.
    pub fn ensure_msaa_targets(
        &mut self,
        requested_samples: u32,
        color_format: wgpu::TextureFormat,
    ) -> Option<&MsaaTargets> {
        let extent = (self.config.width.max(1), self.config.height.max(1));
        msaa_resources::ensure_msaa_targets(
            &self.device,
            extent,
            &self.msaa_supported_sample_counts,
            &mut self.msaa_targets,
            requested_samples,
            color_format,
        )
    }

    /// View of the R32F MSAA depth resolve temp when [`Self::ensure_msaa_depth_resolve_r32_view`] has run.
    pub(crate) fn msaa_depth_resolve_r32_view_ref(&self) -> Option<&wgpu::TextureView> {
        self.msaa_depth_resolve_r32.as_ref().map(|(_, v)| v)
    }

    /// Multisampled targets when MSAA is active for the swapchain path.
    pub(crate) fn msaa_targets_ref(&self) -> Option<&MsaaTargets> {
        self.msaa_targets.as_ref()
    }

    /// Ensures 2-layer (D2Array) multisampled color/depth targets for the OpenXR stereo path.
    ///
    /// - `requested_samples` is clamped against [`Self::msaa_supported_sample_counts_stereo`].
    /// - `extent` is per-eye pixel size from the OpenXR swapchain.
    /// - Returns [`None`] when MSAA is off or unsupported, in which case the caller renders directly
    ///   to the single-sample XR swapchain.
    pub fn ensure_msaa_stereo_targets(
        &mut self,
        requested_samples: u32,
        color_format: wgpu::TextureFormat,
        extent: (u32, u32),
    ) -> Option<&MsaaStereoTargets> {
        msaa_resources::ensure_msaa_stereo_targets(
            &self.device,
            extent,
            &self.msaa_supported_sample_counts_stereo,
            &mut self.msaa_stereo_targets,
            requested_samples,
            color_format,
        )
    }

    /// Ensures a 2-layer [`wgpu::TextureFormat::R32Float`] temp for stereo MSAA depth resolve.
    ///
    /// Matches the per-eye extent of [`Self::ensure_msaa_stereo_targets`]; reallocates on size change.
    pub(crate) fn ensure_msaa_stereo_depth_resolve(
        &mut self,
        extent: (u32, u32),
    ) -> Option<&MsaaStereoDepthResolveR32> {
        msaa_resources::ensure_msaa_stereo_depth_resolve(
            &self.device,
            extent,
            &mut self.msaa_stereo_depth_resolve_r32,
        )
    }

    /// Multisampled 2-layer targets when stereo MSAA is active for the OpenXR path.
    pub(crate) fn msaa_stereo_targets_ref(&self) -> Option<&MsaaStereoTargets> {
        self.msaa_stereo_targets.as_ref()
    }

    /// R32F resolve temp for stereo MSAA depth when [`Self::ensure_msaa_stereo_depth_resolve`] has run.
    pub(crate) fn msaa_stereo_depth_resolve_ref(&self) -> Option<&MsaaStereoDepthResolveR32> {
        self.msaa_stereo_depth_resolve_r32.as_ref()
    }

    /// Ensures a [`wgpu::TextureFormat::Depth32Float`] attachment exists for the current surface extent.
    ///
    /// Call after [`Self::reconfigure`] or when the swapchain size may have changed.
    ///
    /// Returns an error string only if the depth attachment could not be read after allocation (defensive).
    pub fn ensure_depth_view(&mut self) -> Result<&wgpu::TextureView, &'static str> {
        self.ensure_depth_target().map(|(_, v)| v)
    }

    /// Ensures the main depth attachment exists and returns both the texture and its default view.
    ///
    /// Returns an error string only if the depth attachment could not be read after allocation (defensive).
    pub fn ensure_depth_target(
        &mut self,
    ) -> Result<(&wgpu::Texture, &wgpu::TextureView), &'static str> {
        depth_attachment::ensure_depth_target(
            &self.device,
            &self.limits,
            self.config.width,
            self.config.height,
            &mut self.depth_attachment,
            &mut self.depth_extent_px,
        )
    }

    /// Acquires the next frame, reconfiguring once on [`wgpu::CurrentSurfaceTexture::Lost`] or
    /// [`wgpu::CurrentSurfaceTexture::Outdated`].
    pub fn acquire_with_recovery(
        &mut self,
        window: &Window,
    ) -> Result<wgpu::SurfaceTexture, wgpu::CurrentSurfaceTexture> {
        surface_acquire::acquire_with_recovery(self, window)
    }
}
