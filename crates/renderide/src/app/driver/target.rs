//! Window, GPU, and OpenXR target creation for the winit driver.

use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use winit::event_loop::{ActiveEventLoop, OwnedDisplayHandle};
use winit::monitor::Fullscreen;
#[cfg(target_os = "windows")]
use winit::platform::windows::WindowAttributesWindows;
use winit::window::{Window, WindowAttributes};

use crate::diagnostics::crash_context::{self, TargetMode};
use crate::frontend::input::enable_ime_on_window;
use crate::frontend::output_device::head_output_device_wants_openxr;
use crate::gpu::{GpuContext, GpuError, WindowDisplayHandle};
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, RendererInitData};
use crate::xr::XrSessionBundle;

use super::super::bootstrap::GpuStartupConfig;
use super::super::exit::ExitReason;
use super::shutdown::GracefulShutdown;

/// Upper bound on the device-wide GPU drain performed before tearing down the OpenXR
/// swapchain. Bounded so a stuck driver does not block the rest of the shutdown
/// sequence; on timeout we proceed and rely on the wgpu drop-callback (which now skips
/// `vkDestroyImage` for runtime-owned images) to keep teardown safe.
const GPU_SHUTDOWN_POLL_TIMEOUT: Duration = Duration::from_secs(1);

/// Fully initialized windowed render target.
pub(super) struct RenderTarget {
    window: Arc<dyn Window>,
    gpu: GpuContext,
    output_device: HeadOutputDevice,
    mode: RenderTargetMode,
}

/// Device mode for the windowed render target.
pub(super) enum RenderTargetMode {
    /// Ordinary desktop swapchain rendering.
    Desktop,
    /// OpenXR headset rendering plus desktop mirror surface.
    Openxr { session: Box<XrSessionBundle> },
}

/// Target creation failure before the app can enter its redraw loop.
#[derive(Debug, Error)]
pub(super) enum TargetInitError {
    /// Winit rejected main-window creation.
    #[error("create_window failed: {0}")]
    WindowCreate(String),
    /// Desktop GPU initialization failed.
    #[error("GPU init failed: {0}")]
    DesktopGpu(#[source] GpuError),
    /// OpenXR bootstrap failed.
    #[error("OpenXR init failed: {0}. Renderer aborting because VR was requested")]
    OpenxrInit(String),
    /// OpenXR GPU was created but could not present to the mirror surface.
    #[error(
        "OpenXR mirror surface failed: {0}. Renderer aborting because falling back after partial OpenXR init is unsafe"
    )]
    OpenxrMirrorSurface(#[source] GpuError),
}

impl TargetInitError {
    /// Exit reason that matches the target initialization failure.
    pub(super) const fn exit_reason(&self) -> ExitReason {
        match self {
            Self::WindowCreate(_) => ExitReason::WindowCreateFailed,
            Self::DesktopGpu(_) => ExitReason::DesktopGpuInitFailed,
            Self::OpenxrInit(_) => ExitReason::OpenxrInitFailed,
            Self::OpenxrMirrorSurface(_) => ExitReason::OpenxrMirrorSurfaceFailed,
        }
    }
}

impl RenderTarget {
    /// Creates the window, selects desktop vs OpenXR mode, and attaches the GPU to the runtime.
    pub(super) fn create(
        event_loop: &dyn ActiveEventLoop,
        runtime: &mut RendererRuntime,
        startup_gpu: GpuStartupConfig,
        display_handle: &OwnedDisplayHandle,
    ) -> Result<Self, TargetInitError> {
        let window = create_main_window(event_loop)?;
        let output_device = effective_output_device_for_gpu(runtime.pending_init());

        if let Some(init) = runtime.take_pending_init() {
            apply_window_title_from_init(&window, &init);
            if let Some(icon) = init.set_window_icon {
                runtime.queue_window_icon_request(icon);
            }
        }

        let (gpu, mode) = if head_output_device_wants_openxr(output_device) {
            crash_context::set_target_mode(TargetMode::OpenXr);
            create_openxr_target(&window, startup_gpu)?
        } else {
            crash_context::set_target_mode(TargetMode::Desktop);
            create_desktop_target(&window, startup_gpu, display_handle)?
        };

        runtime.attach_gpu(&gpu);
        enable_ime_on_window(window.as_ref());

        Ok(Self {
            window,
            gpu,
            output_device,
            mode,
        })
    }

    /// Main winit window.
    pub(super) fn window(&self) -> &Arc<dyn Window> {
        &self.window
    }

    /// Returns whether the main winit window is currently fullscreen.
    pub(super) fn is_fullscreen(&self) -> bool {
        self.window.fullscreen().is_some()
    }

    /// Toggles borderless fullscreen and returns the requested fullscreen state.
    pub(super) fn toggle_borderless_fullscreen(&self) -> bool {
        let fullscreen = !self.is_fullscreen();
        self.set_borderless_fullscreen(fullscreen);
        fullscreen
    }

    /// Applies borderless fullscreen when `fullscreen` is true, otherwise restores windowed mode.
    pub(super) fn set_borderless_fullscreen(&self, fullscreen: bool) {
        let fullscreen = fullscreen.then(|| Fullscreen::Borderless(None));
        self.window.set_fullscreen(fullscreen);
    }

    /// Active GPU context.
    pub(super) fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    /// Mutable active GPU context.
    pub(super) fn gpu_mut(&mut self) -> &mut GpuContext {
        &mut self.gpu
    }

    /// Host-requested output device that selected this target.
    pub(super) const fn output_device(&self) -> HeadOutputDevice {
        self.output_device
    }

    /// Mutable OpenXR session state when the target is in OpenXR mode.
    pub(super) fn xr_session_mut(&mut self) -> Option<&mut XrSessionBundle> {
        match &mut self.mode {
            RenderTargetMode::Desktop => None,
            RenderTargetMode::Openxr { session } => Some(session.as_mut()),
        }
    }

    /// Shared OpenXR session state when the target is in OpenXR mode.
    pub(super) fn xr_session(&self) -> Option<&XrSessionBundle> {
        match &self.mode {
            RenderTargetMode::Desktop => None,
            RenderTargetMode::Openxr { session } => Some(session.as_ref()),
        }
    }

    /// Mutable GPU and OpenXR session pair.
    pub(super) fn openxr_parts_mut(&mut self) -> Option<(&mut GpuContext, &mut XrSessionBundle)> {
        match &mut self.mode {
            RenderTargetMode::Desktop => None,
            RenderTargetMode::Openxr { session } => Some((&mut self.gpu, session.as_mut())),
        }
    }

    /// Reconfigures swapchain/depth using explicit physical dimensions.
    pub(super) fn reconfigure_physical_size(&mut self, width: u32, height: u32) {
        profiling::scope!("startup::reconfigure_gpu");
        self.gpu.reconfigure(width, height);
    }

    /// Reconfigures using the live window size, falling back to cached surface extent.
    pub(super) fn reconfigure_for_window(&mut self) {
        profiling::scope!("startup::reconfigure_gpu");
        let (width, height) = self
            .gpu
            .window_surface_size()
            .unwrap_or_else(|| self.gpu.surface_extent_px());
        self.gpu.reconfigure(width, height);
    }

    /// Advances graceful target shutdown and returns `true` once resources are quiescent.
    pub(super) fn poll_graceful_shutdown(&mut self, shutdown: &mut GracefulShutdown) -> bool {
        profiling::scope!("app::target_graceful_shutdown");
        self.gpu.wait_for_previous_present();
        match &mut self.mode {
            RenderTargetMode::Desktop => true,
            RenderTargetMode::Openxr { session } => {
                let quiesced = poll_openxr_shutdown(&mut session.handles.xr_session, shutdown);
                if quiesced {
                    poll_gpu_for_openxr_swapchain_shutdown(&self.gpu);
                }
                quiesced
            }
        }
    }
}

/// Returns whether shutdown should drain device work before OpenXR swapchain teardown.
fn should_poll_gpu_for_openxr_shutdown(device_lost: bool) -> bool {
    !device_lost
}

/// Drains GPU work that may still reference OpenXR swapchain images before session teardown.
fn poll_gpu_for_openxr_swapchain_shutdown(gpu: &GpuContext) {
    if !should_poll_gpu_for_openxr_shutdown(gpu.device_lost()) {
        logger::warn!(
            "OpenXR shutdown: skipping device poll(Wait) because the wgpu device is already lost"
        );
        return;
    }

    let poll_type = wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(GPU_SHUTDOWN_POLL_TIMEOUT),
    };
    if let Err(error) = gpu.device().poll(poll_type) {
        logger::warn!(
            "OpenXR shutdown: device poll(Wait) failed before swapchain teardown: {error:?}"
        );
    }
}

fn poll_openxr_shutdown(
    xr_session: &mut crate::xr::XrSessionState,
    shutdown: &mut GracefulShutdown,
) -> bool {
    xr_session.begin_shutdown();

    if !xr_session.poll_finalize_pending() {
        return false;
    }

    if let Err(error) = xr_session.poll_events() {
        logger::warn!("OpenXR shutdown poll_events failed: {error:?}");
    }
    if xr_session.shutdown_quiesced() {
        return true;
    }

    if !shutdown.openxr_exit_requested() {
        if let Err(error) = xr_session.request_exit_for_shutdown() {
            logger::warn!("OpenXR shutdown request_exit failed: {error:?}");
        }
        shutdown.mark_openxr_exit_requested();
    }

    if let Err(error) = xr_session.poll_events() {
        logger::warn!("OpenXR shutdown poll_events after request_exit failed: {error:?}");
    }
    xr_session.shutdown_quiesced()
}

#[cfg(test)]
mod tests {
    use super::should_poll_gpu_for_openxr_shutdown;

    #[test]
    fn openxr_shutdown_polls_gpu_while_device_is_healthy() {
        assert!(should_poll_gpu_for_openxr_shutdown(false));
    }

    #[test]
    fn openxr_shutdown_skips_gpu_poll_after_device_loss() {
        assert!(!should_poll_gpu_for_openxr_shutdown(true));
    }
}

fn create_main_window(
    event_loop: &dyn ActiveEventLoop,
) -> Result<Arc<dyn Window>, TargetInitError> {
    let attrs = WindowAttributes::default()
        .with_title("Renderide")
        .with_maximized(true)
        .with_visible(true);
    #[cfg(target_os = "windows")]
    let attrs = attrs.with_platform_attributes(Box::new(
        WindowAttributesWindows::default().with_use_system_scroll_speed(false),
    ));

    event_loop
        .create_window(attrs)
        .map(Arc::from)
        .map_err(|error| TargetInitError::WindowCreate(error.to_string()))
}

fn create_desktop_target(
    window: &Arc<dyn Window>,
    startup_gpu: GpuStartupConfig,
    display_handle: &OwnedDisplayHandle,
) -> Result<(GpuContext, RenderTargetMode), TargetInitError> {
    let display_handle =
        WindowDisplayHandle::from_winit(display_handle).map_err(TargetInitError::DesktopGpu)?;
    pollster::block_on(GpuContext::new(
        Arc::clone(window),
        display_handle,
        startup_gpu.vsync,
        startup_gpu.max_frame_latency,
        startup_gpu.gpu_validation_layers,
        startup_gpu.power_preference,
        startup_gpu.graphics_api,
    ))
    .map(|gpu| {
        logger::info!("GPU initialized (desktop)");
        (gpu, RenderTargetMode::Desktop)
    })
    .map_err(TargetInitError::DesktopGpu)
}

fn create_openxr_target(
    window: &Arc<dyn Window>,
    startup_gpu: GpuStartupConfig,
) -> Result<(GpuContext, RenderTargetMode), TargetInitError> {
    if !startup_gpu.graphics_api.is_openxr_compatible() {
        logger::warn!(
            "Configured graphics_api={} is incompatible with the OpenXR path; using Vulkan for OpenXR startup.",
            startup_gpu.graphics_api.as_persist_str()
        );
    }
    let handles = crate::xr::init_wgpu_openxr(
        startup_gpu.gpu_validation_layers,
        startup_gpu.power_preference,
    )
    .map_err(|error| TargetInitError::OpenxrInit(error.to_string()))?;
    let gpu = GpuContext::new_from_openxr_bootstrap(
        &handles.wgpu_instance,
        &handles.wgpu_adapter,
        Arc::clone(&handles.device),
        Arc::clone(&handles.queue),
        Arc::clone(window),
        startup_gpu.vsync,
        startup_gpu.max_frame_latency,
    )
    .map_err(TargetInitError::OpenxrMirrorSurface)?;

    logger::info!("GPU initialized (OpenXR Vulkan device + mirror surface)");
    Ok((
        gpu,
        RenderTargetMode::Openxr {
            session: Box::new(XrSessionBundle::new(handles)),
        },
    ))
}

fn effective_output_device_for_gpu(pending: Option<&RendererInitData>) -> HeadOutputDevice {
    pending.map_or(HeadOutputDevice::Screen, |init| init.output_device)
}

fn apply_window_title_from_init(window: &Arc<dyn Window>, init: &RendererInitData) {
    if let Some(ref title) = init.window_title {
        window.set_title(title);
    }
}

#[cfg(test)]
mod effective_output_device_tests {
    use super::effective_output_device_for_gpu;
    use crate::shared::{HeadOutputDevice, RendererInitData};

    #[test]
    fn none_falls_back_to_screen() {
        assert_eq!(
            effective_output_device_for_gpu(None),
            HeadOutputDevice::Screen
        );
    }

    #[test]
    fn some_screen_returns_screen() {
        let init = RendererInitData {
            output_device: HeadOutputDevice::Screen,
            ..Default::default()
        };
        assert_eq!(
            effective_output_device_for_gpu(Some(&init)),
            HeadOutputDevice::Screen
        );
    }

    #[test]
    fn some_vr_device_is_passed_through() {
        let init = RendererInitData {
            output_device: HeadOutputDevice::SteamVR,
            ..Default::default()
        };
        assert_eq!(
            effective_output_device_for_gpu(Some(&init)),
            HeadOutputDevice::SteamVR
        );
    }

    #[test]
    fn some_autodetect_is_passed_through() {
        let init = RendererInitData {
            output_device: HeadOutputDevice::Autodetect,
            ..Default::default()
        };
        assert_eq!(
            effective_output_device_for_gpu(Some(&init)),
            HeadOutputDevice::Autodetect
        );
    }
}
