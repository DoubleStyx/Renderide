//! Renderer process application boundary: bootstrap, headless driver, and winit event-loop driver.
//!
//! The main window is created maximized via [`winit::window::Window::default_attributes`] and
//! [`with_maximized(true)`](winit::window::WindowAttributes::with_maximized), which winit maps to
//! the appropriate Win32, X11, and Wayland behavior.
//!
//! When the host selects a VR [`HeadOutputDevice`](crate::shared::HeadOutputDevice), the Vulkan
//! device may come from [`crate::xr::init_wgpu_openxr`]; the mirror window uses the same device.
//! OpenXR success path state (handles, stereo swapchain/depth, mirror blit) lives in
//! [`crate::xr::XrSessionBundle`] as the app driver's OpenXR render target mode.
//! Each rendered VR frame samples OpenXR `wait_frame` / `locate_views` before sending the next
//! lock-step `FrameStartData`, so headset pose in [`InputState::vr`](crate::shared::InputState)
//! comes from the current HMD tick. When coupled lock-step is waiting for the host, that wait runs
//! before `xrBeginFrame` so the compositor reprojects the last submitted frame instead of receiving
//! an empty frame.
//! The desktop window uses the normal render graph when VR is inactive. When `vr_active` and multiview
//! are available, the headset path renders once to the OpenXR array swapchain and ends the frame with a
//! projection layer; the desktop window shows a **blit of the left-eye** HMD output (no second world render).
//! When the HMD path does not run, the window is cleared for that frame.
//!
//! VR **IPC input** (a non-empty [`InputState::vr`](crate::shared::InputState)) is sent whenever
//! the session output device is VR-capable so the host can create headset devices. If OpenXR
//! init or mirror-surface creation fails after VR was requested, the app exits with a VR startup
//! code instead of silently demoting a partially initialized OpenXR session.
//!
//! ## Process exit visibility (crashes, panics, signals)
//!
//! Fatal faults, panics ([`std::panic::set_hook`]), and graceful shutdown (Unix signals /
//! Windows Ctrl+C) are installed as separate bootstrap services because each layer has different
//! safety constraints.

mod bootstrap;
mod driver;
mod exit;
mod frame_clock;
mod headless;
mod redraw_plan;
mod window_icon;

pub use bootstrap::run;
pub use exit::RunExit;
