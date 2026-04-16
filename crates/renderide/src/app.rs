//! Winit-driven application: startup ([`startup::run`]), [`renderide_app::RenderideApp`] implementing
//! [`winit::application::ApplicationHandler`], plus [`frame_loop`] and [`frame_pacing`] helpers.
//!
//! The main window is created maximized via [`winit::window::Window::default_attributes`] and
//! [`with_maximized(true)`](winit::window::WindowAttributes::with_maximized), which winit maps to
//! the appropriate Win32, X11, and Wayland behavior.
//!
//! When the host selects a VR [`HeadOutputDevice`](crate::shared::HeadOutputDevice), the Vulkan
//! device may come from [`crate::xr::init_wgpu_openxr`]; the mirror window uses the same device.
//! OpenXR success path state (handles, stereo swapchain/depth, mirror blit) lives in
//! [`crate::xr::XrSessionBundle`] as [`crate::app::renderide_app::RenderideApp`]'s `xr_session` field.
//! Each frame: OpenXR `wait_frame` / `locate_views` run **before** lock-step `pre_frame` so headset
//! pose in [`InputState::vr`](crate::shared::InputState) matches the same `locate_views` snapshot.
//! The desktop window uses the normal render graph when VR is inactive. When `vr_active` and multiview
//! are available, the headset path renders once to the OpenXR array swapchain and ends the frame with a
//! projection layer; the desktop window shows a **blit of the left-eye** HMD output (no second world render).
//! When the HMD path does not run, the window is cleared for that frame.
//!
//! VR **IPC input** (a non-empty [`InputState::vr`](crate::shared::InputState)) is sent whenever
//! [`RenderideApp`](renderide_app::RenderideApp)’s session output device is VR-capable so the host can create headset devices. If OpenXR
//! init fails, the app falls back to desktop GPU while still sending VR IPC input when the session
//! device is VR-capable.
//!
//! ## Process exit visibility (crashes, panics, signals)
//!
//! See [`startup`] module documentation for how fatal faults ([`crate::fatal_crash_log`]), panics
//! ([`std::panic::set_hook`]), and graceful shutdown (Unix signals / Windows Ctrl+C) are handled as
//! separate layers.

mod frame_loop;
mod frame_pacing;
mod renderide_app;
mod startup;
mod window_icon;

pub use startup::run;
