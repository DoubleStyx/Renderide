//! Rendering algebra for HUD windows and tabs.
//!
//! Two traits compose into one declarative HUD:
//!
//! - [`HudWindow`] wraps an ImGui window envelope (title, anchor, flags, optional close-button
//!   open flag, background alpha) around a body. It exists so each top-level window's
//!   positioning and styling lives in one impl rather than scattered through
//!   [`crate::diagnostics::DebugHud::encode_overlay`].
//! - [`TabView`] is the leaf flavor for tabs in a tab-bar: "render this tab body when the tab is
//!   active." A single window -- the **Renderide debug** main panel -- composes a fixed list of
//!   `TabView` impls.
//!
//! Both traits use [generic associated types][gat] so each impl can describe its own borrowed
//! data lifetime without forcing trait-object dispatch. Window/tab dispatch is by static enum
//! and direct method call (see [`super::windows::main_debug`]) so the
//! `Box<dyn HudWindow<...>>` GAT pain is avoided.
//!
//! [gat]: https://blog.rust-lang.org/2022/10/28/gats-stabilization.html

use imgui::WindowFlags;

use super::layout::{Viewport, WindowSlot};

/// A top-level HUD window: ImGui envelope (title, anchor, flags, open flag) + body.
///
/// The dispatcher (see [`crate::diagnostics::DebugHud::encode_overlay`]) calls
/// [`Self::read_open_flag`] before opening the ImGui window, hands the resulting `&mut bool` to
/// `Window::opened` for the close-button handling, then calls [`Self::write_open_flag`] after
/// the body returns to persist any user-driven close-button change. Splitting the open-flag
/// access into read/write halves keeps the body's `&mut Self::State` borrow disjoint from the
/// open-flag borrow.
pub trait HudWindow {
    /// Borrowed snapshot data the window body reads.
    type Data<'a>;
    /// Mutable per-window UI state.
    type State;

    /// ImGui window title (also used as the window id).
    fn title(&self) -> &str;

    /// First-use placement resolved against the current viewport.
    fn anchor(&self, viewport: Viewport) -> WindowSlot;

    /// ImGui window flags (defaults: `ALWAYS_AUTO_RESIZE | NO_SAVED_SETTINGS`).
    fn flags(&self) -> WindowFlags {
        WindowFlags::ALWAYS_AUTO_RESIZE | WindowFlags::NO_SAVED_SETTINGS
    }

    /// Background alpha (defaults to a translucent overlay so renderer output stays visible).
    fn bg_alpha(&self) -> f32 {
        0.72
    }

    /// Read the current value of the close-button flag, or `None` for windows whose visibility
    /// is gated solely by [`crate::config::RendererSettings`] flags.
    fn read_open_flag(&self, _state: &Self::State) -> Option<bool> {
        None
    }

    /// Persist the (possibly-changed) close-button flag back into state.
    fn write_open_flag(&self, _state: &mut Self::State, _value: bool) {}

    /// Render the window body. Called inside the ImGui window scope.
    fn body(&self, ui: &imgui::Ui, data: Self::Data<'_>, state: &mut Self::State);
}

/// A tab body inside a tab-bar.
///
/// Tab labels are owned by the [`super::windows::main_debug::DebugTab`] enum (the registry that
/// dispatches each tab variant to its concrete impl) so they live alongside the dispatch order
/// in one place; the body trait only carries the per-tab data shape and `render` method.
pub trait TabView {
    /// Borrowed snapshot data the tab body reads.
    type Data<'a>;
    /// Mutable per-window UI state.
    type State;

    /// Render the tab body when the tab is active.
    fn render(&self, ui: &imgui::Ui, data: Self::Data<'_>, state: &mut Self::State);
}
