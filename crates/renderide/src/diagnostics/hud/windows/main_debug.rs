//! **Renderide debug** main panel -- anchored top-right window with one [`TabView`] per concern.
//!
//! The window envelope is the [`HudWindow`] impl on [`MainDebugWindow`]; the body iterates each
//! tab (`Stats / Shader routes / Draw state / GPU memory / GPU passes`) by static dispatch.

pub mod draw_state;
pub mod gpu_memory;
pub mod gpu_passes;
pub mod shader_routes;
pub mod stats;

use imgui::{TabItem, TabItemFlags, WindowFlags};

use crate::config::DebugHudMainTab;
use crate::diagnostics::{FrameDiagnosticsSnapshot, RendererInfoSnapshot};
use crate::profiling::GpuPassEntry;

use super::super::layout::{self, Viewport, WindowAnchor, WindowSlot};
use super::super::state::HudUiState;
use super::super::view::{HudWindow, TabView};

pub use draw_state::DrawStateTab;
pub use gpu_memory::GpuMemoryTab;
pub use gpu_passes::GpuPassesTab;
pub use shader_routes::ShaderRoutesTab;
pub use stats::StatsTab;

/// Borrowed snapshots fed to [`MainDebugWindow`].
pub struct MainDebugWindowData<'a> {
    /// Stats tab payload (`Frame index`, GPU adapter, scene counts, materials, graph).
    pub renderer_info: Option<&'a RendererInfoSnapshot>,
    /// Frame-scoped tab payload (host metrics, draw stats, shader routes, GPU memory).
    pub frame_diagnostics: Option<&'a FrameDiagnosticsSnapshot>,
    /// Per-pass GPU timings for the **GPU passes** tab.
    pub gpu_pass_timings: &'a [GpuPassEntry],
}

/// **Renderide debug** HUD window -- anchored top-right tabbed panel.
pub struct MainDebugWindow;

impl HudWindow for MainDebugWindow {
    type Data<'a> = MainDebugWindowData<'a>;
    type State = HudUiState;

    fn title(&self) -> &str {
        "Renderide debug"
    }

    fn anchor(&self, viewport: Viewport) -> WindowSlot {
        WindowAnchor::TopRight {
            width: layout::MAIN_DEBUG_PANEL_W,
        }
        .resolve(viewport)
    }

    fn flags(&self) -> WindowFlags {
        WindowFlags::NO_FOCUS_ON_APPEARING | WindowFlags::NO_NAV
    }

    fn body(&self, ui: &imgui::Ui, data: Self::Data<'_>, state: &mut Self::State) {
        if let Some(_tab_bar) = ui.tab_bar("debug_tabs") {
            for &tab in DebugTab::ALL {
                profiling::scope!("hud::render_tab");
                let tab_id = tab.persisted_tab();
                let flags = if state.main_tab_restore_pending && state.main_tab == tab_id {
                    TabItemFlags::SET_SELECTED
                } else {
                    TabItemFlags::empty()
                };
                if let Some(_tab_item) = TabItem::new(tab.label()).flags(flags).begin(ui) {
                    state.main_tab = tab_id;
                    state.main_tab_restore_pending = false;
                    tab.render(ui, &data, state);
                }
            }
        }
    }
}

/// Static-dispatch registry for the main panel's tabs.
///
/// Iteration order in [`Self::ALL`] is the user-visible tab order. Each variant projects the
/// data subset it consumes from the shared [`MainDebugWindowData`] in [`Self::render`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugTab {
    /// **Stats** tab -- frame index, GPU adapter, host metrics, draw stats, resources.
    Stats,
    /// **Shader routes** tab -- host shader -> renderer pipeline routing list.
    ShaderRoutes,
    /// **Draw state** tab -- sorted mesh draws with material pipeline state.
    DrawState,
    /// **GPU memory** tab -- full wgpu allocator report.
    GpuMemory,
    /// **GPU passes** tab -- per-pass GPU timing breakdown.
    GpuPasses,
}

impl DebugTab {
    /// Static dispatch order. Drives the user-visible left-to-right tab order.
    pub const ALL: &'static [Self] = &[
        Self::Stats,
        Self::ShaderRoutes,
        Self::DrawState,
        Self::GpuMemory,
        Self::GpuPasses,
    ];

    /// ImGui tab label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Stats => "Stats",
            Self::ShaderRoutes => "Shader routes",
            Self::DrawState => "Draw state",
            Self::GpuMemory => "GPU memory",
            Self::GpuPasses => "GPU passes",
        }
    }

    /// Stable config enum for this tab.
    pub fn persisted_tab(self) -> DebugHudMainTab {
        match self {
            Self::Stats => DebugHudMainTab::Stats,
            Self::ShaderRoutes => DebugHudMainTab::ShaderRoutes,
            Self::DrawState => DebugHudMainTab::DrawState,
            Self::GpuMemory => DebugHudMainTab::GpuMemory,
            Self::GpuPasses => DebugHudMainTab::GpuPasses,
        }
    }

    /// Render this tab. Projects the right [`MainDebugWindowData`] sub-fields per variant.
    pub fn render(self, ui: &imgui::Ui, data: &MainDebugWindowData<'_>, state: &mut HudUiState) {
        match self {
            Self::Stats => {
                StatsTab.render(ui, (data.renderer_info, data.frame_diagnostics), state);
            }
            Self::ShaderRoutes => ShaderRoutesTab.render(ui, data.frame_diagnostics, state),
            Self::DrawState => DrawStateTab.render(ui, data.frame_diagnostics, state),
            Self::GpuMemory => GpuMemoryTab.render(ui, data.frame_diagnostics, state),
            Self::GpuPasses => GpuPassesTab.render(ui, data.gpu_pass_timings, state),
        }
    }
}
