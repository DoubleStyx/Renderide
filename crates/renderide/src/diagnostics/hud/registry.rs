//! Static-dispatch registry for HUD windows and tabs.
//!
//! [`DebugWindow`] enumerates every top-level overlay window the diagnostics layer can render.
//! [`OverlayFeatureFlags`] captures which windows are enabled by [`crate::config::RendererSettings`]
//! at the start of a HUD frame. The dispatch loop in
//! [`crate::diagnostics::DebugHud::encode_overlay`] iterates [`DebugWindow::ALL`] and calls a
//! `match` per variant -- no `Box<dyn HudWindow<...>>` GAT pain, exhaustiveness-checked at compile
//! time, zero overhead.

use crate::config::RendererSettingsHandle;

/// Enumerates every top-level HUD window. Iterate [`Self::ALL`] for declarative dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugWindow {
    /// **Frame timing** overlay: FPS, CPU/GPU per-frame ms, RAM/VRAM, frametime sparkline.
    FrameTiming,
    /// **Feedback / Bug Report** overlay: quick links for reporting and discussion.
    Feedback,
    /// **Renderide debug** main panel (Stats / Shader routes / Draw state / GPU memory / GPU passes).
    Main,
    /// **Scene transforms** overlay: per-render-space world TRS tables.
    SceneTransforms,
    /// **Textures** overlay: texture pool listing with current-view filtering.
    Textures,
    /// **Renderer config** overlay: editable [`crate::config::RendererSettings`] with disk sync.
    RendererConfig,
}

impl DebugWindow {
    /// Static dispatch order -- controls draw order and tab ordering.
    pub const ALL: &'static [Self] = &[
        Self::FrameTiming,
        Self::Main,
        Self::SceneTransforms,
        Self::Textures,
        Self::RendererConfig,
        Self::Feedback,
    ];

    /// Returns `true` when this window should render this frame.
    ///
    /// The master ImGui visibility toggle hides every window when off.
    /// The optional debug windows are gated by their dedicated
    /// [`crate::config::DebugSettings`] flag. **Renderer config** has no per-window settings gate
    /// so configuration remains reachable whenever ImGui is visible.
    pub fn enabled(self, flags: OverlayFeatureFlags) -> bool {
        if !flags.imgui_visible {
            return false;
        }
        match self {
            Self::FrameTiming => flags.frame_timing,
            Self::Feedback => flags.links,
            Self::Main => flags.main,
            Self::SceneTransforms => flags.scene_transforms,
            Self::Textures => flags.textures,
            Self::RendererConfig => true,
        }
    }
}

/// Per-frame snapshot of which optional HUD windows are enabled by
/// [`crate::config::DebugSettings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayFeatureFlags {
    /// Master ImGui overlay visibility toggle.
    pub imgui_visible: bool,
    /// **Frame timing** window enabled, before master visibility gating.
    pub frame_timing: bool,
    /// **Renderide debug** main panel enabled, before master visibility gating.
    pub main: bool,
    /// **Scene transforms** window enabled, before master visibility gating.
    pub scene_transforms: bool,
    /// **Textures** window enabled, before master visibility gating.
    pub textures: bool,
    /// **Feedback / Bug Report** links panel enabled, before master visibility gating.
    pub links: bool,
}

impl Default for OverlayFeatureFlags {
    fn default() -> Self {
        Self {
            imgui_visible: true,
            frame_timing: false,
            main: false,
            scene_transforms: false,
            textures: false,
            links: true,
        }
    }
}

impl OverlayFeatureFlags {
    /// Snapshot the debug HUD visibility flags from the current settings handle.
    ///
    /// When the read lock cannot be acquired (poisoned), defaults to the lightweight windows
    /// enabled and the expensive debug-content windows off.
    pub fn from_settings(settings: &RendererSettingsHandle) -> Self {
        settings
            .read()
            .map(|g| OverlayFeatureFlags {
                imgui_visible: g.debug.hud.imgui_visible,
                frame_timing: g.debug.debug_hud_frame_timing,
                main: g.debug.debug_hud_enabled,
                scene_transforms: g.debug.debug_hud_transforms,
                textures: g.debug.debug_hud_textures,
                links: g.debug.debug_hud_links,
            })
            .unwrap_or(OverlayFeatureFlags {
                imgui_visible: true,
                frame_timing: true,
                main: false,
                scene_transforms: false,
                textures: false,
                links: true,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{DebugWindow, OverlayFeatureFlags};

    const ALL_OFF: OverlayFeatureFlags = OverlayFeatureFlags {
        imgui_visible: true,
        frame_timing: false,
        main: false,
        scene_transforms: false,
        textures: false,
        links: false,
    };
    const ALL_ON: OverlayFeatureFlags = OverlayFeatureFlags {
        imgui_visible: true,
        frame_timing: true,
        main: true,
        scene_transforms: true,
        textures: true,
        links: true,
    };

    fn only(window: DebugWindow) -> OverlayFeatureFlags {
        let mut f = ALL_OFF;
        match window {
            DebugWindow::FrameTiming => f.frame_timing = true,
            DebugWindow::Feedback => f.links = true,
            DebugWindow::Main => f.main = true,
            DebugWindow::SceneTransforms => f.scene_transforms = true,
            DebugWindow::Textures => f.textures = true,
            DebugWindow::RendererConfig => {}
        }
        f
    }

    #[test]
    fn master_visibility_disables_every_window() {
        let flags = OverlayFeatureFlags {
            imgui_visible: false,
            ..ALL_ON
        };
        for &w in DebugWindow::ALL {
            assert!(
                !w.enabled(flags),
                "{w:?} must be disabled when ImGui is hidden"
            );
        }
    }

    #[test]
    fn renderer_config_window_is_enabled_when_master_visible_regardless_of_debug_flags() {
        assert!(DebugWindow::RendererConfig.enabled(ALL_OFF));
    }

    #[test]
    fn feedback_window_gates_on_links_flag() {
        assert!(!DebugWindow::Feedback.enabled(ALL_OFF));
        assert!(DebugWindow::Feedback.enabled(only(DebugWindow::Feedback)));
    }

    #[test]
    fn each_debug_window_gates_on_its_own_flag() {
        for &w in DebugWindow::ALL {
            if w == DebugWindow::RendererConfig {
                continue;
            }
            let f = only(w);
            assert!(w.enabled(f), "{w:?} should enable when its flag is on");
            for &other in DebugWindow::ALL {
                if other == w || other == DebugWindow::RendererConfig {
                    continue;
                }
                assert!(
                    !other.enabled(f),
                    "{other:?} must remain disabled when only {w:?}'s flag is on"
                );
            }
        }
    }

    #[test]
    fn all_lists_every_variant_exactly_once() {
        let mut counts = [0usize; 6];
        for &w in DebugWindow::ALL {
            let idx = match w {
                DebugWindow::FrameTiming => 0,
                DebugWindow::Feedback => 1,
                DebugWindow::Main => 2,
                DebugWindow::SceneTransforms => 3,
                DebugWindow::Textures => 4,
                DebugWindow::RendererConfig => 5,
            };
            counts[idx] += 1;
        }
        for c in counts {
            assert_eq!(c, 1);
        }
    }
}
