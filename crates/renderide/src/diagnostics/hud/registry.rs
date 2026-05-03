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
    /// **Frame timing** overlay: FPS, CPU/GPU per-frame ms, frametime sparkline.
    FrameTiming,
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
    ];

    /// Returns `true` when this window should render this frame.
    ///
    /// The four debug windows are gated by their dedicated [`crate::config::DebugSettings`] flag.
    /// **Renderer config** has no settings gate -- its visibility is driven by the close-button
    /// open flag persisted in [`crate::diagnostics::HudUiState::renderer_config_open`].
    pub fn enabled(self, flags: OverlayFeatureFlags) -> bool {
        match self {
            Self::FrameTiming => flags.frame_timing,
            Self::Main => flags.main,
            Self::SceneTransforms => flags.scene_transforms,
            Self::Textures => flags.textures,
            Self::RendererConfig => true,
        }
    }
}

/// Per-frame snapshot of which optional HUD windows are enabled by
/// [`crate::config::DebugSettings`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OverlayFeatureFlags {
    /// **Frame timing** window enabled.
    pub frame_timing: bool,
    /// **Renderide debug** main panel enabled.
    pub main: bool,
    /// **Scene transforms** window enabled.
    pub scene_transforms: bool,
    /// **Textures** window enabled.
    pub textures: bool,
}

impl OverlayFeatureFlags {
    /// Snapshot the four `debug.debug_hud_*` flags from the current settings handle.
    ///
    /// When the read lock cannot be acquired (poisoned), defaults to `frame_timing = true` and
    /// the rest off so the renderer's lightweight overlay still appears.
    pub fn from_settings(settings: &RendererSettingsHandle) -> Self {
        settings
            .read()
            .map(|g| OverlayFeatureFlags {
                frame_timing: g.debug.debug_hud_frame_timing,
                main: g.debug.debug_hud_enabled,
                scene_transforms: g.debug.debug_hud_transforms,
                textures: g.debug.debug_hud_textures,
            })
            .unwrap_or(OverlayFeatureFlags {
                frame_timing: true,
                main: false,
                scene_transforms: false,
                textures: false,
            })
    }

    /// `true` when at least one of the four debug-content windows is enabled.
    ///
    /// Used by [`crate::diagnostics::DebugHud::has_visible_content`] to skip the entire HUD
    /// command encoder + GPU profiler query wrap when no debug windows are open.
    pub fn any_debug_content(self) -> bool {
        self.frame_timing || self.main || self.scene_transforms || self.textures
    }
}

#[cfg(test)]
mod tests {
    use super::{DebugWindow, OverlayFeatureFlags};

    const ALL_OFF: OverlayFeatureFlags = OverlayFeatureFlags {
        frame_timing: false,
        main: false,
        scene_transforms: false,
        textures: false,
    };
    const ALL_ON: OverlayFeatureFlags = OverlayFeatureFlags {
        frame_timing: true,
        main: true,
        scene_transforms: true,
        textures: true,
    };

    fn only(window: DebugWindow) -> OverlayFeatureFlags {
        let mut f = ALL_OFF;
        match window {
            DebugWindow::FrameTiming => f.frame_timing = true,
            DebugWindow::Main => f.main = true,
            DebugWindow::SceneTransforms => f.scene_transforms = true,
            DebugWindow::Textures => f.textures = true,
            DebugWindow::RendererConfig => {}
        }
        f
    }

    #[test]
    fn renderer_config_window_is_always_enabled_regardless_of_flags() {
        assert!(DebugWindow::RendererConfig.enabled(ALL_OFF));
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
    fn any_debug_content_truth_table() {
        assert!(!ALL_OFF.any_debug_content());
        assert!(only(DebugWindow::FrameTiming).any_debug_content());
        assert!(only(DebugWindow::Main).any_debug_content());
        assert!(only(DebugWindow::SceneTransforms).any_debug_content());
        assert!(only(DebugWindow::Textures).any_debug_content());
        assert!(ALL_ON.any_debug_content());
    }

    #[test]
    fn all_lists_every_variant_exactly_once() {
        let mut counts = [0usize; 5];
        for &w in DebugWindow::ALL {
            let idx = match w {
                DebugWindow::FrameTiming => 0,
                DebugWindow::Main => 1,
                DebugWindow::SceneTransforms => 2,
                DebugWindow::Textures => 3,
                DebugWindow::RendererConfig => 4,
            };
            counts[idx] += 1;
        }
        for c in counts {
            assert_eq!(c, 1);
        }
    }
}
