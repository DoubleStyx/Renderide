//! Runtime-owned renderer configuration handles.

use std::path::PathBuf;

use crate::config::{RendererSettingsHandle, save_renderer_settings};
use crate::shared::{DesktopConfig, QualityConfig, SkinWeightMode};

/// Minimum positive host desktop frame-rate cap accepted from [`DesktopConfig`].
const MIN_HOST_DESKTOP_FPS_CAP: u32 = 5;

/// Effective foreground and background desktop frame-pacing caps for the app driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DesktopFramePacingCaps {
    /// FPS cap used while the OS reports the renderer window has keyboard focus.
    pub(crate) foreground_fps_cap: u32,
    /// FPS cap used while the OS reports the renderer window does not have keyboard focus.
    pub(crate) background_fps_cap: u32,
}

/// Host-supplied desktop frame-pacing overrides from [`DesktopConfig`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct HostDesktopFramePacingCaps {
    /// Positive host foreground cap, after Unity-compatible minimum clamping.
    foreground_fps_cap: Option<u32>,
    /// Positive host background cap, after Unity-compatible minimum clamping.
    background_fps_cap: Option<u32>,
}

impl HostDesktopFramePacingCaps {
    /// Converts host desktop config fields into optional frame-pacing caps.
    fn from_desktop_config(cfg: &DesktopConfig) -> Self {
        Self {
            foreground_fps_cap: sanitized_host_fps_cap(cfg.maximum_foreground_framerate),
            background_fps_cap: sanitized_host_fps_cap(cfg.maximum_background_framerate),
        }
    }

    /// Resolves optional host caps over renderer-config fallback caps.
    fn resolve(self, fallback: DesktopFramePacingCaps) -> DesktopFramePacingCaps {
        DesktopFramePacingCaps {
            foreground_fps_cap: self
                .foreground_fps_cap
                .unwrap_or(fallback.foreground_fps_cap),
            background_fps_cap: self
                .background_fps_cap
                .unwrap_or(fallback.background_fps_cap),
        }
    }
}

/// Returns a positive host cap with Unity-compatible minimum clamping.
fn sanitized_host_fps_cap(raw: Option<i32>) -> Option<u32> {
    raw.filter(|cap| *cap > 0)
        .map(|cap| (cap as u32).max(MIN_HOST_DESKTOP_FPS_CAP))
}

/// Settings handle, persistence path, and config-write suppression state.
pub(in crate::runtime) struct RuntimeConfigState {
    /// Process-wide renderer settings shared with the HUD and frame loop.
    pub(in crate::runtime) settings: RendererSettingsHandle,
    /// Target path for persisting renderer settings from the ImGui config window.
    config_save_path: PathBuf,
    /// When true, ImGui and config save helpers must not overwrite `config.toml`.
    suppress_renderer_config_disk_writes: bool,
    /// Optional host frame-pacing overrides from the latest [`DesktopConfig`].
    host_desktop_caps: HostDesktopFramePacingCaps,
    /// Effective host skinning quality mode from the latest [`QualityConfig`].
    host_skin_weight_mode: SkinWeightMode,
}

impl RuntimeConfigState {
    /// Creates runtime config state from the loaded settings handle and save path.
    pub(in crate::runtime) fn new(
        settings: RendererSettingsHandle,
        config_save_path: PathBuf,
    ) -> Self {
        Self {
            settings,
            config_save_path,
            suppress_renderer_config_disk_writes: false,
            host_desktop_caps: HostDesktopFramePacingCaps::default(),
            host_skin_weight_mode: SkinWeightMode::Unlimited,
        }
    }

    /// Applies host desktop frame-pacing overrides without mutating persisted renderer settings.
    pub(in crate::runtime) fn apply_host_desktop_config(&mut self, cfg: &DesktopConfig) {
        self.host_desktop_caps = HostDesktopFramePacingCaps::from_desktop_config(cfg);
    }

    /// Applies host rendering quality state without mutating persisted renderer settings.
    pub(in crate::runtime) fn apply_host_quality_config(&mut self, cfg: &QualityConfig) {
        self.host_skin_weight_mode = cfg.skin_weight_mode;
    }

    /// Returns desktop frame-pacing caps after applying host overrides over renderer settings.
    pub(in crate::runtime) fn desktop_frame_pacing_caps(&self) -> DesktopFramePacingCaps {
        let fallback = self
            .settings
            .read()
            .map(|settings| DesktopFramePacingCaps {
                foreground_fps_cap: settings.display.focused_fps_cap,
                background_fps_cap: settings.display.unfocused_fps_cap,
            })
            .unwrap_or(DesktopFramePacingCaps {
                foreground_fps_cap: 0,
                background_fps_cap: 0,
            });
        self.host_desktop_caps.resolve(fallback)
    }

    /// Effective host-owned skin weight mode used for mesh skinning.
    pub(in crate::runtime) fn skin_weight_mode(&self) -> SkinWeightMode {
        self.host_skin_weight_mode
    }

    /// Cloned config save path for backend HUD attach.
    pub(in crate::runtime) fn cloned_config_save_path(&self) -> PathBuf {
        self.config_save_path.clone()
    }

    /// Sets whether renderer config disk writes are blocked.
    pub(in crate::runtime) fn set_suppress_renderer_config_disk_writes(&mut self, value: bool) {
        self.suppress_renderer_config_disk_writes = value;
    }

    /// Whether renderer config disk writes are blocked.
    pub(in crate::runtime) fn suppress_renderer_config_disk_writes(&self) -> bool {
        self.suppress_renderer_config_disk_writes
    }

    /// Toggles the master ImGui overlay visibility setting and persists it when allowed.
    pub(in crate::runtime) fn toggle_imgui_visibility(&self) -> Option<bool> {
        let Ok(mut settings) = self.settings.write() else {
            logger::warn!(
                "Failed to toggle ImGui visibility: renderer settings store is unavailable"
            );
            return None;
        };

        settings.debug.hud.imgui_visible = !settings.debug.hud.imgui_visible;
        let visible = settings.debug.hud.imgui_visible;

        if self.suppress_renderer_config_disk_writes {
            logger::error!(
                "Refusing to save renderer config to {}: disk writes suppressed after startup extract failure",
                self.config_save_path.display()
            );
            return Some(visible);
        }

        if let Err(e) = save_renderer_settings(&self.config_save_path, &settings) {
            logger::warn!(
                "Failed to save renderer config to {}: {e}",
                self.config_save_path.display()
            );
        }

        Some(visible)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::{Arc, RwLock};

    use crate::config::RendererSettings;
    use crate::shared::{DesktopConfig, QualityConfig, SkinWeightMode};

    use super::{DesktopFramePacingCaps, RuntimeConfigState, sanitized_host_fps_cap};

    #[test]
    fn toggle_imgui_visibility_updates_memory_and_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        let state = RuntimeConfigState::new(Arc::clone(&settings), path.clone());

        assert_eq!(state.toggle_imgui_visibility(), Some(false));
        assert!(
            !settings
                .read()
                .expect("settings read")
                .debug
                .hud
                .imgui_visible
        );

        let text = std::fs::read_to_string(path).expect("read saved config");
        let saved: RendererSettings = toml::from_str(&text).expect("decode saved config");
        assert!(!saved.debug.hud.imgui_visible);
    }

    #[test]
    fn toggle_imgui_visibility_respects_disk_write_suppression() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        let mut state = RuntimeConfigState::new(Arc::clone(&settings), path.clone());
        state.set_suppress_renderer_config_disk_writes(true);

        assert_eq!(state.toggle_imgui_visibility(), Some(false));
        assert!(
            !settings
                .read()
                .expect("settings read")
                .debug
                .hud
                .imgui_visible
        );
        assert!(!path.exists());
    }

    #[test]
    fn toggle_imgui_visibility_keeps_memory_change_when_save_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config-as-dir");
        std::fs::create_dir_all(&path).expect("create config dir");
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        let state = RuntimeConfigState::new(Arc::clone(&settings), path);

        assert_eq!(state.toggle_imgui_visibility(), Some(false));
        assert!(
            !settings
                .read()
                .expect("settings read")
                .debug
                .hud
                .imgui_visible
        );
    }

    #[test]
    fn host_desktop_caps_override_positive_values_without_mutating_settings() {
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        {
            let mut settings = settings.write().expect("settings write");
            settings.display.focused_fps_cap = 144;
            settings.display.unfocused_fps_cap = 30;
        }
        let mut state = RuntimeConfigState::new(Arc::clone(&settings), PathBuf::new());

        state.apply_host_desktop_config(&DesktopConfig {
            maximum_foreground_framerate: Some(90),
            maximum_background_framerate: Some(1),
            v_sync: false,
        });

        assert_eq!(
            state.desktop_frame_pacing_caps(),
            DesktopFramePacingCaps {
                foreground_fps_cap: 90,
                background_fps_cap: 5,
            }
        );
        let (focused_fps_cap, unfocused_fps_cap) = {
            let settings = settings.read().expect("settings read");
            (
                settings.display.focused_fps_cap,
                settings.display.unfocused_fps_cap,
            )
        };
        assert_eq!(focused_fps_cap, 144);
        assert_eq!(unfocused_fps_cap, 30);
    }

    #[test]
    fn host_desktop_caps_ignore_none_zero_and_negative_values() {
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        {
            let mut settings = settings.write().expect("settings write");
            settings.display.focused_fps_cap = 240;
            settings.display.unfocused_fps_cap = 60;
        }
        let mut state = RuntimeConfigState::new(settings, PathBuf::new());

        state.apply_host_desktop_config(&DesktopConfig {
            maximum_foreground_framerate: Some(0),
            maximum_background_framerate: Some(-10),
            v_sync: false,
        });

        assert_eq!(
            state.desktop_frame_pacing_caps(),
            DesktopFramePacingCaps {
                foreground_fps_cap: 240,
                background_fps_cap: 60,
            }
        );
    }

    #[test]
    fn host_desktop_cap_sanitizer_matches_unity_minimum() {
        assert_eq!(sanitized_host_fps_cap(None), None);
        assert_eq!(sanitized_host_fps_cap(Some(0)), None);
        assert_eq!(sanitized_host_fps_cap(Some(-1)), None);
        assert_eq!(sanitized_host_fps_cap(Some(1)), Some(5));
        assert_eq!(sanitized_host_fps_cap(Some(30)), Some(30));
    }

    #[test]
    fn defaults_to_unlimited_skinning_before_host_config() {
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        let state = RuntimeConfigState::new(settings, PathBuf::new());

        assert_eq!(state.skin_weight_mode(), SkinWeightMode::Unlimited);
    }

    #[test]
    fn quality_config_updates_skin_weight_mode() {
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        let mut state = RuntimeConfigState::new(settings, PathBuf::new());

        state.apply_host_quality_config(&QualityConfig {
            skin_weight_mode: SkinWeightMode::TwoBones,
            ..Default::default()
        });

        assert_eq!(state.skin_weight_mode(), SkinWeightMode::TwoBones);
    }
}
