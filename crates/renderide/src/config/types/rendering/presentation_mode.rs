//! Swapchain presentation mode (`[rendering] presentation_mode`).

use crate::labeled_enum;

labeled_enum! {
    /// Swapchain presentation mode persisted in `config.toml` as `[rendering] presentation_mode`.
    ///
    /// Values mirror [`wgpu::PresentMode`] while keeping renderer-owned fallback behavior for
    /// explicit modes that are not supported by the active surface. Defaults to [`Self::Immediate`]
    /// to preserve Renderide's previous low-latency desktop behavior.
    pub enum PresentationModeSetting: "presentation mode (`auto_vsync` / `auto_no_vsync` / `fifo` / `fifo_relaxed` / `immediate` / `mailbox`)" {
        default => Immediate;

        /// Lets wgpu choose `FifoRelaxed` when supported and `Fifo` otherwise.
        AutoVsync => {
            persist: "auto_vsync",
            label: "Auto VSync",
            aliases: ["auto-vsync", "autovsync"],
        },
        /// Lets wgpu choose `Immediate`, then `Mailbox`, then `Fifo`.
        AutoNoVsync => {
            persist: "auto_no_vsync",
            label: "Auto no VSync",
            aliases: ["auto-no-vsync", "autonovsync"],
        },
        /// FIFO vblank presentation. This is the traditional VSync option.
        Fifo => {
            persist: "fifo",
            label: "FIFO (traditional VSync)",
        },
        /// Adaptive FIFO presentation, falling back to FIFO when unsupported.
        FifoRelaxed => {
            persist: "fifo_relaxed",
            label: "FIFO relaxed",
            aliases: ["fifo-relaxed", "fiforelaxed"],
        },
        /// Immediate presentation. This is the traditional no-VSync option.
        Immediate => {
            persist: "immediate",
            label: "Immediate (traditional no VSync)",
        },
        /// Mailbox presentation, falling back to `Fifo` when unsupported.
        Mailbox => {
            persist: "mailbox",
            label: "Mailbox",
        },
    }
}

impl PresentationModeSetting {
    /// Resolves this setting to a [`wgpu::PresentMode`] accepted by the active surface.
    ///
    /// Wgpu's `Auto*` modes and `Fifo` are globally valid, so those are passed through directly.
    /// Other explicit modes are selected only when advertised by
    /// [`wgpu::SurfaceCapabilities::present_modes`], with `Fifo` as the guaranteed fallback.
    pub fn resolve_present_mode(self, supported: &[wgpu::PresentMode]) -> wgpu::PresentMode {
        use wgpu::PresentMode::{AutoNoVsync, AutoVsync, Fifo, FifoRelaxed, Immediate, Mailbox};
        match self {
            Self::AutoVsync => AutoVsync,
            Self::AutoNoVsync => AutoNoVsync,
            Self::Fifo => Fifo,
            Self::FifoRelaxed => first_supported_present_mode(&[FifoRelaxed, Fifo], supported),
            Self::Immediate => first_supported_present_mode(&[Immediate, Mailbox, Fifo], supported),
            Self::Mailbox => first_supported_present_mode(&[Mailbox, Fifo], supported),
        }
    }

    /// Whether this mode should let presentation pacing drive desktop redraw cadence.
    pub fn is_presentation_paced(self) -> bool {
        match self {
            Self::AutoVsync | Self::Fifo | Self::FifoRelaxed | Self::Mailbox => true,
            Self::AutoNoVsync | Self::Immediate => false,
        }
    }
}

/// Walks `preferred` in order and returns the first variant present in `supported`, falling back
/// to [`wgpu::PresentMode::Fifo`] when nothing matches.
fn first_supported_present_mode(
    preferred: &[wgpu::PresentMode],
    supported: &[wgpu::PresentMode],
) -> wgpu::PresentMode {
    preferred
        .iter()
        .copied()
        .find(|m| supported.contains(m))
        .unwrap_or(wgpu::PresentMode::Fifo)
}

#[cfg(test)]
mod tests {
    use super::PresentationModeSetting;
    use crate::config::types::RendererSettings;
    use wgpu::PresentMode;

    #[test]
    fn default_is_immediate() {
        assert_eq!(
            PresentationModeSetting::default(),
            PresentationModeSetting::Immediate
        );
    }

    #[test]
    fn explicit_modes_choose_supported_present_modes() {
        let supported = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ];

        assert_eq!(
            PresentationModeSetting::Fifo.resolve_present_mode(&supported),
            PresentMode::Fifo
        );
        assert_eq!(
            PresentationModeSetting::FifoRelaxed.resolve_present_mode(&supported),
            PresentMode::FifoRelaxed
        );
        assert_eq!(
            PresentationModeSetting::Immediate.resolve_present_mode(&supported),
            PresentMode::Immediate
        );
        assert_eq!(
            PresentationModeSetting::Mailbox.resolve_present_mode(&supported),
            PresentMode::Mailbox
        );
    }

    #[test]
    fn auto_modes_pass_through_without_capability_lookup() {
        assert_eq!(
            PresentationModeSetting::AutoVsync.resolve_present_mode(&[]),
            PresentMode::AutoVsync
        );
        assert_eq!(
            PresentationModeSetting::AutoNoVsync.resolve_present_mode(&[]),
            PresentMode::AutoNoVsync
        );
    }

    #[test]
    fn immediate_falls_through_to_mailbox_then_fifo() {
        let mailbox_only = [PresentMode::Mailbox, PresentMode::Fifo];
        assert_eq!(
            PresentationModeSetting::Immediate.resolve_present_mode(&mailbox_only),
            PresentMode::Mailbox
        );
        let fifo_only = [PresentMode::Fifo];
        assert_eq!(
            PresentationModeSetting::Immediate.resolve_present_mode(&fifo_only),
            PresentMode::Fifo
        );
    }

    #[test]
    fn optional_vblank_modes_fall_back_to_fifo() {
        let fifo_only = [PresentMode::Fifo];
        assert_eq!(
            PresentationModeSetting::FifoRelaxed.resolve_present_mode(&fifo_only),
            PresentMode::Fifo
        );
        assert_eq!(
            PresentationModeSetting::Mailbox.resolve_present_mode(&fifo_only),
            PresentMode::Fifo
        );
    }

    #[test]
    fn empty_supported_list_falls_back_to_fifo_for_explicit_surface_modes() {
        for mode in [
            PresentationModeSetting::Fifo,
            PresentationModeSetting::FifoRelaxed,
            PresentationModeSetting::Immediate,
            PresentationModeSetting::Mailbox,
        ] {
            assert_eq!(
                mode.resolve_present_mode(&[]),
                PresentMode::Fifo,
                "mode {mode:?} must terminate at Fifo when nothing is advertised"
            );
        }
    }

    #[test]
    fn presentation_paced_modes_are_identified_for_redraw_planning() {
        for mode in [
            PresentationModeSetting::AutoVsync,
            PresentationModeSetting::Fifo,
            PresentationModeSetting::FifoRelaxed,
            PresentationModeSetting::Mailbox,
        ] {
            assert!(
                mode.is_presentation_paced(),
                "{mode:?} should let presentation pace redraws"
            );
        }
        for mode in [
            PresentationModeSetting::AutoNoVsync,
            PresentationModeSetting::Immediate,
        ] {
            assert!(
                !mode.is_presentation_paced(),
                "{mode:?} should use renderer FPS caps"
            );
        }
    }

    #[test]
    fn presentation_mode_tokens_load() {
        for &mode in PresentationModeSetting::ALL {
            let toml = format!(
                "[rendering]\npresentation_mode = \"{}\"\n",
                mode.persist_str()
            );
            let parsed: RendererSettings = toml::from_str(&toml).expect("presentation mode token");
            assert_eq!(parsed.rendering.presentation_mode, mode);
        }
    }

    #[test]
    fn old_vsync_key_is_ignored() {
        let parsed: RendererSettings =
            toml::from_str("[rendering]\nvsync = true\n").expect("unknown old key ignored");
        assert_eq!(
            parsed.rendering.presentation_mode,
            PresentationModeSetting::Immediate
        );
    }

    #[test]
    fn presentation_mode_serializes_under_new_key() {
        let mut s = RendererSettings::default();
        s.rendering.presentation_mode = PresentationModeSetting::Fifo;
        let toml = toml::to_string(&s).expect("serialize");
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert_eq!(
            back.rendering.presentation_mode,
            PresentationModeSetting::Fifo
        );
        assert!(
            toml.contains("presentation_mode = \"fifo\""),
            "expected `presentation_mode` in serialized TOML, got: {toml}"
        );
        assert!(
            !toml.contains("vsync"),
            "old `vsync` key must not be emitted, got: {toml}"
        );
    }
}
