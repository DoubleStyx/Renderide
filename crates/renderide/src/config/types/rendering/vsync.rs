//! Swapchain vsync mode (`[rendering] vsync`).

use crate::labeled_enum;

labeled_enum! {
    /// Swapchain vsync mode persisted in `config.toml` as `[rendering] vsync`.
    ///
    /// Three values matching what desktop and VR titles typically expose: **Off** (tearing,
    /// lowest latency), **On** (no tearing, low latency -- prefers `Mailbox` over `Fifo`),
    /// **Auto** (vsync when the renderer hits the deadline, tear instead of stutter when it
    /// misses -- `FifoRelaxed`). Defaults to [`Self::Off`].
    ///
    /// Resolution to a [`wgpu::PresentMode`] happens in [`VsyncMode::resolve_present_mode`],
    /// which probes the surface's actual capabilities rather than trusting wgpu's `Auto*`
    /// shortcuts (those always pick `Fifo` for vsync-on, leaving no-tearing behind a deeper
    /// compositor queue than necessary).
    ///
    /// The bool-shape branch lets the historical `vsync = true / false` syntax keep loading
    /// without manual migration; the alias list further covers the pre-rename
    /// `vsync = "adaptive"` / `"fifo_relaxed"` tokens.
    pub enum VsyncMode: "vsync mode (`off` / `on` / `auto`)" {
        default    => Off;
        bool_true  => On;
        bool_false => Off;

        /// No vsync. Lowest latency, may tear; CPU/GPU run uncapped. Resolves to `Immediate`
        /// when the surface advertises it, otherwise falls through `Mailbox` and finally `Fifo`.
        Off => {
            persist: "off",
            label: "Off",
            aliases: ["false", "0", "no", "none"],
        },
        /// Vsync without tearing. Resolves to `Mailbox` when the surface advertises it
        /// (low-latency no-tear presentation), otherwise falls back to `Fifo`. Prefer this
        /// over the deprecated `wgpu::PresentMode::AutoVsync` mapping which always picks `Fifo`.
        On => {
            persist: "on",
            label: "On",
            aliases: ["true", "1", "yes", "vsync", "fifo"],
        },
        /// Adaptive vsync. Resolves to `FifoRelaxed` when supported (vsync until a frame
        /// misses its deadline, then tears once instead of waiting another full vblank),
        /// otherwise falls back to `Fifo`.
        Auto => {
            persist: "auto",
            label: "Auto",
            aliases: ["adaptive", "fifo_relaxed", "fiforelaxed", "relaxed"],
        },
    }
}

impl VsyncMode {
    /// Resolves this mode to a [`wgpu::PresentMode`] that the surface actually supports, using
    /// explicit low-latency preference chains rather than wgpu's lazy `Auto*` shortcuts.
    ///
    /// Each variant walks an ordered preference list and picks the first entry present in
    /// `supported` ([`wgpu::SurfaceCapabilities::present_modes`]). [`wgpu::PresentMode::Fifo`]
    /// is required to be supported by every conformant surface ([wgpu spec][1]), so the chain
    /// always terminates.
    ///
    /// | Variant            | Preference order                            | Behavior                                                          |
    /// | ------------------ | ------------------------------------------- | ----------------------------------------------------------------- |
    /// | [`Self::Off`]      | `Immediate` -> `Mailbox` -> `Fifo`            | Lowest latency; tears                                             |
    /// | [`Self::On`]       | `Mailbox` -> `Fifo`                          | No-tear vsync without the FIFO queue depth                        |
    /// | [`Self::Auto`]     | `FifoRelaxed` -> `Fifo`                      | Vsync until a frame misses; then tear once instead of half-rate   |
    ///
    /// Unlike `wgpu::PresentMode::AutoVsync` (which always resolves to plain `Fifo`) the
    /// [`Self::On`] arm probes for `Mailbox` first, which avoids the extra queueing on desktop
    /// backends that expose it while retaining a mandatory `Fifo` fallback.
    ///
    /// [1]: https://www.w3.org/TR/webgpu/#dom-gpupresentmode-fifo
    pub fn resolve_present_mode(self, supported: &[wgpu::PresentMode]) -> wgpu::PresentMode {
        use wgpu::PresentMode::*;
        match self {
            Self::Off => first_supported_present_mode(&[Immediate, Mailbox, Fifo], supported),
            Self::On => first_supported_present_mode(&[Mailbox, Fifo], supported),
            Self::Auto => first_supported_present_mode(&[FifoRelaxed, Fifo], supported),
        }
    }
}

/// Walks `preferred` in order and returns the first variant present in `supported`, falling
/// back to [`wgpu::PresentMode::Fifo`] when nothing matches.
///
/// `Fifo` is the unconditional fallback because every conformant surface advertises it; see
/// [`VsyncMode::resolve_present_mode`] for the per-mode preference chains that route through here.
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
    use super::VsyncMode;
    use crate::config::types::RendererSettings;
    use wgpu::PresentMode;

    #[test]
    fn off_prefers_immediate_when_supported() {
        let supported = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::Fifo,
        ];
        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&supported),
            PresentMode::Immediate
        );
    }

    #[test]
    fn modes_choose_preferred_modes_when_everything_is_supported() {
        let supported = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ];

        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&supported),
            PresentMode::Immediate
        );
        assert_eq!(
            VsyncMode::On.resolve_present_mode(&supported),
            PresentMode::Mailbox
        );
        assert_eq!(
            VsyncMode::Auto.resolve_present_mode(&supported),
            PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn off_falls_through_to_mailbox_then_fifo() {
        let mailbox_only = [PresentMode::Mailbox, PresentMode::Fifo];
        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&mailbox_only),
            PresentMode::Mailbox
        );
        let fifo_only = [PresentMode::Fifo];
        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&fifo_only),
            PresentMode::Fifo
        );
    }

    #[test]
    fn on_prefers_mailbox_over_fifo() {
        let supported = [PresentMode::Mailbox, PresentMode::Fifo];
        assert_eq!(
            VsyncMode::On.resolve_present_mode(&supported),
            PresentMode::Mailbox
        );
    }

    #[test]
    fn on_falls_back_to_fifo_when_mailbox_missing() {
        let no_mailbox = [PresentMode::Fifo, PresentMode::FifoRelaxed];
        assert_eq!(
            VsyncMode::On.resolve_present_mode(&no_mailbox),
            PresentMode::Fifo
        );
    }

    #[test]
    fn auto_prefers_fifo_relaxed_when_supported() {
        let supported = [PresentMode::Fifo, PresentMode::FifoRelaxed];
        assert_eq!(
            VsyncMode::Auto.resolve_present_mode(&supported),
            PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn auto_falls_back_to_fifo_when_relaxed_missing() {
        let fifo_only = [PresentMode::Fifo];
        assert_eq!(
            VsyncMode::Auto.resolve_present_mode(&fifo_only),
            PresentMode::Fifo
        );
    }

    #[test]
    fn empty_supported_list_falls_back_to_fifo() {
        for mode in VsyncMode::ALL.iter().copied() {
            assert_eq!(
                mode.resolve_present_mode(&[]),
                PresentMode::Fifo,
                "mode {mode:?} must terminate at Fifo when nothing is advertised"
            );
        }
    }

    #[test]
    fn legacy_adaptive_token_loads_as_auto() {
        let toml = "[rendering]\nvsync = \"adaptive\"\n";
        let parsed: RendererSettings = toml::from_str(toml).expect("legacy adaptive token");
        assert_eq!(parsed.rendering.vsync, VsyncMode::Auto);
    }

    #[test]
    fn legacy_relaxed_aliases_load_as_auto() {
        for token in ["fifo_relaxed", "fiforelaxed", "relaxed"] {
            let toml = format!("[rendering]\nvsync = \"{token}\"\n");
            let parsed: RendererSettings = toml::from_str(&toml).expect("relaxed alias");
            assert_eq!(
                parsed.rendering.vsync,
                VsyncMode::Auto,
                "token `{token}` must map to Auto"
            );
        }
    }

    #[test]
    fn legacy_boolean_shape_loads() {
        let on: RendererSettings =
            toml::from_str("[rendering]\nvsync = true\n").expect("bool true");
        assert_eq!(on.rendering.vsync, VsyncMode::On);
        let off: RendererSettings =
            toml::from_str("[rendering]\nvsync = false\n").expect("bool false");
        assert_eq!(off.rendering.vsync, VsyncMode::Off);
    }

    #[test]
    fn auto_serializes_as_snake_case() {
        let mut s = RendererSettings::default();
        s.rendering.vsync = VsyncMode::Auto;
        let toml = toml::to_string(&s).expect("serialize");
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert_eq!(back.rendering.vsync, VsyncMode::Auto);
        assert!(
            toml.contains("vsync = \"auto\""),
            "expected snake_case `auto` in serialized TOML, got: {toml}"
        );
    }
}
