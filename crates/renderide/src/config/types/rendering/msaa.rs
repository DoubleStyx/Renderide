//! MSAA sample count for the main desktop swapchain forward path.

use crate::labeled_enum;

labeled_enum! {
    /// MSAA sample count for the main desktop swapchain forward path
    /// ([`super::RenderingSettings::msaa`]).
    ///
    /// Tiers stop at **8x**; higher modes are not exposed (and are rarely supported for common
    /// surface formats on desktop GPUs). Older `msaa = "x16"` configs continue to load as
    /// [`Self::X8`] via the alias list.
    pub enum MsaaSampleCount: "MSAA sample count" {
        default => Off;

        /// No multisampling (`sample_count` 1).
        Off => {
            persist: "off",
            label: "1x (off)",
            aliases: ["1", "1x", "none"],
        },
        /// 2x MSAA.
        X2 => {
            persist: "x2",
            label: "2x",
            aliases: ["2", "2x"],
        },
        /// 4x MSAA.
        X4 => {
            persist: "x4",
            label: "4x",
            aliases: ["4", "4x"],
        },
        /// 8x MSAA (largest tier in settings; the GPU may still cap lower).
        X8 => {
            persist: "x8",
            label: "8x",
            aliases: ["8", "8x", "x16", "16", "16x"],
        },
    }
}

impl MsaaSampleCount {
    /// Requested [`wgpu::RenderPipeline`] / attachment sample count (`1` = off).
    pub fn as_count(self) -> u32 {
        match self {
            Self::Off => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }

    /// Stable string for TOML / UI (`off`, `x2`, ...). Historical alias for [`Self::persist_str`].
    pub fn as_persist_str(self) -> &'static str {
        self.persist_str()
    }

    /// Parses case-insensitive persisted or UI tokens. Historical alias for
    /// [`Self::parse_persist`].
    pub fn from_persist_str(s: &str) -> Option<Self> {
        Self::parse_persist(s)
    }
}

#[cfg(test)]
mod tests {
    use super::MsaaSampleCount;
    use crate::config::types::RendererSettings;

    #[test]
    fn msaa_sample_count_from_persist_str_aliases_and_counts() {
        assert_eq!(
            MsaaSampleCount::from_persist_str("off"),
            Some(MsaaSampleCount::Off)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("1x"),
            Some(MsaaSampleCount::Off)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("X2"),
            Some(MsaaSampleCount::X2)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("4"),
            Some(MsaaSampleCount::X4)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("x16"),
            Some(MsaaSampleCount::X8)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("16x"),
            Some(MsaaSampleCount::X8)
        );
        assert_eq!(MsaaSampleCount::from_persist_str("bogus"), None);

        assert_eq!(MsaaSampleCount::Off.as_count(), 1);
        assert_eq!(MsaaSampleCount::X2.as_count(), 2);
        assert_eq!(MsaaSampleCount::X4.as_count(), 4);
        assert_eq!(MsaaSampleCount::X8.as_count(), 8);
        assert_eq!(MsaaSampleCount::X8.as_persist_str(), "x8");
    }

    #[test]
    fn msaa_all_variants_persist_str_round_trip() {
        for v in MsaaSampleCount::ALL.iter().copied() {
            let s = v.as_persist_str();
            assert_eq!(
                MsaaSampleCount::from_persist_str(s),
                Some(v),
                "round-trip failed for {s}"
            );
        }
    }

    #[test]
    fn msaa_toml_round_trip_all_variants() {
        for v in MsaaSampleCount::ALL.iter().copied() {
            let mut s = RendererSettings::default();
            s.rendering.msaa = v;
            let toml = toml::to_string(&s).expect("serialize");
            let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
            assert_eq!(back.rendering.msaa, v);
        }
    }

    #[test]
    fn msaa_labels_are_non_empty() {
        for v in MsaaSampleCount::ALL.iter().copied() {
            assert!(!v.label().is_empty());
        }
    }
}
