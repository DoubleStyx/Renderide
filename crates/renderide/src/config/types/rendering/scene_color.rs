//! Intermediate scene color format for the forward pass (pre-compose, pre-post-processing).

use crate::labeled_enum;
use wgpu::TextureFormat;

labeled_enum! {
    /// Intermediate scene color format for the forward pass (pre-compose, pre-post-processing).
    ///
    /// Persist tokens match the on-disk format that the original `#[serde(rename_all =
    /// "snake_case")]` derive produced — serde inserts an underscore before each internal
    /// uppercase letter, so `Rgba16Float` writes as `"rgba16_float"`. The compact-suffix form
    /// (`"rgba16float"`) is accepted as an alias for forgiveness.
    pub enum SceneColorFormat: "scene-color format" {
        default => Rgba16Float;

        /// `rgba16_float`: wide dynamic range and alpha (default HDR scene target).
        Rgba16Float => {
            persist: "rgba16_float",
            label: "RGBA16Float (HDR scene)",
            aliases: ["rgba16float"],
        },
        /// `rg11b10_float`: lower bandwidth; no distinct alpha channel (avoid with premultiplied
        /// transparency).
        Rg11b10Float => {
            persist: "rg11b10_float",
            label: "RG11B10Float (packed HDR)",
            aliases: ["rg11b10float"],
        },
        /// `rgba8_unorm`: LDR scene color (debug / parity).
        Rgba8Unorm => {
            persist: "rgba8_unorm",
            label: "RGBA8 UNORM (LDR scene)",
            aliases: ["rgba8unorm"],
        },
    }
}

impl SceneColorFormat {
    /// [`wgpu::TextureFormat`] for graph transients and forward color attachments.
    pub fn wgpu_format(self) -> TextureFormat {
        match self {
            Self::Rgba16Float => TextureFormat::Rgba16Float,
            Self::Rg11b10Float => TextureFormat::Rg11b10Ufloat,
            Self::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SceneColorFormat;
    use crate::config::types::RendererSettings;
    use wgpu::TextureFormat;

    #[test]
    fn scene_color_format_wgpu_mapping() {
        assert_eq!(
            SceneColorFormat::Rgba16Float.wgpu_format(),
            TextureFormat::Rgba16Float
        );
        assert_eq!(
            SceneColorFormat::Rg11b10Float.wgpu_format(),
            TextureFormat::Rg11b10Ufloat
        );
        assert_eq!(
            SceneColorFormat::Rgba8Unorm.wgpu_format(),
            TextureFormat::Rgba8Unorm
        );
    }

    #[test]
    fn scene_color_format_all_covers_every_variant() {
        for v in SceneColorFormat::ALL.iter().copied() {
            let _ = v.wgpu_format();
            assert!(!v.label().is_empty());
        }
    }

    #[test]
    fn scene_color_format_toml_roundtrip() {
        let mut s = RendererSettings::default();
        s.rendering.scene_color_format = SceneColorFormat::Rg11b10Float;
        let toml = toml::to_string(&s).expect("serialize");
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert_eq!(
            back.rendering.scene_color_format,
            SceneColorFormat::Rg11b10Float
        );
    }
}
